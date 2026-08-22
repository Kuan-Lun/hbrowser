"""Chrome for Testing 自動下載管理器"""

import errno
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlopen

from ..utils import (
    Deadline,
    get_chrome_executable_name,
    get_platform,
    setup_logger,
)

logger = setup_logger(__name__)

CHROME_FOR_TESTING_API = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)
_NETWORK_STALL_TIMEOUT_SECONDS = 5.0
_DEFAULT_INSTALL_DEADLINE_SECONDS = 300.0
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_CACHE_LOCK_FILENAME = ".hbrowser-install.lock"
_CACHE_LOCK_POLL_SECONDS = 0.05
_INSTALL_MARKER_FILENAME = ".hbrowser-install-complete.json"
_INSTALL_MARKER_SCHEMA = 1
_INSTALL_STAGING_PREFIX = ".hbrowser-chrome-staging-"


class ChromePaths(NamedTuple):
    """Chrome 的執行檔路徑"""

    chrome: str
    version: str


def _get_cache_dir() -> Path:
    """
    取得快取目錄路徑（~/.cache/chrome-for-testing）

    使用使用者家目錄下的 .cache，避免放在可能受雲端同步
    （iCloud、SynologyDrive 等）管理的路徑下，
    因為 macOS file provider 會導致 Chrome 執行檔無法正常啟動。

    Returns:
        快取目錄的 Path 物件
    """
    cache_dir = Path.home() / ".cache" / "chrome-for-testing"
    return cache_dir


def create_chrome_install_staging_root() -> Path:
    """Create one parent-owned staging generation beside the Chrome cache."""

    cache_dir = _get_cache_dir().absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=_INSTALL_STAGING_PREFIX,
            dir=cache_dir,
        )
    ).absolute()


def _validate_chrome_install_staging_root(
    staging_root: Path,
    *,
    cache_dir: Path,
) -> Path:
    candidate = staging_root.absolute()
    if candidate.parent != cache_dir.absolute() or not candidate.name.startswith(
        _INSTALL_STAGING_PREFIX
    ):
        raise ValueError("Chrome staging root is outside the owned cache namespace")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        raise ValueError("Chrome staging root does not exist") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Chrome staging root is not an owned directory")
    if any(candidate.iterdir()):
        raise ValueError("Chrome staging root must be empty before installation")
    return candidate


def _remove_legacy_staging_roots(
    cache_dir: Path,
    *,
    version: str,
    deadline: Deadline,
) -> None:
    """Remove pre-owner staging left by older workers while holding the cache lock."""

    for candidate in cache_dir.glob(f".{version}.staging-*"):
        _require_install_budget(deadline, "legacy staging cleanup")
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            logger.warning("Ignoring non-directory Chrome staging path: %s", candidate)
            continue
        shutil.rmtree(candidate)
        _require_install_budget(deadline, "legacy staging cleanup")


def _network_timeout(deadline: Deadline) -> float:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise TimeoutError("Chrome installation exceeded its semantic deadline")
    return min(_NETWORK_STALL_TIMEOUT_SECONDS, remaining)


def _require_install_budget(deadline: Deadline, phase: str) -> float:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise TimeoutError(
            f"Chrome installation exceeded its semantic deadline during {phase}"
        )
    return remaining


def _lock_cache_descriptor(descriptor: int, *, deadline: Deadline) -> None:
    if os.name == "posix":
        import fcntl

        while True:
            remaining = _require_install_budget(deadline, "cache lock acquisition")
            try:
                fcntl.lockf(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                    0,
                    0,
                    os.SEEK_SET,
                )
                return
            except BlockingIOError:
                time.sleep(min(_CACHE_LOCK_POLL_SECONDS, remaining))
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            remaining = _require_install_budget(deadline, "cache lock acquisition")
            try:
                getattr(msvcrt, "locking")(
                    descriptor,
                    getattr(msvcrt, "LK_NBLCK"),
                    1,
                )
                return
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EDEADLK}:
                    raise
                time.sleep(min(_CACHE_LOCK_POLL_SECONDS, remaining))
    raise RuntimeError(f"Unsupported Chrome cache lock platform: {os.name}")


def _unlock_cache_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.lockf(descriptor, fcntl.LOCK_UN, 0, 0, os.SEEK_SET)
        return
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_UNLCK"), 1)
        return
    raise RuntimeError(f"Unsupported Chrome cache lock platform: {os.name}")


@contextmanager
def _locked_chrome_cache(cache_dir: Path, *, deadline: Deadline) -> Iterator[None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / _CACHE_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        _lock_cache_descriptor(descriptor, deadline=deadline)
        locked = True
        _require_install_budget(deadline, "cache lock receipt")
        yield
    finally:
        if locked:
            _unlock_cache_descriptor(descriptor)
        os.close(descriptor)


def _installation_is_complete(
    version_dir: Path,
    chrome_path: Path,
    *,
    version: str,
) -> bool:
    if not chrome_path.is_file():
        return False
    marker_path = version_dir / _INSTALL_MARKER_FILENAME
    try:
        marker: object = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError, OSError, TypeError, ValueError:
        return False
    return marker == {"schema": _INSTALL_MARKER_SCHEMA, "version": version}


def _write_installation_marker(
    version_dir: Path,
    *,
    version: str,
    deadline: Deadline,
) -> None:
    _require_install_budget(deadline, "installation receipt write")
    marker_path = version_dir / _INSTALL_MARKER_FILENAME
    temporary_marker = version_dir / f".{_INSTALL_MARKER_FILENAME}.{os.getpid()}.tmp"
    temporary_marker.write_text(
        json.dumps(
            {"schema": _INSTALL_MARKER_SCHEMA, "version": version},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _require_install_budget(deadline, "installation receipt publication")
    os.replace(temporary_marker, marker_path)
    _require_install_budget(deadline, "installation receipt publication")


def _read_response_chunk(
    response: Any,
    *,
    deadline: Deadline,
    phase: str,
) -> bytes:
    remaining = _require_install_budget(deadline, phase)
    raw_socket = getattr(
        getattr(getattr(response, "fp", None), "raw", None),
        "_sock",
        None,
    )
    settimeout = getattr(raw_socket, "settimeout", None)
    if callable(settimeout):
        settimeout(min(_NETWORK_STALL_TIMEOUT_SECONDS, remaining))
    # HTTPResponse.read() may wait for its entire requested size while a peer
    # trickles bytes. read1() returns after one buffered/socket read so the
    # absolute deadline is reconciled between chunks.
    read = getattr(response, "read1", response.read)
    chunk: bytes = read(_DOWNLOAD_CHUNK_BYTES)
    if deadline.expired:
        raise TimeoutError(
            f"Chrome installation exceeded its semantic deadline during {phase}"
        )
    return chunk


def _fetch_stable_version_info(deadline: Deadline) -> dict[str, Any]:
    """
    從 Chrome for Testing API 獲取 stable 版本資訊

    Returns:
        包含版本和下載連結的字典
    """
    logger.debug("Fetching Chrome for Testing stable version info")
    with urlopen(
        CHROME_FOR_TESTING_API,
        timeout=_network_timeout(deadline),
    ) as response:
        chunks = list[bytes]()
        while chunk := _read_response_chunk(
            response,
            deadline=deadline,
            phase="metadata download",
        ):
            chunks.append(chunk)
        data: dict[str, Any] = json.loads(b"".join(chunks).decode("utf-8"))

    stable: dict[str, Any] = data["channels"]["Stable"]
    return stable


def _download_and_extract(
    url: str,
    dest_dir: Path,
    desc: str,
    *,
    deadline: Deadline,
) -> None:
    """
    下載 zip 檔案並解壓縮

    macOS 使用 ditto 解壓以正確保留 symlinks 和 .app bundle 結構，
    其他平台使用 Python zipfile。

    下載/解壓在 dest_dir 旁邊的暫存目錄進行，只有完全成功才會把結果移進
    dest_dir；中途失敗時暫存目錄會被整個丟棄，dest_dir 不會留下半成品，
    避免 ensure_chrome_installed 之後把半成品誤判成「已安裝」。

    Args:
        url: 下載連結
        dest_dir: 目標目錄
        desc: 描述（用於日誌）
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=dest_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "temp.zip"

        logger.info(f"Downloading {desc}...")
        logger.debug("Download request started: artifact=%s", desc)
        with (
            urlopen(url, timeout=_network_timeout(deadline)) as response,
            zip_path.open("wb") as output,
        ):
            while chunk := _read_response_chunk(
                response,
                deadline=deadline,
                phase=f"{desc} download",
            ):
                output.write(chunk)

        logger.info(f"Extracting {desc}...")
        if platform.system() == "Darwin":
            try:
                subprocess.run(
                    ["ditto", "-xk", str(zip_path), str(tmp_dir)],
                    check=True,
                    timeout=_require_install_budget(deadline, "extraction"),
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"{desc} extraction exceeded its semantic deadline"
                ) from None
        else:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    if deadline.expired:
                        raise TimeoutError(
                            f"{desc} extraction exceeded its semantic deadline"
                        )
                    zf.extract(member, tmp_dir)
        zip_path.unlink()

        for extracted in tmp_dir.iterdir():
            _require_install_budget(deadline, "artifact installation")
            shutil.move(str(extracted), str(dest_dir / extracted.name))

    logger.debug(f"{desc} extracted to {dest_dir}")


def _make_executable(path: Path) -> None:
    """設定檔案為可執行"""
    if platform.system() != "Windows":
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_all_files_executable(directory: Path, *, deadline: Deadline) -> None:
    """將目錄底下所有檔案都設定為可執行

    只用於 zipfile.extractall 解壓的平台：zipfile 不會保留 Unix 執行權限
    位元，導致 chrome 主執行檔以外的輔助執行檔（例如
    chrome_crashpad_handler）解壓後失去執行權限，在 spawn 該輔助程序時
    直接 FATAL abort。對整個資料夾的檔案統一補上執行權限可以一次涵蓋所有
    現有與未來新增的輔助執行檔，不需要逐一列名。

    macOS 使用 ditto 解壓，會正確保留原始權限位元，不需要（也不應該）
    呼叫此函式，否則會替原本沒有執行權限的資源檔案多加上 +x。
    """
    for path in directory.rglob("*"):
        _require_install_budget(deadline, "executable permission setup")
        if path.is_file():
            _make_executable(path)


def _remove_quarantine(path: Path, *, deadline: Deadline) -> None:
    """移除 macOS 的 quarantine 屬性（僅 macOS）

    使用 -dr 只刪除 com.apple.quarantine，
    而非 -cr 清除所有屬性（會破壞 code signing）。
    """
    if platform.system() != "Darwin":
        return
    try:
        result = subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine", str(path)],
            check=False,
            capture_output=True,
            timeout=_require_install_budget(deadline, "quarantine cleanup"),
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            "Chrome quarantine cleanup exceeded its semantic deadline"
        ) from None
    if result.returncode != 0:
        # 下載下來的檔案本來就常常沒有 quarantine attribute（urlretrieve 不會
        # 像瀏覽器下載一樣加上它），xattr 在這種情況下回非 0 是預期內的，
        # 所以記 debug 而非 warning，但仍保留訊息以便真的有問題時可以查。
        logger.debug(
            f"xattr -dr returned non-zero for {path}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    else:
        logger.debug(f"Removed quarantine attribute: {path}")


def _find_download_url(downloads: list[dict[str, str]], plat: str) -> str | None:
    """從下載列表中找到對應平台的 URL"""
    for item in downloads:
        if item["platform"] == plat:
            return item["url"]
    return None


def ensure_chrome_installed(
    force_download: bool = False,
    *,
    deadline: Deadline | None = None,
    staging_root: Path | None = None,
) -> ChromePaths:
    """
    確保 Chrome 已安裝

    如果快取中已有對應版本，直接回傳路徑。
    否則自動下載最新的 stable 版本。

    Args:
        force_download: 強制重新下載

    Returns:
        ChromePaths 包含 chrome 的執行檔路徑和版本
    """
    install_deadline = (
        Deadline.after(_DEFAULT_INSTALL_DEADLINE_SECONDS)
        if deadline is None
        else deadline.bounded(_DEFAULT_INSTALL_DEADLINE_SECONDS)
    )
    plat = get_platform()
    cache_dir = _get_cache_dir().absolute()
    supplied_staging_root = (
        None
        if staging_root is None
        else _validate_chrome_install_staging_root(
            staging_root,
            cache_dir=cache_dir,
        )
    )

    logger.debug(f"Platform: {plat}")
    logger.debug(f"Cache directory: {cache_dir}")

    # 獲取最新版本資訊
    version_info = _fetch_stable_version_info(install_deadline)
    version = version_info["version"]
    logger.debug(f"Latest stable version: {version}")

    version_dir = cache_dir / version

    # Chrome 路徑
    chrome_folder = f"chrome-{plat}"
    chrome_exe_name = get_chrome_executable_name(plat)
    chrome_path = version_dir / chrome_folder / chrome_exe_name

    with _locked_chrome_cache(cache_dir, deadline=install_deadline):
        _remove_legacy_staging_roots(
            cache_dir,
            version=version,
            deadline=install_deadline,
        )
        backup_dir = cache_dir / f".{version}.previous"
        backup_chrome_path = backup_dir / chrome_folder / chrome_exe_name
        final_complete = _installation_is_complete(
            version_dir,
            chrome_path,
            version=version,
        )
        backup_complete = _installation_is_complete(
            backup_dir,
            backup_chrome_path,
            version=version,
        )

        # A worker killed between final->backup and staging->final leaves the
        # previous complete generation recoverable. Never treat an executable
        # path alone as proof: chmod/xattr may not have completed yet.
        if not final_complete and backup_complete:
            if version_dir.exists():
                _require_install_budget(install_deadline, "invalid cache cleanup")
                shutil.rmtree(version_dir)
            os.replace(backup_dir, version_dir)
            _require_install_budget(install_deadline, "cache recovery")
            final_complete = True

        if final_complete and not force_download:
            if backup_dir.exists():
                _require_install_budget(install_deadline, "stale backup cleanup")
                shutil.rmtree(backup_dir)
                _require_install_budget(install_deadline, "stale backup cleanup")
            logger.debug(f"Using cached Chrome {version}")
        else:
            downloads = version_info["downloads"]
            chrome_url = _find_download_url(downloads["chrome"], plat)
            if not chrome_url:
                raise RuntimeError(f"No Chrome download found for platform: {plat}")

            active_staging_root = (
                create_chrome_install_staging_root()
                if supplied_staging_root is None
                else supplied_staging_root
            )
            staging_version = active_staging_root / version
            staging_chrome_path = staging_version / chrome_folder / chrome_exe_name
            try:
                _download_and_extract(
                    chrome_url,
                    staging_version,
                    "Chrome",
                    deadline=install_deadline,
                )
                if platform.system() != "Darwin":
                    _make_all_files_executable(
                        staging_version / chrome_folder,
                        deadline=install_deadline,
                    )
                else:
                    _require_install_budget(
                        install_deadline,
                        "executable permission setup",
                    )
                    _make_executable(staging_chrome_path)
                _remove_quarantine(
                    staging_version / chrome_folder,
                    deadline=install_deadline,
                )
                if not staging_chrome_path.is_file():
                    raise RuntimeError("Chrome installation produced no executable")
                _write_installation_marker(
                    staging_version,
                    version=version,
                    deadline=install_deadline,
                )
                if not _installation_is_complete(
                    staging_version,
                    staging_chrome_path,
                    version=version,
                ):
                    raise RuntimeError("Chrome staging receipt validation failed")

                _require_install_budget(install_deadline, "cache publication")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                    _require_install_budget(
                        install_deadline,
                        "stale backup cleanup",
                    )
                if version_dir.exists():
                    os.replace(version_dir, backup_dir)
                os.replace(staging_version, version_dir)
                _require_install_budget(install_deadline, "cache publication")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                    _require_install_budget(
                        install_deadline,
                        "old cache cleanup",
                    )
            finally:
                if active_staging_root.exists():
                    shutil.rmtree(active_staging_root)

            logger.info("Chrome is ready")

    _require_install_budget(install_deadline, "final validation")
    return ChromePaths(
        chrome=str(chrome_path),
        version=version,
    )
