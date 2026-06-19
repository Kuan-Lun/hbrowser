"""Chrome for Testing 自動下載管理器"""

import json
import platform
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlopen, urlretrieve

from ..utils import (
    get_chrome_executable_name,
    get_platform,
    setup_logger,
)

logger = setup_logger(__name__)

CHROME_FOR_TESTING_API = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)


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


def _fetch_stable_version_info() -> dict[str, Any]:
    """
    從 Chrome for Testing API 獲取 stable 版本資訊

    Returns:
        包含版本和下載連結的字典
    """
    logger.info("Fetching Chrome for Testing stable version info...")
    with urlopen(CHROME_FOR_TESTING_API, timeout=30) as response:
        data: dict[str, Any] = json.loads(response.read().decode("utf-8"))

    stable: dict[str, Any] = data["channels"]["Stable"]
    return stable


def _download_and_extract(url: str, dest_dir: Path, desc: str) -> None:
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
        logger.debug(f"URL: {url}")
        urlretrieve(url, zip_path)

        logger.info(f"Extracting {desc}...")
        if platform.system() == "Darwin":
            subprocess.run(
                ["ditto", "-xk", str(zip_path), str(tmp_dir)],
                check=True,
            )
        else:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
        zip_path.unlink()

        for extracted in tmp_dir.iterdir():
            shutil.move(str(extracted), str(dest_dir / extracted.name))

    logger.debug(f"{desc} extracted to {dest_dir}")


def _make_executable(path: Path) -> None:
    """設定檔案為可執行"""
    if platform.system() != "Windows":
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _remove_quarantine(path: Path) -> None:
    """移除 macOS 的 quarantine 屬性（僅 macOS）

    使用 -dr 只刪除 com.apple.quarantine，
    而非 -cr 清除所有屬性（會破壞 code signing）。
    """
    if platform.system() != "Darwin":
        return
    result = subprocess.run(
        ["xattr", "-dr", "com.apple.quarantine", str(path)],
        check=False,
        capture_output=True,
    )
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


def ensure_chrome_installed(force_download: bool = False) -> ChromePaths:
    """
    確保 Chrome 已安裝

    如果快取中已有對應版本，直接回傳路徑。
    否則自動下載最新的 stable 版本。

    Args:
        force_download: 強制重新下載

    Returns:
        ChromePaths 包含 chrome 的執行檔路徑和版本
    """
    plat = get_platform()
    cache_dir = _get_cache_dir()

    logger.info(f"Platform: {plat}")
    logger.debug(f"Cache directory: {cache_dir}")

    # 獲取最新版本資訊
    version_info = _fetch_stable_version_info()
    version = version_info["version"]
    logger.info(f"Latest stable version: {version}")

    version_dir = cache_dir / version

    # Chrome 路徑
    chrome_folder = f"chrome-{plat}"
    chrome_exe_name = get_chrome_executable_name(plat)
    chrome_path = version_dir / chrome_folder / chrome_exe_name

    # 檢查是否需要下載
    need_chrome = force_download or not chrome_path.exists()

    if need_chrome:
        if force_download and version_dir.exists():
            logger.info("Force download requested, removing existing cache...")
            shutil.rmtree(version_dir)

        version_dir.mkdir(parents=True, exist_ok=True)

        downloads = version_info["downloads"]

        # 下載 Chrome
        chrome_url = _find_download_url(downloads["chrome"], plat)
        if not chrome_url:
            raise RuntimeError(f"No Chrome download found for platform: {plat}")
        _download_and_extract(chrome_url, version_dir, "Chrome")
        _make_executable(chrome_path)
        _remove_quarantine(version_dir / chrome_folder)

        logger.info("Chrome is ready")
    else:
        logger.info(f"Using cached Chrome {version}")

    return ChromePaths(
        chrome=str(chrome_path),
        version=version,
    )
