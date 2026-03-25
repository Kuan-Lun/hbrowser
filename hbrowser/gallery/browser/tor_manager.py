"""Tor Browser + GeckoDriver 自動下載管理器"""

import atexit
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import Request, urlopen, urlretrieve

from ..utils import setup_logger

logger = setup_logger(__name__)

TOR_DOWNLOADS_API = (
    "https://aus1.torproject.org/torbrowser/update_3/release/downloads.json"
)

GECKODRIVER_RELEASES_API = (
    "https://api.github.com/repos/mozilla/geckodriver/releases/latest"
)


class TorPaths(NamedTuple):
    """Tor Browser 和 GeckoDriver 的執行檔路徑"""

    browser: str
    geckodriver: str
    tor_binary: str
    browser_omni_ja: str
    version: str


def _get_platform() -> str:
    """
    根據作業系統和架構取得對應的平台代碼

    Returns:
        Tor Browser 平台代碼 (linux-x86_64, macos, win64, win32)
    """
    system = platform.system()
    machine = platform.machine().lower()

    match system:
        case "Darwin":
            return "macos"
        case "Linux":
            if machine in ("x86_64", "amd64"):
                return "linux-x86_64"
            else:
                return "linux-i686"
        case "Windows":
            if platform.machine().endswith("64"):
                return "win64"
            else:
                return "win32"
        case _:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")


def _get_geckodriver_platform() -> str:
    """
    取得 geckodriver 對應的平台檔名片段

    Returns:
        geckodriver 平台代碼 (linux64, macos-aarch64, macos, win64 等)
    """
    system = platform.system()
    machine = platform.machine().lower()

    match system:
        case "Darwin":
            if machine == "arm64":
                return "macos-aarch64"
            else:
                return "macos"
        case "Linux":
            if machine in ("x86_64", "amd64"):
                return "linux64"
            elif machine == "aarch64":
                return "linux-aarch64"
            else:
                return "linux32"
        case "Windows":
            if platform.machine().endswith("64"):
                return "win64"
            else:
                return "win32"
        case _:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")


def _get_cache_dir() -> Path:
    """
    取得快取目錄路徑（~/.cache/tor-browser）

    使用使用者家目錄下的 .cache，避免放在可能受雲端同步
    （iCloud、SynologyDrive 等）管理的路徑下。

    Returns:
        快取目錄的 Path 物件
    """
    return Path.home() / ".cache" / "tor-browser"


def _get_tor_browser_binary(plat: str, base_dir: Path) -> str:
    """取得 Tor Browser 中 Firefox 執行檔的相對路徑"""
    if plat.startswith("win"):
        return str(base_dir / "Browser" / "firefox.exe")
    elif plat == "macos":
        return str(base_dir / "Tor Browser.app" / "Contents" / "MacOS" / "firefox")
    else:
        # Linux
        return str(base_dir / "tor-browser" / "Browser" / "firefox")


def _get_tor_binary(plat: str, base_dir: Path) -> str:
    """取得 tor SOCKS proxy 執行檔的路徑"""
    if plat.startswith("win"):
        return str(base_dir / "Browser" / "TorBrowser" / "Tor" / "tor.exe")
    elif plat == "macos":
        return str(base_dir / "Tor Browser.app" / "Contents" / "MacOS" / "Tor" / "tor")
    else:
        # Linux
        return str(base_dir / "tor-browser" / "Browser" / "TorBrowser" / "Tor" / "tor")


def _get_browser_omni_ja(plat: str, base_dir: Path) -> str:
    """取得 Tor Browser 的 browser/omni.ja 路徑（包含預設 profile 設定）"""
    if plat.startswith("win"):
        return str(base_dir / "Browser" / "browser" / "omni.ja")
    elif plat == "macos":
        return str(
            base_dir
            / "Tor Browser.app"
            / "Contents"
            / "Resources"
            / "browser"
            / "omni.ja"
        )
    else:
        # Linux
        return str(base_dir / "tor-browser" / "Browser" / "browser" / "omni.ja")


def _get_geckodriver_executable_name(plat: str) -> str:
    """取得 geckodriver 執行檔名稱"""
    if plat.startswith("win"):
        return "geckodriver.exe"
    else:
        return "geckodriver"


# ---- 版本資訊取得 ----


def _fetch_tor_version_info() -> dict[str, Any]:
    """
    從 Tor Project API 獲取最新 stable 版本資訊

    Returns:
        包含版本和下載連結的字典
    """
    logger.info("Fetching Tor Browser stable version info...")
    with urlopen(TOR_DOWNLOADS_API, timeout=30) as response:
        data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    logger.info(f"Latest Tor Browser version: {data['version']}")
    return data


def _fetch_geckodriver_release_info() -> dict[str, Any]:
    """
    從 GitHub API 獲取最新 geckodriver release 資訊

    Returns:
        包含版本和下載資源的字典
    """
    logger.info("Fetching latest geckodriver release info...")
    req = Request(GECKODRIVER_RELEASES_API)
    req.add_header("Accept", "application/vnd.github.v3+json")
    with urlopen(req, timeout=30) as response:
        data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    logger.info(f"Latest geckodriver version: {data['tag_name']}")
    return data


# ---- 下載與解壓 ----


def _download_file(url: str, dest: Path, desc: str) -> None:
    """下載檔案"""
    logger.info(f"Downloading {desc}...")
    logger.debug(f"URL: {url}")
    urlretrieve(url, dest)
    logger.debug(f"Downloaded to {dest}")


def _extract_tar_xz(archive: Path, dest_dir: Path, desc: str) -> None:
    """解壓 .tar.xz 檔案（Linux Tor Browser）"""
    logger.info(f"Extracting {desc} (.tar.xz)...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(dest_dir)
    archive.unlink()
    logger.debug(f"{desc} extracted to {dest_dir}")


def _extract_dmg(dmg_path: Path, dest_dir: Path, desc: str) -> None:
    """解壓 .dmg 檔案（macOS Tor Browser）"""
    logger.info(f"Extracting {desc} (.dmg)...")
    mount_point = Path(tempfile.mkdtemp(prefix="tor_dmg_"))

    try:
        # 掛載 DMG
        subprocess.run(
            [
                "hdiutil",
                "attach",
                str(dmg_path),
                "-nobrowse",
                "-readonly",
                "-mountpoint",
                str(mount_point),
            ],
            check=True,
            capture_output=True,
        )

        # 複製 Tor Browser.app
        src_app = mount_point / "Tor Browser.app"
        dest_app = dest_dir / "Tor Browser.app"
        if dest_app.exists():
            shutil.rmtree(dest_app)
        shutil.copytree(str(src_app), str(dest_app), symlinks=True)

    finally:
        # 確保卸載 DMG
        subprocess.run(
            ["hdiutil", "detach", str(mount_point)],
            check=False,
            capture_output=True,
        )
        if mount_point.exists():
            shutil.rmtree(mount_point, ignore_errors=True)

    dmg_path.unlink()
    logger.debug(f"{desc} extracted to {dest_dir}")


def _extract_windows_exe(exe_path: Path, dest_dir: Path, desc: str) -> None:
    """解壓 Windows Tor Browser 安裝程式（NSIS 自解壓）"""
    logger.info(f"Extracting {desc} (Windows installer)...")
    subprocess.run(
        [str(exe_path), "/S", f"/D={dest_dir}"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    exe_path.unlink()
    logger.debug(f"{desc} extracted to {dest_dir}")


def _extract_tar_gz(archive: Path, dest_dir: Path, desc: str) -> None:
    """解壓 .tar.gz 檔案（Linux/macOS geckodriver）"""
    logger.info(f"Extracting {desc} (.tar.gz)...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir)
    archive.unlink()
    logger.debug(f"{desc} extracted to {dest_dir}")


def _extract_zip(archive: Path, dest_dir: Path, desc: str) -> None:
    """解壓 .zip 檔案（Windows geckodriver）"""
    logger.info(f"Extracting {desc} (.zip)...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    archive.unlink()
    logger.debug(f"{desc} extracted to {dest_dir}")


# ---- 檔案權限工具 ----


def _make_executable(path: Path) -> None:
    """設定檔案為可執行"""
    if platform.system() != "Windows":
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _remove_quarantine(path: Path) -> None:
    """移除 macOS 的 quarantine 屬性（僅 macOS）"""
    if platform.system() != "Darwin":
        return
    subprocess.run(
        ["xattr", "-dr", "com.apple.quarantine", str(path)],
        check=False,
        capture_output=True,
    )
    logger.debug(f"Removed quarantine attribute: {path}")


# ---- GeckoDriver PID 副本管理 ----


def _is_pid_alive(pid: int) -> bool:
    """檢查指定的 PID 是否仍在運行"""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cleanup_stale_geckodrivers(running_dir: Path) -> None:
    """清理已結束進程留下的 geckodriver 副本"""
    if not running_dir.exists():
        return

    for path in running_dir.iterdir():
        name = path.stem
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue

        try:
            pid = int(parts[1])
        except ValueError:
            continue

        if not _is_pid_alive(pid):
            try:
                path.unlink()
                logger.debug(f"Cleaned up stale geckodriver copy: {path.name}")
            except OSError:
                pass


def _create_geckodriver_copy(source: Path, running_dir: Path, plat: str) -> Path:
    """
    建立 geckodriver 的 PID 專屬副本

    Args:
        source: 原始 geckodriver 路徑
        running_dir: running 目錄路徑
        plat: 平台代碼

    Returns:
        副本的路徑
    """
    running_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    if plat.startswith("win"):
        copy_name = f"geckodriver_{pid}.exe"
    else:
        copy_name = f"geckodriver_{pid}"

    copy_path = running_dir / copy_name
    shutil.copy2(source, copy_path)
    _make_executable(copy_path)

    def _cleanup() -> None:
        try:
            if copy_path.exists():
                copy_path.unlink()
                logger.debug(f"Cleaned up geckodriver copy: {copy_path.name}")
        except OSError:
            pass

    atexit.register(_cleanup)

    logger.debug(f"Created geckodriver copy: {copy_path.name}")
    return copy_path


# ---- Tor Browser 下載 ----


def _download_tor_browser(
    downloads: dict[str, Any], plat: str, version_dir: Path
) -> None:
    """下載並解壓 Tor Browser"""
    platform_downloads: dict[str, Any] = downloads.get(plat, {})
    all_lang: dict[str, str] = platform_downloads.get("ALL", {})
    url: str | None = all_lang.get("binary")

    if not url:
        raise RuntimeError(f"No Tor Browser download found for platform: {plat}")

    archive_path = version_dir / url.rsplit("/", 1)[-1]
    _download_file(url, archive_path, "Tor Browser")

    # 根據平台解壓
    if plat == "macos":
        _extract_dmg(archive_path, version_dir, "Tor Browser")
        _remove_quarantine(version_dir / "Tor Browser.app")
    elif plat.startswith("linux"):
        _extract_tar_xz(archive_path, version_dir, "Tor Browser")
    elif plat.startswith("win"):
        _extract_windows_exe(archive_path, version_dir, "Tor Browser")
    else:
        raise RuntimeError(f"Unsupported platform for extraction: {plat}")


# ---- GeckoDriver 下載 ----


def _download_geckodriver(
    release_info: dict[str, Any], geckodriver_plat: str, version_dir: Path
) -> None:
    """下載並解壓 geckodriver"""
    assets: list[dict[str, Any]] = release_info["assets"]

    # 找到對應平台的 asset
    target_asset: dict[str, str] | None = None
    for asset in assets:
        name: str = asset["name"]
        # 跳過簽名檔
        if name.endswith(".asc"):
            continue
        if geckodriver_plat in name:
            target_asset = asset
            break

    if not target_asset:
        raise RuntimeError(
            f"No geckodriver download found for platform: {geckodriver_plat}"
        )

    url = target_asset["browser_download_url"]
    filename = target_asset["name"]
    archive_path = version_dir / filename
    _download_file(url, archive_path, "geckodriver")

    # 根據檔案類型解壓
    if filename.endswith(".tar.gz"):
        _extract_tar_gz(archive_path, version_dir, "geckodriver")
    elif filename.endswith(".zip"):
        _extract_zip(archive_path, version_dir, "geckodriver")
    else:
        raise RuntimeError(f"Unknown geckodriver archive format: {filename}")


# ---- 主要入口點 ----


def ensure_tor_installed(force_download: bool = False) -> TorPaths:
    """
    確保 Tor Browser 和 GeckoDriver 已安裝

    如果快取中已有對應版本，直接回傳路徑。
    否則自動下載最新的 stable 版本。

    回傳的 geckodriver 路徑為 PID 專屬副本，
    避免多個進程同時使用同一個 geckodriver 執行檔的衝突。

    Args:
        force_download: 強制重新下載

    Returns:
        TorPaths 包含瀏覽器、geckodriver 和 tor 的執行檔路徑
    """
    plat = _get_platform()
    geckodriver_plat = _get_geckodriver_platform()
    cache_dir = _get_cache_dir()

    logger.info(f"Platform: {plat}")
    logger.debug(f"Cache directory: {cache_dir}")

    # ---- Tor Browser ----
    tor_info = _fetch_tor_version_info()
    tor_version = tor_info["version"]
    version_dir = cache_dir / tor_version

    browser_path = Path(_get_tor_browser_binary(plat, version_dir))
    tor_binary_path = Path(_get_tor_binary(plat, version_dir))

    need_tor = force_download or not browser_path.exists()

    if need_tor:
        if force_download and version_dir.exists():
            logger.info("Force download requested, removing existing cache...")
            shutil.rmtree(version_dir)

        version_dir.mkdir(parents=True, exist_ok=True)
        _download_tor_browser(tor_info["downloads"], plat, version_dir)
        _make_executable(browser_path)
        _make_executable(tor_binary_path)
        logger.info("Tor Browser is ready")
    else:
        logger.info(f"Using cached Tor Browser {tor_version}")

    # ---- GeckoDriver ----
    geckodriver_dir = cache_dir / "geckodriver"
    geckodriver_release = _fetch_geckodriver_release_info()
    geckodriver_version = geckodriver_release["tag_name"]
    geckodriver_version_dir = geckodriver_dir / geckodriver_version

    geckodriver_exe = _get_geckodriver_executable_name(plat)
    geckodriver_path = geckodriver_version_dir / geckodriver_exe

    need_geckodriver = force_download or not geckodriver_path.exists()

    if need_geckodriver:
        if force_download and geckodriver_version_dir.exists():
            shutil.rmtree(geckodriver_version_dir)

        geckodriver_version_dir.mkdir(parents=True, exist_ok=True)
        _download_geckodriver(
            geckodriver_release, geckodriver_plat, geckodriver_version_dir
        )
        _make_executable(geckodriver_path)
        _remove_quarantine(geckodriver_path)
        logger.info("GeckoDriver is ready")
    else:
        logger.info(f"Using cached geckodriver {geckodriver_version}")

    # 清理已結束進程的 geckodriver 副本
    running_dir = cache_dir / "running"
    _cleanup_stale_geckodrivers(running_dir)

    # 建立 PID 專屬的 geckodriver 副本
    geckodriver_copy = _create_geckodriver_copy(geckodriver_path, running_dir, plat)

    omni_ja_path = Path(_get_browser_omni_ja(plat, version_dir))

    return TorPaths(
        browser=str(browser_path),
        geckodriver=str(geckodriver_copy),
        tor_binary=str(tor_binary_path),
        browser_omni_ja=str(omni_ja_path),
        version=tor_version,
    )
