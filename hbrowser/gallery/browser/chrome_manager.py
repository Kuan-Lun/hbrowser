"""Chrome for Testing 自動下載管理器"""

import atexit
import json
import os
import platform
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlopen, urlretrieve

from ..utils import setup_logger

logger = setup_logger(__name__)

CHROME_FOR_TESTING_API = (
    "https://googlechromelabs.github.io/chrome-for-testing/"
    "last-known-good-versions-with-downloads.json"
)


class ChromePaths(NamedTuple):
    """Chrome 和 ChromeDriver 的執行檔路徑"""

    chrome: str
    chromedriver: str
    version: str


def _get_platform() -> str:
    """
    根據作業系統和架構取得對應的平台代碼

    Returns:
        平台代碼 (mac-arm64, mac-x64, linux64, win64, win32)
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        # macOS
        if machine == "arm64":
            return "mac-arm64"
        else:
            return "mac-x64"
    elif system == "Linux":
        return "linux64"
    elif system == "Windows":
        # 檢查是否為 64 位元
        if platform.machine().endswith("64"):
            return "win64"
        else:
            return "win32"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


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


def _get_chrome_executable_name(plat: str) -> str:
    """取得 Chrome 執行檔名稱"""
    if plat.startswith("win"):
        return "chrome.exe"
    elif plat.startswith("mac"):
        return "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    else:
        return "chrome"


def _get_chromedriver_executable_name(plat: str) -> str:
    """取得 ChromeDriver 執行檔名稱"""
    if plat.startswith("win"):
        return "chromedriver.exe"
    else:
        return "chromedriver"


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

    Args:
        url: 下載連結
        dest_dir: 目標目錄
        desc: 描述（用於日誌）
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "temp.zip"

    logger.info(f"Downloading {desc}...")
    logger.debug(f"URL: {url}")

    urlretrieve(url, zip_path)

    logger.info(f"Extracting {desc}...")
    if platform.system() == "Darwin":
        subprocess.run(
            ["ditto", "-xk", str(zip_path), str(dest_dir)],
            check=True,
        )
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

    # 刪除 zip 檔案
    zip_path.unlink()
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
    subprocess.run(
        ["xattr", "-dr", "com.apple.quarantine", str(path)],
        check=False,
        capture_output=True,
    )
    logger.debug(f"Removed quarantine attribute: {path}")


def _find_download_url(downloads: list[dict[str, str]], plat: str) -> str | None:
    """從下載列表中找到對應平台的 URL"""
    for item in downloads:
        if item["platform"] == plat:
            return item["url"]
    return None


def _is_pid_alive(pid: int) -> bool:
    """檢查指定的 PID 是否仍在運行"""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cleanup_stale_drivers(running_dir: Path) -> None:
    """清理已結束進程留下的 chromedriver 副本"""
    if not running_dir.exists():
        return

    for path in running_dir.iterdir():
        # 檔名格式: chromedriver_<pid> 或 chromedriver_<pid>.exe
        name = path.stem  # 去掉 .exe 後綴
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
                logger.debug(f"Cleaned up stale chromedriver copy: {path.name}")
            except OSError:
                pass


def _create_chromedriver_copy(
    source: Path, running_dir: Path, plat: str
) -> Path:
    """
    建立 chromedriver 的 PID 專屬副本

    Args:
        source: 原始 chromedriver 路徑
        running_dir: running 目錄路徑
        plat: 平台代碼

    Returns:
        副本的路徑
    """
    running_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    if plat.startswith("win"):
        copy_name = f"chromedriver_{pid}.exe"
    else:
        copy_name = f"chromedriver_{pid}"

    copy_path = running_dir / copy_name
    shutil.copy2(source, copy_path)
    _make_executable(copy_path)

    # 註冊 atexit 清理
    def _cleanup() -> None:
        try:
            if copy_path.exists():
                copy_path.unlink()
                logger.debug(f"Cleaned up chromedriver copy: {copy_path.name}")
        except OSError:
            pass

    atexit.register(_cleanup)

    logger.debug(f"Created chromedriver copy: {copy_path.name}")
    return copy_path


def ensure_chrome_installed(force_download: bool = False) -> ChromePaths:
    """
    確保 Chrome 和 ChromeDriver 已安裝

    如果快取中已有對應版本，直接回傳路徑。
    否則自動下載最新的 stable 版本。

    回傳的 chromedriver 路徑為 PID 專屬副本，
    避免多個進程同時使用同一個 chromedriver 執行檔的衝突。

    Args:
        force_download: 強制重新下載

    Returns:
        ChromePaths 包含 chrome 和 chromedriver 的執行檔路徑
    """
    plat = _get_platform()
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
    chrome_exe_name = _get_chrome_executable_name(plat)
    chrome_path = version_dir / chrome_folder / chrome_exe_name

    # ChromeDriver 原始路徑
    chromedriver_folder = f"chromedriver-{plat}"
    chromedriver_exe_name = _get_chromedriver_executable_name(plat)
    chromedriver_path = version_dir / chromedriver_folder / chromedriver_exe_name

    # 檢查是否需要下載
    need_chrome = force_download or not chrome_path.exists()
    need_chromedriver = force_download or not chromedriver_path.exists()

    if need_chrome or need_chromedriver:
        if force_download and version_dir.exists():
            logger.info("Force download requested, removing existing cache...")
            shutil.rmtree(version_dir)

        version_dir.mkdir(parents=True, exist_ok=True)

        downloads = version_info["downloads"]

        # 下載 Chrome
        if need_chrome:
            chrome_url = _find_download_url(downloads["chrome"], plat)
            if not chrome_url:
                raise RuntimeError(f"No Chrome download found for platform: {plat}")
            _download_and_extract(chrome_url, version_dir, "Chrome")
            _make_executable(chrome_path)
            _remove_quarantine(version_dir / chrome_folder)

        # 下載 ChromeDriver
        if need_chromedriver:
            chromedriver_url = _find_download_url(downloads["chromedriver"], plat)
            if not chromedriver_url:
                raise RuntimeError(
                    f"No ChromeDriver download found for platform: {plat}"
                )
            _download_and_extract(chromedriver_url, version_dir, "ChromeDriver")
            _make_executable(chromedriver_path)
            _remove_quarantine(chromedriver_path)

        logger.info("Chrome and ChromeDriver are ready")
    else:
        logger.info(f"Using cached Chrome {version}")

    # 清理已結束進程的 chromedriver 副本
    running_dir = cache_dir / "running"
    _cleanup_stale_drivers(running_dir)

    # 建立 PID 專屬的 chromedriver 副本
    chromedriver_copy = _create_chromedriver_copy(
        chromedriver_path, running_dir, plat
    )

    return ChromePaths(
        chrome=str(chrome_path),
        chromedriver=str(chromedriver_copy),
        version=version,
    )
