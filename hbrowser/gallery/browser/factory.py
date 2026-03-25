"""瀏覽器 WebDriver 工廠"""

import atexit
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService

from ..utils import setup_logger
from .ban_handler import handle_ban_decorator
from .tor_manager import ensure_tor_installed

logger = setup_logger(__name__)

# Tor SOCKS proxy 預設端口
_TOR_SOCKS_PORT = 9150


def _find_available_port(start: int = 9150) -> int:
    """找到一個可用的端口"""
    import socket

    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in range {start}-{start + 99}")


def _start_tor_process(tor_binary: str, socks_port: int) -> subprocess.Popen[bytes]:
    """
    啟動 Tor SOCKS proxy 進程

    Args:
        tor_binary: tor 執行檔路徑
        socks_port: SOCKS proxy 端口

    Returns:
        tor 進程的 Popen 物件
    """
    import fcntl
    import select

    # 建立臨時資料目錄
    data_dir = Path(tempfile.mkdtemp(prefix="tor_data_"))

    logger.info(f"Starting Tor process on SOCKS port {socks_port}...")
    tor_process = subprocess.Popen(
        [
            tor_binary,
            "--SocksPort",
            str(socks_port),
            "--DataDirectory",
            str(data_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 等待 Tor bootstrap 完成（使用非阻塞 I/O）
    bootstrap_timeout = 120
    start_time = time.time()

    if tor_process.stdout is None:
        raise RuntimeError("Failed to capture Tor process output")

    # 設定 stdout 為非阻塞模式
    fd = tor_process.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    buffer = b""
    last_status_time = start_time
    last_bootstrap_pct = "0%"
    status_interval = 10  # 每 10 秒顯示一次狀態

    while time.time() - start_time < bootstrap_timeout:
        if tor_process.poll() is not None:
            raise RuntimeError(
                f"Tor process exited unexpectedly with code {tor_process.returncode}"
            )

        now = time.time()

        # 每 10 秒顯示一次等待狀態
        if now - last_status_time >= status_interval:
            elapsed = int(now - start_time)
            remaining = bootstrap_timeout - elapsed
            logger.info(
                f"Tor bootstrapping... {last_bootstrap_pct} "
                f"({elapsed}s elapsed, {remaining}s remaining)"
            )
            last_status_time = now

        # 等待最多 1 秒看是否有新輸出
        ready, _, _ = select.select([tor_process.stdout], [], [], 1.0)
        if not ready:
            continue

        chunk = tor_process.stdout.read(4096)
        if not chunk:
            continue

        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            if "Bootstrapped 100%" in line:
                elapsed = int(time.time() - start_time)
                logger.info(f"Tor bootstrap completed successfully ({elapsed}s)")
                return tor_process

            # 擷取目前的 bootstrap 百分比
            if "Bootstrapped " in line:
                import re

                match = re.search(r"Bootstrapped (\d+%)", line)
                if match:
                    last_bootstrap_pct = match.group(1)

            logger.debug(f"Tor: {line}")

    # 超時
    tor_process.terminate()
    raise RuntimeError(f"Tor failed to bootstrap within {bootstrap_timeout} seconds")


def _get_local_ip() -> str:
    """透過直接連線取得本機的公開 IP"""
    with urlopen("https://api.ipify.org", timeout=10) as response:
        ip: str = response.read().decode("utf-8").strip()
    return ip


def _get_tor_ip(driver: Any) -> str:
    """透過 Tor Browser 取得 Tor 出口節點的 IP"""
    driver.get("https://api.ipify.org")
    ip: str = driver.find_element("tag name", "body").text.strip()
    return ip


def _verify_tor_ip(driver: Any) -> None:
    """
    驗證 Tor 連線的 IP 與本機 IP 不同

    Raises:
        RuntimeError: 如果 Tor IP 與本機 IP 相同（代表 Tor 未正確運作）
    """
    try:
        local_ip = _get_local_ip()
        tor_ip = _get_tor_ip(driver)

        if local_ip == tor_ip:
            raise RuntimeError(
                f"Tor IP safety check failed: Tor IP ({tor_ip}) is the same "
                f"as local IP ({local_ip}). Tor may not be working properly."
            )

        logger.info(f"Tor IP verified: {tor_ip} (local: {local_ip})")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not verify Tor IP (non-fatal): {e}")


def create_driver(headless: bool = True) -> Any:
    """
    創建 WebDriver 實例（使用 Tor Browser）

    Args:
        headless: 是否使用無頭模式

    Returns:
        配置好的 WebDriver 實例
    """
    logger.info(f"Creating WebDriver (headless: {headless})")

    # 確保 Tor Browser 和 GeckoDriver 已安裝
    tor_paths = ensure_tor_installed()

    # 啟動 Tor SOCKS proxy
    socks_port = _find_available_port(_TOR_SOCKS_PORT)
    tor_process = _start_tor_process(tor_paths.tor_binary, socks_port)

    # 註冊 atexit 確保 Tor 進程被清理
    def _cleanup_tor() -> None:
        try:
            tor_process.terminate()
            tor_process.wait(timeout=5)
        except Exception:
            tor_process.kill()

    atexit.register(_cleanup_tor)

    # 設定 Firefox 選項
    options = FirefoxOptions()
    options.binary_location = tor_paths.browser

    # Tor SOCKS proxy 設定
    options.set_preference("network.proxy.type", 1)
    options.set_preference("network.proxy.socks", "127.0.0.1")
    options.set_preference("network.proxy.socks_port", socks_port)
    options.set_preference("network.proxy.socks_remote_dns", True)
    # 不使用系統 proxy 設定
    options.set_preference("network.proxy.no_proxies_on", "")

    # 禁用自動更新和遙測
    options.set_preference("app.update.enabled", False)
    options.set_preference("datareporting.policy.dataSubmissionEnabled", False)
    options.set_preference("toolkit.telemetry.enabled", False)

    # 頁面加載策略
    options.page_load_strategy = "normal"

    # Headless 模式
    if headless:
        options.add_argument("-headless")

    # 視窗大小
    options.add_argument("--width=1600")
    options.add_argument("--height=900")

    # macOS: 避免系統鑰匙圈提示
    if platform.system() == "Darwin":
        os.environ.setdefault("MOZ_HEADLESS_WIDTH", "1600")
        os.environ.setdefault("MOZ_HEADLESS_HEIGHT", "900")

    # 使用 GeckoDriver 初始化 Firefox WebDriver
    service = FirefoxService(executable_path=tor_paths.geckodriver)

    logger.debug("Initializing Tor Browser driver...")
    driver = Firefox(service=service, options=options)
    logger.info("Tor Browser driver initialized successfully")

    # 將 tor 進程附加到 driver 上，以便在 driver.quit() 時清理
    driver._tor_process = tor_process

    # 包裝 quit() 以同時清理 tor 進程
    original_quit = driver.quit

    def _quit_with_tor_cleanup() -> None:
        try:
            original_quit()
        finally:
            try:
                tor_process.terminate()
                tor_process.wait(timeout=5)
            except Exception:
                tor_process.kill()

    driver.quit = _quit_with_tor_cleanup

    # 添加 ban 檢查裝飾器
    driver.myget = handle_ban_decorator(driver)

    # 驗證 Tor IP 與本機 IP 不同
    _verify_tor_ip(driver)

    return driver
