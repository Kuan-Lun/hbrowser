"""Tor 進程管理"""

import atexit
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from ..utils import setup_logger

logger = setup_logger(__name__)

# Tor SOCKS proxy 預設端口
TOR_SOCKS_PORT = 9150

# Tor 執行檔路徑（使用者需自行安裝 Tor Browser）
_TOR_BINARY_CANDIDATES: dict[str, list[str]] = {
    "Darwin": [
        "/Applications/Tor Browser.app/Contents/MacOS/Tor/tor",
    ],
    "Linux": [
        "/usr/bin/tor",
        os.path.expanduser("~/tor-browser/Browser/TorBrowser/Tor/tor"),
    ],
    "Windows": [
        os.path.expandvars(
            r"%USERPROFILE%\Desktop\Tor Browser\Browser\TorBrowser\Tor\tor.exe"
        ),
        os.path.expandvars(r"%APPDATA%\Tor Browser\Browser\TorBrowser\Tor\tor.exe"),
        r"C:\Program Files\Tor Browser\Browser\TorBrowser\Tor\tor.exe",
        r"C:\Program Files (x86)\Tor Browser\Browser\TorBrowser\Tor\tor.exe",
    ],
}


def _find_tor_binary() -> str:
    """找到系統上的 tor 執行檔

    也可透過環境變數 TOR_BINARY_PATH 指定路徑。
    """
    # 優先使用環境變數
    env_path = os.getenv("TOR_BINARY_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    plat = platform.system()
    candidates = _TOR_BINARY_CANDIDATES.get(plat, [])
    for path in candidates:
        if os.path.isfile(path):
            return path

    searched = (
        ", ".join(candidates) if candidates else f"(unsupported platform: {plat})"
    )
    raise FileNotFoundError(
        f"Tor binary not found. Searched: {searched}\n"
        "Please install Tor Browser (https://www.torproject.org/download/) "
        "or set TOR_BINARY_PATH environment variable."
    )


def find_available_port(start: int = 9150) -> int:
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


def _start_tor_process(socks_port: int) -> subprocess.Popen[bytes]:
    """
    啟動 Tor SOCKS proxy 進程

    Args:
        socks_port: SOCKS proxy 端口

    Returns:
        tor 進程的 Popen 物件
    """
    import re
    import threading
    from collections import deque

    tor_binary = _find_tor_binary()
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

    if tor_process.stdout is None:
        raise RuntimeError("Failed to capture Tor process output")

    # 使用背景 thread 讀取 stdout（跨平台，避免 fcntl/select）
    line_queue: deque[str] = deque()
    reader_done = threading.Event()

    def _reader() -> None:
        assert tor_process.stdout is not None
        for raw_line in tor_process.stdout:
            line_queue.append(raw_line.decode("utf-8", errors="replace").strip())
        reader_done.set()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    # 等待 Tor bootstrap 完成
    bootstrap_timeout = 120
    start_time = time.time()
    last_status_time = start_time
    last_bootstrap_pct = "0%"

    while time.time() - start_time < bootstrap_timeout:
        if tor_process.poll() is not None:
            raise RuntimeError(
                f"Tor process exited unexpectedly "
                f"with code {tor_process.returncode}"
            )

        now = time.time()
        if now - last_status_time >= 10:
            elapsed = int(now - start_time)
            remaining = bootstrap_timeout - elapsed
            logger.info(
                f"Tor bootstrapping... {last_bootstrap_pct} "
                f"({elapsed}s elapsed, {remaining}s remaining)"
            )
            last_status_time = now

        # 處理佇列中的所有行
        while line_queue:
            line = line_queue.popleft()
            if not line:
                continue

            if "Bootstrapped 100%" in line:
                elapsed = int(time.time() - start_time)
                logger.info(f"Tor bootstrap completed successfully ({elapsed}s)")
                return tor_process

            if "Bootstrapped " in line:
                match = re.search(r"Bootstrapped (\d+%)", line)
                if match:
                    last_bootstrap_pct = match.group(1)

            logger.debug(f"Tor: {line}")

        time.sleep(0.5)

    tor_process.terminate()
    raise RuntimeError(f"Tor failed to bootstrap within {bootstrap_timeout} seconds")


def should_use_tor() -> bool:
    """判斷是否啟用 Tor

    優先使用 USE_TOR 環境變數；未設定時自動偵測 tor 執行檔。
    """
    use_tor_env = os.getenv("USE_TOR")
    if use_tor_env is not None:
        return use_tor_env.lower() not in ("0", "false", "no", "off")

    # 未設定環境變數：有 tor 就用，沒有就不用
    try:
        _find_tor_binary()
        return True
    except FileNotFoundError:
        return False


def start_tor_with_retry(
    socks_port: int,
    max_retries: int = 3,
    retry_wait: int = 300,
) -> subprocess.Popen[bytes]:
    """啟動 Tor 並在失敗時重試，同時註冊 atexit 清理。

    Args:
        socks_port: SOCKS proxy 端口
        max_retries: 最大重試次數
        retry_wait: 重試等待秒數（預設 5 分鐘）

    Returns:
        tor 進程的 Popen 物件
    """
    tor_process: subprocess.Popen[bytes] | None = None
    for attempt in range(1, max_retries + 1):
        try:
            tor_process = _start_tor_process(socks_port)
            break
        except RuntimeError:
            if attempt == max_retries:
                raise
            logger.warning(
                f"Tor bootstrap failed (attempt {attempt}/{max_retries}), "
                f"retrying in {retry_wait // 60} minutes..."
            )
            time.sleep(retry_wait)

    assert tor_process is not None

    # 註冊 atexit 確保 Tor 進程被清理
    _tor = tor_process

    def _cleanup_tor() -> None:
        try:
            _tor.terminate()
            _tor.wait(timeout=5)
        except Exception:
            _tor.kill()

    atexit.register(_cleanup_tor)
    return tor_process


def verify_tor_ip(driver: Any) -> None:
    """驗證 Tor 連線的 IP 與本機 IP 不同"""
    try:
        with urlopen("https://api.ipify.org", timeout=10) as response:
            local_ip: str = response.read().decode("utf-8").strip()

        driver.get("https://api.ipify.org")
        tor_ip: str = driver.find_element("tag name", "body").text.strip()

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
