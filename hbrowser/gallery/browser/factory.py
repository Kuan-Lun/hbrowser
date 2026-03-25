"""瀏覽器 WebDriver 工廠"""

import atexit
import os
import platform
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import undetected_chromedriver as uc
from fake_useragent import UserAgent
from undetected_chromedriver.patcher import Patcher

from ..utils import setup_logger
from .ban_handler import handle_ban_decorator
from .chrome_manager import ensure_chrome_installed

logger = setup_logger(__name__)

# Tor SOCKS proxy 預設端口
_TOR_SOCKS_PORT = 9150

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
        os.path.expandvars(
            r"%APPDATA%\Tor Browser\Browser\TorBrowser\Tor\tor.exe"
        ),
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
        ", ".join(candidates) if candidates
        else f"(unsupported platform: {plat})"
    )
    raise FileNotFoundError(
        f"Tor binary not found. Searched: {searched}\n"
        "Please install Tor Browser (https://www.torproject.org/download/) "
        "or set TOR_BINARY_PATH environment variable."
    )


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
            line_queue.append(
                raw_line.decode("utf-8", errors="replace").strip()
            )
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
                logger.info(
                    f"Tor bootstrap completed successfully ({elapsed}s)"
                )
                return tor_process

            if "Bootstrapped " in line:
                match = re.search(r"Bootstrapped (\d+%)", line)
                if match:
                    last_bootstrap_pct = match.group(1)

            logger.debug(f"Tor: {line}")

        time.sleep(0.5)

    tor_process.terminate()
    raise RuntimeError(
        f"Tor failed to bootstrap within {bootstrap_timeout} seconds"
    )


def _verify_tor_ip(driver: Any) -> None:
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


def _create_proxy_extension(
    proxy_host: str, proxy_port: int, proxy_user: str, proxy_pass: str
) -> str:
    """
    創建一個 Chrome 擴充功能來處理代理認證

    Returns:
        擴充功能 zip 檔案的路徑
    """
    manifest_json = """
{
    "version": "1.0.0",
    "manifest_version": 2,
    "name": "Chrome Proxy",
    "permissions": [
        "proxy",
        "tabs",
        "unlimitedStorage",
        "storage",
        "<all_urls>",
        "webRequest",
        "webRequestBlocking"
    ],
    "background": {
        "scripts": ["background.js"]
    },
    "minimum_chrome_version":"22.0.0"
}
"""

    background_js = f"""
var config = {{
        mode: "fixed_servers",
        rules: {{
          singleProxy: {{
            scheme: "http",
            host: "{proxy_host}",
            port: parseInt({proxy_port})
          }},
          bypassList: ["localhost"]
        }}
      }};

chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});

function callbackFn(details) {{
    return {{
        authCredentials: {{
            username: "{proxy_user}",
            password: "{proxy_pass}"
        }}
    }};
}}

chrome.webRequest.onAuthRequired.addListener(
            callbackFn,
            {{urls: ["<all_urls>"]}},
            ['blocking']
);
"""

    # 創建臨時目錄
    plugin_dir = tempfile.mkdtemp()

    # 寫入 manifest.json
    with open(os.path.join(plugin_dir, "manifest.json"), "w") as f:
        f.write(manifest_json)

    # 寫入 background.js
    with open(os.path.join(plugin_dir, "background.js"), "w") as f:
        f.write(background_js)

    # 創建 zip 檔案
    plugin_file = os.path.join(tempfile.gettempdir(), "proxy_auth_plugin.zip")
    with zipfile.ZipFile(plugin_file, "w") as zp:
        zp.write(os.path.join(plugin_dir, "manifest.json"), "manifest.json")
        zp.write(os.path.join(plugin_dir, "background.js"), "background.js")

    return plugin_file


def create_driver(headless: bool = True) -> Any:
    """
    創建 WebDriver 實例

    Args:
        headless: 是否使用無頭模式

    Returns:
        配置好的 WebDriver 實例
    """
    logger.info(f"Creating WebDriver (headless: {headless})")

    # 啟動 Tor SOCKS proxy
    socks_port = _find_available_port(_TOR_SOCKS_PORT)
    tor_process = _start_tor_process(socks_port)

    # 註冊 atexit 確保 Tor 進程被清理
    def _cleanup_tor() -> None:
        try:
            tor_process.terminate()
            tor_process.wait(timeout=5)
        except Exception:
            tor_process.kill()

    atexit.register(_cleanup_tor)

    # 設定瀏覽器參數
    options = uc.ChromeOptions()

    # 住宅代理設定（從環境變數讀取）
    rp_username = os.getenv("RP_USERNAME")
    rp_password = os.getenv("RP_PASSWORD")
    rp_dns = os.getenv("RP_DNS")

    proxy_extension = None
    if rp_username and rp_password and rp_dns:
        # 解析代理地址和端口
        if ":" in rp_dns:
            proxy_host, proxy_port = rp_dns.split(":", 1)
        else:
            proxy_host = rp_dns
            proxy_port = "8080"

        logger.info(f"Using residential proxy: {rp_username}@{proxy_host}:{proxy_port}")

        # 創建代理認證擴充功能
        proxy_extension = _create_proxy_extension(
            proxy_host=proxy_host,
            proxy_port=int(proxy_port),
            proxy_user=rp_username,
            proxy_pass=rp_password,
        )
        logger.debug(f"Proxy extension created at: {proxy_extension}")
    else:
        # 使用 Tor SOCKS proxy
        options.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
        logger.info(f"Using Tor SOCKS proxy on port {socks_port}")

    # 檢測是否為 Linux + Xvfb 環境
    is_xvfb_env = (
        platform.system() == "Linux"
        and os.environ.get("DISPLAY")
        and ":" in os.environ.get("DISPLAY", "")
    )

    # 基本設定
    # 注意：如果使用代理擴充功能，不能禁用擴充功能
    if not proxy_extension:
        options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")  # 解決DevToolsActivePort文件不存在的問題
    options.add_argument("--window-size=1600,900")
    options.add_argument("--disable-dev-shm-usage")

    # Headless 模式設定
    if headless:
        options.add_argument("--headless=new")  # 使用新的無頭模式

        # 檢測是否為 Linux server 環境（通常沒有 GPU）
        # 在 Linux 且檢測到 DISPLAY 環境變數為空或 Xvfb 時，認為是 server 環境
        is_linux_server = platform.system() == "Linux" and (
            not os.environ.get("DISPLAY") or "Xvfb" in os.environ.get("DISPLAY", "")
        )

        # 只在 Linux server 環境下添加 GPU 相關參數
        # 在 macOS/Windows 或有實體顯示的 Linux 桌面環境，
        # 不添加這些參數以保持更真實的瀏覽器指紋
        if is_linux_server:
            options.add_argument("--disable-gpu")  # 無 GPU 環境必須
            options.add_argument("--disable-software-rasterizer")

    # Xvfb 環境下不添加 --disable-gpu 參數
    # 原因：讓 Chrome 使用 SwiftShader 軟體渲染可能有更自然的指紋
    # 明確禁用 GPU 反而容易被 Cloudflare 偵測
    if is_xvfb_env and not headless:
        logger.info(
            "Detected Xvfb environment, "
            "using default GPU settings for better fingerprint"
        )

    # 反偵測參數 - 降低被 Cloudflare 識別的機率
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")

    # User Agent
    options.add_argument("user-agent={ua}".format(ua=UserAgent()["google chrome"]))

    # 頁面加載策略
    options.page_load_strategy = "normal"  # 等待加载图片normal eager none

    # 如果有代理擴充功能，添加到選項中
    if proxy_extension:
        options.add_extension(proxy_extension)

    # macOS: 避免 Chrome for Testing 存取系統鑰匙圈時彈出授權提示
    if platform.system() == "Darwin":
        options.add_argument("--use-mock-keychain")

    # 確保 Chrome 和 ChromeDriver 已安裝
    chrome_paths = ensure_chrome_installed()
    options.binary_location = chrome_paths.chrome

    # macOS: 先用 Patcher 手動 patch chromedriver，再 codesign
    # 這樣 uc.Chrome() 偵測到已 patch 就不會再修改二進位，簽名保持有效
    if platform.system() == "Darwin":
        patcher = Patcher(executable_path=chrome_paths.chromedriver)
        if not patcher.is_binary_patched():
            patcher.patch_exe()
            logger.debug("ChromeDriver patched for undetected-chromedriver")
        subprocess.run(
            ["codesign", "--force", "--sign", "-", chrome_paths.chromedriver],
            check=False,
            capture_output=True,
        )
        logger.debug("ChromeDriver codesigned")

    # 隱私強化 - 禁用可能洩漏資訊的 API
    options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
    options.add_argument("--enforce-webrtc-ip-permission-check")
    options.add_argument("--webrtc-ip-handling-policy=disable_non_proxied_udp")

    # 使用 undetected-chromedriver 初始化 WebDriver
    # 注意: undetected-chromedriver 已經內建處理了
    # excludeSwitches 和 useAutomationExtension
    # 所以我們不需要手動設定這些選項
    logger.debug("Initializing Chrome driver...")
    driver = uc.Chrome(
        options=options,
        use_subprocess=True,
        driver_executable_path=chrome_paths.chromedriver,
    )
    logger.info("Chrome driver initialized successfully")

    # 隱私強化 - 透過 CDP 禁用洩漏資訊的 API
    driver.execute_cdp_cmd("Emulation.setGeolocationOverride", {})
    driver.execute_cdp_cmd(
        "Network.setUserAgentOverride",
        {"userAgent": driver.execute_script("return navigator.userAgent")},
    )

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
