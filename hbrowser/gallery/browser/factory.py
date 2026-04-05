"""瀏覽器工廠 - 使用 zendriver (CDP-based)"""

import os
import platform
import subprocess
from typing import Any

import zendriver as zd
from fake_useragent import UserAgent

from ..utils import setup_logger
from .chrome_manager import ensure_chrome_installed
from .proxy import (
    configure_proxy,
    find_available_port,
    has_residential_proxy,
    verify_proxy_ip,
)
from .tor import (
    TOR_SOCKS_PORT,
    should_use_tor,
    start_tor_with_retry,
)

logger = setup_logger(__name__)


def _build_config(
    headless: bool,
    proxy_extension: str | None,
    use_tor: bool = False,
    socks_port: int | None = None,
    chrome_path: str | None = None,
) -> zd.Config:
    """建立 zendriver Config（代理、基本、headless、隱私）。"""
    config = zd.Config()

    # Chrome 執行檔路徑
    if chrome_path:
        config.browser_executable_path = chrome_path

    # Headless 模式
    config.headless = headless

    # 隱私強化 - 禁用 WebRTC（防止 IP 洩漏）
    config.disable_webrtc = True

    # User Agent
    config.user_agent = UserAgent()["google chrome"]

    # 代理設定
    if proxy_extension:
        logger.info("Using residential proxy extension")
        config.add_extension(proxy_extension)
    elif use_tor and socks_port is not None:
        config.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
        logger.info(f"Using Tor SOCKS proxy on port {socks_port}")
    else:
        logger.info("No proxy configured (direct connection)")

    # 檢測是否為 Linux + Xvfb 環境
    is_xvfb_env = (
        platform.system() == "Linux"
        and os.environ.get("DISPLAY")
        and ":" in os.environ.get("DISPLAY", "")
    )

    # 基本設定
    if not proxy_extension:
        config.add_argument("--disable-extensions")
    config.sandbox = False  # 解決 DevToolsActivePort 文件不存在的問題
    config.add_argument("--window-size=1600,900")
    config.add_argument("--disable-dev-shm-usage")

    # Headless 模式下的 GPU 設定
    if headless:
        is_linux_server = platform.system() == "Linux" and (
            not os.environ.get("DISPLAY") or "Xvfb" in os.environ.get("DISPLAY", "")
        )
        if is_linux_server:
            config.add_argument("--disable-gpu")
            config.add_argument("--disable-software-rasterizer")

    if is_xvfb_env and not headless:
        logger.info(
            "Detected Xvfb environment, "
            "using default GPU settings for better fingerprint"
        )

    # 反偵測參數
    config.add_argument("--disable-blink-features=AutomationControlled")
    config.add_argument("--disable-infobars")
    config.add_argument("--disable-notifications")
    config.add_argument("--disable-popup-blocking")

    # 禁用密碼儲存提示與翻譯提示
    config.add_argument("--disable-save-password-bubble")
    config.add_argument("--disable-translate")
    config.add_argument("--password-store=basic")

    # macOS: 避免 Chrome for Testing 存取系統鑰匙圈時彈出授權提示
    if platform.system() == "Darwin":
        config.add_argument("--use-mock-keychain")

    # 隱私強化
    config.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
    config.add_argument("--enforce-webrtc-ip-permission-check")
    config.add_argument("--webrtc-ip-handling-policy=disable_non_proxied_udp")

    return config


async def _post_create_setup(
    browser: zd.Browser,
    page: zd.Tab,
    tor_process: subprocess.Popen[bytes] | None,
    use_tor: bool,
) -> None:
    """browser 建立後的設定：CDP 隱私強化、Tor 清理、ban 檢查、IP 驗證。"""
    from zendriver import cdp

    # 隱私強化 - 透過 CDP 禁用地理定位
    await page.send(cdp.emulation.set_geolocation_override())

    # 覆寫 User-Agent（透過 CDP）
    ua = await page.evaluate("navigator.userAgent")
    await page.send(cdp.network.set_user_agent_override(user_agent=ua))

    # 記錄 tor process 以便清理
    if tor_process is not None:
        browser._tor_process = tor_process

    # 驗證 Tor IP 與本機 IP 不同（僅在使用 Tor 且無住宅代理時）
    if use_tor and not has_residential_proxy():
        await verify_proxy_ip(browser, page)


async def create_browser(
    headless: bool = True,
) -> tuple[zd.Browser, zd.Tab]:
    """
    創建 zendriver Browser 實例

    Args:
        headless: 是否使用無頭模式

    Returns:
        (browser, page) tuple
    """
    logger.info(f"Creating browser (headless: {headless})")

    # 1. Tor 設定
    use_tor = should_use_tor()
    tor_process: subprocess.Popen[bytes] | None = None
    socks_port: int | None = None
    if use_tor:
        socks_port = find_available_port(TOR_SOCKS_PORT)
        tor_process = start_tor_with_retry(socks_port)

    # 2. 代理設定
    proxy_extension = configure_proxy()

    # 3. 確保 Chrome 已安裝
    chrome_paths = ensure_chrome_installed()

    # 4. 建立 Config
    config = _build_config(
        headless, proxy_extension, use_tor, socks_port, chrome_paths.chrome
    )

    # 5. 啟動 Browser
    logger.debug("Initializing browser...")
    browser = await zd.start(config=config)
    page = browser.main_tab
    logger.info("Browser initialized successfully")

    # 6. 後置設定
    await _post_create_setup(browser, page, tor_process, use_tor)

    return browser, page


async def stop_browser(browser: Any) -> None:
    """停止 browser 並清理資源（包含 Tor 進程）"""
    tor_process = getattr(browser, "_tor_process", None)
    try:
        await browser.stop()
    finally:
        if tor_process is not None:
            try:
                tor_process.terminate()
                tor_process.wait(timeout=5)
            except Exception:
                tor_process.kill()
