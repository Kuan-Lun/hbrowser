"""瀏覽器工廠"""

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
    config = zd.Config()

    if chrome_path:
        config.browser_executable_path = chrome_path

    config.headless = headless
    config.disable_webrtc = True
    config.user_agent = UserAgent()["google chrome"]

    if proxy_extension:
        logger.info("Using residential proxy extension")
        config.add_extension(proxy_extension)
    elif use_tor and socks_port is not None:
        config.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
        logger.info(f"Using Tor SOCKS proxy on port {socks_port}")
    else:
        logger.info("No proxy configured (direct connection)")

    is_xvfb_env = (
        platform.system() == "Linux"
        and os.environ.get("DISPLAY")
        and ":" in os.environ.get("DISPLAY", "")
    )

    if not proxy_extension:
        config.add_argument("--disable-extensions")
    config.sandbox = False
    config.add_argument("--window-size=1600,900")
    config.add_argument("--disable-dev-shm-usage")

    if headless:
        is_linux_server = platform.system() == "Linux" and (
            not os.environ.get("DISPLAY") or "Xvfb" in os.environ.get("DISPLAY", "")
        )
        if is_linux_server:
            config.add_argument("--disable-gpu")
            config.add_argument("--disable-software-rasterizer")

    if is_xvfb_env and not headless:
        # Xvfb 環境讓 Chrome 使用 SwiftShader 軟體渲染，刻意不加 --disable-gpu，
        # 明確禁用 GPU 反而容易被 Cloudflare 偵測。
        logger.info(
            "Detected Xvfb environment, "
            "using default GPU settings for better fingerprint"
        )

    config.add_argument("--disable-blink-features=AutomationControlled")
    config.add_argument("--disable-infobars")
    config.add_argument("--disable-notifications")
    config.add_argument("--disable-popup-blocking")

    config.add_argument("--disable-save-password-bubble")
    config.add_argument("--disable-translate")
    config.add_argument("--password-store=basic")

    if platform.system() == "Darwin":
        # 避免 Chrome for Testing 存取系統鑰匙圈時彈出授權提示
        config.add_argument("--use-mock-keychain")

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
    from zendriver import cdp

    await page.send(cdp.emulation.set_geolocation_override())

    ua = await page.evaluate("navigator.userAgent")
    await page.send(cdp.network.set_user_agent_override(user_agent=ua))

    if tor_process is not None:
        browser._tor_process = tor_process

    if use_tor and not has_residential_proxy():
        await verify_proxy_ip(browser, page)


async def create_browser(
    headless: bool = True,
) -> tuple[zd.Browser, zd.Tab]:
    """創建 zendriver Browser 實例。

    Returns:
        (browser, page) tuple
    """
    logger.info(f"Creating browser (headless: {headless})")

    use_tor = should_use_tor()
    tor_process: subprocess.Popen[bytes] | None = None
    socks_port: int | None = None
    if use_tor:
        socks_port = find_available_port(TOR_SOCKS_PORT)
        tor_process = start_tor_with_retry(socks_port)

    proxy_extension = configure_proxy()

    chrome_paths = ensure_chrome_installed()

    config = _build_config(
        headless, proxy_extension, use_tor, socks_port, chrome_paths.chrome
    )

    logger.debug("Initializing browser...")
    browser = await zd.start(config=config)
    page = browser.main_tab
    logger.info("Browser initialized successfully")

    await _post_create_setup(browser, page, tor_process, use_tor)

    return browser, page


async def stop_browser(browser: Any) -> None:
    """停止 browser 並清理資源（包含 Tor 進程）。"""
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
