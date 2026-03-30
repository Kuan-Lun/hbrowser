"""瀏覽器 WebDriver 工廠"""

import os
import platform
import subprocess
from typing import Any

import undetected_chromedriver as uc
from fake_useragent import UserAgent
from undetected_chromedriver.patcher import Patcher

from ..utils import setup_logger
from .ban_handler import handle_ban_decorator
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


def _configure_chrome_options(
    headless: bool,
    proxy_extension: str | None,
    use_tor: bool = False,
    socks_port: int | None = None,
) -> uc.ChromeOptions:
    """建立並設定 Chrome 選項（代理、基本、headless、反偵測、隱私）。"""
    options = uc.ChromeOptions()

    # 代理設定
    if proxy_extension:
        logger.info("Using residential proxy extension")
    elif use_tor and socks_port is not None:
        options.add_argument(f"--proxy-server=socks5://127.0.0.1:{socks_port}")
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

    # 禁用密碼儲存提示與翻譯提示
    options.add_experimental_option(
        "prefs",
        {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            "translate_blocked_languages": ["en", "ja", "zh-TW", "zh-CN"],
            "translate": {"enabled": False},
        },
    )

    # 隱私強化 - 禁用可能洩漏資訊的 API
    options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
    options.add_argument("--enforce-webrtc-ip-permission-check")
    options.add_argument("--webrtc-ip-handling-policy=disable_non_proxied_udp")

    return options


def _patch_chromedriver(chromedriver_path: str) -> None:
    """macOS: patch chromedriver 並 codesign，確保簽名有效。"""
    if platform.system() != "Darwin":
        return

    patcher = Patcher(executable_path=chromedriver_path)
    if not patcher.is_binary_patched():
        patcher.patch_exe()
        logger.debug("ChromeDriver patched for undetected-chromedriver")
    subprocess.run(
        ["codesign", "--force", "--sign", "-", chromedriver_path],
        check=False,
        capture_output=True,
    )
    logger.debug("ChromeDriver codesigned")


def _post_create_setup(
    driver: Any,
    tor_process: subprocess.Popen[bytes] | None,
    use_tor: bool,
) -> None:
    """driver 建立後的設定：CDP 隱私強化、Tor 清理包裝、ban 檢查、IP 驗證。"""
    # 隱私強化 - 透過 CDP 禁用洩漏資訊的 API
    driver.execute_cdp_cmd("Emulation.setGeolocationOverride", {})
    driver.execute_cdp_cmd(
        "Network.setUserAgentOverride",
        {"userAgent": driver.execute_script("return navigator.userAgent")},
    )

    # 包裝 quit() 以同時清理 tor 進程
    if tor_process is not None:
        original_quit = driver.quit
        _tor = tor_process

        def _quit_with_tor_cleanup() -> None:
            try:
                original_quit()
            finally:
                try:
                    _tor.terminate()
                    _tor.wait(timeout=5)
                except Exception:
                    _tor.kill()

        driver.quit = _quit_with_tor_cleanup

    # 添加 ban 檢查裝飾器
    driver.myget = handle_ban_decorator(driver)

    # 驗證 Tor IP 與本機 IP 不同（僅在使用 Tor 且無住宅代理時）
    if use_tor and not has_residential_proxy():
        verify_proxy_ip(driver)


def create_driver(headless: bool = True) -> Any:
    """
    創建 WebDriver 實例

    Args:
        headless: 是否使用無頭模式

    Returns:
        配置好的 WebDriver 實例
    """
    logger.info(f"Creating WebDriver (headless: {headless})")

    # 1. Tor 設定
    use_tor = should_use_tor()
    tor_process: subprocess.Popen[bytes] | None = None
    socks_port: int | None = None
    if use_tor:
        socks_port = find_available_port(TOR_SOCKS_PORT)
        tor_process = start_tor_with_retry(socks_port)

    # 2. 代理設定
    proxy_extension = configure_proxy()

    # 3. Chrome 選項
    options = _configure_chrome_options(headless, proxy_extension, use_tor, socks_port)

    # 4. 確保 Chrome 和 ChromeDriver 已安裝
    chrome_paths = ensure_chrome_installed()
    options.binary_location = chrome_paths.chrome

    # 5. Patch ChromeDriver (macOS)
    _patch_chromedriver(chrome_paths.chromedriver)

    # 6. 初始化 WebDriver
    logger.debug("Initializing Chrome driver...")
    driver = uc.Chrome(
        options=options,
        use_subprocess=True,
        driver_executable_path=chrome_paths.chromedriver,
    )
    logger.info("Chrome driver initialized successfully")

    # 7. 後置設定
    _post_create_setup(driver, tor_process, use_tor)

    return driver
