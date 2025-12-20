"""瀏覽器 WebDriver 工廠"""

import os
import platform
import tempfile
import zipfile
from typing import Any

import undetected_chromedriver as uc
from fake_useragent import UserAgent

from ..utils import setup_logger
from .ban_handler import handle_ban_decorator

logger = setup_logger(__name__)


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
        logger.info(
            "No residential proxy configured "
            "(set RP_USERNAME, RP_PASSWORD, RP_DNS to enable)"
        )

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

    # 使用 undetected-chromedriver 初始化 WebDriver
    # 注意: undetected-chromedriver 已經內建處理了
    # excludeSwitches 和 useAutomationExtension
    # 所以我們不需要手動設定這些選項
    logger.debug("Initializing Chrome driver...")
    driver = uc.Chrome(options=options, use_subprocess=True)
    logger.info("Chrome driver initialized successfully")

    # 添加 ban 檢查裝飾器
    driver.myget = handle_ban_decorator(driver)

    return driver
