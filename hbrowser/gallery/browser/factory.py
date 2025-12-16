"""瀏覽器 WebDriver 工廠"""

import os
import platform
from typing import Any

import undetected_chromedriver as uc
from fake_useragent import UserAgent

from .ban_handler import handle_ban_decorator


def create_driver(headless: bool = True, logcontrol: Any = None) -> Any:
    """
    創建 WebDriver 實例

    Args:
        headless: 是否使用無頭模式
        logcontrol: 日誌控制函數

    Returns:
        配置好的 WebDriver 實例
    """
    # 設定瀏覽器參數
    options = uc.ChromeOptions()

    # 檢測是否為 Linux + Xvfb 環境
    is_xvfb_env = (
        platform.system() == "Linux"
        and os.environ.get("DISPLAY")
        and ":" in os.environ.get("DISPLAY", "")
    )

    # 基本設定
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
        # 在 macOS/Windows 或有實體顯示的 Linux 桌面環境，不添加這些參數以保持更真實的瀏覽器指紋
        if is_linux_server:
            options.add_argument("--disable-gpu")  # 無 GPU 環境必須
            options.add_argument("--disable-software-rasterizer")

    # Xvfb 環境下不添加 --disable-gpu 參數
    # 原因：讓 Chrome 使用 SwiftShader 軟體渲染可能有更自然的指紋
    # 明確禁用 GPU 反而容易被 Cloudflare 偵測
    if is_xvfb_env and not headless:
        print(
            "Detected Xvfb environment, using default GPU settings for better fingerprint..."
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

    # 使用 undetected-chromedriver 初始化 WebDriver
    # 注意: undetected-chromedriver 已經內建處理了 excludeSwitches 和 useAutomationExtension
    # 所以我們不需要手動設定這些選項
    driver = uc.Chrome(options=options, use_subprocess=True)

    # 添加 ban 檢查裝飾器
    driver.myget = handle_ban_decorator(driver, logcontrol)

    return driver
