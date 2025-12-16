"""瀏覽器 WebDriver 工廠"""
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

    # 基本設定
    options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")  # 解決DevToolsActivePort文件不存在的問題
    options.add_argument("--window-size=1600,900")
    options.add_argument("--disable-dev-shm-usage")

    # Headless 模式設定
    if headless:
        options.add_argument("--headless=new")  # 使用新的無頭模式
        # 在 headless 模式下添加更多反偵測參數
        options.add_argument("--disable-gpu")  # 無 GPU 環境必須
        options.add_argument("--disable-software-rasterizer")

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
