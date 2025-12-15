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
    options.add_argument("--disable-extensions")
    if headless:
        options.add_argument("--headless=new")  # 使用新的無頭模式
    options.add_argument("--no-sandbox")  # 解決DevToolsActivePort文件不存在的問題
    options.add_argument("--window-size=1600,900")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent={ua}".format(ua=UserAgent()["google chrome"]))
    options.page_load_strategy = "normal"  # 等待加载图片normal eager none

    # 使用 undetected-chromedriver 初始化 WebDriver
    driver = uc.Chrome(options=options, use_subprocess=True)

    # 添加 ban 檢查裝飾器
    driver.myget = handle_ban_decorator(driver, logcontrol)

    return driver
