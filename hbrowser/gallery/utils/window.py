"""瀏覽器視窗相關工具函數"""

from typing import Any


def find_new_window(existing_windows: set[str], driver: Any) -> str | None:
    """
    找到新開啟的視窗

    Args:
        existing_windows: 已存在的視窗集合
        driver: Selenium WebDriver 實例

    Returns:
        新視窗的 handle，如果沒有則返回 None
    """
    current_windows = set(driver.window_handles)
    new_windows = current_windows - existing_windows
    return next(iter(new_windows or []), None)
