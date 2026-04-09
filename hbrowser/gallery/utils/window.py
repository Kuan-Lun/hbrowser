"""瀏覽器視窗相關工具函數"""

import asyncio
from typing import Any

import zendriver as zd


async def wait_for_new_tab(
    browser: zd.Browser,
    existing_tabs: set[Any],
    timeout: float = 10.0,
) -> zd.Tab | None:
    """
    等待新的 tab 開啟

    Args:
        browser: zendriver Browser 實例
        existing_tabs: 已存在的 tab 集合（target_id）
        timeout: 超時時間（秒）

    Returns:
        新 tab，如果超時則返回 None
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await browser.update_targets()
        current_tabs = {
            t.target.target_id for t in browser.tabs if t.target is not None
        }
        new_tabs = current_tabs - existing_tabs
        if new_tabs:
            new_tab_id = next(iter(new_tabs))
            for tab in browser.tabs:
                if tab.target is not None and tab.target.target_id == new_tab_id:
                    return tab
        await asyncio.sleep(0.2)
    return None
