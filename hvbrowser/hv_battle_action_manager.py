import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from hbrowser.gallery.element_action import ElementAction

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard

# 用輕量 JS 計算頁面 HTML 的快速 hash（cyrb53 變體），偵測任何頁面變化
# - 純長度比對會漏掉「字數相同但內容變化」的情況（例如 HP 50% → 60%）
# - 完整 SHA hash 在大頁面上成本高
# - cyrb53 是 53-bit 快速非加密 hash，碰撞率極低，計算成本接近 length
# 加上 null guard 避免在頁面 reload/navigation 過程中 document.body 暫時為 null
_PAGE_HASH_JS = """
(() => {
    if (!document.body) return 0;
    const s = document.body.innerHTML;
    let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
    for (let i = 0; i < s.length; i++) {
        const ch = s.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
    h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
    h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return 4294967296 * (2097151 & h2) + (h1 >>> 0);
})()
"""


class ElementActionManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard
        self._action = ElementAction(driver.page)

    @property
    def page(self) -> Any:
        return self.hvdriver.page

    async def _click(self, element: Any) -> None:
        await self._action.click(element)

    # --- Resilient helpers (delegate to generic ElementAction) ---
    async def click_resilient(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        retries: int = 3,
        delay: float = 0.1,
    ) -> None:
        """Click element returned by get_element with retries."""
        await self._action.click_resilient(get_element, retries=retries, delay=delay)

    async def click_until(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        condition: Callable[[], Coroutine[Any, Any, bool]],
        max_attempts: int = 5,
        delay: float = 0.1,
        timeout: float = 0.3,
    ) -> None:
        """Click element repeatedly until condition returns True."""
        await self._action.click_until(
            get_element,
            condition,
            max_attempts=max_attempts,
            delay=delay,
            timeout=timeout,
        )

    async def click_locator(
        self,
        selector: str,
        retries: int = 3,
        wait_timeout: float = 2.0,
        delay: float = 0.1,
    ) -> None:
        """Wait for element to be clickable by CSS selector, then click."""
        await self._action.click_locator(
            selector, retries=retries, wait_timeout=wait_timeout, delay=delay
        )

    # --- Battle-specific methods ---
    async def click_and_wait_log_locator(
        self,
        selector: str,
        is_retry: bool = True,
        stale_retries: int = 3,
        timeout: float = 5.0,
        check_interval: float = 0.3,
    ) -> None:
        """
        點擊元素並等待戰鬥 log 更新。

        使用輕量 JS 輪詢 battlelog innerHTML 偵測變化，
        避免頻繁呼叫 get_content() + parse_snapshot() 阻塞 CDP 連線。
        """
        # 用輕量 JS 取得 pre-click log 快照
        pre_log = await self.hvdriver.page.evaluate(_PAGE_HASH_JS)

        # 點擊
        await self._action.click_locator(
            selector, retries=stale_retries, wait_timeout=2.0, delay=0.1
        )

        # 輪詢：用輕量 JS 偵測 log 變化
        waited = 0.0
        while True:
            await asyncio.sleep(check_interval)
            waited += check_interval
            current_log = await self.hvdriver.page.evaluate(_PAGE_HASH_JS)
            if current_log != pre_log:
                break
            if waited >= timeout:
                if is_retry:
                    await self.hvdriver.page.reload()
                    return await self.click_and_wait_log_locator(
                        selector,
                        is_retry=False,
                        stale_retries=stale_retries,
                        timeout=timeout,
                        check_interval=check_interval,
                    )
                else:
                    raise TimeoutError("Battle action timeout waiting for log update")
