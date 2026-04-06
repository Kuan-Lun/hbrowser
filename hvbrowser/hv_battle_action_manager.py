import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from hbrowser.gallery.element_action import ElementAction

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard

# cyrb53 hash of document.body.innerHTML，用於偵測頁面變化。
# 回傳 0 表示 document.body 尚未就緒。
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
        stale_retries: int = 3,
        timeout: float = 5.0,
        check_interval: float = 0.3,
    ) -> None:
        """
        點擊元素並等待頁面變化。

        使用 cyrb53 hash 比對 document.body.innerHTML 偵測任何頁面變化。
        timeout 後直接 raise，由外層 battle() 的 recovery 機制統一處理。
        """
        # 用輕量 JS 取得 pre-click hash 快照
        pre_hash = await self.hvdriver.page.evaluate(_PAGE_HASH_JS)

        # 點擊
        await self._action.click_locator(
            selector, retries=stale_retries, wait_timeout=2.0, delay=0.1
        )

        # 輪詢：用 hash 偵測頁面變化
        waited = 0.0
        while True:
            await asyncio.sleep(check_interval)
            waited += check_interval
            current_hash = await self.hvdriver.page.evaluate(_PAGE_HASH_JS)
            if current_hash != pre_hash:
                return
            if waited >= timeout:
                raise TimeoutError(
                    f"Battle action timeout waiting for page change "
                    f"after clicking {selector}"
                )
