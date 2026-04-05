import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from hbrowser.gallery.element_action import ElementAction

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard

# 用輕量 JS 取得頁面 HTML 長度，偵測任何頁面變化（log 更新、HP 變化等）
_PAGE_LENGTH_JS = "document.body.innerHTML.length"


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
        pre_log = await self.hvdriver.page.evaluate(_PAGE_LENGTH_JS)

        # 點擊
        await self._action.click_locator(
            selector, retries=stale_retries, wait_timeout=2.0, delay=0.1
        )

        # 輪詢：用輕量 JS 偵測 log 變化
        waited = 0.0
        while True:
            await asyncio.sleep(check_interval)
            waited += check_interval
            current_log = await self.hvdriver.page.evaluate(_PAGE_LENGTH_JS)
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
