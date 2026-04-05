import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from hv_bie import parse_snapshot

from hbrowser.gallery.element_action import ElementAction

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard


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
        check_interval: float = 0.05,
    ) -> None:
        """
        Like click_and_wait_log but takes a CSS selector so we can re-find
        element if it turns stale or after a refresh.
        """
        # Pre-click snapshot
        html = await self.hvdriver.page.get_content()
        pre_lines = self.battle_dashboard.log_entries.get_new_lines(
            parse_snapshot(html)
        )

        # Wait for element to be clickable, then click with retries
        await self._action.click_locator(
            selector, retries=stale_retries, wait_timeout=2.0, delay=0.1
        )

        waited = 0.0
        while True:
            current_html = await self.hvdriver.page.get_content()
            current_lines = self.battle_dashboard.log_entries.get_new_lines(
                parse_snapshot(current_html)
            )
            if pre_lines != current_lines:
                break
            await asyncio.sleep(check_interval)
            waited += check_interval
            if waited >= timeout:
                if is_retry:
                    # Soft recovery: browser refresh, then attempt once more
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
