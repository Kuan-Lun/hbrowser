import time
from collections.abc import Callable
from typing import Any

from hv_bie import parse_snapshot
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from hbrowser.gallery.element_action import ElementAction

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard


class ElementActionManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard
        self._action = ElementAction(driver.driver)

    @property
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    def _click(self, element: WebElement) -> None:
        self._action.click(element)

    # --- Resilient helpers (delegate to generic ElementAction) ---
    def click_resilient(
        self,
        get_element: Callable[[], WebElement],
        retries: int = 3,
        delay: float = 0.1,
    ) -> None:
        """Click element returned by get_element with stale/interactable retries."""
        self._action.click_resilient(get_element, retries=retries, delay=delay)

    def click_until(
        self,
        get_element: Callable[[], WebElement],
        condition: Callable[[], bool],
        max_attempts: int = 5,
        delay: float = 0.1,
        timeout: float = 0.3,
    ) -> None:
        """Click element repeatedly until condition returns True."""
        self._action.click_until(
            get_element,
            condition,
            max_attempts=max_attempts,
            delay=delay,
            timeout=timeout,
        )

    # --- Battle-specific methods ---
    def click_and_wait_log_locator(
        self,
        by: str | By,
        value: str,
        is_retry: bool = True,
        stale_retries: int = 3,
        timeout: float = 5.0,
        check_interval: float = 0.05,
    ) -> None:
        """
        Like click_and_wait_log but takes a locator so we can re-find
        element if it turns stale or after a refresh.
        """
        # Pre-click snapshot
        html = self.battle_dashboard.log_entries.get_new_lines(
            parse_snapshot(self.hvdriver.driver.page_source)
        )

        # Wait for element to be clickable, then click with stale retries
        self._action.click_locator(
            by, value, retries=stale_retries, wait_timeout=2.0, delay=0.1
        )

        waited = 0.0
        while html == self.battle_dashboard.log_entries.get_new_lines(
            parse_snapshot(self.hvdriver.driver.page_source)
        ):
            time.sleep(check_interval)
            waited += check_interval
            if waited >= timeout:
                if is_retry:
                    # Soft recovery: browser refresh, then attempt once more
                    # (no infinite recursion)
                    self.hvdriver.driver.refresh()
                    return self.click_and_wait_log_locator(
                        by,
                        value,
                        is_retry=False,
                        stale_retries=stale_retries,
                        timeout=timeout,
                        check_interval=check_interval,
                    )
                else:
                    raise TimeoutError("Battle action timeout waiting for log update")
