import time
from collections.abc import Callable
from typing import Any

from hv_bie import parse_snapshot
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard


class ElementActionManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard

    @property
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    def _click(self, element: WebElement) -> None:
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()

    # --- Resilient helpers ---
    def click_resilient(
        self,
        get_element: Callable[[], WebElement],
        retries: int = 3,
        delay: float = 0.1,
    ) -> None:
        """Click element returned by get_element with stale retries."""
        last_err: Exception | None = None
        for i in range(retries):
            try:
                element = get_element()
                self._click(element)
                return
            except StaleElementReferenceException as e:
                last_err = e
                time.sleep(delay)
                continue
        if last_err:
            raise last_err

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

        # Try clicking with stale retries
        for attempt in range(stale_retries):
            try:
                element = self.driver.find_element(by, value)
                self._click(element)
                break
            except StaleElementReferenceException:
                if attempt == stale_retries - 1:
                    raise
                time.sleep(0.05)
        else:
            # Should not reach here; defensive
            return

        # Short settle
        time.sleep(0.05)

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

        # Ensure slight post-update stability
        time.sleep(0.01)
