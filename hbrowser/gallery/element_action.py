"""通用的 Selenium 元素操作工具，提供帶重試機制的點擊功能"""

import time
from collections.abc import Callable

from selenium.common.exceptions import (
    ElementNotInteractableException,
    StaleElementReferenceException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class ElementAction:
    """通用的元素操作類，提供帶重試和等待機制的點擊功能。

    只依賴 Selenium WebDriver，不綁定任何業務邏輯。
    """

    def __init__(self, driver: object) -> None:
        self._driver = driver

    @property
    def driver(self) -> object:
        return self._driver

    def click(self, element: WebElement) -> None:
        """使用 ActionChains 點擊元素"""
        actions = ActionChains(self._driver)
        actions.move_to_element(element).click().perform()

    def click_resilient(
        self,
        get_element: Callable[[], WebElement],
        retries: int = 3,
        delay: float = 0.1,
    ) -> None:
        """點擊元素，自動重試 stale/interactable 錯誤"""
        last_err: Exception | None = None
        for i in range(retries):
            try:
                element = get_element()
                self.click(element)
                return
            except (
                StaleElementReferenceException,
                ElementNotInteractableException,
            ) as e:
                last_err = e
                time.sleep(delay)
                continue
        if last_err:
            raise last_err

    def click_until(
        self,
        get_element: Callable[[], WebElement],
        condition: Callable[[], bool],
        max_attempts: int = 5,
        delay: float = 0.1,
        timeout: float = 0.3,
    ) -> None:
        """重複點擊直到條件成立。

        每次點擊後等待頁面內容變化，避免意外雙重觸發。
        """
        for _ in range(max_attempts):
            if condition():
                return
            old_source = self._driver.page_source  # type: ignore[attr-defined]
            try:
                self.click_resilient(get_element, retries=3, delay=delay)
            except (
                StaleElementReferenceException,
                ElementNotInteractableException,
            ):
                continue
            deadline = time.monotonic() + timeout
            while self._driver.page_source == old_source:  # type: ignore[attr-defined]
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)

    def click_locator(
        self,
        by: str | By,
        value: str,
        retries: int = 3,
        wait_timeout: float = 2.0,
        delay: float = 0.1,
    ) -> None:
        """透過 locator 找到元素，等待可點擊後點擊，自動重試"""
        for attempt in range(retries):
            try:
                element = WebDriverWait(self._driver, wait_timeout).until(
                    EC.element_to_be_clickable((by, value))
                )
                self.click(element)
                return
            except (
                StaleElementReferenceException,
                ElementNotInteractableException,
            ):
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
