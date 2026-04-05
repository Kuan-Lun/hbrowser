"""通用的元素操作工具，提供帶重試機制的點擊功能（async zendriver 版本）"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import zendriver as zd


class ElementAction:
    """通用的元素操作類，提供帶重試和等待機制的點擊功能。

    只依賴 zendriver Tab，不綁定任何業務邏輯。
    """

    def __init__(self, page: zd.Tab) -> None:
        self._page = page

    @property
    def page(self) -> zd.Tab:
        return self._page

    async def click(self, element: Any) -> None:
        """點擊元素，先嘗試滾動到可見位置再模擬滑鼠點擊，失敗時用 JS click"""
        try:
            await element.scroll_into_view()
            await element.mouse_move()
            await element.mouse_click()
        except Exception as e:
            if "could not find position" in str(e):
                await element.apply("(el) => el.click()")
            else:
                raise

    async def click_resilient(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        retries: int = 3,
        delay: float = 0.1,
    ) -> None:
        """點擊元素，自動重試錯誤"""
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                element = await get_element()
                await self.click(element)
                return
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay)
                continue
        if last_err:
            raise last_err

    async def click_until(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        condition: Callable[[], Coroutine[Any, Any, bool]],
        max_attempts: int = 5,
        delay: float = 0.1,
        timeout: float = 0.3,
    ) -> None:
        """重複點擊直到條件成立。

        每次點擊後等待頁面內容變化，避免意外雙重觸發。
        """
        for _ in range(max_attempts):
            if await condition():
                return
            old_source = await self._page.get_content()
            try:
                await self.click_resilient(get_element, retries=3, delay=delay)
            except Exception:
                continue
            deadline = asyncio.get_event_loop().time() + timeout
            while await self._page.get_content() == old_source:
                if asyncio.get_event_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.05)

    async def click_locator(
        self,
        selector: str,
        retries: int = 3,
        wait_timeout: float = 2.0,
        delay: float = 0.1,
    ) -> None:
        """透過 CSS selector 找到元素，等待後點擊，自動重試"""
        for attempt in range(retries):
            try:
                element = await self._page.select(selector, timeout=wait_timeout)
                await self.click(element)
                return
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
