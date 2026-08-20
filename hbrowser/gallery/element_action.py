"""通用元素操作工具：可重試查找，但絕不重放失敗的點擊。"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import zendriver as zd

from .utils import (
    ZendriverOperationTimeout,
    is_browser_generation_error,
    wait_for_zendriver,
)
from .utils.mutation import wait_for_zendriver_mutation

_SELECT_WATCHDOG_MARGIN_SECONDS = 2.0


class ElementAction:
    """通用元素操作類，提供可重試查找和明確的點擊時限。

    只依賴 zendriver Tab，不綁定任何業務邏輯。

    page 透過 callable 動態取得，而不是在建構時固定捕捉：呼叫端（例如瀏覽器
    重啟後 driver.page 被換成新物件）不需要重新建立 ElementAction 實例。
    """

    def __init__(self, page_provider: Callable[[], zd.Tab]) -> None:
        self._page_provider = page_provider

    @property
    def page(self) -> zd.Tab:
        return self._page_provider()

    async def click(self, element: Any, *, operation_timeout: float) -> None:
        """捲動後觸發 DOM click()，並以明確的 protocol watchdog 約束操作。"""
        await wait_for_zendriver_mutation(
            element.apply(
                "(el) => { el.scrollIntoView({block: 'center'}); el.click(); }"
            ),
            timeout=operation_timeout,
            owner=element,
            operation="Element click",
        )

    async def click_resilient(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        retries: int = 3,
        delay: float = 0.1,
        *,
        operation_timeout: float,
    ) -> None:
        """重試安全的元素查找；一旦點擊開始，任何錯誤都立即傳出。"""
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                element = await get_element()
            except ZendriverOperationTimeout:
                raise
            except Exception as e:
                if is_browser_generation_error(e):
                    raise
                last_err = e
                await asyncio.sleep(delay)
                continue
            # Once click starts, every failure has an unknown remote outcome.
            # Retrying here could execute the same mutation twice.
            await self.click(
                element,
                operation_timeout=operation_timeout,
            )
            return
        if last_err:
            raise last_err

    async def click_until(
        self,
        get_element: Callable[[], Coroutine[Any, Any, Any]],
        condition: Callable[[], Coroutine[Any, Any, bool]],
        max_attempts: int = 5,
        delay: float = 0.1,
        timeout: float = 0.3,
        content_read_timeout: float = 5.0,
        *,
        operation_timeout: float,
    ) -> None:
        """重複點擊直到條件成立。

        只有前一次點擊已明確成功回覆、而本地條件仍不成立時，才依此 API
        對 cycling/pane UI 的既定語意發出下一次點擊。任何點擊錯誤都立即
        終止，絕不重放 outcome-unknown mutation。
        """
        for _ in range(max_attempts):
            if await condition():
                return
            old_source = await wait_for_zendriver(
                self.page.get_content(),
                timeout=content_read_timeout,
                owner=self.page,
            )
            await self.click_resilient(
                get_element,
                retries=3,
                delay=delay,
                operation_timeout=operation_timeout,
            )
            deadline = asyncio.get_event_loop().time() + timeout
            while (
                await wait_for_zendriver(
                    self.page.get_content(),
                    timeout=content_read_timeout,
                    owner=self.page,
                )
                == old_source
            ):
                if asyncio.get_event_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.05)

    async def click_locator(
        self,
        selector: str,
        retries: int = 3,
        wait_timeout: float = 2.0,
        delay: float = 0.1,
        *,
        operation_timeout: float,
    ) -> None:
        """透過 CSS selector 找到元素，等待後點擊，自動重試"""
        for attempt in range(retries):
            try:
                element = await wait_for_zendriver(
                    self.page.select(selector, timeout=wait_timeout),
                    timeout=wait_timeout + _SELECT_WATCHDOG_MARGIN_SECONDS,
                    owner=self.page,
                )
            except ZendriverOperationTimeout:
                raise
            except Exception as e:
                if is_browser_generation_error(e) or attempt == retries - 1:
                    raise
                await asyncio.sleep(delay)
                continue
            # Element lookup is safe to retry. Click invocation is not.
            await self.click(
                element,
                operation_timeout=operation_timeout,
            )
            return
