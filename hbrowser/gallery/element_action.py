"""通用元素操作工具：只提供一次性、不可重放的點擊。"""

from collections.abc import Callable
from typing import Any

import zendriver as zd

from .utils.mutation import wait_for_zendriver_mutation


class ElementAction:
    """通用元素操作類，提供明確且不可重放的點擊時限。

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
