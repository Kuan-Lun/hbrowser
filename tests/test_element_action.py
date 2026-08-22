import asyncio
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import zendriver as zd

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.element_action import ElementAction
from hbrowser.gallery.utils.protocol import (
    ZendriverOperationTimeout,
    _begin_zendriver_retirement,
)


class _StuckElement:
    """An element whose apply() never resolves, simulating an unresponsive tab."""

    def __init__(self, tab: Any = None) -> None:
        self._tab = tab

    async def apply(self, js_function: str) -> Any:
        del js_function
        await asyncio.Event().wait()


class _RespondingElement:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def apply(self, js_function: str) -> None:
        self.calls.append(js_function)


class ElementActionClickTests(unittest.IsolatedAsyncioTestCase):
    async def test_click_times_out_instead_of_hanging_forever(self) -> None:
        page = cast(zd.Tab, SimpleNamespace())
        action = ElementAction(lambda: page)
        element = _StuckElement(page)

        with self.assertRaises(ZendriverOperationTimeout):
            await action.click(element, operation_timeout=0.05)
        await _begin_zendriver_retirement(page).retire_operations(
            quiescent_connections=(),
        )

    async def test_click_succeeds_when_element_responds(self) -> None:
        page = cast(zd.Tab, SimpleNamespace())
        action = ElementAction(lambda: page)
        element = _RespondingElement()

        await action.click(element, operation_timeout=1.0)

        self.assertEqual(len(element.calls), 1)

    async def test_generic_click_failure_is_redacted_and_generation_terminal(
        self,
    ) -> None:
        sensitive_error = RuntimeError("secret click payload")
        element = SimpleNamespace(apply=AsyncMock(side_effect=sensitive_error))
        action = ElementAction(lambda: cast(zd.Tab, SimpleNamespace()))

        with self.assertRaises(BrowserMutationOutcomeUnknownError) as raised:
            await action.click(element, operation_timeout=1.0)

        element.apply.assert_awaited_once()
        self.assertNotIn("secret click payload", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    async def test_click_tracks_stale_element_under_its_original_browser(
        self,
    ) -> None:
        browser_a = SimpleNamespace()
        browser_b = SimpleNamespace()
        page_a = SimpleNamespace(
            browser=browser_a,
            websocket=object(),
            mapper={},
        )
        page_b = SimpleNamespace(
            browser=browser_b,
            websocket=object(),
            mapper={},
        )
        element_from_a = _StuckElement(page_a)
        action = ElementAction(lambda: cast(zd.Tab, page_b))

        with self.assertRaises(ZendriverOperationTimeout):
            await action.click(element_from_a, operation_timeout=0)

        retirement_a = _begin_zendriver_retirement(browser_a)
        self.assertEqual(retirement_a.owned_connections(), (page_a,))
        self.assertNotIn("_hbrowser_zendriver_lifecycle", vars(browser_b))

        await retirement_a.retire_operations(
            quiescent_connections=(page_a,),
        )


if __name__ == "__main__":
    unittest.main()
