import asyncio
import unittest
from typing import Any, cast

import zendriver as zd

from hbrowser.gallery.element_action import ElementAction


class _StuckElement:
    """An element whose apply() never resolves, simulating an unresponsive tab."""

    async def apply(self, js_function: str) -> Any:
        await asyncio.Event().wait()


class _RespondingElement:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def apply(self, js_function: str) -> Any:
        self.calls.append(js_function)
        return None


class _HangingPage:
    """A page whose get_content()/select() never resolve."""

    async def get_content(self) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def select(self, selector: str, timeout: float) -> Any:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _RespondingPage:
    def __init__(self, *, content: str = "content", element: Any = None) -> None:
        self.content = content
        self.element = element if element is not None else _RespondingElement()
        self.select_calls: list[str] = []

    async def get_content(self) -> str:
        return self.content

    async def select(self, selector: str, timeout: float) -> Any:
        self.select_calls.append(selector)
        return self.element


class ElementActionClickTests(unittest.IsolatedAsyncioTestCase):
    async def test_click_times_out_instead_of_hanging_forever(self) -> None:
        action = ElementAction(lambda: (_ for _ in ()).throw(AssertionError()))

        with self.assertRaises(TimeoutError):
            await action.click(_StuckElement(), timeout=0.05)

    async def test_click_succeeds_when_element_responds(self) -> None:
        action = ElementAction(lambda: (_ for _ in ()).throw(AssertionError()))
        element = _RespondingElement()

        await action.click(element, timeout=1.0)

        self.assertEqual(len(element.calls), 1)


class ElementActionClickLocatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_times_out_instead_of_hanging_forever(self) -> None:
        action = ElementAction(lambda: cast(zd.Tab, _HangingPage()))

        with self.assertRaises(TimeoutError):
            await action.click_locator("#sel", retries=1, wait_timeout=0.05, delay=0)

    async def test_succeeds_when_element_found(self) -> None:
        element = _RespondingElement()
        action = ElementAction(lambda: cast(zd.Tab, _RespondingPage(element=element)))

        await action.click_locator("#sel", wait_timeout=1.0)

        self.assertEqual(len(element.calls), 1)


class ElementActionClickUntilTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_read_times_out_instead_of_hanging_forever(self) -> None:
        action = ElementAction(lambda: cast(zd.Tab, _HangingPage()))

        async def get_element() -> Any:
            raise AssertionError("should not be reached before the content read")

        async def condition() -> bool:
            return False

        with self.assertRaises(TimeoutError):
            await action.click_until(
                get_element,
                condition,
                max_attempts=1,
                content_read_timeout=0.05,
            )

    async def test_returns_once_condition_is_already_true(self) -> None:
        page = _RespondingPage()
        action = ElementAction(lambda: cast(zd.Tab, page))

        async def get_element() -> Any:
            raise AssertionError("should not click when condition is already true")

        async def condition() -> bool:
            return True

        await action.click_until(get_element, condition, max_attempts=1)


if __name__ == "__main__":
    unittest.main()
