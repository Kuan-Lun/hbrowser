import asyncio
import unittest
from typing import Any

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


if __name__ == "__main__":
    unittest.main()
