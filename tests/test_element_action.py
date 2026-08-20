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

    async def test_click_resilient_does_not_retry_unknown_outcome(self) -> None:
        page = cast(zd.Tab, object())
        action = ElementAction(lambda: page)
        get_element_calls = 0
        click_calls = 0

        async def get_element() -> _RespondingElement:
            nonlocal get_element_calls
            get_element_calls += 1
            return _RespondingElement()

        async def unknown_outcome_click(
            element: Any,
            *,
            operation_timeout: float,
        ) -> None:
            nonlocal click_calls
            del element
            del operation_timeout
            click_calls += 1
            raise ZendriverOperationTimeout(timeout_seconds=3.0)

        action.click = unknown_outcome_click  # type: ignore[method-assign]
        with self.assertRaises(ZendriverOperationTimeout):
            await action.click_resilient(
                get_element,
                retries=3,
                delay=0,
                operation_timeout=1,
            )

        self.assertEqual(get_element_calls, 1)
        self.assertEqual(click_calls, 1)

    async def test_click_resilient_does_not_retry_timed_out_lookup(self) -> None:
        page = cast(zd.Tab, object())
        action = ElementAction(lambda: page)
        element = _RespondingElement()
        get_element_calls = 0

        async def get_element() -> _RespondingElement:
            nonlocal get_element_calls
            get_element_calls += 1
            if get_element_calls == 1:
                raise ZendriverOperationTimeout(timeout_seconds=1.0)
            return element

        with self.assertRaises(ZendriverOperationTimeout):
            await action.click_resilient(
                get_element,
                retries=2,
                delay=0,
                operation_timeout=1,
            )

        self.assertEqual(get_element_calls, 1)
        self.assertEqual(len(element.calls), 0)

    async def test_click_resilient_retries_safe_lookup_error(self) -> None:
        page = cast(zd.Tab, object())
        action = ElementAction(lambda: page)
        element = _RespondingElement()
        get_element_calls = 0

        async def get_element() -> _RespondingElement:
            nonlocal get_element_calls
            get_element_calls += 1
            if get_element_calls == 1:
                raise RuntimeError("element not ready")
            return element

        await action.click_resilient(
            get_element,
            retries=2,
            delay=0,
            operation_timeout=1,
        )

        self.assertEqual(get_element_calls, 2)
        self.assertEqual(len(element.calls), 1)

    async def test_click_resilient_does_not_replay_generic_click_failure(self) -> None:
        action = ElementAction(lambda: cast(zd.Tab, object()))
        click_calls = 0

        async def get_element() -> _RespondingElement:
            return _RespondingElement()

        async def failed_click(
            element: Any,
            *,
            operation_timeout: float,
        ) -> None:
            nonlocal click_calls
            del element, operation_timeout
            click_calls += 1
            raise RuntimeError("click outcome unknown")

        action.click = failed_click  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            await action.click_resilient(
                get_element,
                retries=3,
                delay=0,
                operation_timeout=1,
            )

        self.assertEqual(click_calls, 1)


class ElementActionClickLocatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_times_out_instead_of_hanging_forever(self) -> None:
        action = ElementAction(lambda: cast(zd.Tab, _HangingPage()))

        with self.assertRaises(TimeoutError):
            await action.click_locator(
                "#sel",
                retries=1,
                wait_timeout=0.05,
                delay=0,
                operation_timeout=1,
            )

    async def test_succeeds_when_element_found(self) -> None:
        element = _RespondingElement()
        action = ElementAction(lambda: cast(zd.Tab, _RespondingPage(element=element)))

        await action.click_locator(
            "#sel",
            wait_timeout=1.0,
            operation_timeout=1,
        )

        self.assertEqual(len(element.calls), 1)

    async def test_does_not_retry_a_click_with_unknown_outcome(self) -> None:
        element = _RespondingElement()
        page = _RespondingPage(element=element)
        action = ElementAction(lambda: cast(zd.Tab, page))
        click_calls = 0

        async def unknown_outcome_click(
            clicked_element: Any,
            *,
            operation_timeout: float,
        ) -> None:
            nonlocal click_calls
            self.assertIs(clicked_element, element)
            del operation_timeout
            click_calls += 1
            raise ZendriverOperationTimeout(timeout_seconds=3.0)

        action.click = unknown_outcome_click  # type: ignore[method-assign,assignment]
        with self.assertRaises(ZendriverOperationTimeout):
            await action.click_locator(
                "#sel",
                retries=3,
                wait_timeout=1.0,
                delay=0,
                operation_timeout=1,
            )

        self.assertEqual(page.select_calls, ["#sel"])
        self.assertEqual(click_calls, 1)


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
                operation_timeout=1,
            )

    async def test_returns_once_condition_is_already_true(self) -> None:
        page = _RespondingPage()
        action = ElementAction(lambda: cast(zd.Tab, page))

        async def get_element() -> Any:
            raise AssertionError("should not click when condition is already true")

        async def condition() -> bool:
            return True

        await action.click_until(
            get_element,
            condition,
            max_attempts=1,
            operation_timeout=1,
        )

    async def test_does_not_retry_a_click_with_unknown_outcome(self) -> None:
        page = _RespondingPage()
        action = ElementAction(lambda: cast(zd.Tab, page))
        get_element_calls = 0

        async def get_element() -> _RespondingElement:
            nonlocal get_element_calls
            get_element_calls += 1
            return page.element

        async def condition() -> bool:
            return False

        async def unknown_outcome_click(
            element: Any,
            *,
            operation_timeout: float,
        ) -> None:
            del element
            del operation_timeout
            raise ZendriverOperationTimeout(timeout_seconds=3.0)

        action.click = unknown_outcome_click  # type: ignore[method-assign]
        with self.assertRaises(ZendriverOperationTimeout):
            await action.click_until(
                get_element,
                condition,
                max_attempts=3,
                delay=0,
                operation_timeout=1,
            )

        self.assertEqual(get_element_calls, 1)


if __name__ == "__main__":
    unittest.main()
