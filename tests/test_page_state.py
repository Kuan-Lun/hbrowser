from __future__ import annotations

import asyncio
import time
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from zendriver import cdp

from hbrowser.gallery.utils import (
    Deadline,
    PageStateTimeout,
    mutate_and_wait_for_navigation,
    mutate_and_wait_for_new_tab,
    navigate_and_wait,
    reload_and_wait,
    wait_for_selector,
    wait_for_zendriver,
)
from hbrowser.gallery.utils import page_state as page_state_module
from hbrowser.gallery.utils.protocol import (
    ZendriverOperationTimeout,
    ZendriverOwnerRetiredError,
)


class _EventPage:
    def __init__(self) -> None:
        self.browser = SimpleNamespace()
        self.websocket = object()
        self.mapper: dict[str, object] = {}
        self.handlers: dict[type[Any], list[Any]] = defaultdict(list)
        self.loader_id = "old-loader"
        self.url = "https://example.test/old"
        self.ready_state = "complete"
        self.navigation_loader: str | None = "new-loader"
        self.emit_navigation = True
        self.emit_stale_while_enabling = False
        self.queue_stale_while_enabling = False
        self.delay_same_document = 0.0
        self.reload_payload: dict[str, Any] | None = None
        self.selector_results: list[Any] = []
        self.selector_queries = 0

    def add_handler(self, event_type: type[Any], handler: Any) -> None:
        self.handlers[event_type].append(handler)

    def remove_handlers(self, event_type: type[Any], handler: Any) -> None:
        self.handlers[event_type].remove(handler)

    async def emit(self, event_type: type[Any], event: Any) -> None:
        for handler in tuple(self.handlers[event_type]):
            await handler(event, self)

    async def _emit_cross_document(self, loader_id: str) -> None:
        await self.emit(
            cdp.page.FrameNavigated,
            SimpleNamespace(
                frame=SimpleNamespace(id_="main-frame", loader_id=loader_id)
            ),
        )
        await self.emit(
            cdp.page.LifecycleEvent,
            SimpleNamespace(
                frame_id="main-frame",
                loader_id=loader_id,
                name="DOMContentLoaded",
            ),
        )

    async def send(self, command: Any) -> Any:
        payload = next(command)
        method = payload["method"]
        if method == "Page.getFrameTree":
            return SimpleNamespace(
                frame=SimpleNamespace(
                    id_="main-frame",
                    loader_id=self.loader_id,
                )
            )
        if method == "Page.setLifecycleEventsEnabled":
            if self.emit_stale_while_enabling:
                await self._emit_cross_document("stale-loader")
            if self.queue_stale_while_enabling:
                asyncio.create_task(self._emit_cross_document("queued-stale-loader"))
            return None
        if method == "Page.navigate":
            self.url = payload["params"]["url"]
            loader_id = self.navigation_loader
            if self.emit_navigation:
                if loader_id is None:

                    async def emit_same_document() -> None:
                        if self.delay_same_document:
                            await asyncio.sleep(self.delay_same_document)
                        await self.emit(
                            cdp.page.NavigatedWithinDocument,
                            SimpleNamespace(
                                frame_id="main-frame",
                                url=self.url,
                            ),
                        )

                    asyncio.create_task(emit_same_document())
                else:
                    self.loader_id = loader_id
                    await self._emit_cross_document(loader_id)
            return "main-frame", loader_id, None, False
        if method == "Page.reload":
            self.reload_payload = payload
            if self.emit_navigation:
                self.loader_id = "reload-loader"
                await self._emit_cross_document(self.loader_id)
            return None
        if method == "DOM.enable":
            return None
        raise AssertionError(f"Unexpected CDP method: {method}")

    async def evaluate(self, _script: str) -> dict[str, str]:
        return {"url": self.url, "readyState": self.ready_state}

    async def query_selector(self, _selector: str) -> Any:
        self.selector_queries += 1
        result = self.selector_results.pop(0)
        if result is None:
            asyncio.create_task(self.emit(cdp.dom.DocumentUpdated, SimpleNamespace()))
        return result


class _ExpiredAfterCommandDeadline:
    """A deterministic deadline that expires after an immediate command."""

    def bounded(self, _seconds: float) -> _ExpiredAfterCommandDeadline:
        return self

    def remaining(self) -> float:
        return 1.0

    @property
    def expired(self) -> bool:
        return True


class _RemainingDeadline:
    def __init__(self, *remaining: float) -> None:
        self._remaining = deque(remaining)

    def bounded(self, _seconds: float) -> _RemainingDeadline:
        return self

    def remaining(self) -> float:
        return self._remaining.popleft()


class _ExpiresDuringCommandDeadline:
    """Expose a tiny budget that expires only after one command starts."""

    def __init__(self) -> None:
        self._expiration_checks = 0

    def bounded(self, _seconds: float) -> _ExpiresDuringCommandDeadline:
        return self

    def remaining(self) -> float:
        return 0.001

    @property
    def expired(self) -> bool:
        self._expiration_checks += 1
        return self._expiration_checks > 1


class PageStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_tab_mutation_late_ack_is_not_accepted(self) -> None:
        mutation = AsyncMock(return_value="ack")
        browser = SimpleNamespace(connection=None, tabs=[])

        async def complete_mutation(awaitable: Any, **_kwargs: Any) -> Any:
            return await awaitable

        with (
            patch(
                "hbrowser.gallery.utils.window.wait_for_zendriver_mutation",
                new=AsyncMock(side_effect=complete_mutation),
            ),
            self.assertRaisesRegex(TimeoutError, "completed after"),
        ):
            await mutate_and_wait_for_new_tab(
                cast(Any, browser),
                set(),
                mutation,
                owner=object(),
                operation="Popup click",
                deadline=_RemainingDeadline(1.0, 0.0),  # type: ignore[arg-type]
            )

        mutation.assert_awaited_once_with()

    async def test_fallback_navigation_rejects_a_late_command_success(self) -> None:
        browser = SimpleNamespace()
        page = SimpleNamespace(
            browser=browser,
            websocket=object(),
            mapper={},
            get=AsyncMock(return_value=None),
        )

        with self.assertRaisesRegex(PageStateTimeout, "completed after"):
            await navigate_and_wait(
                page,
                "https://example.test/new",
                deadline=_ExpiredAfterCommandDeadline(),  # type: ignore[arg-type]
            )

    async def test_done_lifecycle_future_is_rejected_after_deadline(self) -> None:
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)

        with self.assertRaisesRegex(PageStateTimeout, "completed after"):
            await page_state_module._wait_for_future(
                future,
                _ExpiredAfterCommandDeadline(),  # type: ignore[arg-type]
                description="Late lifecycle",
            )

    async def test_dom_match_is_rejected_after_deadline(self) -> None:
        page = _EventPage()
        page.selector_results = [object()]

        with self.assertRaisesRegex(PageStateTimeout, "completed after"):
            await wait_for_selector(
                page,
                "#late",
                deadline=_ExpiredAfterCommandDeadline(),  # type: ignore[arg-type]
            )

    async def test_navigation_accepts_lifecycle_before_command_reply(self) -> None:
        page = _EventPage()

        receipt = await navigate_and_wait(
            page,
            "https://example.test/new",
            deadline=Deadline.after(0.5),
        )

        self.assertEqual(receipt.loader_id, "new-loader")
        self.assertEqual(receipt.url, "https://example.test/new")
        self.assertEqual(page.handlers[cdp.page.FrameNavigated], [])
        self.assertEqual(page.handlers[cdp.page.LifecycleEvent], [])

    async def test_setup_event_before_trigger_cannot_complete_navigation(self) -> None:
        page = _EventPage()
        page.emit_stale_while_enabling = True
        page.emit_navigation = False

        with self.assertRaises(PageStateTimeout):
            await navigate_and_wait(
                page,
                "https://example.test/new",
                deadline=Deadline.after(0.03),
            )

        # A semantic state timeout leaves the proven transport generation usable.
        result = await wait_for_zendriver(
            asyncio.sleep(0, result="healthy"),
            timeout=0.1,
            owner=page,
        )
        self.assertEqual(result, "healthy")

    async def test_queued_setup_event_is_drained_before_trigger(self) -> None:
        page = _EventPage()
        page.queue_stale_while_enabling = True
        page.emit_navigation = False

        with self.assertRaises(PageStateTimeout):
            await navigate_and_wait(
                page,
                "https://example.test/new",
                deadline=Deadline.after(0.03),
            )

    async def test_expired_deadline_does_not_invoke_mutation(self) -> None:
        page = SimpleNamespace()
        mutation = AsyncMock()

        with self.assertRaises(PageStateTimeout):
            await mutate_and_wait_for_navigation(
                page,
                mutation,
                owner=page,
                operation="Expired mutation",
                deadline=Deadline(0.0),
            )

        mutation.assert_not_awaited()

    async def test_same_document_ack_waits_for_matching_event(self) -> None:
        page = _EventPage()
        page.navigation_loader = None
        page.delay_same_document = 0.03
        started = time.monotonic()

        receipt = await navigate_and_wait(
            page,
            "https://example.test/old#new-fragment",
            deadline=Deadline.after(0.5),
        )

        self.assertGreaterEqual(time.monotonic() - started, 0.02)
        self.assertIsNone(receipt.loader_id)
        self.assertEqual(receipt.url, "https://example.test/old#new-fragment")

    async def test_reload_guards_the_previous_loader_and_waits_for_new_one(
        self,
    ) -> None:
        page = _EventPage()

        receipt = await reload_and_wait(page, deadline=Deadline.after(0.5))

        assert page.reload_payload is not None
        self.assertEqual(page.reload_payload["params"]["loaderId"], "old-loader")
        self.assertEqual(receipt.loader_id, "reload-loader")

    async def test_selector_uses_dom_signal_and_immediate_queries(self) -> None:
        page = _EventPage()
        element = object()
        page.selector_results = [None, element]

        with patch.object(page_state_module, "_DOM_RECONCILIATION_SECONDS", 1.0):
            result = await asyncio.wait_for(
                wait_for_selector(
                    page,
                    "#ready",
                    deadline=Deadline.after(0.5),
                ),
                timeout=0.25,
            )

        self.assertIs(result, element)
        self.assertEqual(page.selector_queries, 2)
        for event_type in page_state_module._DOM_CHANGE_EVENTS:
            self.assertEqual(page.handlers[event_type], [])

    async def test_close_to_deadline_reply_is_a_healthy_semantic_timeout(
        self,
    ) -> None:
        browser = SimpleNamespace()

        async def completed_query(_selector: str) -> object:
            return object()

        page = SimpleNamespace(browser=browser, query_selector=completed_query)
        with (
            patch.object(
                page_state_module,
                "ZENDRIVER_COMMAND_TIMEOUT_SECONDS",
                0.05,
            ),
            self.assertRaisesRegex(PageStateTimeout, "completed after"),
        ):
            await wait_for_selector(
                page,
                "#late",
                deadline=_ExpiresDuringCommandDeadline(),  # type: ignore[arg-type]
            )

        result = await wait_for_zendriver(
            asyncio.sleep(0, result="healthy"),
            timeout=0.05,
            owner=page,
        )
        self.assertEqual(result, "healthy")

    async def test_43ms_remaining_does_not_become_a_protocol_watchdog(
        self,
    ) -> None:
        browser = SimpleNamespace()

        async def reply_after_semantic_deadline(_selector: str) -> object:
            await asyncio.sleep(0.06)
            return object()

        page = SimpleNamespace(
            browser=browser,
            query_selector=reply_after_semantic_deadline,
        )
        with (
            patch.object(
                page_state_module,
                "ZENDRIVER_COMMAND_TIMEOUT_SECONDS",
                0.2,
            ),
            self.assertRaises(PageStateTimeout),
        ):
            await wait_for_selector(
                page,
                "#late",
                deadline=Deadline.after(0.043),
            )

        result = await wait_for_zendriver(
            asyncio.sleep(0, result="healthy"),
            timeout=0.05,
            owner=page,
        )
        self.assertEqual(result, "healthy")

    async def test_no_event_at_deadline_does_not_submit_a_final_query(self) -> None:
        query_selector = AsyncMock(return_value=None)
        page = SimpleNamespace(query_selector=query_selector)

        with self.assertRaisesRegex(PageStateTimeout, "did not appear"):
            await wait_for_selector(
                page,
                "#absent",
                deadline=Deadline.after(0.02),
            )

        query_selector.assert_awaited_once_with("#absent")

    async def test_final_interval_rechecks_deadline_before_timing_out(self) -> None:
        protocol_wait = AsyncMock(return_value=(set(), set()))

        with (
            patch(
                "hbrowser.gallery.utils.page_state.asyncio.wait",
                protocol_wait,
            ),
            self.assertRaisesRegex(PageStateTimeout, "did not appear"),
        ):
            await page_state_module._wait_for_dom_change(
                asyncio.Event(),
                deadline=_RemainingDeadline(0.043, 0.01, 0.0),  # type: ignore[arg-type]
                description="Selector '#absent'",
            )

        self.assertEqual(protocol_wait.await_count, 2)
        self.assertEqual(
            [call.kwargs["timeout"] for call in protocol_wait.await_args_list],
            [0.043, 0.01],
        )

    async def test_hung_dom_query_uses_watchdog_and_retires_generation(
        self,
    ) -> None:
        browser = SimpleNamespace()
        release_query = asyncio.Event()

        async def hung_query(_selector: str) -> None:
            await release_query.wait()

        page = SimpleNamespace(browser=browser, query_selector=hung_query)
        try:
            with (
                patch.object(
                    page_state_module,
                    "ZENDRIVER_COMMAND_TIMEOUT_SECONDS",
                    0.03,
                ),
                self.assertRaises(ZendriverOperationTimeout) as raised,
            ):
                await wait_for_selector(
                    page,
                    "#hung",
                    deadline=Deadline.after(0.005),
                )

            self.assertEqual(raised.exception.timeout_seconds, 0.03)
            with self.assertRaises(ZendriverOwnerRetiredError):
                await wait_for_zendriver(
                    asyncio.sleep(0),
                    timeout=0.05,
                    owner=page,
                )
        finally:
            release_query.set()
            await asyncio.sleep(0)

    async def test_command_timeout_still_retires_generation(self) -> None:
        page = _EventPage()

        async def hang(_command: Any) -> Any:
            await asyncio.Event().wait()

        page.send = hang  # type: ignore[assignment]
        with self.assertRaises(ZendriverOperationTimeout):
            await navigate_and_wait(
                page,
                "https://example.test/new",
                deadline=Deadline.after(0.02),
            )


if __name__ == "__main__":
    unittest.main()
