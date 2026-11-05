from __future__ import annotations

import asyncio
import json
import time
import unittest
from collections import defaultdict, deque
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from zendriver import cdp
from zendriver.core.connection import Connection, ProtocolException, Transaction

from hbrowser.gallery.utils import (
    Deadline,
    PageStateTimeout,
    mutate_and_wait_for_navigation,
    mutate_and_wait_for_new_tab,
    navigate_and_wait,
    open_tab_and_wait,
    reload_and_wait,
    wait_for_selector,
    wait_for_zendriver,
)
from hbrowser.gallery.utils import page_state as page_state_module
from hbrowser.gallery.utils import window as window_module
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
        self.serialized_navigation_url: str | None = None
        self.navigation_event_url: str | None = None
        self.navigation_requests: list[str] = []
        self.reload_payload: dict[str, Any] | None = None
        self.selector_results: list[Any] = []
        self.selector_queries = 0
        self.document_objects: dict[str, dict[str, str]] = {}
        self.released_document_objects: list[str] = []
        self.snapshot_object_ids: list[str] = []
        self._next_object_id = 0
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _schedule(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def add_handler(self, event_type: type[Any], handler: Any) -> None:
        self.handlers[event_type].append(handler)

    def remove_handlers(self, event_type: type[Any], handler: Any) -> None:
        self.handlers[event_type].remove(handler)

    async def emit(self, event_type: type[Any], event: Any) -> None:
        for handler in tuple(self.handlers[event_type]):
            await handler(event, self)

    async def _emit_cross_document(self, loader_id: str) -> None:
        self.loader_id = loader_id
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
        match method:
            case "Page.getFrameTree":
                return SimpleNamespace(
                    frame=SimpleNamespace(
                        id_="main-frame",
                        loader_id=self.loader_id,
                    )
                )
            case "Page.setLifecycleEventsEnabled":
                if self.emit_stale_while_enabling:
                    await self._emit_cross_document("stale-loader")
                if self.queue_stale_while_enabling:
                    self._schedule(self._emit_cross_document("queued-stale-loader"))
                return None
            case "Page.createIsolatedWorld":
                assert payload["params"]["frameId"] == "main-frame"
                assert payload["params"]["worldName"] == "hbrowser.navigation-url"
                return cdp.runtime.ExecutionContextId(123)
            case "Runtime.evaluate":
                if payload["params"]["expression"] == "document":
                    assert payload["params"]["returnByValue"] is False
                    self._next_object_id += 1
                    object_id = f"document-{self._next_object_id}"
                    self.document_objects[object_id] = {
                        "url": self.url,
                        "readyState": self.ready_state,
                    }
                    return (
                        SimpleNamespace(
                            object_id=cdp.runtime.RemoteObjectId(object_id)
                        ),
                        None,
                    )
                assert payload["params"]["contextId"] == 123
                assert payload["params"]["returnByValue"] is True
                expression = payload["params"]["expression"]
                requested_url = json.loads(
                    expression.removeprefix("new URL(").removesuffix(").href")
                )
                return (
                    SimpleNamespace(
                        value=self.serialized_navigation_url or requested_url
                    ),
                    None,
                )
            case "Runtime.callFunctionOn":
                params = payload["params"]
                object_id = params["objectId"]
                assert "this.URL" in params["functionDeclaration"]
                assert "this.readyState" in params["functionDeclaration"]
                assert "contextId" not in params
                self.snapshot_object_ids.append(object_id)
                return SimpleNamespace(value=self.document_objects[object_id]), None
            case "Runtime.releaseObject":
                object_id = payload["params"]["objectId"]
                del self.document_objects[object_id]
                self.released_document_objects.append(object_id)
                return None
            case "Page.navigate":
                requested_url = payload["params"]["url"]
                self.navigation_requests.append(requested_url)
                self.url = (
                    self.navigation_event_url
                    or self.serialized_navigation_url
                    or requested_url
                )
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

                        self._schedule(emit_same_document())
                    else:
                        self.loader_id = loader_id
                        await self._emit_cross_document(loader_id)
                return "main-frame", loader_id, None, False
            case "Page.reload":
                self.reload_payload = payload
                if self.emit_navigation:
                    self.loader_id = "reload-loader"
                    await self._emit_cross_document(self.loader_id)
                return None
            case "DOM.enable":
                return None
            case _:
                raise AssertionError(f"Unexpected CDP method: {method}")

    async def evaluate(self, _script: str) -> dict[str, str]:
        return {"url": self.url, "readyState": self.ready_state}

    async def query_selector(self, _selector: str) -> Any:
        self.selector_queries += 1
        result = self.selector_results.pop(0)
        if result is None:
            self._schedule(self.emit(cdp.dom.DocumentUpdated, SimpleNamespace()))
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
    @staticmethod
    async def _trigger_navigation(
        operation: str, page: _EventPage, deadline: Deadline
    ) -> None:
        if operation == "navigate":
            await navigate_and_wait(page, "https://example.test/new", deadline=deadline)
        elif operation == "reload":
            await reload_and_wait(page, deadline=deadline)
        else:
            await mutate_and_wait_for_navigation(
                page,
                AsyncMock(return_value="accepted click"),
                owner=SimpleNamespace(_tab=page),
                operation="Login click",
                deadline=deadline,
            )

    async def test_unfinished_acknowledged_navigation_rejects_later_work(self) -> None:
        for operation in ("navigate", "reload", "click"):
            with self.subTest(operation=operation):
                page = _EventPage()
                page.emit_navigation = False
                with self.assertRaises(PageStateTimeout):
                    await self._trigger_navigation(
                        operation, page, Deadline.after(0.02)
                    )
                later_work = AsyncMock()
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(later_work(), timeout=1, owner=page)
                later_work.assert_not_awaited()
                self.assertFalse(any(page.handlers.values()))

    async def test_cancelled_navigation_retires_its_original_generation(self) -> None:
        for operation in ("navigate", "reload", "click"):
            with self.subTest(operation=operation):
                page = _EventPage()
                page.emit_navigation = False
                original_browser = page.browser
                replacement = SimpleNamespace()
                waiting = asyncio.Event()
                original_wait = page_state_module._NavigationObserver.wait

                async def wait_for_lifecycle(
                    observer: Any,
                    *args: Any,
                    _waiting: asyncio.Event = waiting,
                    _wait: Any = original_wait,
                    **kwargs: Any,
                ) -> None:
                    _waiting.set()
                    await _wait(observer, *args, **kwargs)

                with patch.object(
                    page_state_module._NavigationObserver,
                    "wait",
                    new=wait_for_lifecycle,
                ):
                    task = asyncio.create_task(
                        self._trigger_navigation(operation, page, Deadline.after(5))
                    )
                    try:
                        await asyncio.wait_for(waiting.wait(), timeout=1)
                        page.browser = replacement
                        task.cancel("abandon acknowledged navigation")
                        with self.assertRaises(asyncio.CancelledError) as raised:
                            await task
                    finally:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(
                    raised.exception.args, ("abandon acknowledged navigation",)
                )
                later_work = AsyncMock()
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(
                        later_work(), timeout=1, owner=original_browser
                    )
                later_work.assert_not_awaited()
                await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=replacement)
                self.assertFalse(any(page.handlers.values()))

    async def test_failed_navigation_snapshot_retires_generation(self) -> None:
        page = _EventPage()
        page.ready_state = "loading"
        with self.assertRaises(PageStateTimeout):
            await navigate_and_wait(
                page, "https://example.test/new", deadline=Deadline.after(1)
            )
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_navigation_setup_failure_does_not_retire_generation(self) -> None:
        page = _EventPage()
        with (
            patch.object(
                page_state_module,
                "_enable_lifecycle_events",
                side_effect=ValueError("invalid local setup"),
            ),
            self.assertRaisesRegex(ValueError, "invalid local setup"),
        ):
            await navigate_and_wait(
                page, "https://example.test/new", deadline=Deadline.after(1)
            )
        self.assertFalse(page.navigation_requests)
        self.assertFalse(any(page.handlers.values()))
        await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_created_target_without_inventory_retires_generation(self) -> None:
        browser = SimpleNamespace(targets=[])
        connection = _EventPage()
        connection.browser = browser
        browser.connection = connection
        accepted = AsyncMock(return_value="created-target")
        with (
            patch.object(connection, "send", new=accepted),
            self.assertRaises(PageStateTimeout),
        ):
            await open_tab_and_wait(browser, deadline=Deadline.after(0.02))
        accepted.assert_awaited_once()
        self.assertFalse(any(connection.handlers.values()))
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=browser)

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
                owner=browser,
                operation="Popup click",
                deadline=_RemainingDeadline(1.0, 0.0),  # type: ignore[arg-type]
            )

        mutation.assert_awaited_once_with()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=browser)

    async def test_acknowledged_popup_inventory_timeout_retires_original_browser(
        self,
    ) -> None:
        for events in (False, True):
            with self.subTest(events=events):
                page = _EventPage()
                browser = page.browser
                browser.connection = page if events else None
                browser.tabs = []
                replacement = SimpleNamespace()

                async def mutate(
                    page: _EventPage = page, replacement: Any = replacement
                ) -> str:
                    page.browser = replacement
                    return "ack"

                result = await mutate_and_wait_for_new_tab(
                    cast(Any, browser),
                    set(),
                    mutate,
                    owner=page,
                    operation="Popup click",
                    deadline=Deadline.after(0.01),
                )

                self.assertEqual(result, ("ack", None))
                self.assertFalse(any(page.handlers.values()))
                later = AsyncMock()
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(later(), timeout=1, owner=browser)
                later.assert_not_awaited()
                await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=replacement)

    async def test_acknowledged_popup_inventory_success_keeps_generation_live(
        self,
    ) -> None:
        page = _EventPage()
        browser = page.browser
        browser.connection = page
        browser.tabs = []
        tab = SimpleNamespace(target=SimpleNamespace(target_id="new-tab"))

        async def mutate() -> str:
            browser.tabs.append(tab)
            return "ack"

        self.assertEqual(
            await mutate_and_wait_for_new_tab(
                cast(Any, browser),
                set(),
                mutate,
                owner=page,
                operation="Popup click",
                deadline=Deadline.after(1),
            ),
            ("ack", tab),
        )
        self.assertFalse(any(page.handlers.values()))
        await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=browser)

    async def test_cancelled_popup_wait_retires_original_generation(self) -> None:
        page = _EventPage()
        browser = page.browser
        browser.connection = page
        browser.tabs = []
        replacement_browser = SimpleNamespace()
        waiting = asyncio.Event()
        original_wait = window_module._wait_for_inventory_change

        async def wait_for_inventory(*args: Any, **kwargs: Any) -> Any:
            waiting.set()
            return await original_wait(*args, **kwargs)

        mutation = AsyncMock(return_value="ack")
        with patch.object(
            window_module, "_wait_for_inventory_change", side_effect=wait_for_inventory
        ):
            task = asyncio.create_task(
                mutate_and_wait_for_new_tab(
                    cast(Any, browser),
                    set(),
                    mutation,
                    owner=page,
                    operation="Popup click",
                    deadline=Deadline.after(5),
                )
            )
            await asyncio.wait_for(waiting.wait(), timeout=1)
            # Retirement must remain bound to the original browser even when
            # a page owner is rebound after its mutation acknowledgement.
            page.browser = replacement_browser
            task.cancel("caller cancellation")
            with self.assertRaises(asyncio.CancelledError) as raised:
                await task

        self.assertEqual(raised.exception.args, ("caller cancellation",))
        mutation.assert_awaited_once_with()
        self.assertTrue(all(not handlers for handlers in page.handlers.values()))
        new_work = AsyncMock()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(new_work(), timeout=1, owner=browser)
        new_work.assert_not_awaited()
        await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=replacement_browser)

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
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=browser)

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
        self.assertEqual(page.snapshot_object_ids, ["document-1"])
        self.assertEqual(page.released_document_objects, ["document-1"])
        self.assertFalse(page.document_objects)

    async def test_navigation_receipt_rejects_intervening_document_generations(
        self,
    ) -> None:
        for operation in ("navigate", "reload", "click"):
            for transition in ("different loader", "A-B-A", "reused loader"):
                for phase in ("capture", "snapshot", "after snapshot"):
                    with self.subTest(
                        operation=operation, transition=transition, phase=phase
                    ):
                        await self._assert_document_transition_rejected(
                            operation, transition, phase
                        )

    async def _assert_document_transition_rejected(
        self, operation: str, transition: str, phase: str
    ) -> None:
        page = _EventPage()
        original_send = page.send
        original_browser = page.browser
        injected = False
        target_url = "https://example.test/new"
        mutation = AsyncMock()

        async def change_document() -> None:
            nonlocal injected
            injected = True
            original_loader = page.loader_id
            # The URL is intentionally identical: URL equality does not prove
            # document identity, including a restored BFCache document A.
            page.url = target_url
            if transition != "reused loader":
                await page._emit_cross_document("intervening-B")
            if transition != "different loader":
                await page._emit_cross_document(original_loader)

        async def send(command: Any) -> Any:
            payload = next(command)
            method = payload["method"]
            is_capture = (
                method == "Runtime.evaluate"
                and payload["params"]["expression"] == "document"
            )
            if not injected and (
                (phase == "capture" and is_capture)
                or (phase == "snapshot" and method == "Runtime.callFunctionOn")
            ):
                await change_document()
            result = await original_send(iter((payload,)))
            if (
                not injected
                and phase == "after snapshot"
                and method == "Runtime.callFunctionOn"
            ):
                # Match Zendriver's queued async event callbacks rather than
                # requiring immediate inline event delivery.
                page._schedule(change_document())
            return result

        async def click() -> str:
            page.url = target_url
            await page._emit_cross_document("click-A")
            return "accepted once"

        mutation.side_effect = click
        with patch.object(page, "send", new=send):
            with self.assertRaisesRegex(PageStateTimeout, "document changed"):
                if operation == "navigate":
                    await navigate_and_wait(
                        page, target_url, deadline=Deadline.after(1)
                    )
                elif operation == "reload":
                    await reload_and_wait(page, deadline=Deadline.after(1))
                else:
                    await mutate_and_wait_for_navigation(
                        page,
                        mutation,
                        owner=page,
                        operation="One click",
                        deadline=Deadline.after(1),
                    )
        self.assertTrue(injected)
        self.assertFalse(page.document_objects)
        self.assertFalse(any(page.handlers.values()))
        if operation == "navigate":
            self.assertEqual(page.navigation_requests, [target_url])
        if operation == "click":
            mutation.assert_awaited_once_with()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(
                asyncio.sleep(0), timeout=1, owner=original_browser
            )

    async def test_document_snapshot_uses_and_releases_captured_remote_object(
        self,
    ) -> None:
        page = _EventPage()
        original_url = page.url
        original_send = page.send

        async def send(command: Any) -> Any:
            payload = next(command)
            result = await original_send(iter((payload,)))
            if payload["method"] == "Runtime.evaluate":
                # The next default execution context may belong to another
                # document. callFunctionOn must still address the captured one.
                page.url = "https://example.test/other-document"
                page.ready_state = "loading"
            return result

        with patch.object(page, "send", new=send):
            snapshot = await page_state_module._read_document_snapshot(
                page, Deadline.after(1)
            )
        self.assertEqual(snapshot, (original_url, "complete"))
        self.assertEqual(page.snapshot_object_ids, ["document-1"])
        self.assertEqual(page.released_document_objects, ["document-1"])
        self.assertFalse(page.document_objects)

    async def test_snapshot_protocol_failure_releases_object_and_retires_browser(
        self,
    ) -> None:
        page = _EventPage()
        original_send = page.send

        async def send(command: Any) -> Any:
            payload = next(command)
            if payload["method"] == "Runtime.callFunctionOn":
                return SimpleNamespace(value=None), {"message": "Document destroyed"}
            return await original_send(iter((payload,)))

        with patch.object(page, "send", new=send), self.assertRaises(ProtocolException):
            await navigate_and_wait(
                page, "https://example.test/new", deadline=Deadline.after(1)
            )
        self.assertFalse(page.document_objects)
        self.assertEqual(page.released_document_objects, ["document-1"])
        self.assertFalse(any(page.handlers.values()))
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_same_document_receipt_rejects_subsequent_fragment_change(
        self,
    ) -> None:
        page = _EventPage()
        page.navigation_loader = None
        original_send = page.send

        async def send(command: Any) -> Any:
            payload = next(command)
            if payload["method"] == "Runtime.callFunctionOn":
                page.url = "https://example.test/old#other"
                await page.emit(
                    cdp.page.NavigatedWithinDocument,
                    SimpleNamespace(frame_id="main-frame", url=page.url),
                )
            return await original_send(iter((payload,)))

        with (
            patch.object(page, "send", new=send),
            self.assertRaisesRegex(PageStateTimeout, "document changed"),
        ):
            await navigate_and_wait(
                page, "https://example.test/old#requested", deadline=Deadline.after(1)
            )
        self.assertFalse(page.document_objects)
        self.assertFalse(any(page.handlers.values()))

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

        # A healthy transport is not proof that its document mutation settled.
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=0.1, owner=page)

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
        await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

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

    async def test_same_document_pre_ack_events_cannot_substitute_requested_url(
        self,
    ) -> None:
        page = _EventPage()
        page.navigation_loader = None
        requested_url = "https://example.test/old#requested"
        other_url = "https://example.test/old#other"
        original_send = page.send

        async def send(command: Any) -> Any:
            payload = next(command)
            if payload["method"] == "Page.navigate":
                page.navigation_requests.append(payload["params"]["url"])
                # Both events precede the command reply and expect_loader().
                # Finding the requested URL somewhere in history is not proof
                # that it is still the document's current navigation result.
                for url in (requested_url, other_url):
                    page.url = url
                    await page.emit(
                        cdp.page.NavigatedWithinDocument,
                        SimpleNamespace(frame_id="main-frame", url=url),
                    )
                return "main-frame", None, None, False
            return await original_send(iter((payload,)))

        with patch.object(page, "send", new=send), self.assertRaises(PageStateTimeout):
            await navigate_and_wait(page, requested_url, deadline=Deadline.after(0.05))

        self.assertEqual(page.navigation_requests, [requested_url])
        self.assertEqual(page.url, other_url)
        self.assertFalse(page.snapshot_object_ids)
        self.assertFalse(any(page.handlers.values()))
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_same_document_pre_ack_latest_matching_target_can_complete(
        self,
    ) -> None:
        page = _EventPage()
        page.navigation_loader = None
        requested_url = "https://example.test/old#requested"
        original_send = page.send

        async def send(command: Any) -> Any:
            payload = next(command)
            if payload["method"] == "Page.navigate":
                page.navigation_requests.append(payload["params"]["url"])
                for url in ("https://example.test/old#intermediate", requested_url):
                    page.url = url
                    await page.emit(
                        cdp.page.NavigatedWithinDocument,
                        SimpleNamespace(frame_id="main-frame", url=url),
                    )
                return "main-frame", None, None, False
            return await original_send(iter((payload,)))

        with patch.object(page, "send", new=send):
            receipt = await navigate_and_wait(
                page, requested_url, deadline=Deadline.after(1)
            )

        self.assertIsNone(receipt.loader_id)
        self.assertEqual(receipt.url, requested_url)
        self.assertEqual(page.navigation_requests, [requested_url])
        self.assertFalse(page.document_objects)
        self.assertFalse(any(page.handlers.values()))

    async def test_same_document_uses_browser_serialized_url(self) -> None:
        for requested, serialized in (
            ("https://example.test#fragment", "https://example.test/#fragment"),
            ("https://EXAMPLE.test:443/#fragment", "https://example.test/#fragment"),
            ("https://example.test/a/../#fragment", "https://example.test/#fragment"),
            ("https://example.test/#a b", "https://example.test/#a%20b"),
            ('https://example.test/#a"b', "https://example.test/#a%22b"),
        ):
            with self.subTest(requested=requested):
                page = _EventPage()
                page.navigation_loader = None
                page.serialized_navigation_url = serialized
                receipt = await navigate_and_wait(
                    page, requested, deadline=Deadline.after(0.5)
                )
                self.assertEqual(receipt.url, serialized)
                self.assertEqual(page.navigation_requests, [requested])
                self.assertFalse(any(page.handlers.values()))
                await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_same_document_does_not_accept_a_different_serialized_fragment(
        self,
    ) -> None:
        page = _EventPage()
        page.navigation_loader = None
        page.serialized_navigation_url = "https://example.test/#expected"
        page.navigation_event_url = "https://example.test/#other"
        with self.assertRaises(PageStateTimeout):
            await navigate_and_wait(
                page, "https://example.test#expected", deadline=Deadline.after(0.03)
            )
        self.assertFalse(any(page.handlers.values()))

    async def test_same_document_rejects_invalid_serialization(self) -> None:
        for value, error in (
            (None, None),
            ("", None),
            (123, None),
            (None, {"message": "serialization failed"}),
        ):
            with self.subTest(value=value, error=error):
                page = _EventPage()
                page.navigation_loader = None

                async def send(
                    command: Any,
                    result_value: Any = value,
                    result_error: Any = error,
                    target: _EventPage = page,
                ) -> Any:
                    payload = next(command)
                    if payload["method"] == "Runtime.evaluate":
                        return SimpleNamespace(value=result_value), result_error
                    return await _EventPage.send(target, iter((payload,)))

                expected_error = ProtocolException if error else TypeError
                with (
                    patch.object(page, "send", new=send),
                    self.assertRaises(expected_error),
                ):
                    await navigate_and_wait(
                        page,
                        "https://example.test/#fragment",
                        deadline=Deadline.after(1),
                    )
                self.assertFalse(any(page.handlers.values()))
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_same_document_rejects_serialization_after_deadline(self) -> None:
        page = _EventPage()
        page.navigation_loader = None
        with self.assertRaisesRegex(PageStateTimeout, "serialization.*deadline"):
            await navigate_and_wait(
                page,
                "https://example.test/#fragment",
                deadline=_ExpiredAfterCommandDeadline(),  # type: ignore[arg-type]
            )
        self.assertFalse(any(page.handlers.values()))

    async def test_cross_document_navigation_does_not_serialize_url(self) -> None:
        page = _EventPage()
        with patch.object(page_state_module, "_serialize_navigation_url") as serialize:
            receipt = await navigate_and_wait(
                page, "https://example.test/new", deadline=Deadline.after(1)
            )
        self.assertEqual(receipt.loader_id, "new-loader")
        serialize.assert_not_called()

    async def test_reload_guards_the_previous_loader_and_waits_for_new_one(
        self,
    ) -> None:
        page = _EventPage()

        receipt = await reload_and_wait(page, deadline=Deadline.after(0.5))

        assert page.reload_payload is not None
        self.assertEqual(page.reload_payload["params"]["loaderId"], "old-loader")
        self.assertEqual(receipt.loader_id, "reload-loader")

    async def test_selector_after_navigation_settles_real_zendriver_handlers(
        self,
    ) -> None:
        for initially_missing in (False, True):
            with self.subTest(initially_missing=initially_missing):
                connection = Connection("ws://offline.invalid")
                connection.websocket = cast(Any, object())
                connection.listener = cast(Any, SimpleNamespace(running=True))
                element = object()
                results = deque([None, element] if initially_missing else [element])

                async def transport(
                    transaction: Transaction[Any],
                    target: Connection = connection,
                ) -> None:
                    response = (
                        {"result": {"type": "number", "value": 1}}
                        if transaction.method == "Runtime.evaluate"
                        else {}
                    )
                    transaction(result=response)
                    assert transaction.id is not None
                    target.mapper.pop(transaction.id)

                async def query(
                    _selector: str,
                    target: Connection = connection,
                    matches: deque[object] = results,
                ) -> object | None:
                    await target.send(cdp.runtime.evaluate("1", return_by_value=True))
                    return matches.popleft()

                async def observed(_event: object) -> None:
                    pass

                cast(Any, connection).query_selector = query
                with patch.object(connection, "_send_transaction", new=transport):
                    # Real registration/removal establishes the old domain state.
                    connection.add_handler(cdp.page.LifecycleEvent, observed)
                    await connection.send(
                        cdp.runtime.evaluate("1", return_by_value=True)
                    )
                    connection.remove_handlers(cdp.page.LifecycleEvent, observed)
                    self.assertIn(cdp.page, connection.enabled_domains)
                    result = await wait_for_selector(
                        connection, "#ready", deadline=Deadline.after(1)
                    )
                self.assertIs(result, element)
                self.assertFalse(connection.mapper)

    async def test_missing_selector_cleanup_covers_timeout_and_cancellation(
        self,
    ) -> None:
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                page = _EventPage()
                subscribed = asyncio.Event()

                async def send(
                    command: Any,
                    target: _EventPage = page,
                    signal: asyncio.Event = subscribed,
                ) -> Any:
                    result = await _EventPage.send(target, command)
                    if any(
                        target.handlers[event]
                        for event in page_state_module._DOM_CHANGE_EVENTS
                    ):
                        signal.set()
                    return result

                with (
                    patch.object(page, "send", new=send),
                    patch.object(
                        page, "query_selector", new=AsyncMock(return_value=None)
                    ),
                ):
                    task = asyncio.create_task(
                        wait_for_selector(
                            page,
                            "#missing",
                            deadline=Deadline.after(0.04),
                        )
                    )
                    await asyncio.wait_for(subscribed.wait(), timeout=1)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        with self.assertRaises(PageStateTimeout):
                            await task
                for event in page_state_module._DOM_CHANGE_EVENTS:
                    self.assertEqual(page.handlers[event], [])

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
