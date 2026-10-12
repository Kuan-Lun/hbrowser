"""Lifecycle- and DOM-event-driven page state transitions.

Zendriver's ``Tab.get`` waits for a quiet websocket and ``Tab.reload`` only
sends ``Page.reload``.  Neither operation proves that the replacement document
has a usable execution context.  The helpers here keep the short CDP command
watchdog separate from the longer semantic page-state deadline.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from zendriver import cdp
from zendriver.core.connection import ProtocolException

from .deadline import Deadline
from .mutation import wait_for_zendriver_mutation
from .protocol import (
    MAX_ZENDRIVER_COMMAND_TIMEOUT_SECONDS,
    _begin_zendriver_retirement,
    _resolve_owner,
    wait_for_zendriver,
)

ZENDRIVER_COMMAND_TIMEOUT_SECONDS = MAX_ZENDRIVER_COMMAND_TIMEOUT_SECONDS
DEFAULT_NAVIGATION_DEADLINE_SECONDS = 10.0
_READY_LIFECYCLE_NAMES = frozenset({"DOMContentLoaded", "load"})
_READY_DOCUMENT_STATES = frozenset({"interactive", "complete"})
_DOM_RECONCILIATION_SECONDS = 0.25
_DOCUMENT_SNAPSHOT_SCRIPT = """(() => ({
    url: window.location.href,
    readyState: document.readyState
}))()"""


class PageStateTimeout(TimeoutError):
    """A healthy transport did not expose the requested semantic page state."""


@contextmanager
def _navigation_completion(browser: Any) -> Iterator[None]:
    """Require semantic completion after ACK for the captured generation."""

    try:
        yield
    except BaseException:
        # A live transport and a successful command reply do not settle the
        # document mutation. Keep the original browser terminal even if a tab
        # owner was rebound while its navigation was completing.
        _begin_zendriver_retirement(browser)
        raise


def _page_state_deadline(deadline: Deadline | None) -> Deadline:
    """Apply the policy ceiling without resetting a caller's earlier deadline."""

    if deadline is None:
        return Deadline.after(DEFAULT_NAVIGATION_DEADLINE_SECONDS)
    return deadline.bounded(DEFAULT_NAVIGATION_DEADLINE_SECONDS)


def _require_deadline(deadline: Deadline, description: str) -> None:
    if deadline.expired:
        raise PageStateTimeout(f"{description} completed after its semantic deadline")


@dataclass(frozen=True, slots=True)
class NavigationReceipt:
    frame_id: str
    loader_id: str | None
    url: str
    ready_state: str


def _supports_cdp_events(page: Any) -> bool:
    return (
        callable(getattr(page, "add_handler", None))
        and callable(getattr(page, "remove_handlers", None))
        and inspect.iscoroutinefunction(getattr(page, "send", None))
    )


def _command_timeout(
    deadline: Deadline,
    requested_timeout: float = ZENDRIVER_COMMAND_TIMEOUT_SECONDS,
) -> float:
    if (
        isinstance(requested_timeout, bool)
        or not isinstance(requested_timeout, int | float)
        or not math.isfinite(float(requested_timeout))
        or requested_timeout < 0
    ):
        raise ValueError("CDP command timeout must be finite and non-negative")
    if requested_timeout > ZENDRIVER_COMMAND_TIMEOUT_SECONDS:
        raise ValueError(
            "CDP command timeout must not exceed "
            f"{ZENDRIVER_COMMAND_TIMEOUT_SECONDS:g} seconds"
        )
    remaining = deadline.remaining()
    if remaining <= 0:
        raise PageStateTimeout("page-state deadline expired before a CDP command")
    return min(ZENDRIVER_COMMAND_TIMEOUT_SECONDS, requested_timeout, remaining)


async def _read_document_snapshot(page: Any, deadline: Deadline) -> tuple[str, str]:
    command_timeout = _command_timeout(deadline)
    value = await wait_for_zendriver(
        page.evaluate(_DOCUMENT_SNAPSHOT_SCRIPT),
        timeout=command_timeout,
        owner=page,
    )
    _require_deadline(deadline, "Document snapshot")
    if not isinstance(value, dict):
        raise TypeError("document snapshot was not an object")
    url = value.get("url")
    ready_state = value.get("readyState")
    if not isinstance(url, str) or not isinstance(ready_state, str):
        raise TypeError("document snapshot contained invalid fields")
    return url, ready_state


async def _wait_for_future(
    future: asyncio.Future[None],
    deadline: Deadline,
    *,
    description: str,
) -> None:
    remaining = deadline.remaining()
    if remaining <= 0:
        future.cancel()
        raise PageStateTimeout(f"{description} exceeded its semantic deadline")
    done, _ = await asyncio.wait((future,), timeout=remaining)
    if not done:
        future.cancel()
        raise PageStateTimeout(f"{description} exceeded its semantic deadline")
    future.result()
    _require_deadline(deadline, description)


class _NavigationObserver:
    def __init__(
        self,
        page: Any,
        *,
        main_frame_id: str,
        previous_loader_id: str | None,
        infer_loader: bool,
        allow_same_document: bool,
    ) -> None:
        self._page = page
        self._main_frame_id = main_frame_id
        self._previous_loader_id = previous_loader_id
        self._infer_loader = infer_loader
        self._expects_same_document = allow_same_document
        self._expected_loader_id: str | None = None
        self._ready_lifecycles: set[tuple[str, str]] = set()
        self._same_document_urls: list[str] = []
        self._expected_same_document_url: str | None = None
        self._armed = False
        self._future = asyncio.get_running_loop().create_future()

    async def frame_navigated(self, event: Any, _connection: Any = None) -> None:
        if not self._armed:
            return
        frame = event.frame
        frame_id = str(frame.id_)
        loader_id = str(frame.loader_id)
        if frame_id != self._main_frame_id:
            return
        if (
            self._previous_loader_id is not None
            and loader_id == self._previous_loader_id
        ):
            return
        if self._infer_loader and self._expected_loader_id is None:
            self._expected_loader_id = loader_id
        self._complete_if_ready()

    async def lifecycle_event(self, event: Any, _connection: Any = None) -> None:
        if not self._armed:
            return
        if event.name not in _READY_LIFECYCLE_NAMES:
            return
        self._ready_lifecycles.add((str(event.frame_id), str(event.loader_id)))
        self._complete_if_ready()

    async def navigated_within_document(
        self,
        event: Any,
        _connection: Any = None,
    ) -> None:
        if not self._armed:
            return
        if str(event.frame_id) != self._main_frame_id:
            return
        self._same_document_urls.append(str(event.url))
        self._complete_if_ready()

    def expect_loader(
        self,
        loader_id: str | None,
        *,
        same_document_url: str | None = None,
    ) -> None:
        if loader_id is None:
            self._expects_same_document = True
            self._expected_same_document_url = same_document_url
        else:
            self._expected_loader_id = loader_id
        self._complete_if_ready()

    @property
    def loader_id(self) -> str | None:
        return self._expected_loader_id

    @property
    def main_frame_id(self) -> str:
        return self._main_frame_id

    def arm(self) -> None:
        """Start accepting events immediately before the triggering mutation."""

        self._expected_loader_id = None
        self._ready_lifecycles.clear()
        self._same_document_urls.clear()
        self._armed = True

    def _complete_if_ready(self) -> None:
        if self._future.done():
            return
        if self._expects_same_document and self._same_document_urls:
            expected_url = self._expected_same_document_url
            if expected_url is None or expected_url in self._same_document_urls:
                self._future.set_result(None)
                return
        loader_id = self._expected_loader_id
        if loader_id is None:
            return
        if (self._main_frame_id, loader_id) in self._ready_lifecycles:
            self._future.set_result(None)

    async def wait(self, deadline: Deadline, *, description: str) -> None:
        await _wait_for_future(self._future, deadline, description=description)

    def install(self) -> None:
        self._page.add_handler(cdp.page.FrameNavigated, self.frame_navigated)
        self._page.add_handler(cdp.page.LifecycleEvent, self.lifecycle_event)
        self._page.add_handler(
            cdp.page.NavigatedWithinDocument,
            self.navigated_within_document,
        )

    def remove(self) -> None:
        self._page.remove_handlers(cdp.page.FrameNavigated, self.frame_navigated)
        self._page.remove_handlers(cdp.page.LifecycleEvent, self.lifecycle_event)
        self._page.remove_handlers(
            cdp.page.NavigatedWithinDocument,
            self.navigated_within_document,
        )


async def _main_frame_identity(
    page: Any,
    deadline: Deadline,
) -> tuple[str, str | None]:
    command_timeout = _command_timeout(deadline)
    frame_tree = await wait_for_zendriver(
        page.send(cdp.page.get_frame_tree()),
        timeout=command_timeout,
        owner=page,
    )
    frame = frame_tree.frame
    loader_id = getattr(frame, "loader_id", None)
    return str(frame.id_), None if loader_id is None else str(loader_id)


async def _enable_lifecycle_events(page: Any, deadline: Deadline) -> None:
    command_timeout = _command_timeout(deadline)
    await wait_for_zendriver(
        page.send(cdp.page.set_lifecycle_events_enabled(True)),
        timeout=command_timeout,
        owner=page,
    )


async def _finish_navigation(
    page: Any,
    observer: _NavigationObserver,
    deadline: Deadline,
    *,
    description: str,
) -> NavigationReceipt:
    await observer.wait(deadline, description=description)
    url, ready_state = await _read_document_snapshot(page, deadline)
    if ready_state not in _READY_DOCUMENT_STATES:
        raise PageStateTimeout(
            f"{description} produced document.readyState={ready_state!r}"
        )
    return NavigationReceipt(
        frame_id=observer.main_frame_id,
        loader_id=observer.loader_id,
        url=url,
        ready_state=ready_state,
    )


async def _serialize_navigation_url(
    page: Any, url: str, main_frame_id: str, deadline: Deadline
) -> str:
    """Match Chrome's URL serialization without using the page's globals."""

    command_timeout = _command_timeout(deadline)
    context_id = await wait_for_zendriver(
        page.send(
            cdp.page.create_isolated_world(
                cdp.page.FrameId(main_frame_id),
                world_name="hbrowser.navigation-url",
            )
        ),
        timeout=command_timeout,
        owner=page,
    )
    command_timeout = _command_timeout(deadline)
    result, error = await wait_for_zendriver(
        page.send(
            cdp.runtime.evaluate(
                f"new URL({json.dumps(url)}).href",
                context_id=context_id,
                return_by_value=True,
            )
        ),
        timeout=command_timeout,
        owner=page,
    )
    _require_deadline(deadline, "Navigation URL serialization")
    if error is not None:
        raise ProtocolException(error)
    serialized = result.value
    if not isinstance(serialized, str) or not serialized:
        raise TypeError("Chrome did not return a serialized navigation URL")
    return serialized


async def navigate_and_wait(
    page: Any,
    url: str,
    *,
    deadline: Deadline | None = None,
) -> NavigationReceipt:
    """Navigate and await the matching main-frame document lifecycle."""

    operation_deadline = _page_state_deadline(deadline)
    if not _supports_cdp_events(page):
        command_timeout = _command_timeout(operation_deadline)
        original_browser, _ = _resolve_owner(page)
        await wait_for_zendriver_mutation(
            page.get(url),
            timeout=command_timeout,
            owner=page,
            operation="Page navigation",
        )
        with _navigation_completion(original_browser):
            _require_deadline(operation_deadline, "Page navigation")
            return NavigationReceipt("", None, url, "complete")

    main_frame_id, previous_loader_id = await _main_frame_identity(
        page,
        operation_deadline,
    )
    observer = _NavigationObserver(
        page,
        main_frame_id=main_frame_id,
        previous_loader_id=previous_loader_id,
        infer_loader=False,
        allow_same_document=False,
    )
    observer.install()
    try:
        await _enable_lifecycle_events(page, operation_deadline)
        # Flush callbacks queued by setup while the observer is still gated.
        await asyncio.sleep(0)
        observer.arm()
        command_timeout = _command_timeout(operation_deadline)
        original_browser, _ = _resolve_owner(page)
        (
            frame_id,
            loader_id,
            error_text,
            is_download,
        ) = await wait_for_zendriver_mutation(
            page.send(cdp.page.navigate(url)),
            timeout=command_timeout,
            owner=page,
            operation="Page navigation",
        )
        with _navigation_completion(original_browser):
            if error_text:
                raise RuntimeError("Chrome rejected page navigation")
            if is_download:
                raise RuntimeError("Page navigation became a download")
            if str(frame_id) != main_frame_id:
                raise RuntimeError("Chrome navigated an unexpected frame")
            expected_url = (
                await _serialize_navigation_url(
                    page, url, main_frame_id, operation_deadline
                )
                if loader_id is None
                else None
            )
            observer.expect_loader(
                None if loader_id is None else str(loader_id),
                same_document_url=expected_url,
            )
            return await _finish_navigation(
                page,
                observer,
                operation_deadline,
                description="Page navigation",
            )
    finally:
        observer.remove()


async def reload_and_wait(
    page: Any,
    *,
    deadline: Deadline | None = None,
) -> NavigationReceipt:
    """Reload exactly the observed loader and await its replacement document."""

    operation_deadline = _page_state_deadline(deadline)
    if not _supports_cdp_events(page):
        command_timeout = _command_timeout(operation_deadline)
        original_browser, _ = _resolve_owner(page)
        await wait_for_zendriver_mutation(
            page.reload(),
            timeout=command_timeout,
            owner=page,
            operation="Page reload",
        )
        with _navigation_completion(original_browser):
            _require_deadline(operation_deadline, "Page reload")
            return NavigationReceipt("", None, "", "complete")

    main_frame_id, previous_loader_id = await _main_frame_identity(
        page,
        operation_deadline,
    )
    observer = _NavigationObserver(
        page,
        main_frame_id=main_frame_id,
        previous_loader_id=previous_loader_id,
        infer_loader=True,
        allow_same_document=False,
    )
    observer.install()
    try:
        await _enable_lifecycle_events(page, operation_deadline)
        await asyncio.sleep(0)
        observer.arm()
        command_timeout = _command_timeout(operation_deadline)
        original_browser, _ = _resolve_owner(page)
        await wait_for_zendriver_mutation(
            page.send(
                cdp.page.reload(
                    ignore_cache=True,
                    loader_id=(
                        None
                        if previous_loader_id is None
                        else cdp.network.LoaderId(previous_loader_id)
                    ),
                )
            ),
            timeout=command_timeout,
            owner=page,
            operation="Page reload",
        )
        with _navigation_completion(original_browser):
            return await _finish_navigation(
                page,
                observer,
                operation_deadline,
                description="Page reload",
            )
    finally:
        observer.remove()


def _browser_target(browser: Any, target_id: str) -> Any | None:
    for target in getattr(browser, "targets", ()):
        candidate_id = getattr(target, "target_id", None)
        if candidate_id is None:
            candidate = getattr(target, "target", None)
            candidate_id = getattr(candidate, "target_id", None)
        if candidate_id is not None and str(candidate_id) == target_id:
            return target
    return None


async def open_tab_and_wait(
    browser: Any,
    *,
    url: str = "about:blank",
    deadline: Deadline | None = None,
) -> Any:
    """Create a target with a short ACK watchdog and await its inventory entry."""

    operation_deadline = _page_state_deadline(deadline)
    connection = getattr(browser, "connection", None)
    if not _supports_cdp_events(connection):
        command_timeout = _command_timeout(operation_deadline)
        target = await wait_for_zendriver_mutation(
            browser.get(url, new_tab=True),
            timeout=command_timeout,
            owner=connection if connection is not None else browser,
            operation="Browser tab creation",
        )
        with _navigation_completion(browser):
            _require_deadline(operation_deadline, "Browser tab creation")
            return target
    event_connection = cast(Any, connection)

    changed = asyncio.Event()
    armed = False
    observed_target_ids: set[str] = set()

    async def on_target(event: Any, _connection: Any = None) -> None:
        if not armed:
            return
        target_info = event.target_info
        observed_target_ids.add(str(target_info.target_id))
        changed.set()

    event_connection.add_handler(cdp.target.TargetCreated, on_target)
    event_connection.add_handler(cdp.target.TargetInfoChanged, on_target)
    try:
        armed = True
        command_timeout = _command_timeout(operation_deadline)
        target_id = str(
            await wait_for_zendriver_mutation(
                event_connection.send(cdp.target.create_target(url)),
                timeout=command_timeout,
                owner=connection,
                operation="Browser tab creation",
            )
        )
        with _navigation_completion(browser):
            while True:
                _require_deadline(operation_deadline, "Browser target discovery")
                target = _browser_target(browser, target_id)
                if target is not None:
                    if hasattr(target, "browser"):
                        target.browser = browser
                    return target
                remaining = operation_deadline.remaining()
                if remaining <= 0:
                    raise PageStateTimeout(
                        "Created browser target did not enter the target inventory"
                    )
                changed.clear()
                # Target events are authoritative.  A short reconciliation yield
                # covers handler scheduling order inside Zendriver's dispatcher.
                wait_seconds = min(0.05, remaining)
                change_task = asyncio.create_task(changed.wait())
                try:
                    await asyncio.wait((change_task,), timeout=wait_seconds)
                finally:
                    if not change_task.done():
                        change_task.cancel()
                        await asyncio.gather(change_task, return_exceptions=True)
                if target_id not in observed_target_ids:
                    continue
    finally:
        event_connection.remove_handlers(cdp.target.TargetCreated, on_target)
        event_connection.remove_handlers(cdp.target.TargetInfoChanged, on_target)


async def mutate_and_wait_for_navigation[ResultT](
    page: Any,
    mutation: Callable[[], Awaitable[ResultT]],
    *,
    owner: Any,
    operation: str,
    deadline: Deadline | None = None,
    command_timeout: float = ZENDRIVER_COMMAND_TIMEOUT_SECONDS,
) -> tuple[ResultT, NavigationReceipt | None]:
    """Pre-arm navigation events, invoke one mutation, and await its document."""

    operation_deadline = _page_state_deadline(deadline)
    if not _supports_cdp_events(page):
        effective_command_timeout = _command_timeout(
            operation_deadline,
            command_timeout,
        )
        original_browser, _ = _resolve_owner(owner)
        result = await wait_for_zendriver_mutation(
            mutation(),
            timeout=effective_command_timeout,
            owner=owner,
            operation=operation,
        )
        with _navigation_completion(original_browser):
            _require_deadline(operation_deadline, operation)
            return result, None

    main_frame_id, previous_loader_id = await _main_frame_identity(
        page,
        operation_deadline,
    )
    observer = _NavigationObserver(
        page,
        main_frame_id=main_frame_id,
        previous_loader_id=previous_loader_id,
        infer_loader=True,
        allow_same_document=True,
    )
    observer.install()
    try:
        await _enable_lifecycle_events(page, operation_deadline)
        await asyncio.sleep(0)
        observer.arm()
        effective_command_timeout = _command_timeout(
            operation_deadline,
            command_timeout,
        )
        original_browser, _ = _resolve_owner(owner)
        result = await wait_for_zendriver_mutation(
            mutation(),
            timeout=effective_command_timeout,
            owner=owner,
            operation=operation,
        )
        with _navigation_completion(original_browser):
            receipt = await _finish_navigation(
                page,
                observer,
                operation_deadline,
                description=operation,
            )
            return result, receipt
    finally:
        observer.remove()


_DOM_CHANGE_EVENTS = (
    cdp.dom.AttributeModified,
    cdp.dom.CharacterDataModified,
    cdp.dom.ChildNodeInserted,
    cdp.dom.DocumentUpdated,
    cdp.dom.SetChildNodes,
    cdp.dom.ShadowRootPushed,
)


async def _run_dom_command(
    page: Any,
    command: Callable[[], Awaitable[Any]],
    *,
    deadline: Deadline,
    description: str,
) -> Any:
    """Run one DOM command under independent transport and semantic bounds.

    A command started before the semantic deadline gets the full protocol
    watchdog.  A successful reply that arrives after the semantic deadline is
    a page-state timeout, not evidence that the browser generation hung.
    """

    _require_deadline(deadline, description)
    result = await wait_for_zendriver(
        command(),
        timeout=ZENDRIVER_COMMAND_TIMEOUT_SECONDS,
        owner=page,
    )
    _require_deadline(deadline, description)
    return result


async def _wait_for_dom_change(
    changed: asyncio.Event,
    *,
    deadline: Deadline,
    description: str,
) -> bool:
    """Return whether an event fired, or false when reconciliation is due."""

    while True:
        remaining = deadline.remaining()
        if remaining <= 0:
            raise PageStateTimeout(f"{description} did not appear before its deadline")
        wait_seconds = min(_DOM_RECONCILIATION_SECONDS, remaining)
        wait_reaches_deadline = remaining <= _DOM_RECONCILIATION_SECONDS
        change_task = asyncio.create_task(changed.wait())
        try:
            done, _ = await asyncio.wait((change_task,), timeout=wait_seconds)
        finally:
            if not change_task.done():
                change_task.cancel()
                await asyncio.gather(change_task, return_exceptions=True)
        if change_task in done:
            change_task.result()
            return True
        if not wait_reaches_deadline:
            return False
        # A timeout intended to reach the semantic deadline must not cause an
        # early PageStateTimeout if the event-loop timer returned slightly
        # early. Keep waiting without issuing another DOM query.


async def _wait_for_dom_query(
    page: Any,
    query: Callable[[], Awaitable[Any]],
    *,
    deadline: Deadline,
    description: str,
) -> Any:
    """Wait on DOM signals, with a low-frequency reconciliation safety net."""

    deadline = _page_state_deadline(deadline)

    supports_events = _supports_cdp_events(page)
    changed = asyncio.Event()

    async def on_dom_change(_event: Any, _connection: Any = None) -> None:
        changed.set()

    observing = False
    try:
        while True:
            changed.clear()
            element = await _run_dom_command(
                page,
                query,
                deadline=deadline,
                description=description,
            )
            match element:
                case list():
                    if element:
                        return element[0]
                case None:
                    pass
                case _:
                    return element

            remaining = deadline.remaining()
            if remaining <= 0:
                raise PageStateTimeout(
                    f"{description} did not appear before its deadline"
                )
            if supports_events and not observing:
                # A normal initial query settles the previous lifecycle subscription.
                # Install only if still missing, then immediately re-query so a DOM
                # change during installation cannot be lost between observers.
                observing = True
                for event_type in _DOM_CHANGE_EVENTS:
                    page.add_handler(event_type, on_dom_change)
                await _run_dom_command(
                    page,
                    lambda: page.send(cdp.dom.enable()),
                    deadline=deadline,
                    description="DOM event subscription",
                )
                continue
            # CDP DOM mutation events are delivered immediately for materialized
            # nodes.  Reconciliation covers browser versions that do not emit an
            # event until their subtree has first been requested.
            await _wait_for_dom_change(
                changed,
                deadline=deadline,
                description=description,
            )
    finally:
        if observing:
            for event_type in _DOM_CHANGE_EVENTS:
                page.remove_handlers(event_type, on_dom_change)


async def wait_for_selector(
    page: Any,
    selector: str,
    *,
    deadline: Deadline,
) -> Any:
    """Wait for a selector under one semantic deadline and short queries."""

    if inspect.iscoroutinefunction(getattr(page, "query_selector", None)):

        def query() -> Awaitable[Any]:
            return cast(Awaitable[Any], page.query_selector(selector))

    else:

        def query() -> Awaitable[Any]:
            return cast(Awaitable[Any], page.select(selector, timeout=0))

    return await _wait_for_dom_query(
        page,
        query,
        deadline=deadline,
        description=f"Selector {selector!r}",
    )


async def wait_for_xpath(
    page: Any,
    expression: str,
    *,
    deadline: Deadline,
) -> Any:
    """Wait for XPath matches without giving Zendriver a long inner poll."""

    return await _wait_for_dom_query(
        page,
        lambda: page.xpath(expression, timeout=0.05),
        deadline=deadline,
        description=f"XPath {expression!r}",
    )


__all__ = [
    "DEFAULT_NAVIGATION_DEADLINE_SECONDS",
    "ZENDRIVER_COMMAND_TIMEOUT_SECONDS",
    "NavigationReceipt",
    "PageStateTimeout",
    "mutate_and_wait_for_navigation",
    "navigate_and_wait",
    "open_tab_and_wait",
    "reload_and_wait",
    "wait_for_selector",
    "wait_for_xpath",
]
