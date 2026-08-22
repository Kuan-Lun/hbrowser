"""Event-driven browser target discovery helpers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

import zendriver as zd
from zendriver import cdp

from .deadline import Deadline
from .mutation import wait_for_zendriver_mutation

_TARGET_COMMAND_TIMEOUT_SECONDS = 5.0
_TARGET_SEMANTIC_DEADLINE_SECONDS = 10.0
_TARGET_RECONCILIATION_SECONDS = 0.05


def _new_tab(browser: Any, existing_tabs: set[Any]) -> Any | None:
    for tab in getattr(browser, "tabs", ()):
        target = getattr(tab, "target", None)
        target_id = getattr(target, "target_id", None)
        if target_id is not None and target_id not in existing_tabs:
            return tab
    return None


async def _wait_for_inventory_change(
    browser: Any,
    changed: asyncio.Event,
    existing_tabs: set[Any],
    deadline: Deadline,
) -> Any | None:
    while True:
        if deadline.remaining() <= 0:
            return None
        tab = _new_tab(browser, existing_tabs)
        if tab is not None:
            if deadline.remaining() <= 0:
                return None
            return tab
        remaining = deadline.remaining()
        if remaining <= 0:
            return None
        changed.clear()
        change_task = asyncio.create_task(changed.wait())
        try:
            await asyncio.wait(
                (change_task,),
                timeout=min(_TARGET_RECONCILIATION_SECONDS, remaining),
            )
        finally:
            if not change_task.done():
                change_task.cancel()
                await asyncio.gather(change_task, return_exceptions=True)


async def mutate_and_wait_for_new_tab[ResultT](
    browser: zd.Browser,
    existing_tabs: set[Any],
    mutation: Callable[[], Awaitable[ResultT]],
    *,
    owner: Any,
    operation: str,
    deadline: Deadline,
) -> tuple[ResultT, zd.Tab | None]:
    """Pre-arm target events, invoke one mutation, then resolve the new tab."""

    deadline = deadline.bounded(_TARGET_SEMANTIC_DEADLINE_SECONDS)

    connection = getattr(browser, "connection", None)
    supports_events = (
        connection is not None
        and callable(getattr(connection, "add_handler", None))
        and callable(getattr(connection, "remove_handlers", None))
        and inspect.iscoroutinefunction(getattr(connection, "send", None))
    )
    changed = asyncio.Event()
    armed = False
    event_connection = cast(Any, connection)

    async def on_target(_event: Any, _connection: Any = None) -> None:
        if armed:
            changed.set()

    if supports_events:
        event_connection.add_handler(cdp.target.TargetCreated, on_target)
        event_connection.add_handler(cdp.target.TargetInfoChanged, on_target)
    try:
        armed = True
        command_timeout = min(
            _TARGET_COMMAND_TIMEOUT_SECONDS,
            deadline.remaining(),
        )
        if command_timeout <= 0:
            raise TimeoutError("New-tab deadline expired before its mutation")
        result = await wait_for_zendriver_mutation(
            mutation(),
            timeout=command_timeout,
            owner=owner,
            operation=operation,
        )
        if deadline.remaining() <= 0:
            raise TimeoutError(f"{operation} completed after its deadline")
        tab = await _wait_for_inventory_change(
            browser,
            changed,
            existing_tabs,
            deadline,
        )
        return result, tab
    finally:
        if supports_events:
            event_connection.remove_handlers(cdp.target.TargetCreated, on_target)
            event_connection.remove_handlers(cdp.target.TargetInfoChanged, on_target)


__all__ = ["mutate_and_wait_for_new_tab"]
