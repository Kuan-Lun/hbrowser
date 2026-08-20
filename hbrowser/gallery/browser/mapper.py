"""Bounded cleanup for Zendriver connection response mappers."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from zendriver.core.connection import EventTransaction

from ..utils import setup_logger

logger = setup_logger(__name__)

ZENDRIVER_MAPPER_CLEANUP_INTERVAL_SECONDS = 60.0
_JANITOR_TASK_ATTRIBUTE = "_hbrowser_zendriver_mapper_janitor_task"

type Sleep = Callable[[float], Awaitable[None]]


def _validate_cleanup_interval(interval: float) -> float:
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError(
            "Zendriver mapper cleanup interval must be finite and positive"
        )
    return float(interval)


def prune_zendriver_connection_mapper(connection: Any) -> int:
    """Remove completed event records without touching command transactions.

    Zendriver removes ordinary command transactions before completing them, but
    retains every completed ``EventTransaction`` in the same mapper.  Cancelled
    command transactions are completed futures too and may still receive a late
    wire response, so a broad ``Future.done()`` predicate is unsafe here.
    """

    mapper = getattr(connection, "mapper", None)
    if not isinstance(mapper, MutableMapping):
        return 0

    removed = 0
    for message_id, transaction in tuple(mapper.items()):
        if not (isinstance(transaction, EventTransaction) and transaction.done()):
            continue
        # A cleanup pass contains no awaits.  Retaining the identity check also
        # makes the mutation safe if a custom mapper replaces an entry while its
        # snapshot is being inspected.
        if mapper.get(message_id) is transaction:
            del mapper[message_id]
            removed += 1
    return removed


def prune_zendriver_browser_mappers(browser: Any) -> int:
    """Prune the browser-level connection and every current target once."""

    connections = (
        getattr(browser, "connection", None),
        *tuple(getattr(browser, "targets", ())),
    )
    seen: set[int] = set()
    removed = 0
    for connection in connections:
        if connection is None:
            continue
        identity = id(connection)
        if identity in seen:
            continue
        seen.add(identity)
        removed += prune_zendriver_connection_mapper(connection)
    return removed


async def run_zendriver_mapper_janitor(
    browser: Any,
    *,
    interval: float = ZENDRIVER_MAPPER_CLEANUP_INTERVAL_SECONDS,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Periodically bound all mapper event records for one live browser."""

    cleanup_interval = _validate_cleanup_interval(interval)
    while True:
        await sleep(cleanup_interval)
        try:
            removed = prune_zendriver_browser_mappers(browser)
        except Exception as error:
            logger.warning(
                "Zendriver mapper cleanup failed: error_type=%s",
                type(error).__name__,
            )
            continue
        if removed:
            logger.debug(
                "Pruned completed Zendriver event transactions: count=%d",
                removed,
            )


def start_zendriver_mapper_janitor(
    browser: Any,
    *,
    interval: float = ZENDRIVER_MAPPER_CLEANUP_INTERVAL_SECONDS,
) -> asyncio.Task[None]:
    """Run one initial prune and start the browser-owned periodic task."""

    cleanup_interval = _validate_cleanup_interval(interval)
    existing = getattr(browser, _JANITOR_TASK_ATTRIBUTE, None)
    if existing is not None:
        if not isinstance(existing, asyncio.Task):
            raise RuntimeError("Browser mapper janitor state is invalid")
        if not existing.done():
            return existing
        # Observe any terminal state before replacing a stopped task.
        try:
            existing.result()
        except asyncio.CancelledError:
            pass
        except Exception as error:
            logger.warning(
                "Previous Zendriver mapper janitor failed: error_type=%s",
                type(error).__name__,
            )

    try:
        removed = prune_zendriver_browser_mappers(browser)
    except Exception as error:
        logger.warning(
            "Initial Zendriver mapper cleanup failed: error_type=%s",
            type(error).__name__,
        )
        removed = 0
    if removed:
        logger.debug(
            "Pruned completed Zendriver event transactions at startup: count=%d",
            removed,
        )
    task = asyncio.create_task(
        run_zendriver_mapper_janitor(browser, interval=cleanup_interval),
        name=f"zendriver-mapper-janitor-{id(browser):x}",
    )
    try:
        setattr(browser, _JANITOR_TASK_ATTRIBUTE, task)
    except BaseException:
        task.cancel()
        raise
    return task


async def stop_zendriver_mapper_janitor(browser: Any) -> None:
    """Cancel and observe the browser's janitor without swallowing callers."""

    task = getattr(browser, _JANITOR_TASK_ATTRIBUTE, None)
    if task is None:
        return
    if not isinstance(task, asyncio.Task):
        raise RuntimeError("Browser mapper janitor state is invalid")
    janitor_task: asyncio.Task[Any] = task
    if not janitor_task.done():
        janitor_task.cancel()

    # ``gather(..., return_exceptions=True)`` consumes the janitor's deliberate
    # cancellation while an independent cancellation of this caller still
    # propagates normally.
    outcome: Any = None
    try:
        outcomes = await asyncio.gather(janitor_task, return_exceptions=True)
        outcome = outcomes[0]
    finally:
        if (
            getattr(browser, _JANITOR_TASK_ATTRIBUTE, None) is janitor_task
            and janitor_task.done()
        ):
            setattr(browser, _JANITOR_TASK_ATTRIBUTE, None)
    if isinstance(outcome, BaseException) and not isinstance(
        outcome, asyncio.CancelledError
    ):
        raise outcome


__all__ = [
    "ZENDRIVER_MAPPER_CLEANUP_INTERVAL_SECONDS",
    "prune_zendriver_browser_mappers",
    "prune_zendriver_connection_mapper",
    "run_zendriver_mapper_janitor",
    "start_zendriver_mapper_janitor",
    "stop_zendriver_mapper_janitor",
]
