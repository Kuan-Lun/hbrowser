"""Cancellation-safe timeouts for Zendriver protocol operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

# Zendriver keeps an in-flight Transaction in its response mapper.  Retain
# timed-out operations until their late response arrives, then observe any
# exception so asyncio does not report an orphaned task.  A command that never
# completes remains retained until event-loop shutdown by design: cancelling it
# while the listener is live would recreate the mapper-corruption bug.
_BACKGROUND_ZENDRIVER_OPERATIONS: set[asyncio.Future[Any]] = set()


def _observe_zendriver_completion(operation: asyncio.Future[Any]) -> None:
    _BACKGROUND_ZENDRIVER_OPERATIONS.discard(operation)
    if operation.cancelled():
        return
    try:
        operation.exception()
    except BaseException:
        # Observing completion keeps the event loop clean.  An active caller
        # still receives the same exception from ``operation.result()``.
        pass


async def wait_for_zendriver[T](
    awaitable: Awaitable[T],
    *,
    timeout: float,
) -> T:
    """Bound a Zendriver await without cancelling its protocol transaction.

    Zendriver 0.15 keeps cancelled Transactions in its mapper.  When Chrome
    later replies, its listener calls ``set_result`` on that cancelled Future
    and terminates with ``InvalidStateError``.  ``asyncio.wait`` provides the
    required timeout without propagating cancellation to the protocol Future.
    """

    operation = asyncio.ensure_future(awaitable)
    _BACKGROUND_ZENDRIVER_OPERATIONS.add(operation)
    operation.add_done_callback(_observe_zendriver_completion)

    done, _ = await asyncio.wait((operation,), timeout=timeout)
    if not done:
        raise TimeoutError
    return operation.result()
