"""Wait for owned work without losing cancellation or consuming its outcome."""

from __future__ import annotations

import asyncio


async def wait_owned_task_completion[T](
    task: asyncio.Task[T],
    caller_cancellation: asyncio.CancelledError | None = None,
) -> asyncio.CancelledError | None:
    """Settle bounded owned work; its caller decides result/error precedence.

    The task owns its execution and cleanup deadline. Cancelling this waiter
    cannot cancel or detach that work. Unlike shield, wait does not propagate
    the child's exception before the caller can combine it with cancellation.
    """

    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as error:
            if caller_cancellation is None:
                caller_cancellation = error
    return caller_cancellation
