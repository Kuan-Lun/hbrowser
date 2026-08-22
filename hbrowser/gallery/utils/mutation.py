"""Strict boundary for state-changing Zendriver operations."""

import asyncio
import inspect
import math
from collections.abc import Awaitable
from typing import Any

from ...exceptions import BrowserMutationOutcomeUnknownError
from .browser_generation import is_browser_generation_error
from .protocol import (
    _retire_validated_zendriver_owner,
    _validate_zendriver_operation,
    wait_for_zendriver,
)


async def wait_for_zendriver_mutation[ResultT](
    awaitable: Awaitable[ResultT],
    *,
    timeout: float,
    owner: Any,
    operation: str,
    completion_deadline_expires_at: float | None = None,
) -> ResultT:
    """Run one mutation and make an unclassified failure generation-terminal.

    ``awaitable`` must represent a state-changing operation that has already
    been invoked.  Protocol/lifecycle failures retain their exact type.  Any
    other failure cannot prove whether the remote mutation took effect, so it
    is converted to the common outcome-unknown marker without retaining a
    possibly sensitive protocol payload.  When an absolute completion deadline
    is supplied, an acknowledgement delivered at or after that boundary is
    likewise outcome-unknown and retires the captured browser generation.
    """
    try:
        if completion_deadline_expires_at is not None and (
            isinstance(completion_deadline_expires_at, bool)
            or not isinstance(completion_deadline_expires_at, int | float)
            or not math.isfinite(completion_deadline_expires_at)
        ):
            raise ValueError(
                "completion_deadline_expires_at must be a finite monotonic time"
            )
    except Exception:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise

    # Keep caller/owner validation outside the mutation-failure branch. Invalid
    # local arguments reject and dispose an unscheduled coroutine without
    # retiring a browser generation that never accepted the operation.
    _, browser, connection, lifecycle = _validate_zendriver_operation(
        awaitable,
        timeout=timeout,
        owner=owner,
    )

    try:
        result = await wait_for_zendriver(
            awaitable,
            timeout=timeout,
            owner=owner,
        )
        if (
            completion_deadline_expires_at is not None
            and asyncio.get_running_loop().time() >= completion_deadline_expires_at
        ):
            raise TimeoutError(
                f"{operation} acknowledgement arrived after its semantic deadline"
            )
        return result
    except Exception as error:
        # Once a valid mutation awaitable has run, every failure that does not
        # prove a clean result makes this exact owner generation unusable.
        _retire_validated_zendriver_owner(
            browser=browser,
            connection=connection,
            lifecycle=lifecycle,
            owner=owner,
        )
        if is_browser_generation_error(error):
            raise

    raise BrowserMutationOutcomeUnknownError(
        f"{operation} was invoked, but its outcome is unknown"
    )
