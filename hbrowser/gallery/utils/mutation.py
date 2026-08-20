"""Strict boundary for state-changing Zendriver operations."""

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
) -> ResultT:
    """Run one mutation and make an unclassified failure generation-terminal.

    ``awaitable`` must represent a state-changing operation that has already
    been invoked.  Protocol/lifecycle failures retain their exact type.  Any
    other failure cannot prove whether the remote mutation took effect, so it
    is converted to the common outcome-unknown marker without retaining a
    possibly sensitive protocol payload.
    """
    # Keep caller/owner validation outside the mutation-failure branch. Invalid
    # local arguments reject and dispose an unscheduled coroutine without
    # retiring a browser generation that never accepted the operation.
    _, browser, connection, lifecycle = _validate_zendriver_operation(
        awaitable,
        timeout=timeout,
        owner=owner,
    )

    try:
        return await wait_for_zendriver(
            awaitable,
            timeout=timeout,
            owner=owner,
        )
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
