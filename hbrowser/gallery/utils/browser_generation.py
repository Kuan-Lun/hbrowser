"""Classify failures that make one browser generation unsafe to reuse."""

from websockets.exceptions import ConnectionClosed

from ...exceptions import BrowserMutationOutcomeUnknownError
from .protocol import ZendriverOperationTimeout, ZendriverOwnerRetiredError

_BROWSER_GENERATION_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionClosed,
    BrowserMutationOutcomeUnknownError,
    ZendriverOperationTimeout,
    ZendriverOwnerRetiredError,
)


def is_browser_generation_error(error: BaseException) -> bool:
    """Return whether an error chain makes the browser generation unusable."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, _BROWSER_GENERATION_ERRORS):
            return True
        if candidate.__cause__ is not None:
            pending.append(candidate.__cause__)
        if candidate.__context__ is not None:
            pending.append(candidate.__context__)
    return False
