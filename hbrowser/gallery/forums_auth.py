"""Fail-closed authentication-state detection for the E-Hentai Forums."""

from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from .utils import wait_for_zendriver

FORUMS_HOST = "forums.e-hentai.org"
_AUTH_STATE_READ_TIMEOUT_SECONDS = 5.0


class ForumsAuthState(StrEnum):
    GUEST = "guest"
    AUTHENTICATED = "authenticated"
    UNKNOWN = "unknown"


async def detect_forums_auth_state(page: Any) -> ForumsAuthState:
    """Classify the current Forums page using both origin and DOM markers."""
    current_url = await wait_for_zendriver(
        page.evaluate("window.location.href"),
        timeout=_AUTH_STATE_READ_TIMEOUT_SECONDS,
    )
    if not isinstance(current_url, str):
        return ForumsAuthState.UNKNOWN

    parsed_url = urlsplit(current_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != FORUMS_HOST:
        return ForumsAuthState.UNKNOWN

    guest_elements = await wait_for_zendriver(
        page.query_selector_all("#userlinksguest"),
        timeout=_AUTH_STATE_READ_TIMEOUT_SECONDS,
    )
    member_elements = await wait_for_zendriver(
        page.query_selector_all("#userlinks"),
        timeout=_AUTH_STATE_READ_TIMEOUT_SECONDS,
    )
    has_guest_marker = bool(guest_elements)
    has_member_marker = bool(member_elements)

    if has_member_marker and not has_guest_marker:
        return ForumsAuthState.AUTHENTICATED
    if has_guest_marker and not has_member_marker:
        return ForumsAuthState.GUEST
    return ForumsAuthState.UNKNOWN
