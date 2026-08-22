"""Fail-closed authentication-state detection for the E-Hentai Forums."""

from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from .utils import wait_for_zendriver

FORUMS_HOST = "forums.e-hentai.org"
_AUTH_STATE_READ_TIMEOUT_SECONDS = 5.0
_AUTH_STATE_SNAPSHOT_SCRIPT = """(() => ({
    url: window.location.href,
    guest: Boolean(document.querySelector('#userlinksguest')),
    member: Boolean(document.querySelector('#userlinks'))
}))()"""


class ForumsAuthState(StrEnum):
    GUEST = "guest"
    AUTHENTICATED = "authenticated"
    UNKNOWN = "unknown"


async def detect_forums_auth_state(page: Any) -> ForumsAuthState:
    """Classify the current Forums page using both origin and DOM markers."""
    snapshot = await wait_for_zendriver(
        page.evaluate(_AUTH_STATE_SNAPSHOT_SCRIPT),
        timeout=_AUTH_STATE_READ_TIMEOUT_SECONDS,
        owner=page,
    )
    if not isinstance(snapshot, dict):
        return ForumsAuthState.UNKNOWN
    current_url = snapshot.get("url")
    if not isinstance(current_url, str):
        return ForumsAuthState.UNKNOWN

    parsed_url = urlsplit(current_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != FORUMS_HOST:
        return ForumsAuthState.UNKNOWN

    has_guest_marker = snapshot.get("guest") is True
    has_member_marker = snapshot.get("member") is True

    if has_member_marker and not has_guest_marker:
        return ForumsAuthState.AUTHENTICATED
    if has_guest_marker and not has_member_marker:
        return ForumsAuthState.GUEST
    return ForumsAuthState.UNKNOWN
