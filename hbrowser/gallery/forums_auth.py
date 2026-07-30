"""Fail-closed authentication-state detection for the E-Hentai Forums."""

from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

FORUMS_HOST = "forums.e-hentai.org"


class ForumsAuthState(StrEnum):
    GUEST = "guest"
    AUTHENTICATED = "authenticated"
    UNKNOWN = "unknown"


async def detect_forums_auth_state(page: Any) -> ForumsAuthState:
    """Classify the current Forums page using both origin and DOM markers."""
    current_url = await page.evaluate("window.location.href")
    if not isinstance(current_url, str):
        return ForumsAuthState.UNKNOWN

    parsed_url = urlsplit(current_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != FORUMS_HOST:
        return ForumsAuthState.UNKNOWN

    guest_elements = await page.query_selector_all("#userlinksguest")
    member_elements = await page.query_selector_all("#userlinks")
    has_guest_marker = bool(guest_elements)
    has_member_marker = bool(member_elements)

    if has_member_marker and not has_guest_marker:
        return ForumsAuthState.AUTHENTICATED
    if has_guest_marker and not has_member_marker:
        return ForumsAuthState.GUEST
    return ForumsAuthState.UNKNOWN
