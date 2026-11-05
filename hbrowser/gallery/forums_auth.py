"""Fail-closed authentication-state detection for the E-Hentai Forums."""

import json
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from ..exceptions import LoginFailedException
from .utils import Deadline, wait_for_selector, wait_for_zendriver
from .utils.mutation import wait_for_zendriver_mutation

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

    try:
        parsed_url = urlsplit(current_url)
        trusted = (
            parsed_url.scheme == "https"
            and parsed_url.hostname == FORUMS_HOST
            and parsed_url.port in (None, 443)
            and parsed_url.username is None
            and parsed_url.password is None
        )
    except ValueError:
        return ForumsAuthState.UNKNOWN
    if not trusted:
        return ForumsAuthState.UNKNOWN

    has_guest_marker = snapshot.get("guest") is True
    has_member_marker = snapshot.get("member") is True

    if has_member_marker and not has_guest_marker:
        return ForumsAuthState.AUTHENTICATED
    if has_guest_marker and not has_member_marker:
        return ForumsAuthState.GUEST
    return ForumsAuthState.UNKNOWN


async def fill_forums_login_form(
    page: Any, username: str | None, password: str | None, *, deadline: Deadline
) -> None:
    """Check origin and fill both fields in one synchronous main-document action.

    Selector waits only establish readiness. They confer no authority to send
    secrets: navigation may replace their document before the mutation executes.
    The mutation reselects fields and checks the executing document's origin.
    """
    if username is None or password is None:
        raise LoginFailedException("Forums credentials are unavailable")
    for selector in ("[name='UserName']", "[name='PassWord']"):
        await wait_for_selector(page, selector, deadline=deadline)
    timeout = min(5.0, deadline.remaining())
    if timeout <= 0:
        raise TimeoutError("Login-form deadline expired before credential input")
    script = (
        """(() => {
        if (window !== window.top ||
            window.location.origin !== 'https://forums.e-hentai.org') return false;
        """
        # No spread/iterator or credential-bearing page call may precede origin
        # validation: a replacement document controls its own Array prototype.
        + f"const username = {json.dumps(username, ensure_ascii=True)};\n"
        + f"const password = {json.dumps(password, ensure_ascii=True)};\n"
        + """
        const user = document.querySelector('[name="UserName"]');
        const pass = document.querySelector('[name="PassWord"]');
        if (!(user instanceof HTMLInputElement) ||
            !(pass instanceof HTMLInputElement)) return false;
        const setValue = Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype, 'value').set;
        setValue.call(user, username);
        setValue.call(pass, password);
        user.dispatchEvent(new Event('input', {bubbles: true}));
        pass.dispatchEvent(new Event('input', {bubbles: true}));
        return true;
    })()"""
    )
    accepted = await wait_for_zendriver_mutation(
        page.evaluate(script),
        timeout=timeout,
        owner=page,
        operation="Forums credential input",
    )
    if accepted is not True:
        raise LoginFailedException("Login document is not a trusted Forums form")
