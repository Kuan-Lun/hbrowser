"""IP ban 處理邏輯"""

import asyncio
import logging
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype, NavigableString, Tag

from ..utils import (
    Deadline,
    navigate_and_wait,
    reload_and_wait,
    wait_for_zendriver,
)

logger = logging.getLogger(__name__)

_BAN_MESSAGE = "Your IP address has been temporarily banned"
_BAN_NOTICE_PATTERN = re.compile(
    rf"^{re.escape(_BAN_MESSAGE)}(?:[\s.!?:,;()]|$)", re.IGNORECASE
)
_DURATION_UNIT = r"[0-9]+\s+(?:days?|hours?|minutes?)\b"
_NOTICE_DURATION = rf"({_DURATION_UNIT}(?:\s*(?:,\s*|and\s+)?{_DURATION_UNIT})*)"
# A ban replaces the normal page with a standalone text response. These tags
# only lay out that response; named/interactive/user-content regions are not
# authority, regardless of their particular CSS class or position on a page.
_NOTICE_LAYOUT_TAGS = frozenset(
    {
        "div",
        "main",
        "section",
        "p",
        "br",
        "span",
        "b",
        "strong",
        "i",
        "em",
        "small",
        "center",
        "pre",
        "h1",
        "h2",
    }
)
_NOTICE_LAYOUT_ATTRIBUTES = frozenset({"style", "align", "lang", "dir"})
_BLANK_PAGE = "<html><head></head><body></body></html>"
_HOUR_SECONDS = 60 * 60
_BLANK_PAGE_QUICK_RETRIES = 3
_PAGE_READ_TIMEOUT_SECONDS = 5.0
_PAGE_NAVIGATION_DEADLINE_SECONDS = 10.0


async def _read_content_before(
    page: Any,
    *,
    deadline: Deadline,
    description: str,
) -> str:
    read_timeout = min(_PAGE_READ_TIMEOUT_SECONDS, deadline.remaining())
    if read_timeout <= 0:
        raise TimeoutError(f"{description} expired before document read")
    source = await wait_for_zendriver(
        page.get_content(),
        timeout=read_timeout,
        owner=page,
    )
    if deadline.remaining() <= 0:
        raise TimeoutError(f"{description} completed after its deadline")
    if not isinstance(source, str):
        raise TypeError(f"{description} returned non-text document content")
    return source


def parse_ban_time(page_source: str) -> int:
    """
    解析被禁時間

    Args:
        page_source: 頁面源碼

    Returns:
        被禁的秒數
    """

    def calculate(duration_str: str) -> dict[str, int]:
        # Regular expression patterns to capture days, hours, and minutes
        patterns = {
            "days": r"(\d+) day?",
            "hours": r"(\d+) hour?",
            "minutes": r"(\d+) minute?",
        }

        # Dictionary to store the found durations
        durations = {"days": 0, "hours": 0, "minutes": 0}

        # Search for each duration in the string and update the durations dictionary
        for key, pattern in patterns.items():
            match = re.search(pattern, duration_str)
            if match:
                durations[key] = int(match.group(1))

        return durations

    durations = calculate(page_source)
    return 60 * (
        60 * (24 * durations["days"] + durations["hours"]) + durations["minutes"]
    )


@dataclass(frozen=True)
class BanStatus:
    is_banned: bool
    is_blank_page: bool

    @property
    def should_wait(self) -> bool:
        return self.is_banned or self.is_blank_page


def check_ban_status(source: str) -> BanStatus:
    """從頁面原始碼判斷目前是否處於 IP ban 或空白頁狀態。"""
    return BanStatus(
        is_banned=_ban_notice_document(source) is not None,
        is_blank_page=source == _BLANK_PAGE,
    )


def _ban_notice_document(source: str) -> str | None:
    """Recognize a dedicated ban response, never a notice found inside a page.

    The accepted contract is plain/body text with optional presentational
    markup. A general div/p/role is not a server-notice identity. In particular,
    do not remove normal page regions and reinterpret the remainder as a ban.
    """

    document = BeautifulSoup(source, "html.parser")
    if any(len(document.find_all(name)) > 1 for name in ("html", "head", "body")):
        return None
    head = document.head
    if head is not None:
        if head.parent is not document and head.parent is not document.html:
            return None
        head.decompose()
    root = document.body or document.html or document
    child: Tag = root
    for parent in root.parents:
        if parent is not document and parent.name != "html":
            return None
        for sibling in parent.contents:
            if sibling is child or isinstance(sibling, Comment | Doctype):
                continue
            if isinstance(sibling, NavigableString) and not sibling.strip():
                continue
            return None
        child = parent
    if root.name == "body" and set(root.attrs) - _NOTICE_LAYOUT_ATTRIBUTES:
        return None
    for element in root.find_all(True):
        if (
            element.name not in _NOTICE_LAYOUT_TAGS
            or set(element.attrs) - _NOTICE_LAYOUT_ATTRIBUTES
        ):
            return None
    visible_text = " ".join(root.get_text(" ", strip=True).split())
    return visible_text if _BAN_NOTICE_PATTERN.match(visible_text) else None


def _notice_wait_seconds(notice: str) -> int:
    # A duration elsewhere on the page is not authority to sleep. Preserve the
    # standalone "banned for ..." form and the server's explicit expiry clause;
    # consume only their adjacent duration units, not arbitrary trailing prose.
    for prefix in (r"\bban expires in\s+", rf"{re.escape(_BAN_MESSAGE)}\s+for\s+"):
        match = re.search(prefix + _NOTICE_DURATION, notice, re.IGNORECASE)
        if match is not None:
            return parse_ban_time(match.group(1).lower())
    return 0


def format_wait_message(wait_seconds: int, wait_until: datetime) -> str:
    remaining = timedelta(seconds=wait_seconds)
    wait_until_str = wait_until.strftime("%Y-%m-%d %H:%M:%S")
    return f"IP banned, waiting {remaining} (until {wait_until_str}) to retry..."


async def _wait_out_ban(
    wait_seconds: int,
    *,
    deadline: Deadline | None = None,
) -> None:
    """Wait for the server ban without extending an explicit caller deadline."""
    remaining_wait = float(wait_seconds)
    while remaining_wait > 0:
        sleep_seconds = min(remaining_wait, _HOUR_SECONDS)
        if deadline is not None:
            remaining = deadline.remaining()
            if remaining <= 0:
                raise TimeoutError("IP ban wait exceeded the caller deadline")
            sleep_seconds = min(sleep_seconds, remaining)
        await asyncio.sleep(sleep_seconds)
        if deadline is not None and deadline.expired:
            raise TimeoutError("IP ban wait exceeded the caller deadline")
        remaining_wait -= sleep_seconds
        if remaining_wait > 0:
            wait_until = datetime.now() + timedelta(seconds=remaining_wait)
            logger.info(format_wait_message(int(remaining_wait), wait_until))


async def _retry_until_unbanned(
    page: Any,
    source: str,
    *,
    deadline: Deadline | None = None,
) -> None:
    status = check_ban_status(source)
    is_first = True
    while status.is_banned:
        logger.debug(
            "Ban page inspected: bytes=%d banned=%s blank=%s",
            len(source.encode("utf-8", errors="ignore")),
            status.is_banned,
            status.is_blank_page,
        )
        if not is_first:
            logger.warning("Banned again")

        notice = _ban_notice_document(source)
        assert notice is not None
        wait_seconds = _notice_wait_seconds(notice)
        if wait_seconds <= 0:
            raise RuntimeError("IP ban page did not contain a positive wait duration")
        wait_until = datetime.now() + timedelta(seconds=wait_seconds)
        logger.warning(format_wait_message(wait_seconds, wait_until))

        await _wait_out_ban(wait_seconds, deadline=deadline)

        logger.info("Retrying connection")
        retry_deadline = (
            Deadline.after(_PAGE_NAVIGATION_DEADLINE_SECONDS)
            if deadline is None
            else deadline.bounded(_PAGE_NAVIGATION_DEADLINE_SECONDS)
        )
        await reload_and_wait(
            page,
            deadline=retry_deadline,
        )
        source = await _read_content_before(
            page,
            deadline=retry_deadline,
            description="Post-ban reload",
        )
        is_first = False
        status = check_ban_status(source)

        if status.is_blank_page:
            raise RuntimeError(
                "Page is still blank after reloading while waiting out an IP "
                "ban; giving up instead of retrying forever."
            )

    logger.info("IP ban lifted")


async def _resolve_transient_blank_page(
    page: Any,
    source: str,
    *,
    deadline: Deadline | None = None,
) -> str:
    """空白頁面通常只是頁面尚未載入完成（例如剛登入後的重新導向），
    先快速重試幾次排除這種暫時性狀況，避免誤判為長時間 ban。"""
    blank_page_deadline = (
        Deadline.after(_PAGE_NAVIGATION_DEADLINE_SECONDS)
        if deadline is None
        else deadline.bounded(_PAGE_NAVIGATION_DEADLINE_SECONDS)
    )
    for _ in range(_BLANK_PAGE_QUICK_RETRIES):
        if not check_ban_status(source).is_blank_page:
            break
        # The next document lifecycle is the readiness evidence. A fixed sleep
        # before reload only delays the same observation and behaves unlike a
        # human-triggered reload, which waits for the new document itself.
        await reload_and_wait(
            page,
            deadline=blank_page_deadline,
        )
        source = await _read_content_before(
            page,
            deadline=blank_page_deadline,
            description="Blank-page recovery",
        )
    return source


def handle_ban_decorator(
    page: Any,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """
    處理 IP ban 的裝飾器

    Args:
        page: zendriver Tab 實例

    Returns:
        包裝後的 async get 函數
    """

    async def myget(url: str, *, deadline: Deadline | None = None) -> None:
        navigation_deadline = (
            Deadline.after(_PAGE_NAVIGATION_DEADLINE_SECONDS)
            if deadline is None
            else deadline.bounded(_PAGE_NAVIGATION_DEADLINE_SECONDS)
        )
        await navigate_and_wait(
            page,
            url,
            deadline=navigation_deadline,
        )
        source = await _read_content_before(
            page,
            deadline=navigation_deadline,
            description="Page navigation",
        )
        if check_ban_status(source).is_blank_page:
            source = await _resolve_transient_blank_page(
                page,
                source,
                deadline=navigation_deadline,
            )
        status = check_ban_status(source)
        if status.is_blank_page:
            raise RuntimeError(
                "Page remained blank after bounded lifecycle-aware reloads; "
                "no evidence supports treating it as an IP ban"
            )
        if status.is_banned:
            await _retry_until_unbanned(page, source, deadline=deadline)

    return myget
