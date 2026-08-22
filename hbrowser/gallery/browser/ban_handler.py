"""IP ban 處理邏輯"""

import asyncio
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..utils import (
    Deadline,
    navigate_and_wait,
    reload_and_wait,
    setup_logger,
    wait_for_zendriver,
)

logger = setup_logger(__name__)

_BAN_MESSAGE = "Your IP address has been temporarily banned"
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
        is_banned=_BAN_MESSAGE in source,
        is_blank_page=source == _BLANK_PAGE,
    )


def format_wait_message(wait_seconds: int, wait_until: datetime) -> str:
    remaining = timedelta(seconds=wait_seconds)
    wait_until_str = wait_until.strftime("%Y-%m-%d %H:%M:%S")
    return f"IP banned, waiting {remaining} (until {wait_until_str}) to retry..."


async def _wait_out_ban(wait_seconds: int) -> None:
    """以一小時為單位睡眠等待，每小時記一次目前還剩多久。"""
    while wait_seconds > _HOUR_SECONDS:
        await asyncio.sleep(_HOUR_SECONDS)
        wait_seconds -= _HOUR_SECONDS
        wait_until = datetime.now() + timedelta(seconds=wait_seconds)
        logger.info(format_wait_message(wait_seconds, wait_until))
    # This is a server-declared ban duration, not a CDP command watchdog.
    await asyncio.sleep(wait_seconds)


async def _retry_until_unbanned(page: Any, source: str) -> None:
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

        wait_seconds = parse_ban_time(source)
        if wait_seconds <= 0:
            raise RuntimeError("IP ban page did not contain a positive wait duration")
        wait_until = datetime.now() + timedelta(seconds=wait_seconds)
        logger.warning(format_wait_message(wait_seconds, wait_until))

        await _wait_out_ban(wait_seconds)

        logger.info("Retrying connection")
        retry_deadline = Deadline.after(_PAGE_NAVIGATION_DEADLINE_SECONDS)
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
            await _retry_until_unbanned(page, source)

    return myget
