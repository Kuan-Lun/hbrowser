"""IP ban 處理邏輯"""

import asyncio
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..utils import setup_logger

logger = setup_logger(__name__)

_BAN_MESSAGE = "Your IP address has been temporarily banned"
_BLANK_PAGE = "<html><head></head><body></body></html>"
_EMPTY_PAGE_WAIT_SECONDS = 4 * 60 * 60
_HOUR_SECONDS = 60 * 60
_RETRY_BUFFER_SECONDS = 15 * 60


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
    await asyncio.sleep(wait_seconds + _RETRY_BUFFER_SECONDS)


async def _retry_until_unbanned(page: Any, source: str) -> None:
    status = check_ban_status(source)
    is_first = True
    while status.should_wait:
        logger.debug(f"Page source: {source[:200]}...")
        if not is_first:
            logger.warning("Banned again")

        wait_seconds = (
            _EMPTY_PAGE_WAIT_SECONDS if status.is_blank_page else parse_ban_time(source)
        )
        wait_until = datetime.now() + timedelta(seconds=wait_seconds)
        logger.warning(format_wait_message(wait_seconds, wait_until))

        await _wait_out_ban(wait_seconds)

        logger.info("Retrying connection")
        await page.reload()
        source = await page.get_content()
        is_first = False
        status = check_ban_status(source)

        if status.is_blank_page:
            raise RuntimeError(
                "Page is still blank after reloading while waiting out an IP "
                "ban; giving up instead of retrying forever."
            )

    logger.info("IP ban lifted")


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

    async def myget(*args: Any, **kwargs: Any) -> None:
        await page.get(*args, **kwargs)
        source = await page.get_content()
        if check_ban_status(source).should_wait:
            await _retry_until_unbanned(page, source)

    return myget
