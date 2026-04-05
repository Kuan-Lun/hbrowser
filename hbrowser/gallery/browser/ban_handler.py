"""IP ban 處理邏輯"""

import asyncio
import re
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from typing import Any

from ..utils import setup_logger

logger = setup_logger(__name__)


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

    async def banningcheck() -> None:
        def banningmsg() -> str:
            a = timedelta(seconds=wait_seconds)
            wait_until_str = wait_until.strftime("%Y-%m-%d %H:%M:%S")
            msg = f"IP banned, waiting {a} (until {wait_until_str}) to retry..."
            return msg

        def whilecheck() -> bool:
            return whilecheckban() or whilechecknothing()

        def whilecheckban() -> bool:
            return baningmsg in source

        def whilechecknothing() -> bool:
            return bool(nothing == source)

        source = await page.get_content()
        nothing = "<html><head></head><body></body></html>"
        baningmsg = "Your IP address has been temporarily banned"
        onehour = 60 * 60

        if whilecheck():
            isfirst = True
            isnothing = nothing == source
            while whilecheck():
                logger.debug(f"Page source: {source[:200]}...")
                if not isfirst:
                    logger.warning("Banned again")
                if isnothing:
                    wait_seconds = 4 * onehour
                else:
                    wait_seconds = parse_ban_time(source)
                wait_until = datetime.now() + timedelta(seconds=wait_seconds)
                logger.warning(banningmsg())

                while wait_seconds > onehour:
                    await asyncio.sleep(onehour)
                    wait_seconds -= onehour
                    logger.info(banningmsg())
                await asyncio.sleep(wait_seconds + 15 * 60)
                wait_seconds = 0
                logger.info("Retrying connection")
                await page.reload()
                source = await page.get_content()
                isfirst = False
                if isnothing:
                    logger.error("Empty page, stopping retry")
                    raise RuntimeError()
            logger.info("IP ban lifted")
        else:
            return

    async def myget(*args: Any, **kwargs: Any) -> None:
        await page.get(*args, **kwargs)
        await banningcheck()

    return myget
