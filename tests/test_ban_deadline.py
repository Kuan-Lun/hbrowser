from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from hbrowser.gallery.browser import ban_handler
from hbrowser.gallery.utils import Deadline
from hbrowser.gallery.utils import deadline as deadline_module

_BAN = "Your IP address has been temporarily banned for "


@dataclass
class _Clock:
    now: float = 1000.0
    sleeps: list[float] = field(default_factory=list)
    reload_deadlines: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    async def navigate(self, _page: object, _url: str, *, deadline: Deadline) -> None:
        self.now += 2

    async def reload(self, _page: object, *, deadline: Deadline) -> None:
        self.reload_deadlines.append(deadline.expires_at)
        self.now += 4


class BanDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def _run_bans(
        self, sources: list[str], *, seconds: float | None, expires: bool
    ) -> _Clock:
        clock = _Clock()
        caller_deadline = None if seconds is None else Deadline(clock.now + seconds)
        with (
            patch.object(
                deadline_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
            ),
            patch.object(ban_handler, "asyncio", SimpleNamespace(sleep=clock.sleep)),
            patch.object(ban_handler, "navigate_and_wait", new=clock.navigate),
            patch.object(ban_handler, "reload_and_wait", new=clock.reload),
            patch.object(
                ban_handler, "_read_content_before", new=AsyncMock(side_effect=sources)
            ),
        ):
            navigate = ban_handler.handle_ban_decorator(object())
            if expires:
                with self.assertRaisesRegex(TimeoutError, "ban"):
                    await navigate("https://example.invalid/", deadline=caller_deadline)
            else:
                await navigate("https://example.invalid/", deadline=caller_deadline)
        return clock

    async def test_hour_ban_cannot_extend_a_ten_second_caller_deadline(self) -> None:
        clock = await self._run_bans(
            [_BAN + "1 hour", "ready"], seconds=10, expires=True
        )
        self.assertEqual(clock.sleeps, [8])
        self.assertEqual(clock.now, 1010)
        self.assertEqual(clock.reload_deadlines, [])

    async def test_repeated_bans_and_reloads_share_the_original_deadline(self) -> None:
        clock = await self._run_bans(
            [_BAN + "1 minute"] * 4 + ["ready"], seconds=150, expires=True
        )
        self.assertEqual(clock.sleeps, [60, 60, 20])
        self.assertEqual(clock.now, 1150)
        self.assertEqual(clock.reload_deadlines, [1072, 1136])

    async def test_longer_explicit_deadline_is_not_shortened_to_first_navigation(
        self,
    ) -> None:
        clock = await self._run_bans(
            [_BAN + "1 minute", "ready"], seconds=100, expires=False
        )
        self.assertEqual(clock.sleeps, [60])
        self.assertEqual(clock.reload_deadlines, [1072])
        self.assertEqual(clock.now, 1066)

    async def test_without_caller_deadline_server_ban_duration_is_preserved(
        self,
    ) -> None:
        clock = await self._run_bans(
            [_BAN + "2 hours", "ready"], seconds=None, expires=False
        )
        self.assertEqual(clock.sleeps, [3600, 3600])
        self.assertEqual(clock.now, 8206)
        self.assertEqual(clock.reload_deadlines, [8212])

    async def test_cancellation_during_ban_wait_never_reloads(self) -> None:
        entered = asyncio.Event()

        async def block(_seconds: float) -> None:
            entered.set()
            await asyncio.Future[None]()

        with (
            patch.object(ban_handler, "asyncio", SimpleNamespace(sleep=block)),
            patch.object(ban_handler, "navigate_and_wait", new=AsyncMock()),
            patch.object(ban_handler, "reload_and_wait", new=AsyncMock()) as reload,
            patch.object(
                ban_handler,
                "_read_content_before",
                new=AsyncMock(return_value=_BAN + "1 hour"),
            ),
        ):
            task = asyncio.create_task(
                ban_handler.handle_ban_decorator(object())(
                    "https://example.invalid/", deadline=Deadline.after(10)
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            reload.assert_not_awaited()
