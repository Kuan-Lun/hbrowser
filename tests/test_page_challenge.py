from __future__ import annotations

import logging
import unittest
from collections import deque
from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, patch

from hbrowser.exceptions import BrowserIdentityApplyException, LoginFailedException
from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrResult,
    FlareSolverrSessionUnavailable,
    FlareSolverrSolveReceipt,
)
from hbrowser.gallery.captcha.models import ChallengeDetection, Kind
from hbrowser.gallery.captcha.page_challenge import PageChallengeHandler

FORUMS_URL = "https://forums.e-hentai.org/"


def _receipt(generation: int) -> FlareSolverrSolveReceipt:
    return FlareSolverrSolveReceipt(
        result=FlareSolverrResult(cookies=[], user_agent=f"agent-{generation}"),
        session_generation=generation,
    )


class _Detector:
    def __init__(self, kinds: list[Kind], events: list[str]) -> None:
        self._kinds = deque(kinds)
        self._events = events
        self.pages: list[object] = []
        self.timeouts: list[float] = []

    async def detect(
        self,
        page: object,
        timeout: float = 2.0,
    ) -> ChallengeDetection:
        if not self._kinds:
            raise AssertionError("Unexpected challenge detection")
        kind = self._kinds.popleft()
        self.pages.append(page)
        self.timeouts.append(timeout)
        self._events.append(f"detect:{kind}")
        return ChallengeDetection(url=FORUMS_URL, kind=kind)


class _Solver:
    def __init__(
        self,
        events: list[str],
        *,
        receipt: FlareSolverrSolveReceipt | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._receipt = receipt
        self._error = error
        self.solve_calls: list[str] = []
        self.mark_calls: list[FlareSolverrSolveReceipt] = []
        self.retire_calls = 0

    async def solve_managed(self, url: str) -> FlareSolverrSolveReceipt:
        self.solve_calls.append(url)
        self._events.append("solve")
        if self._error is not None:
            raise self._error
        if self._receipt is None:
            raise AssertionError("Solver has no configured receipt")
        return self._receipt

    def mark_identity_applied(self, receipt: FlareSolverrSolveReceipt) -> None:
        self.mark_calls.append(receipt)
        self._events.append("mark_identity_applied")

    async def retire(self) -> None:
        self.retire_calls += 1
        self._events.append("retire_solver")


async def _unexpected_callback(*args: object) -> None:
    raise AssertionError(f"Unexpected callback arguments: {args!r}")


class PageChallengeHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_solution_is_applied_committed_and_freshly_verified(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["cf_managed_challenge", "none"], events)
        receipt = _receipt(7)
        solver = _Solver(events, receipt=receipt)

        async def apply_identity(actual: object) -> None:
            self.assertIs(actual, receipt)
            events.append("apply_identity")

        async def navigate(url: str) -> None:
            self.assertEqual(url, FORUMS_URL)
            events.append("navigate")

        save_diagnostic = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=True,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=apply_identity,
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        await handler.resolve(page, FORUMS_URL, detect_timeout=2.5)

        self.assertEqual(
            events,
            [
                "detect:cf_managed_challenge",
                "solve",
                "apply_identity",
                "mark_identity_applied",
                "navigate",
                "detect:none",
            ],
        )
        self.assertEqual(solver.solve_calls, [FORUMS_URL])
        self.assertEqual(solver.mark_calls, [receipt])
        self.assertEqual(solver.retire_calls, 0)
        self.assertEqual(detector.pages, [page, page])
        self.assertEqual(detector.timeouts, [2.5, 2.5])
        save_diagnostic.assert_not_awaited()

    async def test_exhausted_solver_fails_headless_after_one_diagnostic(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["cf_managed_challenge"], events)
        solver = _Solver(
            events,
            error=FlareSolverrSessionUnavailable("request exhausted"),
        )

        async def save_diagnostic(kind: str) -> None:
            self.assertEqual(kind, "challenge_page")
            events.append("save_diagnostic")

        navigate = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=True,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=cast(
                Callable[[object], Awaitable[None]],
                AsyncMock(side_effect=_unexpected_callback),
            ),
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with (
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertLogs("test-managed-challenge", level="WARNING") as captured,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            ["detect:cf_managed_challenge", "solve", "save_diagnostic"],
        )
        self.assertEqual(detector.pages, [page])
        self.assertEqual(solver.mark_calls, [])
        navigate.assert_not_awaited()
        sleep.assert_not_awaited()
        rendered_log = "\n".join(captured.output)
        self.assertIn("Automatic Cloudflare verification failed", rendered_log)
        self.assertNotIn("request failed", rendered_log)

    async def test_solver_is_retired_before_manual_fallback_after_identity_fails(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(
            ["cf_managed_challenge", "cf_managed_challenge", "none"],
            events,
        )
        receipt = _receipt(8)
        solver = _Solver(events, receipt=receipt)

        async def apply_identity(actual: object) -> None:
            self.assertIs(actual, receipt)
            events.append("apply_identity")

        async def navigate(url: str) -> None:
            self.assertEqual(url, FORUMS_URL)
            events.append("navigate")

        async def save_diagnostic(kind: str) -> None:
            self.assertEqual(kind, "challenge_page")
            events.append("save_diagnostic")

        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-page-challenge"),
            headless=False,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=apply_identity,
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with patch(
            "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            [
                "detect:cf_managed_challenge",
                "solve",
                "apply_identity",
                "mark_identity_applied",
                "navigate",
                "detect:cf_managed_challenge",
                "retire_solver",
                "save_diagnostic",
                "detect:none",
            ],
        )
        self.assertEqual(solver.retire_calls, 1)
        sleep.assert_not_awaited()

    async def test_gui_solver_failure_waits_on_the_same_page_for_manual_solution(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(
            ["cf_managed_challenge", "cf_managed_challenge", "none"],
            events,
        )
        solver = _Solver(
            events,
            error=FlareSolverrSessionUnavailable("request exhausted"),
        )

        async def save_diagnostic(kind: str) -> None:
            self.assertEqual(kind, "challenge_page")
            events.append("save_diagnostic")

        navigate = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=False,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=cast(
                Callable[[object], Awaitable[None]],
                AsyncMock(side_effect=_unexpected_callback),
            ),
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with patch(
            "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            [
                "detect:cf_managed_challenge",
                "solve",
                "save_diagnostic",
                "detect:cf_managed_challenge",
                "detect:none",
            ],
        )
        self.assertEqual(detector.pages, [page, page, page])
        self.assertEqual(solver.mark_calls, [])
        navigate.assert_not_awaited()
        sleep.assert_awaited_once_with(1)

    async def test_identity_apply_failure_propagates_without_manual_fallback(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["cf_managed_challenge"], events)
        receipt = _receipt(9)
        solver = _Solver(events, receipt=receipt)

        async def apply_identity(actual: object) -> None:
            self.assertIs(actual, receipt)
            events.append("apply_identity")
            raise BrowserIdentityApplyException("identity is indeterminate")

        navigate = AsyncMock(side_effect=_unexpected_callback)
        save_diagnostic = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=False,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=apply_identity,
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with (
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(BrowserIdentityApplyException),
        ):
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            ["detect:cf_managed_challenge", "solve", "apply_identity"],
        )
        self.assertEqual(solver.mark_calls, [])
        navigate.assert_not_awaited()
        save_diagnostic.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_unexpected_solver_bug_propagates_without_manual_fallback(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["cf_managed_challenge"], events)
        solver = _Solver(events, error=RuntimeError("programming bug"))
        navigate = AsyncMock(side_effect=_unexpected_callback)
        save_diagnostic = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=False,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=cast(
                Callable[[object], Awaitable[None]],
                AsyncMock(side_effect=_unexpected_callback),
            ),
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with (
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(RuntimeError, "programming bug"),
        ):
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(events, ["detect:cf_managed_challenge", "solve"])
        navigate.assert_not_awaited()
        save_diagnostic.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_headless_without_solver_fails_immediately(self) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["cf_managed_challenge"], events)

        async def save_diagnostic(kind: str) -> None:
            self.assertEqual(kind, "challenge_page")
            events.append("save_diagnostic")

        navigate = AsyncMock(side_effect=_unexpected_callback)
        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-managed-challenge"),
            headless=True,
            manual_timeout=30,
            automatic_solver=None,
            apply_identity=cast(
                Callable[[object], Awaitable[None]],
                AsyncMock(side_effect=_unexpected_callback),
            ),
            navigate=navigate,
            save_diagnostic=save_diagnostic,
        )

        with (
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            ["detect:cf_managed_challenge", "save_diagnostic"],
        )
        self.assertEqual(detector.pages, [page])
        navigate.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_non_managed_page_challenge_uses_manual_policy_without_solver(
        self,
    ) -> None:
        for kind in ("turnstile_widget", "recaptcha_v2"):
            with self.subTest(kind=kind):
                events: list[str] = []
                page = object()
                detector = _Detector([kind, "none"], events)
                solver = _Solver(events, receipt=_receipt(1))

                async def save_diagnostic(actual: str) -> None:
                    self.assertEqual(actual, "challenge_page")
                    events.append("save_diagnostic")

                navigate = AsyncMock(side_effect=_unexpected_callback)
                handler = PageChallengeHandler(
                    detector=detector,
                    logger=logging.getLogger("test-page-challenge"),
                    headless=False,
                    manual_timeout=30,
                    automatic_solver=solver,
                    apply_identity=cast(
                        Callable[[object], Awaitable[None]],
                        AsyncMock(side_effect=_unexpected_callback),
                    ),
                    navigate=navigate,
                    save_diagnostic=save_diagnostic,
                )

                with (
                    patch(
                        "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                        new=AsyncMock(),
                    ) as sleep,
                    self.assertLogs("test-page-challenge", level="INFO") as captured,
                ):
                    await handler.resolve(page, FORUMS_URL)

                self.assertEqual(
                    events,
                    [f"detect:{kind}", "save_diagnostic", "detect:none"],
                )
                self.assertEqual(solver.solve_calls, [])
                self.assertEqual(solver.mark_calls, [])
                navigate.assert_not_awaited()
                sleep.assert_not_awaited()
                rendered_log = "\n".join(captured.output)
                self.assertIn(f"kind={kind}", rendered_log)
                self.assertNotIn("Cloudflare managed challenge", rendered_log)

    async def test_non_managed_page_challenge_fails_immediately_headless(
        self,
    ) -> None:
        events: list[str] = []
        page = object()
        detector = _Detector(["turnstile_widget"], events)
        solver = _Solver(events, receipt=_receipt(1))

        async def save_diagnostic(kind: str) -> None:
            self.assertEqual(kind, "challenge_page")
            events.append("save_diagnostic")

        handler = PageChallengeHandler(
            detector=detector,
            logger=logging.getLogger("test-page-challenge"),
            headless=True,
            manual_timeout=30,
            automatic_solver=solver,
            apply_identity=cast(
                Callable[[object], Awaitable[None]],
                AsyncMock(side_effect=_unexpected_callback),
            ),
            navigate=AsyncMock(side_effect=_unexpected_callback),
            save_diagnostic=save_diagnostic,
        )

        with (
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page, FORUMS_URL)

        self.assertEqual(
            events,
            ["detect:turnstile_widget", "save_diagnostic"],
        )
        self.assertEqual(solver.solve_calls, [])
        self.assertEqual(solver.mark_calls, [])
        sleep.assert_not_awaited()
