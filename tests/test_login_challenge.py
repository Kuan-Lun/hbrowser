import logging
import unittest
from collections import deque
from unittest.mock import AsyncMock, patch

from hbrowser.exceptions import (
    BrowserIdentityApplyException,
    LoginFailedException,
    LoginTokenInjectionOutcomeUnknownError,
)
from hbrowser.gallery.browser.flaresolverr import FlareSolverrSessionUnavailable
from hbrowser.gallery.captcha import (
    CaptchaDetector,
    ChallengeDetection,
    LoginChallengeHandler,
)

LOGIN_URL = "https://forums.e-hentai.org/index.php?act=Login&CODE=00"


class _Detector(CaptchaDetector):
    def __init__(self, kind: str) -> None:
        self.detection = ChallengeDetection(url=LOGIN_URL, kind=kind)  # type: ignore[arg-type]

    async def detect(self, page: object, timeout: float = 2.0) -> ChallengeDetection:
        del page, timeout
        return self.detection


class _Page:
    def __init__(self, *results: object) -> None:
        self.results = deque(results)
        self.expressions: list[str] = []

    async def evaluate(self, expression: str) -> object:
        self.expressions.append(expression)
        if not self.results:
            raise AssertionError(f"Unexpected evaluate expression: {expression}")
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class _ScriptedDeadline:
    def __init__(self, *, expired: list[bool], remaining: list[float]) -> None:
        self._expired = deque(expired)
        self._remaining = deque(remaining)

    @property
    def expired(self) -> bool:
        return self._expired.popleft()

    def remaining(self) -> float:
        return self._remaining.popleft()


class _Solver:
    def __init__(
        self,
        *,
        token: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.token = token
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> str | None:
        self.calls.append((url, tabs, timeout_ms))
        if self.error is not None:
            raise self.error
        return self.token


def _handler(
    kind: str,
    *,
    solver: _Solver | None,
    headless: bool = False,
) -> LoginChallengeHandler:
    return LoginChallengeHandler(
        detector=_Detector(kind),
        logger=logging.getLogger("test-login-challenge"),
        headless=headless,
        manual_timeout=30,
        turnstile_tabs=15,
        automatic_solver=solver,
    )


class LoginChallengeHandlerTests(unittest.IsolatedAsyncioTestCase):
    def test_turnstile_tab_count_is_a_bounded_integer(self) -> None:
        for invalid in (0, True, 1.5, float("inf"), 31, 10**9):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                LoginChallengeHandler(
                    detector=_Detector("none"),
                    logger=logging.getLogger("test-login-challenge"),
                    headless=False,
                    manual_timeout=30,
                    turnstile_tabs=invalid,  # type: ignore[arg-type]
                    automatic_solver=None,
                )

    def test_manual_timeout_cannot_exceed_human_wait_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\(0, 180\]"):
            LoginChallengeHandler(
                detector=_Detector("none"),
                logger=logging.getLogger("test-login-challenge"),
                headless=False,
                manual_timeout=181,
                turnstile_tabs=15,
                automatic_solver=None,
            )

    async def test_auto_turnstile_token_is_injected(self) -> None:
        page = _Page("", True)
        solver = _Solver(token="generated-token")

        await _handler("turnstile_widget", solver=solver).resolve(page)

        self.assertEqual(solver.calls, [(LOGIN_URL, 15, 30_000)])
        injection_script = page.expressions[1]
        self.assertIn("cf-turnstile-response", injection_script)
        self.assertIn('"generated-token"', injection_script)
        self.assertIn("new Event('input'", injection_script)
        self.assertIn("new Event('change'", injection_script)
        self.assertNotIn("setAttribute", injection_script)

    async def test_solver_failure_falls_back_to_manual_in_gui(self) -> None:
        page = _Page("", "", "manual-token")
        solver = _Solver(error=FlareSolverrSessionUnavailable("solver unavailable"))

        with patch(
            "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            await _handler("turnstile_widget", solver=solver).resolve(page)

        self.assertEqual(len(solver.calls), 1)
        sleep.assert_awaited_once_with(1)

    async def test_unexpected_solver_bug_propagates_without_manual_fallback(
        self,
    ) -> None:
        page = _Page("")
        solver = _Solver(error=RuntimeError("programming bug"))

        with (
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(RuntimeError, "programming bug"),
        ):
            await _handler("turnstile_widget", solver=solver).resolve(page)

        sleep.assert_not_awaited()

    async def test_identity_apply_failure_never_falls_back_to_manual(self) -> None:
        page = _Page("")
        solver = _Solver(
            error=BrowserIdentityApplyException(
                "Could not apply the FlareSolverr Cloudflare cookies"
            )
        )

        with (
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(BrowserIdentityApplyException),
        ):
            await _handler("turnstile_widget", solver=solver).resolve(page)

        sleep.assert_not_awaited()

    async def test_missing_solver_token_falls_back_to_manual_in_gui(self) -> None:
        page = _Page("", "manual-token")
        solver = _Solver(token=None)

        await _handler("turnstile_widget", solver=solver).resolve(page)

        self.assertEqual(len(solver.calls), 1)

    async def test_headless_unresolved_turnstile_fails_immediately(self) -> None:
        page = _Page("")
        solver = _Solver(token=None)

        with (
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await _handler(
                "turnstile_widget",
                solver=solver,
                headless=True,
            ).resolve(page)

        self.assertEqual(len(solver.calls), 1)
        sleep.assert_not_awaited()

    async def test_token_injection_failure_is_redacted_and_never_falls_back(
        self,
    ) -> None:
        page = _Page(
            "",
            RuntimeError("protocol params included generated-token"),
            "manual-token-must-not-be-read",
        )
        solver = _Solver(token="generated-token")

        with (
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginTokenInjectionOutcomeUnknownError) as raised,
        ):
            await _handler("turnstile_widget", solver=solver).resolve(page)

        self.assertNotIn("generated-token", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual(len(page.expressions), 2)
        sleep.assert_not_awaited()

    async def test_existing_recaptcha_token_needs_no_solver(self) -> None:
        page = _Page("existing-recaptcha-token")
        solver = _Solver(token="must-not-be-used")

        await _handler(
            "recaptcha_v2",
            solver=solver,
            headless=True,
        ).resolve(page)

        self.assertEqual(solver.calls, [])

    async def test_recaptcha_manual_fallback_is_preserved(self) -> None:
        page = _Page("", "manual-recaptcha-token")
        solver = _Solver(token="must-not-be-used")

        await _handler("recaptcha_v2", solver=solver).resolve(page)

        self.assertEqual(solver.calls, [])
        self.assertTrue(
            all(
                "#g-recaptcha-response" in expression for expression in page.expressions
            )
        )

    async def test_expired_manual_deadline_does_not_read_token_again(self) -> None:
        page = _Page()
        handler = _handler("recaptcha_v2", solver=None)
        read_token = AsyncMock(return_value="")

        with (
            patch.object(handler, "_read_response_token", read_token),
            patch(
                "hbrowser.gallery.captcha.login_challenge.Deadline.after",
                return_value=_ScriptedDeadline(expired=[True], remaining=[]),
            ),
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page)

        read_token.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_near_manual_deadline_bounds_token_read_and_sleep(self) -> None:
        page = _Page()
        handler = _handler("recaptcha_v2", solver=None)
        read_token = AsyncMock(return_value="")
        deadline = _ScriptedDeadline(
            expired=[False, True],
            remaining=[0.25, 0.05],
        )

        with (
            patch.object(handler, "_read_response_token", read_token),
            patch(
                "hbrowser.gallery.captcha.login_challenge.Deadline.after",
                return_value=deadline,
            ),
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page)

        self.assertEqual(read_token.await_count, 2)
        self.assertEqual(read_token.await_args_list[1].kwargs["timeout"], 0.25)
        sleep.assert_awaited_once_with(0.05)

    async def test_token_observed_after_manual_deadline_is_not_accepted(self) -> None:
        page = _Page("", "late-token")
        handler = _handler("recaptcha_v2", solver=None)
        deadline = _ScriptedDeadline(
            expired=[False],
            remaining=[0.25, 0.0],
        )

        with (
            patch(
                "hbrowser.gallery.captcha.login_challenge.Deadline.after",
                return_value=deadline,
            ),
            patch(
                "hbrowser.gallery.captcha.login_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await handler.resolve(page)

        sleep.assert_not_awaited()
