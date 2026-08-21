from __future__ import annotations

import inspect
import unittest
from collections import deque
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from hbrowser.exceptions import LoginFailedException
from hbrowser.gallery import driver_base as driver_base_module
from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrResult,
    FlareSolverrSessionScope,
    FlareSolverrSessionUnavailable,
    FlareSolverrSolveReceipt,
)
from hbrowser.gallery.captcha.models import ChallengeDetection, Kind
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.forums_auth import ForumsAuthState

FORUMS_URL = "https://forums.e-hentai.org/"


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class _Detector:
    def __init__(self, kinds: list[Kind]) -> None:
        self._kinds = deque(kinds)
        self.pages: list[object] = []
        self.timeouts: list[float] = []

    async def detect(
        self,
        page: object,
        timeout: float = 2.0,
    ) -> ChallengeDetection:
        if not self._kinds:
            raise AssertionError("Unexpected challenge detection")
        self.pages.append(page)
        self.timeouts.append(timeout)
        return ChallengeDetection(url=FORUMS_URL, kind=self._kinds.popleft())


class _UnavailableScope:
    def __init__(self) -> None:
        self.solve_calls: list[str] = []
        self.mark_calls: list[object] = []

    async def solve_managed(self, url: str) -> object:
        self.solve_calls.append(url)
        raise FlareSolverrSessionUnavailable("request attempts exhausted")

    def mark_identity_applied(self, receipt: object) -> None:
        self.mark_calls.append(receipt)
        raise AssertionError("An unavailable solve cannot be committed")


class _TokenPage:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._results = deque(["", True])
        self.expressions: list[str] = []

    async def evaluate(self, expression: str) -> object:
        self.expressions.append(expression)
        self._events.append(
            "read_token" if len(self.expressions) == 1 else "inject_token"
        )
        if not self._results:
            raise AssertionError("Unexpected page evaluation")
        return self._results.popleft()


class _TurnstileScope:
    def __init__(
        self,
        events: list[str],
        receipt: FlareSolverrSolveReceipt,
    ) -> None:
        self._events = events
        self._receipt = receipt
        self.solve_calls: list[tuple[str, int, int]] = []
        self.mark_calls: list[FlareSolverrSolveReceipt] = []

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> FlareSolverrSolveReceipt:
        self.solve_calls.append((url, tabs, timeout_ms))
        self._events.append("solve_turnstile")
        return self._receipt

    def mark_identity_applied(self, receipt: FlareSolverrSolveReceipt) -> None:
        self.mark_calls.append(receipt)
        self._events.append("mark_identity_applied")


class DriverFlareSolverrCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_login_owns_bounded_scope_and_passes_it_to_login_flow(
        self,
    ) -> None:
        driver = _TestDriver(flaresolverr_session_attempts=5)
        login_flow = AsyncMock()
        driver._login = login_flow  # type: ignore[method-assign]
        scope = Mock(spec=FlareSolverrSessionScope)

        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=scope)
        session_context.__aexit__ = AsyncMock(return_value=None)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.session.return_value = session_context
        client.create_session = AsyncMock(
            side_effect=AssertionError("Driver must not create raw sessions")
        )
        client.destroy_session = AsyncMock(
            side_effect=AssertionError("The scope owns session cleanup")
        )
        client_type = Mock(return_value=client)

        with (
            patch.object(
                driver_base_module,
                "get_flaresolverr_url",
                return_value="http://127.0.0.1:8191/v1",
            ),
            patch.object(
                driver_base_module,
                "should_use_flaresolverr",
                return_value=True,
            ),
            patch.object(
                driver_base_module,
                "FlareSolverrClient",
                client_type,
            ),
        ):
            await driver.login()

        client_type.assert_called_once_with("http://127.0.0.1:8191/v1")
        client.session.assert_called_once_with(max_attempts=5)
        login_flow.assert_awaited_once_with(scope)
        session_context.__aenter__.assert_awaited_once_with()
        session_context.__aexit__.assert_awaited_once_with(None, None, None)
        client.__aenter__.assert_awaited_once_with()
        client.__aexit__.assert_awaited_once_with(None, None, None)
        client.create_session.assert_not_awaited()
        client.destroy_session.assert_not_awaited()

    async def test_login_without_endpoint_passes_none_without_building_client(
        self,
    ) -> None:
        driver = _TestDriver()
        login_flow = AsyncMock()
        driver._login = login_flow  # type: ignore[method-assign]
        client_type = Mock(side_effect=AssertionError("Client must stay disabled"))

        with (
            patch.object(
                driver_base_module,
                "get_flaresolverr_url",
                return_value=None,
            ),
            patch.object(
                driver_base_module,
                "should_use_flaresolverr",
            ) as eligibility,
            patch.object(
                driver_base_module,
                "FlareSolverrClient",
                client_type,
            ),
        ):
            await driver.login()

        login_flow.assert_awaited_once_with(None)
        eligibility.assert_not_called()
        client_type.assert_not_called()

    async def test_ineligible_solver_passes_none_without_building_client(
        self,
    ) -> None:
        driver = _TestDriver()
        login_flow = AsyncMock()
        driver._login = login_flow  # type: ignore[method-assign]
        client_type = Mock(side_effect=AssertionError("Client must stay disabled"))

        with (
            patch.object(
                driver_base_module,
                "get_flaresolverr_url",
                return_value="http://127.0.0.1:8191/v1",
            ),
            patch.object(
                driver_base_module,
                "should_use_flaresolverr",
                return_value=False,
            ),
            patch.object(
                driver_base_module,
                "FlareSolverrClient",
                client_type,
            ),
        ):
            await driver.login()

        login_flow.assert_awaited_once_with(None)
        client_type.assert_not_called()


class DriverPageChallengeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_gui_solver_failure_keeps_browser_and_waits_on_same_page(
        self,
    ) -> None:
        driver = _TestDriver(headless=False)
        page = object()
        navigator = AsyncMock(
            side_effect=AssertionError("Solver failure must not navigate")
        )
        detector = _Detector(["cf_managed_challenge", "cf_managed_challenge", "none"])
        scope = _UnavailableScope()
        driver.page = page
        driver.myget = navigator
        driver.captcha_detector = detector  # type: ignore[assignment]
        diagnostic = AsyncMock(return_value=None)
        driver.save_page_diagnostic = diagnostic  # type: ignore[method-assign]

        with (
            patch.object(
                driver_base_module,
                "create_browser",
                new=AsyncMock(side_effect=AssertionError("Unexpected browser create")),
            ) as create_browser,
            patch.object(
                driver_base_module,
                "stop_browser",
                new=AsyncMock(side_effect=AssertionError("Unexpected browser stop")),
            ) as stop_browser,
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            await driver._handle_page_challenge(
                FORUMS_URL,
                cast(FlareSolverrSessionScope, scope),
                detect_timeout=1.5,
            )

        self.assertIs(driver.page, page)
        self.assertIs(driver.myget, navigator)
        self.assertEqual(detector.pages, [page, page, page])
        self.assertEqual(detector.timeouts, [1.5, 1.5, 1.5])
        self.assertEqual(scope.solve_calls, [FORUMS_URL])
        self.assertEqual(scope.mark_calls, [])
        diagnostic.assert_awaited_once_with("challenge_page")
        navigator.assert_not_awaited()
        sleep.assert_awaited_once_with(1)
        create_browser.assert_not_awaited()
        stop_browser.assert_not_awaited()

    async def test_post_submit_verification_uses_same_scope_before_auth_check(
        self,
    ) -> None:
        events: list[str] = []
        driver = _TestDriver()
        page = object()
        scope = Mock(spec=FlareSolverrSessionScope)
        driver.page = page

        async def navigate(url: str) -> None:
            self.assertEqual(url, FORUMS_URL)
            events.append("navigate")

        async def handle_page_challenge(
            url: str,
            actual_scope: FlareSolverrSessionScope | None,
        ) -> None:
            self.assertEqual(url, FORUMS_URL)
            self.assertIs(actual_scope, scope)
            events.append("handle_page_challenge")

        async def detect_auth(actual_page: object) -> ForumsAuthState:
            self.assertIs(actual_page, page)
            events.append("detect_auth")
            return ForumsAuthState.AUTHENTICATED

        driver.myget = navigate

        with (
            patch.object(
                driver,
                "_handle_page_challenge",
                new=handle_page_challenge,
            ),
            patch.object(
                driver_base_module,
                "detect_forums_auth_state",
                new=detect_auth,
            ),
        ):
            await driver._verify_login_succeeded(scope)

        self.assertEqual(
            events,
            ["navigate", "handle_page_challenge", "detect_auth"],
        )

    async def test_turnstile_adapter_applies_and_anchors_before_injection(
        self,
    ) -> None:
        events: list[str] = []
        driver = _TestDriver(headless=True)
        page = _TokenPage(events)
        receipt = FlareSolverrSolveReceipt(
            result=FlareSolverrResult(
                cookies=[],
                user_agent="test-agent",
                turnstile_token="single-use-token",
            ),
            session_generation=4,
        )
        scope = _TurnstileScope(events, receipt)
        driver.page = page
        driver.captcha_detector = _Detector(["turnstile_widget"])  # type: ignore[assignment]

        async def apply_identity(actual: FlareSolverrSolveReceipt) -> None:
            self.assertIs(actual, receipt)
            events.append("apply_identity")

        with patch.object(
            driver,
            "_apply_flaresolverr_receipt",
            new=apply_identity,
        ):
            await driver._handle_login_challenge(cast(FlareSolverrSessionScope, scope))

        self.assertEqual(
            events,
            [
                "read_token",
                "solve_turnstile",
                "apply_identity",
                "mark_identity_applied",
                "inject_token",
            ],
        )
        self.assertEqual(scope.solve_calls, [(FORUMS_URL, 15, 30_000)])
        self.assertEqual(scope.mark_calls, [receipt])
        self.assertIn("single-use-token", page.expressions[-1])

    async def test_headless_solver_failure_keeps_browser_and_fails_without_sleep(
        self,
    ) -> None:
        driver = _TestDriver(headless=True)
        page = object()
        navigator = AsyncMock(
            side_effect=AssertionError("Solver failure must not navigate")
        )
        detector = _Detector(["cf_managed_challenge"])
        scope = _UnavailableScope()
        driver.page = page
        driver.myget = navigator
        driver.captcha_detector = detector  # type: ignore[assignment]
        diagnostic = AsyncMock(return_value=None)
        driver.save_page_diagnostic = diagnostic  # type: ignore[method-assign]

        with (
            patch.object(
                driver_base_module,
                "create_browser",
                new=AsyncMock(side_effect=AssertionError("Unexpected browser create")),
            ) as create_browser,
            patch.object(
                driver_base_module,
                "stop_browser",
                new=AsyncMock(side_effect=AssertionError("Unexpected browser stop")),
            ) as stop_browser,
            patch(
                "hbrowser.gallery.captcha.page_challenge.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(LoginFailedException),
        ):
            await driver._handle_page_challenge(
                FORUMS_URL,
                cast(FlareSolverrSessionScope, scope),
            )

        self.assertIs(driver.page, page)
        self.assertIs(driver.myget, navigator)
        self.assertEqual(detector.pages, [page])
        self.assertEqual(scope.solve_calls, [FORUMS_URL])
        self.assertEqual(scope.mark_calls, [])
        diagnostic.assert_awaited_once_with("challenge_page")
        navigator.assert_not_awaited()
        sleep.assert_not_awaited()
        create_browser.assert_not_awaited()
        stop_browser.assert_not_awaited()


class DriverChallengeConfigurationTests(unittest.TestCase):
    def test_constructor_exposes_only_separated_challenge_configuration(
        self,
    ) -> None:
        parameters = inspect.signature(Driver.__init__).parameters

        self.assertEqual(
            tuple(parameters),
            (
                "self",
                "headless",
                "flaresolverr_session_attempts",
                "captcha_manual_timeout",
                "turnstile_tabs",
            ),
        )
        self.assertNotIn("proxy_rotator", parameters)
        self.assertNotIn("max_captcha_retries", parameters)

    def test_constructor_requires_positive_integer_session_attempts(self) -> None:
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _TestDriver(flaresolverr_session_attempts=invalid)  # type: ignore[arg-type]

        driver = _TestDriver(flaresolverr_session_attempts=4)
        self.assertEqual(driver.flaresolverr_session_attempts, 4)
