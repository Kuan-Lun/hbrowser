import asyncio
import unittest
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx

from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrClient,
    FlareSolverrConfigurationError,
    FlareSolverrProtocolError,
    FlareSolverrRequestError,
    FlareSolverrResult,
    FlareSolverrSession,
    FlareSolverrSessionOwnershipError,
    FlareSolverrSessionScope,
    FlareSolverrSessionUnavailable,
    FlareSolverrSolveReceipt,
)

LOGIN_URL = "https://forums.e-hentai.org/index.php?act=Login&CODE=00"


class _Response:
    def __init__(
        self,
        data: dict[str, Any],
        *,
        status_error: httpx.HTTPStatusError | None = None,
    ) -> None:
        self._data = data
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict[str, Any]:
        return self._data


class _Created:
    def __init__(self, label: str) -> None:
        self.label = label


class _ListedLatestCreate:
    def __init__(self, *, present: bool) -> None:
        self.present = present


class _HTTP:
    def __init__(
        self,
        *responses: (
            dict[str, Any]
            | _Response
            | _Created
            | _ListedLatestCreate
            | httpx.HTTPError
        ),
    ) -> None:
        self.responses = deque(responses)
        self.posts: list[dict[str, Any]] = []
        self.create_session_ids: list[str] = []
        self.session_labels: dict[str, str] = {}
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: httpx.Timeout | float,
    ) -> _Response:
        self.posts.append(
            {
                "endpoint": endpoint,
                "json": json,
                "timeout": timeout,
            }
        )
        if json.get("cmd") == "sessions.create":
            session_id = json.get("session")
            if not isinstance(session_id, str) or not session_id:
                raise AssertionError("sessions.create requires a caller-owned ID")
            self.create_session_ids.append(session_id)
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        response = self.responses.popleft()
        if isinstance(response, httpx.HTTPError):
            raise response
        if isinstance(response, _Response):
            return response
        if isinstance(response, _Created):
            if json.get("cmd") != "sessions.create":
                raise AssertionError("Created response used for a non-create request")
            session_id = self.create_session_ids[-1]
            self.session_labels[session_id] = response.label
            return _Response({"status": "ok", "session": session_id})
        if isinstance(response, _ListedLatestCreate):
            if json.get("cmd") != "sessions.list":
                raise AssertionError("List response used for a non-list request")
            sessions = [self.create_session_ids[-1]] if response.present else []
            return _Response({"status": "ok", "sessions": sessions})
        return _Response(response)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0


class _ClockDeadline:
    def __init__(self, clock: _Clock, *, expires_at: float) -> None:
        self._clock = clock
        self._expires_at = expires_at

    def remaining(self) -> float:
        return max(0.0, self._expires_at - self._clock.now)

    def bounded(self, seconds: float) -> _ClockDeadline:
        return _ClockDeadline(
            self._clock,
            expires_at=min(self._expires_at, self._clock.now + seconds),
        )


class _SlowFirstRequestHTTP(_HTTP):
    def __init__(
        self,
        clock: _Clock,
        *responses: (
            dict[str, Any]
            | _Response
            | _Created
            | _ListedLatestCreate
            | httpx.HTTPError
        ),
    ) -> None:
        super().__init__(*responses)
        self._clock = clock
        self._advanced = False

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: httpx.Timeout | float,
    ) -> _Response:
        try:
            return await super().post(endpoint, json=json, timeout=timeout)
        finally:
            if json["cmd"] == "request.get" and not self._advanced:
                self._advanced = True
                self._clock.now += 50.0


class _ClockAdvancingHTTP(_HTTP):
    def __init__(
        self,
        clock: _Clock,
        advances: dict[str, float],
        *responses: (
            dict[str, Any]
            | _Response
            | _Created
            | _ListedLatestCreate
            | httpx.HTTPError
        ),
    ) -> None:
        super().__init__(*responses)
        self._clock = clock
        self._advances = advances

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: httpx.Timeout | float,
    ) -> _Response:
        try:
            return await super().post(endpoint, json=json, timeout=timeout)
        finally:
            self._clock.now += self._advances.get(json["cmd"], 0.0)


class _BlockingDestroyHTTP(_HTTP):
    def __init__(
        self,
        *responses: (
            dict[str, Any]
            | _Response
            | _Created
            | _ListedLatestCreate
            | httpx.HTTPError
        ),
    ) -> None:
        super().__init__(*responses)
        self.destroy_started = asyncio.Event()

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: httpx.Timeout | float,
    ) -> _Response:
        if json.get("cmd") != "sessions.destroy":
            return await super().post(endpoint, json=json, timeout=timeout)
        self.posts.append({"endpoint": endpoint, "json": json, "timeout": timeout})
        self.destroy_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _BlockingRequestHTTP(_HTTP):
    def __init__(
        self,
        *responses: (
            dict[str, Any]
            | _Response
            | _Created
            | _ListedLatestCreate
            | httpx.HTTPError
        ),
    ) -> None:
        super().__init__(*responses)
        self.request_started = asyncio.Event()

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: httpx.Timeout | float,
    ) -> _Response:
        if json.get("cmd") != "request.get":
            return await super().post(endpoint, json=json, timeout=timeout)
        self.posts.append({"endpoint": endpoint, "json": json, "timeout": timeout})
        self.request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _http_failure(label: str) -> httpx.ConnectError:
    request = httpx.Request("POST", "http://127.0.0.1:8191/v1")
    return httpx.ConnectError(f"private transport detail: {label}", request=request)


def _http_read_timeout(label: str) -> httpx.ReadTimeout:
    request = httpx.Request("POST", "http://127.0.0.1:8191/v1")
    return httpx.ReadTimeout(
        f"private transport detail: {label}",
        request=request,
    )


def _http_502(secret: str) -> _Response:
    request = httpx.Request("POST", "http://127.0.0.1:8191/v1")
    response = httpx.Response(502, request=request)
    error = httpx.HTTPStatusError(
        f"502 response contained {secret}",
        request=request,
        response=response,
    )
    return _Response({}, status_error=error)


def _created(session_id: str) -> _Created:
    return _Created(session_id)


def _listed_latest(*, present: bool) -> _ListedLatestCreate:
    return _ListedLatestCreate(present=present)


def _destroyed() -> dict[str, Any]:
    return {"status": "ok", "message": ""}


def _commands(http: _HTTP) -> list[str]:
    return [post["json"]["cmd"] for post in http.posts]


def _request_sessions(http: _HTTP) -> list[str]:
    return [
        http.session_labels.get(post["json"]["session"], post["json"]["session"])
        for post in http.posts
        if post["json"]["cmd"] == "request.get"
    ]


def _destroy_sessions(http: _HTTP) -> list[str]:
    return [
        http.session_labels.get(post["json"]["session"], post["json"]["session"])
        for post in http.posts
        if post["json"]["cmd"] == "sessions.destroy"
    ]


def _solution(
    token: object = None,
    *,
    user_agent: str = "test-agent",
) -> dict[str, Any]:
    return {
        "status": "ok",
        "solution": {
            "cookies": [],
            "userAgent": user_agent,
            "turnstile_token": token,
        },
    }


def _own_session(client: FlareSolverrClient, session_id: str) -> None:
    client._owned_session_ids.add(session_id)


class FlareSolverrTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_http_response_is_rejected_before_json_acceptance(self) -> None:
        clock = _Clock()
        deadline = _ClockDeadline(clock, expires_at=50.0)
        response = _Response(_solution())
        response.json = Mock(wraps=response.json)  # type: ignore[method-assign]
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _SlowFirstRequestHTTP(  # type: ignore[assignment]
            clock,
            response,
        )

        with self.assertRaises(FlareSolverrRequestError) as raised:
            await client._post(
                {"cmd": "request.get"},
                deadline=deadline,  # type: ignore[arg-type]
            )

        self.assertEqual(raised.exception.transport_type, "TimeoutError")
        response.json.assert_not_called()

    def test_session_attempt_count_has_a_fixed_production_cap(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")

        with self.assertRaisesRegex(ValueError, r"\[1, 3\]"):
            FlareSolverrSessionScope(client, max_attempts=4)

    async def test_managed_timeout_cannot_exceed_sixty_seconds(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        session = FlareSolverrSession(client, "session")

        with self.assertRaisesRegex(ValueError, r"\(0, 60000\]"):
            await session.get(LOGIN_URL, timeout_ms=60_001)

    async def test_turnstile_timeout_cannot_exceed_thirty_seconds(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        session = FlareSolverrSession(client, "session")

        with self.assertRaisesRegex(ValueError, r"\(0, 30000\]"):
            await session.solve_turnstile(
                LOGIN_URL,
                tabs=15,
                timeout_ms=30_001,
            )

    def test_invalid_endpoint_is_a_sanitized_configuration_error(self) -> None:
        secret = "PRIVATE-MALFORMED-ENDPOINT"

        with self.assertRaises(FlareSolverrConfigurationError) as raised:
            FlareSolverrClient(f"not-a-url-{secret}")

        rendered = f"{raised.exception!r} {raised.exception}"
        self.assertIn("HTTP", rendered)
        self.assertNotIn(secret, rendered)

    def test_non_http_endpoint_is_rejected(self) -> None:
        with self.assertRaises(FlareSolverrConfigurationError):
            FlareSolverrClient("ftp://127.0.0.1/v1")

    def test_only_cloudflare_cookies_are_exported_and_secrets_are_redacted(
        self,
    ) -> None:
        result = FlareSolverrResult(
            cookies=[
                {
                    "name": "ipb_member_id",
                    "value": "application-secret",
                    "domain": ".e-hentai.org",
                },
                {
                    "name": "cf_clearance",
                    "value": "clearance-secret",
                    "domain": ".e-hentai.org",
                },
                {
                    "name": "__cf_bm",
                    "value": "bot-management-secret",
                    "domain": ".e-hentai.org",
                },
            ],
            user_agent="test-agent",
        )

        cookie_params = result.to_cdp_cloudflare_cookie_params()

        self.assertEqual(
            [cookie.name for cookie in cookie_params],
            ["cf_clearance", "__cf_bm"],
        )
        self.assertNotIn("application-secret", repr(result))
        self.assertNotIn("clearance-secret", repr(result))

    async def test_parses_and_redacts_turnstile_token(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(_solution("single-use-token"))
        client._http = http  # type: ignore[assignment]
        _own_session(client, "session-1")

        result = await client.get(
            LOGIN_URL,
            session_id="session-1",
            turnstile_tabs=15,
        )

        self.assertEqual(result.turnstile_token, "single-use-token")
        self.assertNotIn("single-use-token", repr(result))
        payload = http.posts[0]["json"]
        self.assertEqual(payload["session"], "session-1")
        self.assertEqual(payload["tabs_till_verify"], 15)
        self.assertIs(payload["returnOnlyCookies"], True)
        timeout = http.posts[0]["timeout"]
        self.assertIsInstance(timeout, httpx.Timeout)
        self.assertLessEqual(timeout.connect, 5)
        self.assertLessEqual(timeout.write, 5)
        self.assertLessEqual(timeout.pool, 5)
        self.assertGreater(timeout.read, 5)

    async def test_get_rejects_unbounded_turnstile_work(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(_solution("unused"))  # type: ignore[assignment]
        _own_session(client, "session-1")

        for invalid in (0, True, 1.5, float("inf"), 31, 10**9):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                await client.get(
                    LOGIN_URL,
                    session_id="session-1",
                    turnstile_tabs=invalid,  # type: ignore[arg-type]
                )

    async def test_http_transport_close_has_a_command_phase_watchdog(self) -> None:
        async def hang() -> None:
            await asyncio.Event().wait()

        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        close = AsyncMock(side_effect=hang)
        client._http = Mock(aclose=close)

        with (
            patch(
                "hbrowser.gallery.browser.flaresolverr._TRANSPORT_PHASE_TIMEOUT_SECONDS",
                0.01,
            ),
            patch("hbrowser.gallery.browser.flaresolverr.logger") as logger,
        ):
            await client.__aexit__(None, None, None)

        close.assert_awaited_once_with()
        logger.warning.assert_called_once()
        self.assertIsNone(client._http)

    async def test_invalid_or_empty_tokens_are_normalized_to_none(self) -> None:
        for token in ("", None, 123):
            with self.subTest(token=token):
                client = FlareSolverrClient("http://127.0.0.1:8191/v1")
                client._http = _HTTP(_solution(token))  # type: ignore[assignment]
                _own_session(client, "session")

                result = await client.get(LOGIN_URL, session_id="session")

                self.assertIsNone(result.turnstile_token)

    async def test_empty_user_agent_is_rejected(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(  # type: ignore[assignment]
            _solution(user_agent=""),
        )
        _own_session(client, "session")

        with self.assertRaisesRegex(
            FlareSolverrProtocolError,
            "no browser user agent",
        ):
            await client.get(LOGIN_URL, session_id="session")

    async def test_http_502_is_a_sanitized_typed_request_error(self) -> None:
        secret = "PRIVATE-UPSTREAM-RESPONSE-BODY"
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(_http_502(secret))  # type: ignore[assignment]
        _own_session(client, "session")

        with self.assertRaises(FlareSolverrRequestError) as raised:
            await client.get(LOGIN_URL, session_id="session")

        rendered = f"{raised.exception!r} {raised.exception}"
        self.assertIn("502", rendered)
        self.assertNotIn(secret, rendered)
        self.assertEqual(raised.exception.kind, "http_status")
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIsNone(raised.exception.transport_type)

    async def test_service_rejection_is_typed_and_does_not_expose_message(
        self,
    ) -> None:
        secret = "PRIVATE-SERVICE-MESSAGE"
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(  # type: ignore[assignment]
            {"status": "error", "message": secret}
        )
        _own_session(client, "session")

        with self.assertRaises(FlareSolverrRequestError) as raised:
            await client.get(LOGIN_URL, session_id="session")

        rendered = f"{raised.exception!r} {raised.exception}"
        self.assertEqual(raised.exception.kind, "service_status")
        self.assertIsNone(raised.exception.status_code)
        self.assertIsNone(raised.exception.transport_type)
        self.assertNotIn(secret, rendered)

    async def test_ambiguous_direct_get_is_terminal_until_destroy_ack(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _http_read_timeout("request may still be running"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]
        _own_session(client, "owned-session")

        with self.assertRaises(FlareSolverrRequestError):
            await client.get(LOGIN_URL, session_id="owned-session")

        commands_after_timeout = list(_commands(http))
        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.get(LOGIN_URL, session_id="owned-session")
        self.assertEqual(_commands(http), commands_after_timeout)

        await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), ["request.get", "sessions.destroy"])
        self.assertTrue(http.closed)

    async def test_cancelled_direct_get_is_terminal_and_never_replayed(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _BlockingRequestHTTP(_destroyed())
        client._http = http  # type: ignore[assignment]
        _own_session(client, "owned-session")

        request_task = asyncio.create_task(
            client.get(LOGIN_URL, session_id="owned-session")
        )
        await http.request_started.wait()
        request_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request_task

        commands_after_cancel = list(_commands(http))
        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.get(LOGIN_URL, session_id="owned-session")
        self.assertEqual(_commands(http), commands_after_cancel)

        await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), ["request.get", "sessions.destroy"])
        self.assertTrue(http.closed)

    async def test_persistent_session_is_reused_for_turnstile(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(_solution(), _solution(), _solution("token"))
        client._http = http  # type: ignore[assignment]
        _own_session(client, "same-session")
        session = FlareSolverrSession(client, "same-session")

        await session.get("https://forums.e-hentai.org/")
        result = await session.solve_turnstile(LOGIN_URL, tabs=15)

        self.assertEqual(result.turnstile_token, "token")
        self.assertEqual(
            [post["json"]["session"] for post in http.posts],
            ["same-session", "same-session", "same-session"],
        )
        self.assertNotIn("tabs_till_verify", http.posts[0]["json"])
        self.assertNotIn("tabs_till_verify", http.posts[1]["json"])
        self.assertEqual(http.posts[2]["json"]["tabs_till_verify"], 15)

    async def test_missing_turnstile_token_is_retried_once(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(_solution(), _solution(), _solution("token"))
        client._http = http  # type: ignore[assignment]
        _own_session(client, "same-session")
        session = FlareSolverrSession(client, "same-session")

        result = await session.solve_turnstile(LOGIN_URL, tabs=15)

        self.assertEqual(result.turnstile_token, "token")
        self.assertEqual(len(http.posts), 3)
        self.assertNotIn("tabs_till_verify", http.posts[0]["json"])
        self.assertEqual(
            [post["json"]["tabs_till_verify"] for post in http.posts[1:]],
            [15, 15],
        )

    async def test_persistent_session_rejects_user_agent_changes(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(  # type: ignore[assignment]
            _solution(user_agent="first-agent"),
            _solution(user_agent="different-agent"),
        )
        _own_session(client, "same-session")
        session = FlareSolverrSession(client, "same-session")

        await session.get("https://forums.e-hentai.org/")
        with self.assertRaisesRegex(RuntimeError, "changed user agent"):
            await session.solve_turnstile(LOGIN_URL, tabs=15)

    async def test_session_is_destroyed_when_body_raises(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("temporary-session"),
            _solution(),
            {"status": "ok", "message": ""},
        )
        client._http = http  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            async with client.session(max_attempts=1) as scope:
                await scope.solve_managed(LOGIN_URL)
                raise RuntimeError("body failed")

        self.assertEqual(http.posts[-1]["json"]["cmd"], "sessions.destroy")
        self.assertEqual(_destroy_sessions(http), ["temporary-session"])

    async def test_unanchored_request_failure_replaces_session_in_order(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_failure("first session failed before anchoring"),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=2) as scope:
            self.assertIsInstance(scope, FlareSolverrSessionScope)
            receipt = await scope.solve_managed(LOGIN_URL)

        self.assertIsInstance(receipt, FlareSolverrSolveReceipt)
        self.assertEqual(receipt.result.user_agent, "test-agent")
        self.assertEqual(
            _commands(http),
            [
                "sessions.create",
                "request.get",
                "sessions.destroy",
                "sessions.create",
                "request.get",
                "sessions.destroy",
            ],
        )
        self.assertEqual(_request_sessions(http), ["session-1", "session-2"])

    async def test_replacement_request_uses_only_overall_deadline_remaining(
        self,
    ) -> None:
        clock = _Clock()
        solve_deadline = _ClockDeadline(clock, expires_at=65.0)
        cleanup_deadline = _ClockDeadline(clock, expires_at=55.0)
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _SlowFirstRequestHTTP(
            clock,
            _created("session-1"),
            _http_failure("slow first attempt"),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with patch(
            "hbrowser.gallery.browser.flaresolverr.Deadline.after",
            side_effect=[solve_deadline, cleanup_deadline],
        ):
            async with client.session(max_attempts=2) as scope:
                await scope.solve_managed(LOGIN_URL, timeout_ms=60_000)

        requests = [post for post in http.posts if post["json"]["cmd"] == "request.get"]
        self.assertEqual(len(requests), 2)
        self.assertAlmostEqual(requests[0]["timeout"].read, 65.0)
        self.assertAlmostEqual(requests[1]["timeout"].read, 15.0)
        self.assertEqual(requests[0]["json"]["maxTimeout"], 60_000)
        self.assertEqual(requests[1]["json"]["maxTimeout"], 10_000)

    async def test_replacement_session_is_used_for_later_requests(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_failure("replace me"),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _solution("later-token"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=2) as scope:
            first = await scope.solve_managed("https://forums.e-hentai.org/")
            later = await scope.solve_managed(LOGIN_URL)

        self.assertIsInstance(first, FlareSolverrSolveReceipt)
        self.assertEqual(later.result.turnstile_token, "later-token")
        self.assertEqual(first.session_generation, later.session_generation)
        self.assertEqual(
            _request_sessions(http),
            ["session-1", "session-2", "session-2"],
        )
        self.assertEqual(_commands(http).count("sessions.create"), 2)

    async def test_anchored_request_failure_does_not_replace_session(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("anchored-session"),
            _solution(),
            _http_failure("failure after identity anchor"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            receipt = await scope.solve_managed("https://forums.e-hentai.org/")
            scope.mark_identity_applied(receipt)
            with self.assertRaises(FlareSolverrSessionUnavailable) as raised:
                await scope.solve_managed(LOGIN_URL)
            commands_after_failure = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_failure)

        self.assertIsInstance(raised.exception.__cause__, FlareSolverrRequestError)
        self.assertEqual(_commands(http).count("sessions.create"), 1)
        self.assertEqual(
            _request_sessions(http),
            ["anchored-session", "anchored-session"],
        )
        self.assertEqual(
            _commands(http),
            [
                "sessions.create",
                "request.get",
                "request.get",
                "sessions.destroy",
            ],
        )

    async def test_protocol_error_does_not_replace_unanchored_session(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-with-invalid-response"),
            {"status": "ok", "solution": None},
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            with self.assertRaises(FlareSolverrProtocolError):
                await scope.solve_managed(LOGIN_URL)
            commands_after_failure = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_failure)

        self.assertEqual(_commands(http).count("sessions.create"), 1)
        self.assertEqual(
            _commands(http),
            ["sessions.create", "request.get", "sessions.destroy"],
        )

    async def test_session_attempt_budget_is_exact_and_every_session_is_cleaned(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_failure("attempt 1"),
            _destroyed(),
            _created("session-2"),
            _http_failure("attempt 2"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=2) as scope:
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            commands_after_failure = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_failure)

        self.assertEqual(
            _commands(http),
            [
                "sessions.create",
                "request.get",
                "sessions.destroy",
                "sessions.create",
                "request.get",
                "sessions.destroy",
            ],
        )
        self.assertEqual(_request_sessions(http), ["session-1", "session-2"])

    async def test_active_replacement_is_cleaned_when_scope_body_raises(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_failure("replace before body failure"),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            async with client.session(max_attempts=2) as scope:
                await scope.solve_managed(LOGIN_URL)
                raise RuntimeError("body failed")

        self.assertEqual(
            _destroy_sessions(http),
            ["session-1", "session-2"],
        )

    async def test_only_current_generation_receipt_can_anchor_identity(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _solution(user_agent="first-agent"),
            _http_failure("successful solve was not committed"),
            _destroyed(),
            _created("session-2"),
            _solution(user_agent="second-agent"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=2) as scope:
            stale = await scope.solve_managed("https://forums.e-hentai.org/")
            current = await scope.solve_managed(LOGIN_URL)

            self.assertNotEqual(
                stale.session_generation,
                current.session_generation,
            )
            with self.assertRaises(ValueError):
                scope.mark_identity_applied(stale)
            scope.mark_identity_applied(current)

        self.assertEqual(
            _request_sessions(http), ["session-1", "session-1", "session-2"]
        )
        self.assertEqual(
            _commands(http).count("sessions.create"),
            2,
        )

    async def test_session_creation_request_failure_is_retried_within_budget(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _http_failure("first create failed"),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with patch(
            "hbrowser.gallery.browser.flaresolverr.secrets.token_hex",
            side_effect=["failed-session", "replacement-session"],
        ):
            async with client.session(max_attempts=2) as scope:
                receipt = await scope.solve_managed(LOGIN_URL)

        self.assertEqual(receipt.session_generation, 1)
        self.assertEqual(
            _commands(http),
            [
                "sessions.create",
                "sessions.create",
                "request.get",
                "sessions.destroy",
            ],
        )
        self.assertEqual(_request_sessions(http), ["session-2"])
        self.assertEqual(
            http.create_session_ids,
            ["failed-session", "replacement-session"],
        )

    async def test_ambiguous_create_adopts_positive_inventory_without_replay(
        self,
    ) -> None:
        clock = _Clock()
        parent_deadline = _ClockDeadline(clock, expires_at=65.0)
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _http_read_timeout("create response was lost"),
            _listed_latest(present=True),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with patch(
            "hbrowser.gallery.browser.flaresolverr.secrets.token_hex",
            return_value="owned-session",
        ):
            session = await client.create_session(
                deadline=parent_deadline,  # type: ignore[arg-type]
            )

        self.assertEqual(session._session_id, "owned-session")
        self.assertEqual(_commands(http), ["sessions.create", "sessions.list"])
        self.assertEqual(http.create_session_ids, ["owned-session"])
        create_timeout = http.posts[0]["timeout"]
        inventory_timeout = http.posts[1]["timeout"]
        self.assertIsInstance(create_timeout, httpx.Timeout)
        self.assertIsInstance(inventory_timeout, httpx.Timeout)
        self.assertEqual(create_timeout.read, 10.0)
        self.assertEqual(inventory_timeout.read, 5.0)
        self.assertLessEqual(create_timeout.connect, 5.0)
        self.assertLessEqual(create_timeout.write, 5.0)
        self.assertLessEqual(create_timeout.pool, 5.0)

        await client.destroy_session(
            session,
            deadline=parent_deadline,  # type: ignore[arg-type]
        )
        destroy_timeout = http.posts[-1]["timeout"]
        self.assertIsInstance(destroy_timeout, httpx.Timeout)
        self.assertEqual(destroy_timeout.read, 15.0)

    async def test_create_and_late_inventory_share_one_fifteen_second_deadline(
        self,
    ) -> None:
        clock = _Clock()
        parent_deadline = _ClockDeadline(clock, expires_at=65.0)
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _ClockAdvancingHTTP(
            clock,
            {"sessions.create": 10.0, "sessions.list": 5.0},
            _http_read_timeout("create may still be running"),
            _listed_latest(present=True),
        )
        client._http = http  # type: ignore[assignment]

        with (
            patch(
                "hbrowser.gallery.browser.flaresolverr.secrets.token_hex",
                return_value="late-inventory-session",
            ),
            self.assertRaises(FlareSolverrSessionOwnershipError),
        ):
            await client.create_session(
                deadline=parent_deadline,  # type: ignore[arg-type]
            )

        self.assertEqual(clock.now, 15.0)
        self.assertEqual(_commands(http), ["sessions.create", "sessions.list"])
        self.assertEqual(
            client._owned_session_ids,
            {"late-inventory-session"},
        )
        self.assertEqual(
            client._uncertain_create_ids,
            {"late-inventory-session"},
        )

    async def test_ambiguous_create_absence_is_terminal_and_never_replayed(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _http_read_timeout("create may still be running"),
            _listed_latest(present=False),
            _listed_latest(present=False),
        )
        client._http = http  # type: ignore[assignment]
        scope = FlareSolverrSessionScope(client, max_attempts=3)

        with (
            patch(
                "hbrowser.gallery.browser.flaresolverr.secrets.token_hex",
                return_value="pending-session",
            ),
            self.assertRaises(FlareSolverrSessionUnavailable) as raised,
        ):
            await scope.solve_managed(LOGIN_URL)

        self.assertIsInstance(
            raised.exception.__cause__,
            FlareSolverrSessionOwnershipError,
        )
        self.assertEqual(client._uncertain_create_ids, {"pending-session"})
        commands_after_failure = list(_commands(http))
        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.create_session()
        self.assertEqual(_commands(http), commands_after_failure)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await scope.close()

        self.assertEqual(
            _commands(http),
            ["sessions.create", "sessions.list", "sessions.list"],
        )
        self.assertEqual(http.create_session_ids, ["pending-session"])
        self.assertNotIn("sessions.destroy", _commands(http))

    async def test_mismatched_create_identity_is_not_claimed_as_owned(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            {"status": "ok", "session": "unexpected-session"},
            _listed_latest(present=False),
            _listed_latest(present=False),
        )
        client._http = http  # type: ignore[assignment]

        with (
            patch(
                "hbrowser.gallery.browser.flaresolverr.secrets.token_hex",
                return_value="requested-session",
            ),
            self.assertRaises(FlareSolverrSessionOwnershipError),
        ):
            await client.create_session()

        self.assertEqual(
            client._owned_session_ids,
            {"requested-session"},
        )
        self.assertEqual(client._uncertain_create_ids, {"requested-session"})

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)

        self.assertEqual(
            _commands(http),
            [
                "sessions.create",
                "sessions.list",
                "sessions.list",
            ],
        )
        self.assertEqual(client._owned_session_ids, {"requested-session"})
        self.assertNotIn("unexpected-session", client._owned_session_ids)
        self.assertNotIn("sessions.destroy", _commands(http))
        self.assertTrue(http.closed)
        commands_after_exit = list(_commands(http))
        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), commands_after_exit)

    async def test_ambiguous_request_destroy_does_not_replace_session(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_read_timeout("request outcome is unknown"),
            _http_read_timeout("destroy outcome is unknown"),
        )
        client._http = http  # type: ignore[assignment]
        scope = FlareSolverrSessionScope(client, max_attempts=3)

        with self.assertRaises(FlareSolverrSessionUnavailable) as raised:
            await scope.solve_managed(LOGIN_URL)

        self.assertIsInstance(
            raised.exception.__cause__,
            FlareSolverrSessionOwnershipError,
        )
        self.assertEqual(
            _commands(http),
            ["sessions.create", "request.get", "sessions.destroy"],
        )
        self.assertEqual(_commands(http).count("sessions.create"), 1)
        self.assertEqual(len(client._uncertain_destroy_ids), 1)
        commands_after_failure = list(_commands(http))
        with self.assertRaises(FlareSolverrSessionUnavailable):
            await scope.solve_managed(LOGIN_URL)
        self.assertEqual(_commands(http), commands_after_failure)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await scope.close()
        self.assertEqual(_commands(http), commands_after_failure)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), commands_after_failure)
        self.assertTrue(http.closed)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), commands_after_failure)

    async def test_cancelled_destroy_is_tombstoned_and_never_replayed(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _BlockingDestroyHTTP(
            _created("cancelled-destroy-session"),
            _solution(),
        )
        client._http = http  # type: ignore[assignment]
        scope = FlareSolverrSessionScope(client, max_attempts=1)
        await scope.solve_managed(LOGIN_URL)

        close_task = asyncio.create_task(scope.close())
        await http.destroy_started.wait()
        close_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_task

        commands_after_cancel = list(_commands(http))
        self.assertEqual(
            commands_after_cancel,
            ["sessions.create", "request.get", "sessions.destroy"],
        )
        self.assertEqual(len(client._uncertain_destroy_ids), 1)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await scope.close()
        self.assertEqual(_commands(http), commands_after_cancel)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), commands_after_cancel)
        self.assertTrue(http.closed)

        with self.assertRaises(FlareSolverrSessionOwnershipError):
            await client.__aexit__(None, None, None)
        self.assertEqual(_commands(http), commands_after_cancel)

    async def test_retry_log_names_solver_session_without_sensitive_detail(
        self,
    ) -> None:
        secret = "PRIVATE-TRANSPORT-DETAIL"
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_failure(secret),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with self.assertLogs(
            "hbrowser.gallery.browser.flaresolverr",
            level="WARNING",
        ) as captured:
            async with client.session(max_attempts=2) as scope:
                await scope.solve_managed(LOGIN_URL)

        rendered = "\n".join(captured.output)
        self.assertIn("unanchored solver session", rendered)
        self.assertIn("retry=yes", rendered)
        self.assertIn("failure_kind=transport", rendered)
        self.assertIn("transport_type=ConnectError", rendered)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("proxy", rendered.casefold())
        self.assertNotIn("route rotation", rendered.casefold())

    async def test_http_retry_log_keeps_status_but_not_response_detail(self) -> None:
        secret = "PRIVATE-UPSTREAM-RESPONSE-BODY"
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("session-1"),
            _http_502(secret),
            _destroyed(),
            _created("session-2"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with self.assertLogs(
            "hbrowser.gallery.browser.flaresolverr",
            level="WARNING",
        ) as captured:
            async with client.session(max_attempts=2) as scope:
                await scope.solve_managed(LOGIN_URL)

        rendered = "\n".join(captured.output)
        self.assertIn("failure_kind=http_status", rendered)
        self.assertIn("status_code=502", rendered)
        self.assertNotIn(secret, rendered)

    async def test_session_creation_protocol_error_is_terminal(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            {"status": "ok", "session": None},
            _listed_latest(present=False),
            _listed_latest(present=True),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            with self.assertRaises(FlareSolverrSessionUnavailable) as raised:
                await scope.solve_managed(LOGIN_URL)
            self.assertIsInstance(
                raised.exception.__cause__,
                FlareSolverrSessionOwnershipError,
            )
            self.assertEqual(len(client._uncertain_create_ids), 1)
            commands_after_failure = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_failure)

        self.assertEqual(
            _commands(http),
            ["sessions.create", "sessions.list", "sessions.list", "sessions.destroy"],
        )
        self.assertEqual(len(http.create_session_ids), 1)

    async def test_managed_and_turnstile_solves_share_anchored_generation(
        self,
    ) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("shared-session"),
            _solution(),
            _solution(),
            _solution("turnstile-token"),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            managed = await scope.solve_managed("https://forums.e-hentai.org/")
            scope.mark_identity_applied(managed)
            turnstile = await scope.solve_turnstile(LOGIN_URL, tabs=15)

        self.assertEqual(
            managed.session_generation,
            turnstile.session_generation,
        )
        self.assertEqual(turnstile.result.turnstile_token, "turnstile-token")
        self.assertEqual(
            _request_sessions(http),
            ["shared-session", "shared-session", "shared-session"],
        )

    async def test_scope_cleanup_runs_when_body_is_cancelled(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("cancelled-session"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        with self.assertRaises(asyncio.CancelledError):
            async with client.session(max_attempts=1) as scope:
                await scope.solve_managed(LOGIN_URL)
                raise asyncio.CancelledError

        self.assertEqual(
            _commands(http),
            ["sessions.create", "request.get", "sessions.destroy"],
        )

    async def test_retired_scope_destroys_session_and_never_sends_again(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            _created("retired-session"),
            _solution(),
            _destroyed(),
        )
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            receipt = await scope.solve_managed(LOGIN_URL)
            scope.mark_identity_applied(receipt)
            await scope.retire()
            commands_after_retire = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_retire)

        self.assertEqual(
            _commands(http),
            ["sessions.create", "request.get", "sessions.destroy"],
        )
