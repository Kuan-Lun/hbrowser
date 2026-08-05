import asyncio
import unittest
from collections import deque
from typing import Any

import httpx

from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrClient,
    FlareSolverrConfigurationError,
    FlareSolverrProtocolError,
    FlareSolverrRequestError,
    FlareSolverrResult,
    FlareSolverrSession,
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


class _HTTP:
    def __init__(
        self,
        *responses: dict[str, Any] | _Response | httpx.HTTPError,
    ) -> None:
        self.responses = deque(responses)
        self.posts: list[dict[str, Any]] = []

    async def post(
        self,
        endpoint: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> _Response:
        self.posts.append(
            {
                "endpoint": endpoint,
                "json": json,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        response = self.responses.popleft()
        if isinstance(response, httpx.HTTPError):
            raise response
        if isinstance(response, _Response):
            return response
        return _Response(response)


def _http_failure(label: str) -> httpx.ConnectError:
    request = httpx.Request("POST", "http://127.0.0.1:8191/v1")
    return httpx.ConnectError(f"private transport detail: {label}", request=request)


def _http_502(secret: str) -> _Response:
    request = httpx.Request("POST", "http://127.0.0.1:8191/v1")
    response = httpx.Response(502, request=request)
    error = httpx.HTTPStatusError(
        f"502 response contained {secret}",
        request=request,
        response=response,
    )
    return _Response({}, status_error=error)


def _created(session_id: str) -> dict[str, Any]:
    return {"status": "ok", "session": session_id}


def _destroyed() -> dict[str, Any]:
    return {"status": "ok", "message": ""}


def _commands(http: _HTTP) -> list[str]:
    return [post["json"]["cmd"] for post in http.posts]


def _request_sessions(http: _HTTP) -> list[str]:
    return [
        post["json"]["session"]
        for post in http.posts
        if post["json"]["cmd"] == "request.get"
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


class FlareSolverrTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_invalid_or_empty_tokens_are_normalized_to_none(self) -> None:
        for token in ("", None, 123):
            with self.subTest(token=token):
                client = FlareSolverrClient("http://127.0.0.1:8191/v1")
                client._http = _HTTP(_solution(token))  # type: ignore[assignment]

                result = await client.get(LOGIN_URL, session_id="session")

                self.assertIsNone(result.turnstile_token)

    async def test_empty_user_agent_is_rejected(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(  # type: ignore[assignment]
            _solution(user_agent=""),
        )

        with self.assertRaisesRegex(
            FlareSolverrProtocolError,
            "no browser user agent",
        ):
            await client.get(LOGIN_URL, session_id="session")

    async def test_http_502_is_a_sanitized_typed_request_error(self) -> None:
        secret = "PRIVATE-UPSTREAM-RESPONSE-BODY"
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        client._http = _HTTP(_http_502(secret))  # type: ignore[assignment]

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

        with self.assertRaises(FlareSolverrRequestError) as raised:
            await client.get(LOGIN_URL, session_id="session")

        rendered = f"{raised.exception!r} {raised.exception}"
        self.assertEqual(raised.exception.kind, "service_status")
        self.assertIsNone(raised.exception.status_code)
        self.assertIsNone(raised.exception.transport_type)
        self.assertNotIn(secret, rendered)

    async def test_persistent_session_is_reused_for_turnstile(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(_solution(), _solution(), _solution("token"))
        client._http = http  # type: ignore[assignment]
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
        session = FlareSolverrSession(client, "same-session")

        await session.get("https://forums.e-hentai.org/")
        with self.assertRaisesRegex(RuntimeError, "changed user agent"):
            await session.solve_turnstile(LOGIN_URL, tabs=15)

    async def test_session_is_destroyed_when_body_raises(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(
            {"status": "ok", "session": "temporary-session"},
            {"status": "ok", "message": ""},
        )
        client._http = http  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            async with client.session():
                raise RuntimeError("body failed")

        self.assertEqual(http.posts[-1]["json"]["cmd"], "sessions.destroy")
        self.assertEqual(
            http.posts[-1]["json"]["session"],
            "temporary-session",
        )

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
            [
                post["json"]["session"]
                for post in http.posts
                if post["json"]["cmd"] == "sessions.destroy"
            ],
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
        http = _HTTP({"status": "ok", "session": None})
        client._http = http  # type: ignore[assignment]

        async with client.session(max_attempts=3) as scope:
            with self.assertRaises(FlareSolverrProtocolError):
                await scope.solve_managed(LOGIN_URL)
            commands_after_failure = list(_commands(http))
            with self.assertRaises(FlareSolverrSessionUnavailable):
                await scope.solve_managed(LOGIN_URL)
            self.assertEqual(_commands(http), commands_after_failure)

        self.assertEqual(_commands(http), ["sessions.create"])

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
