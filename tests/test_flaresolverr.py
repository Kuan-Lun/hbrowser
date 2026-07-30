import unittest
from collections import deque
from typing import Any

from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrClient,
    FlareSolverrResult,
    FlareSolverrSession,
)

LOGIN_URL = "https://forums.e-hentai.org/index.php?act=Login&CODE=00"


class _Response:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


class _HTTP:
    def __init__(self, *responses: dict[str, Any]) -> None:
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
        return _Response(self.responses.popleft())


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

        with self.assertRaisesRegex(RuntimeError, "no browser user agent"):
            await client.get(LOGIN_URL, session_id="session")

    async def test_persistent_session_is_reused_for_turnstile(self) -> None:
        client = FlareSolverrClient("http://127.0.0.1:8191/v1")
        http = _HTTP(_solution(), _solution("token"))
        client._http = http  # type: ignore[assignment]
        session = FlareSolverrSession(client, "same-session")

        await session.get("https://forums.e-hentai.org/")
        result = await session.solve_turnstile(LOGIN_URL, tabs=15)

        self.assertEqual(result.turnstile_token, "token")
        self.assertEqual(
            [post["json"]["session"] for post in http.posts],
            ["same-session", "same-session"],
        )
        self.assertNotIn("tabs_till_verify", http.posts[0]["json"])
        self.assertEqual(http.posts[1]["json"]["tabs_till_verify"], 15)

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
