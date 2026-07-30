"""FlareSolverr client and persistent browser-session integration."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self

import httpx
from zendriver import cdp

from ..utils import setup_logger
from .proxy import has_residential_proxy
from .tor import should_use_tor

logger = setup_logger(__name__)

_SAME_SITE_MAP = {
    "Strict": cdp.network.CookieSameSite.STRICT,
    "Lax": cdp.network.CookieSameSite.LAX,
    "None": cdp.network.CookieSameSite.NONE,
}
_CLOUDFLARE_COOKIE_PREFIXES = ("cf_", "__cf", "_cf")


@dataclass(frozen=True)
class FlareSolverrResult:
    """The reusable parts of a successful FlareSolverr response."""

    cookies: list[dict[str, Any]] = field(repr=False)
    user_agent: str
    turnstile_token: str | None = field(default=None, repr=False)

    def to_cdp_cloudflare_cookie_params(self) -> list[cdp.network.CookieParam]:
        """Convert only Cloudflare cookies into CDP cookie parameters.

        A FlareSolverr browser has its own guest Forums session. Importing all
        of its cookies would overwrite the authenticated browser's application
        session, especially if another managed challenge appears after login.
        """
        return [
            self._cookie_to_cdp_param(cookie)
            for cookie in self.cookies
            if self._is_cloudflare_cookie(cookie)
        ]

    @staticmethod
    def _is_cloudflare_cookie(cookie: dict[str, Any]) -> bool:
        name = cookie.get("name")
        return isinstance(name, str) and name.startswith(_CLOUDFLARE_COOKIE_PREFIXES)

    @staticmethod
    def _cookie_to_cdp_param(cookie: dict[str, Any]) -> cdp.network.CookieParam:
        expiry = cookie.get("expiry", cookie.get("expires"))
        return cdp.network.CookieParam(
            name=cookie["name"],
            value=cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path"),
            secure=cookie.get("secure"),
            http_only=cookie.get("httpOnly"),
            same_site=_SAME_SITE_MAP.get(cookie.get("sameSite", "")),
            expires=(
                cdp.network.TimeSinceEpoch(expiry) if expiry and expiry > 0 else None
            ),
        )


class FlareSolverrSession:
    """One persistent FlareSolverr browser.

    Keeping the same browser between the managed challenge and the embedded
    login Turnstile is important: the latter must be loaded with the clearance
    cookies and browser identity established by the former.
    """

    def __init__(self, client: FlareSolverrClient, session_id: str) -> None:
        self._client = client
        self._session_id = session_id
        self._has_navigated = False
        self._user_agent: str | None = None

    async def get(
        self,
        url: str,
        *,
        timeout_ms: int = 60_000,
        turnstile_tabs: int | None = None,
    ) -> FlareSolverrResult:
        result = await self._client.get(
            url,
            session_id=self._session_id,
            timeout_ms=timeout_ms,
            turnstile_tabs=turnstile_tabs,
        )
        if self._user_agent is None:
            self._user_agent = result.user_agent
        elif result.user_agent != self._user_agent:
            raise RuntimeError(
                "FlareSolverr changed user agent within a persistent session"
            )
        self._has_navigated = True
        return result

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> FlareSolverrResult:
        """Solve an embedded Turnstile in the current persistent browser."""
        if not self._has_navigated:
            # The first navigation establishes Cloudflare clearance in this
            # FlareSolverr browser. A second navigation is then able to target
            # the embedded widget rather than the top-level managed challenge.
            await self.get(url, timeout_ms=timeout_ms)

        return await self.get(
            url,
            timeout_ms=timeout_ms,
            turnstile_tabs=tabs,
        )


class FlareSolverrClient:
    """Small async client for the subset of FlareSolverr used by HBrowser."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._http = httpx.AsyncClient()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FlareSolverrSession]:
        """Create and always destroy one persistent FlareSolverr browser."""
        session = await self.create_session()
        try:
            yield session
        finally:
            await self.destroy_session(session)

    async def create_session(self) -> FlareSolverrSession:
        """Start one persistent FlareSolverr browser."""
        data = await self._post({"cmd": "sessions.create"}, timeout_seconds=15)
        session_id = data.get("session")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("FlareSolverr did not return a session ID")
        return FlareSolverrSession(self, session_id)

    async def destroy_session(self, session: FlareSolverrSession) -> None:
        """Best-effort cleanup for a persistent FlareSolverr browser."""
        try:
            await self._post(
                {
                    "cmd": "sessions.destroy",
                    "session": session._session_id,
                },
                timeout_seconds=15,
            )
        except Exception as error:
            logger.warning(
                "Failed to destroy FlareSolverr session (%s)",
                type(error).__name__,
            )

    async def get(
        self,
        url: str,
        *,
        session_id: str,
        timeout_ms: int = 60_000,
        turnstile_tabs: int | None = None,
    ) -> FlareSolverrResult:
        payload: dict[str, Any] = {
            "cmd": "request.get",
            "url": url,
            "session": session_id,
            "maxTimeout": timeout_ms,
            "returnOnlyCookies": True,
        }
        if turnstile_tabs is not None:
            if turnstile_tabs < 1:
                raise ValueError("turnstile_tabs must be at least 1")
            payload["tabs_till_verify"] = turnstile_tabs

        data = await self._post(
            payload,
            timeout_seconds=timeout_ms / 1000 + 10,
        )
        solution = data.get("solution")
        if not isinstance(solution, dict):
            raise RuntimeError("FlareSolverr response did not contain a solution")

        cookies = solution.get("cookies", [])
        if not isinstance(cookies, list) or not all(
            isinstance(cookie, dict)
            and isinstance(cookie.get("name"), str)
            and isinstance(cookie.get("value"), str)
            for cookie in cookies
        ):
            raise RuntimeError("FlareSolverr returned invalid cookies")

        user_agent = solution.get("userAgent", "")
        if not isinstance(user_agent, str) or not user_agent:
            raise RuntimeError("FlareSolverr returned no browser user agent")

        token = solution.get("turnstile_token")
        return FlareSolverrResult(
            cookies=cookies,
            user_agent=user_agent,
            turnstile_token=token if isinstance(token, str) and token else None,
        )

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("FlareSolverrClient must be used as an async context")

        response = await self._http.post(
            self.endpoint,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("FlareSolverr returned a non-object response")
        if data.get("status") != "ok":
            message = data.get("message") or "unknown error"
            raise RuntimeError(f"FlareSolverr failed: {message}")
        return data


def get_flaresolverr_url() -> str | None:
    """Read and normalize the optional FlareSolverr `/v1` endpoint."""
    url = os.getenv("FLARESOLVERR_URL")
    if not url:
        return None
    return url.strip().strip("\"'").rstrip("/")


def should_use_flaresolverr() -> bool:
    """Return whether FlareSolverr can share the driver's public IP address."""
    url = get_flaresolverr_url()
    if not url:
        return False
    if should_use_tor():
        logger.warning(
            "FLARESOLVERR_URL is set but Tor is enabled; ignoring FlareSolverr "
            "because the clearance cookie would not match the Tor exit IP."
        )
        return False
    if has_residential_proxy():
        logger.warning(
            "FLARESOLVERR_URL is set but a residential proxy is enabled; "
            "ignoring FlareSolverr because it is not configured with the same "
            "sticky proxy."
        )
        return False
    return True
