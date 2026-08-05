"""FlareSolverr client and persistent browser-session integration."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Self, overload

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
_TURNSTILE_DISCOVERY_ATTEMPTS = 2


class FlareSolverrError(RuntimeError):
    """Base class for sanitized FlareSolverr failures."""


FlareSolverrRequestKind = Literal["http_status", "transport", "service_status"]


class FlareSolverrRequestError(FlareSolverrError):
    """A categorized request failure whose public fields are safe to log."""

    def __init__(
        self,
        *,
        kind: FlareSolverrRequestKind,
        status_code: int | None = None,
        transport_type: str | None = None,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        self.transport_type = transport_type

        if kind == "http_status":
            message = f"FlareSolverr request failed with HTTP status {status_code}"
        elif kind == "transport":
            message = "FlareSolverr transport request failed"
        else:
            message = "FlareSolverr service rejected the request"
        super().__init__(message)


class FlareSolverrConfigurationError(ValueError):
    """The configured endpoint cannot be used as an HTTP service URL."""


class FlareSolverrProtocolError(FlareSolverrError):
    """A response that cannot be trusted as valid FlareSolverr data."""


class FlareSolverrSessionUnavailable(FlareSolverrError):
    """A scoped solve cannot safely obtain another FlareSolverr session."""


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


@dataclass(frozen=True)
class FlareSolverrSolveReceipt:
    """A solve result bound to one scoped FlareSolverr session generation."""

    result: FlareSolverrResult = field(repr=False)
    session_generation: int


class FlareSolverrSession:
    """One persistent FlareSolverr browser.

    Keeping the same browser between the managed challenge and the embedded
    login Turnstile is important: the latter must be loaded with the clearance
    cookies and browser identity established by the former.
    """

    def __init__(self, client: FlareSolverrClient, session_id: str) -> None:
        self._client = client
        self._session_id = session_id
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
            raise FlareSolverrProtocolError(
                "FlareSolverr changed user agent within a persistent session"
            )
        return result

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> FlareSolverrResult:
        """Solve an embedded Turnstile in the current persistent browser."""
        # Warm the exact login page before asking FlareSolverr to inspect it.
        # Its Turnstile implementation checks for the hidden response field
        # immediately after navigation, so a cold, asynchronously rendered
        # widget can otherwise be reported as absent.
        await self.get(url, timeout_ms=timeout_ms)

        result: FlareSolverrResult | None = None
        for _ in range(_TURNSTILE_DISCOVERY_ATTEMPTS):
            result = await self.get(
                url,
                timeout_ms=timeout_ms,
                turnstile_tabs=tabs,
            )
            if result.turnstile_token:
                return result

        assert result is not None
        return result


class FlareSolverrSessionScope:
    """Bounded, replaceable FlareSolverr session ownership for one login.

    A request failure may replace a session only until its identity has been
    applied to the caller's browser. Once anchored, changing the session would
    mix browser identities, so later request failures become terminal for this
    scope. Protocol failures are always terminal because retrying malformed
    data cannot establish a trustworthy identity.
    """

    def __init__(self, client: FlareSolverrClient, *, max_attempts: int) -> None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self._client = client
        self._max_attempts = max_attempts
        self._attempts_used = 0
        self._session: FlareSolverrSession | None = None
        self._session_generation = 0
        self._identity_generation: int | None = None
        self._closed = False
        self._unavailable = False

    async def solve_managed(
        self,
        url: str,
        *,
        timeout_ms: int = 60_000,
    ) -> FlareSolverrSolveReceipt:
        """Solve a managed challenge with bounded unanchored replacement."""
        return await self._solve(
            lambda session: session.get(url, timeout_ms=timeout_ms)
        )

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> FlareSolverrSolveReceipt:
        """Solve Turnstile while preserving the active session identity."""
        return await self._solve(
            lambda session: session.solve_turnstile(
                url,
                tabs=tabs,
                timeout_ms=timeout_ms,
            )
        )

    def mark_identity_applied(self, receipt: FlareSolverrSolveReceipt) -> None:
        """Anchor only a receipt issued for the current session generation."""
        if self._closed:
            raise ValueError("FlareSolverr session scope is closed")
        if self._session is None or (
            receipt.session_generation != self._session_generation
        ):
            raise ValueError(
                "FlareSolverr solve receipt is not from the current session generation"
            )
        self._identity_generation = self._session_generation

    async def close(self) -> None:
        """Best-effort cleanup of the currently owned session."""
        if self._closed:
            return
        self._closed = True
        await self._discard_current_session()

    async def retire(self) -> None:
        """Make this scope terminal and discard its current solver session."""
        if self._closed or self._unavailable:
            return
        await self._quarantine()

    async def _solve(
        self,
        operation: Callable[[FlareSolverrSession], Awaitable[FlareSolverrResult]],
    ) -> FlareSolverrSolveReceipt:
        if self._closed:
            raise FlareSolverrSessionUnavailable("FlareSolverr session scope is closed")
        if self._unavailable:
            raise FlareSolverrSessionUnavailable(
                "FlareSolverr session scope is unavailable"
            )

        while True:
            session = await self._ensure_session()
            generation = self._session_generation
            try:
                result = await operation(session)
            except FlareSolverrProtocolError as error:
                logger.warning(
                    "FlareSolverr protocol validation failed; quarantining "
                    "solver session: session_generation=%s error_type=%s",
                    generation,
                    type(error).__name__,
                )
                await self._quarantine()
                raise
            except FlareSolverrRequestError as error:
                if self._identity_generation == generation:
                    logger.warning(
                        "FlareSolverr request failed after browser identity was "
                        "anchored; quarantining solver session without replacement: "
                        "session_generation=%s failure_kind=%s status_code=%s "
                        "transport_type=%s",
                        generation,
                        error.kind,
                        error.status_code if error.status_code is not None else "none",
                        error.transport_type or "none",
                    )
                    await self._quarantine()
                    raise FlareSolverrSessionUnavailable(
                        "FlareSolverr session failed after its identity was applied"
                    ) from error

                retry = self._attempts_used < self._max_attempts
                logger.warning(
                    "FlareSolverr request failed in unanchored solver session: "
                    "session_attempt=%s/%s retry=%s failure_kind=%s "
                    "status_code=%s transport_type=%s",
                    self._attempts_used,
                    self._max_attempts,
                    "yes" if retry else "no",
                    error.kind,
                    error.status_code if error.status_code is not None else "none",
                    error.transport_type or "none",
                )
                await self._discard_current_session()
                if not retry:
                    self._unavailable = True
                    raise FlareSolverrSessionUnavailable(
                        "FlareSolverr session attempts were exhausted"
                    ) from error
                continue

            return FlareSolverrSolveReceipt(
                result=result,
                session_generation=generation,
            )

    async def _ensure_session(self) -> FlareSolverrSession:
        if self._session is not None:
            return self._session

        while self._attempts_used < self._max_attempts:
            self._attempts_used += 1
            try:
                session = await self._client.create_session()
            except FlareSolverrProtocolError as error:
                logger.warning(
                    "FlareSolverr session creation returned invalid data; "
                    "retry=no error_type=%s",
                    type(error).__name__,
                )
                self._unavailable = True
                raise
            except FlareSolverrRequestError as error:
                retry = self._attempts_used < self._max_attempts
                logger.warning(
                    "FlareSolverr session creation failed: "
                    "session_attempt=%s/%s retry=%s failure_kind=%s "
                    "status_code=%s transport_type=%s",
                    self._attempts_used,
                    self._max_attempts,
                    "yes" if retry else "no",
                    error.kind,
                    error.status_code if error.status_code is not None else "none",
                    error.transport_type or "none",
                )
                if not retry:
                    self._unavailable = True
                    raise FlareSolverrSessionUnavailable(
                        "FlareSolverr session attempts were exhausted"
                    ) from error
                continue

            self._session_generation += 1
            self._session = session
            return session

        self._unavailable = True
        raise FlareSolverrSessionUnavailable(
            "FlareSolverr session attempts were exhausted"
        )

    async def _quarantine(self) -> None:
        self._unavailable = True
        await self._discard_current_session()

    async def _discard_current_session(self) -> None:
        session = self._session
        self._session = None
        self._identity_generation = None
        if session is not None:
            await self._client.destroy_session(session)


class FlareSolverrClient:
    """Small async client for the subset of FlareSolverr used by HBrowser."""

    def __init__(self, endpoint: str) -> None:
        try:
            parsed_endpoint = httpx.URL(endpoint.strip().rstrip("/"))
        except httpx.InvalidURL, TypeError:
            raise FlareSolverrConfigurationError(
                "FlareSolverr endpoint must be a valid HTTP(S) URL"
            ) from None
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.host:
            raise FlareSolverrConfigurationError(
                "FlareSolverr endpoint must be a valid HTTP(S) URL"
            )
        self.endpoint = str(parsed_endpoint)
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

    @overload
    def session(
        self,
        *,
        max_attempts: None = None,
    ) -> AbstractAsyncContextManager[FlareSolverrSession]: ...

    @overload
    def session(
        self,
        *,
        max_attempts: int,
    ) -> AbstractAsyncContextManager[FlareSolverrSessionScope]: ...

    def session(
        self,
        *,
        max_attempts: int | None = None,
    ) -> (
        AbstractAsyncContextManager[FlareSolverrSession]
        | AbstractAsyncContextManager[FlareSolverrSessionScope]
    ):
        """Own either a legacy raw session or a bounded production scope.

        Omitting ``max_attempts`` preserves the original eager raw-session API.
        Passing it opts into a lazy scope that can replace an unanchored session
        after a sanitized request failure.
        """
        if max_attempts is None:
            return self._raw_session()
        return self._session_scope(max_attempts=max_attempts)

    @asynccontextmanager
    async def _raw_session(self) -> AsyncIterator[FlareSolverrSession]:
        raw_session = await self.create_session()
        try:
            yield raw_session
        finally:
            await self.destroy_session(raw_session)

    @asynccontextmanager
    async def _session_scope(
        self,
        *,
        max_attempts: int,
    ) -> AsyncIterator[FlareSolverrSessionScope]:
        scope = FlareSolverrSessionScope(self, max_attempts=max_attempts)
        try:
            yield scope
        finally:
            await scope.close()

    async def create_session(self) -> FlareSolverrSession:
        """Start one persistent FlareSolverr browser."""
        data = await self._post({"cmd": "sessions.create"}, timeout_seconds=15)
        session_id = data.get("session")
        if not isinstance(session_id, str) or not session_id:
            raise FlareSolverrProtocolError(
                "FlareSolverr did not return a valid session ID"
            )
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
            raise FlareSolverrProtocolError(
                "FlareSolverr response did not contain a solution"
            )

        cookies = solution.get("cookies", [])
        if not isinstance(cookies, list) or not all(
            isinstance(cookie, dict)
            and isinstance(cookie.get("name"), str)
            and isinstance(cookie.get("value"), str)
            for cookie in cookies
        ):
            raise FlareSolverrProtocolError("FlareSolverr returned invalid cookies")

        user_agent = solution.get("userAgent", "")
        if not isinstance(user_agent, str) or not user_agent:
            raise FlareSolverrProtocolError(
                "FlareSolverr returned no browser user agent"
            )

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

        try:
            response = await self._http.post(
                self.endpoint,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.InvalidURL:
            raise FlareSolverrConfigurationError(
                "FlareSolverr endpoint must be a valid HTTP(S) URL"
            ) from None
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            raise FlareSolverrRequestError(
                kind="http_status",
                status_code=status_code,
            ) from None
        except httpx.HTTPError as error:
            raise FlareSolverrRequestError(
                kind="transport",
                transport_type=type(error).__name__,
            ) from None

        try:
            data = response.json()
        except TypeError, ValueError:
            raise FlareSolverrProtocolError(
                "FlareSolverr returned invalid JSON"
            ) from None
        if not isinstance(data, dict):
            raise FlareSolverrProtocolError(
                "FlareSolverr returned a non-object response"
            )

        status = data.get("status")
        if not isinstance(status, str):
            raise FlareSolverrProtocolError(
                "FlareSolverr returned an invalid service status"
            )
        if status != "ok":
            raise FlareSolverrRequestError(kind="service_status")
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
