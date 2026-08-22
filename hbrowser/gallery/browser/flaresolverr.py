"""FlareSolverr client and persistent browser-session integration."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import httpx
from zendriver import cdp

from ..challenge_policy import validate_turnstile_tabs
from ..utils import Deadline, setup_logger
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
_TRANSPORT_PHASE_TIMEOUT_SECONDS = 5.0
_SOLVE_RESPONSE_MARGIN_SECONDS = 5.0
_SESSION_LIFECYCLE_TIMEOUT_SECONDS = 15.0
_SESSION_RECONCILIATION_RESERVE_SECONDS = 5.0
_SESSION_DESTROY_ATTEMPTS = 2
_MAX_MANAGED_TIMEOUT_MS = 60_000
_MAX_TURNSTILE_TIMEOUT_MS = 30_000
MAX_FLARESOLVERR_SESSION_ATTEMPTS = 3


def _require_flaresolverr_deadline(deadline: Deadline) -> None:
    if deadline.remaining() <= 0:
        raise FlareSolverrRequestError(
            kind="transport",
            transport_type="TimeoutError",
        )


def _solve_deadline(
    timeout_ms: int,
    *,
    maximum_ms: int = _MAX_MANAGED_TIMEOUT_MS,
) -> Deadline:
    if type(timeout_ms) is not int or not 0 < timeout_ms <= maximum_ms:
        raise ValueError(f"timeout_ms must be an integer in (0, {maximum_ms}]")
    return Deadline.after(timeout_ms / 1000 + _SOLVE_RESPONSE_MARGIN_SECONDS)


def _bounded_solve_deadline(
    timeout_ms: int,
    *,
    maximum_ms: int,
    deadline: Deadline | None,
) -> Deadline:
    if type(timeout_ms) is not int or not 0 < timeout_ms <= maximum_ms:
        raise ValueError(f"timeout_ms must be an integer in (0, {maximum_ms}]")
    if deadline is None:
        return _solve_deadline(timeout_ms, maximum_ms=maximum_ms)
    return deadline.bounded(timeout_ms / 1000 + _SOLVE_RESPONSE_MARGIN_SECONDS)


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


class FlareSolverrSessionOwnershipError(FlareSolverrError):
    """A deterministic FlareSolverr session identity remains unresolved."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__("FlareSolverr session ownership remains unresolved")


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
        deadline: Deadline | None = None,
    ) -> FlareSolverrResult:
        operation_deadline = _bounded_solve_deadline(
            timeout_ms,
            maximum_ms=_MAX_MANAGED_TIMEOUT_MS,
            deadline=deadline,
        )
        result = await self._client.get(
            url,
            session_id=self._session_id,
            timeout_ms=timeout_ms,
            turnstile_tabs=turnstile_tabs,
            deadline=operation_deadline,
        )
        # Keep the browser terminal until all session-level validation and the
        # final deadline boundary accept this receipt.
        self._client._uncertain_request_ids.add(self._session_id)
        if self._user_agent is None:
            self._user_agent = result.user_agent
        elif result.user_agent != self._user_agent:
            raise FlareSolverrProtocolError(
                "FlareSolverr changed user agent within a persistent session"
            )
        _require_flaresolverr_deadline(operation_deadline)
        self._client._uncertain_request_ids.discard(self._session_id)
        return result

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
        deadline: Deadline | None = None,
    ) -> FlareSolverrResult:
        """Solve an embedded Turnstile in the current persistent browser."""
        # Warm the exact login page before asking FlareSolverr to inspect it.
        # Its Turnstile implementation checks for the hidden response field
        # immediately after navigation, so a cold, asynchronously rendered
        # widget can otherwise be reported as absent.
        operation_deadline = _bounded_solve_deadline(
            timeout_ms,
            maximum_ms=_MAX_TURNSTILE_TIMEOUT_MS,
            deadline=deadline,
        )
        await self.get(
            url,
            timeout_ms=timeout_ms,
            deadline=operation_deadline,
        )

        result: FlareSolverrResult | None = None
        for _ in range(_TURNSTILE_DISCOVERY_ATTEMPTS):
            result = await self.get(
                url,
                timeout_ms=timeout_ms,
                turnstile_tabs=tabs,
                deadline=operation_deadline,
            )
            if result.turnstile_token:
                _require_flaresolverr_deadline(operation_deadline)
                return result

        assert result is not None
        _require_flaresolverr_deadline(operation_deadline)
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
        if type(max_attempts) is not int or not (
            1 <= max_attempts <= MAX_FLARESOLVERR_SESSION_ATTEMPTS
        ):
            raise ValueError(
                "max_attempts must be an integer in "
                f"[1, {MAX_FLARESOLVERR_SESSION_ATTEMPTS}]"
            )
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
        deadline = _solve_deadline(
            timeout_ms,
            maximum_ms=_MAX_MANAGED_TIMEOUT_MS,
        )
        return await self._solve(
            lambda session: session.get(
                url,
                timeout_ms=timeout_ms,
                deadline=deadline,
            ),
            deadline=deadline,
        )

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> FlareSolverrSolveReceipt:
        """Solve Turnstile while preserving the active session identity."""
        deadline = _solve_deadline(
            timeout_ms,
            maximum_ms=_MAX_TURNSTILE_TIMEOUT_MS,
        )
        return await self._solve(
            lambda session: session.solve_turnstile(
                url,
                tabs=tabs,
                timeout_ms=timeout_ms,
                deadline=deadline,
            ),
            deadline=deadline,
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
        if self._closed and self._session is None:
            return
        self._closed = True
        await self._discard_current_session()

    async def retire(self) -> None:
        """Make this scope terminal and discard its current solver session."""
        if (self._closed or self._unavailable) and self._session is None:
            return
        await self._quarantine()

    async def _solve(
        self,
        operation: Callable[[FlareSolverrSession], Awaitable[FlareSolverrResult]],
        *,
        deadline: Deadline,
    ) -> FlareSolverrSolveReceipt:
        if self._closed:
            raise FlareSolverrSessionUnavailable("FlareSolverr session scope is closed")
        if self._unavailable:
            raise FlareSolverrSessionUnavailable(
                "FlareSolverr session scope is unavailable"
            )

        while True:
            session = await self._ensure_session(deadline=deadline)
            generation = self._session_generation
            try:
                result = await operation(session)
                receipt = FlareSolverrSolveReceipt(
                    result=result,
                    session_generation=generation,
                )
                _require_flaresolverr_deadline(deadline)
                return receipt
            except asyncio.CancelledError:
                # A cancelled request may still be executing in FlareSolverr.
                # Prevent callers that catch cancellation from reusing it;
                # the surrounding scope/client finalizers retain ownership.
                self._unavailable = True
                raise
            except FlareSolverrProtocolError as error:
                logger.warning(
                    "FlareSolverr protocol validation failed; quarantining "
                    "solver session: session_generation=%s error_type=%s",
                    generation,
                    type(error).__name__,
                )
                await self._quarantine(deadline=deadline)
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
                    await self._quarantine(deadline=deadline)
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
                self._unavailable = True
                try:
                    await self._discard_current_session(deadline=deadline)
                except FlareSolverrSessionOwnershipError as cleanup_error:
                    self._unavailable = True
                    raise FlareSolverrSessionUnavailable(
                        "FlareSolverr session cleanup could not prove ownership "
                        "release"
                    ) from cleanup_error
                if not retry:
                    raise FlareSolverrSessionUnavailable(
                        "FlareSolverr session attempts were exhausted"
                    ) from error
                self._unavailable = False
                continue

    async def _ensure_session(self, *, deadline: Deadline) -> FlareSolverrSession:
        if self._session is not None:
            return self._session

        while self._attempts_used < self._max_attempts:
            self._attempts_used += 1
            try:
                session = await self._client.create_session(deadline=deadline)
            except asyncio.CancelledError:
                self._unavailable = True
                raise
            except FlareSolverrSessionOwnershipError as error:
                logger.warning(
                    "FlareSolverr session creation outcome is unknown; "
                    "retry=no session_attempt=%s/%s",
                    self._attempts_used,
                    self._max_attempts,
                )
                # Retain the deterministic identity so scope/client cleanup can
                # poll for positive creation evidence before destroying it.
                self._session = FlareSolverrSession(self._client, error.session_id)
                self._unavailable = True
                raise FlareSolverrSessionUnavailable(
                    "FlareSolverr session creation ownership is unresolved"
                ) from error
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
            if deadline.remaining() <= 0:
                self._unavailable = True
                raise FlareSolverrSessionUnavailable(
                    "FlareSolverr session creation completed after its deadline"
                ) from FlareSolverrRequestError(
                    kind="transport",
                    transport_type="TimeoutError",
                )
            return session

        self._unavailable = True
        raise FlareSolverrSessionUnavailable(
            "FlareSolverr session attempts were exhausted"
        )

    async def _quarantine(self, *, deadline: Deadline | None = None) -> None:
        self._unavailable = True
        await self._discard_current_session(deadline=deadline)

    async def _discard_current_session(
        self,
        *,
        deadline: Deadline | None = None,
    ) -> None:
        session = self._session
        if session is not None:
            await self._client.destroy_session(session, deadline=deadline)
            if self._session is session:
                self._session = None
                self._identity_generation = None


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
        self._owned_session_ids: set[str] = set()
        self._uncertain_create_ids: set[str] = set()
        self._uncertain_destroy_ids: set[str] = set()
        self._uncertain_request_ids: set[str] = set()

    async def __aenter__(self) -> Self:
        self._http = httpx.AsyncClient()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        if self._http is None:
            if self._owned_session_ids:
                raise FlareSolverrSessionOwnershipError(min(self._owned_session_ids))
            return

        client = self._http
        cleanup_error: BaseException | None = None
        cleanup_deadline = Deadline.after(_SESSION_LIFECYCLE_TIMEOUT_SECONDS)
        for session_id in sorted(self._owned_session_ids):
            try:
                await self._destroy_session_id(
                    session_id,
                    deadline=cleanup_deadline,
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        self._http = None
        try:
            remaining = cleanup_deadline.remaining()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(
                client.aclose(),
                timeout=min(_TRANSPORT_PHASE_TIMEOUT_SECONDS, remaining),
            )
            if cleanup_deadline.remaining() <= 0:
                raise TimeoutError
        except TimeoutError:
            logger.warning(
                "FlareSolverr HTTP transport close exceeded %.1f seconds",
                _TRANSPORT_PHASE_TIMEOUT_SECONDS,
            )
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_error.__traceback__)

    def session(
        self,
        *,
        max_attempts: int,
    ) -> AbstractAsyncContextManager[FlareSolverrSessionScope]:
        """Own one bounded production scope with fail-closed replacement."""
        return self._session_scope(max_attempts=max_attempts)

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

    async def create_session(
        self,
        *,
        deadline: Deadline | None = None,
    ) -> FlareSolverrSession:
        """Start one persistent FlareSolverr browser."""
        if self._http is None:
            raise RuntimeError("FlareSolverrClient must be used as an async context")
        parent_deadline = (
            Deadline.after(_SESSION_LIFECYCLE_TIMEOUT_SECONDS)
            if deadline is None
            else deadline
        )
        operation_deadline = parent_deadline.bounded(_SESSION_LIFECYCLE_TIMEOUT_SECONDS)
        _require_flaresolverr_deadline(operation_deadline)
        unresolved_ids = (
            self._uncertain_create_ids
            | self._uncertain_destroy_ids
            | self._uncertain_request_ids
        )
        if unresolved_ids:
            # A prior lifecycle mutation remains unresolved.  No new identity
            # may overlap that browser generation.
            raise FlareSolverrSessionOwnershipError(min(unresolved_ids))
        session_id = secrets.token_hex(16)
        self._owned_session_ids.add(session_id)
        self._uncertain_create_ids.add(session_id)
        create_budget = max(
            0.0,
            operation_deadline.remaining() - _SESSION_RECONCILIATION_RESERVE_SECONDS,
        )
        await self._establish_session_identity(
            session_id,
            deadline=operation_deadline.bounded(create_budget),
            reconciliation_deadline=operation_deadline,
        )
        # _establish_session_identity returns only with a positive creation
        # receipt.  Persist that fact even if the caller's final acceptance is
        # late, so cleanup does not require another inventory request.
        self._uncertain_create_ids.discard(session_id)
        if operation_deadline.remaining() <= 0 or parent_deadline.remaining() <= 0:
            # The server-side session is known, but this caller may not accept
            # a late lifecycle result.  Retain the ID so scope cleanup can
            # destroy it instead of retrying under a new identity.
            raise FlareSolverrSessionOwnershipError(session_id)
        return FlareSolverrSession(self, session_id)

    @staticmethod
    def _is_pre_dispatch_failure(error: FlareSolverrRequestError) -> bool:
        return error.kind == "transport" and error.transport_type in {
            "ConnectError",
            "ConnectTimeout",
            "PoolTimeout",
        }

    async def _list_sessions(self, *, deadline: Deadline) -> frozenset[str]:
        data = await self._post(
            {"cmd": "sessions.list"},
            deadline=deadline.bounded(_TRANSPORT_PHASE_TIMEOUT_SECONDS),
        )
        sessions = data.get("sessions")
        if not isinstance(sessions, list) or not all(
            isinstance(session_id, str) and session_id for session_id in sessions
        ):
            raise FlareSolverrProtocolError(
                "FlareSolverr returned an invalid session inventory"
            )
        _require_flaresolverr_deadline(deadline)
        return frozenset(sessions)

    async def _establish_session_identity(
        self,
        session_id: str,
        *,
        deadline: Deadline,
        reconciliation_deadline: Deadline,
    ) -> None:
        """Create one caller-chosen ID without replaying an ambiguous request."""

        if deadline.remaining() <= 0:
            self._owned_session_ids.discard(session_id)
            self._uncertain_create_ids.discard(session_id)
            _require_flaresolverr_deadline(deadline)
        try:
            data = await self._post(
                {
                    "cmd": "sessions.create",
                    "session": session_id,
                },
                # Session/browser startup is an external lifecycle phase.  The
                # overall semantic deadline may exceed five seconds; _post
                # still caps connect/write/pool phases at five.
                deadline=deadline,
            )
        except FlareSolverrRequestError as error:
            if self._is_pre_dispatch_failure(error):
                # These failures prove the request never reached FlareSolverr,
                # so a later scope attempt may safely allocate a new nonce.
                self._owned_session_ids.discard(session_id)
                self._uncertain_create_ids.discard(session_id)
                raise
            if await self._created_session_is_visible(
                session_id,
                deadline=reconciliation_deadline,
            ):
                return
            # FlareSolverr's create check/spawn/store sequence is not locked.
            # Even replaying the same ID could start two WebDrivers and orphan
            # the one overwritten in its session registry.  An absent list
            # result can also race this still-running create request, so retain
            # the nonce and fail closed without any create replay.
            raise FlareSolverrSessionOwnershipError(session_id) from error
        except FlareSolverrProtocolError as error:
            if await self._created_session_is_visible(
                session_id,
                deadline=reconciliation_deadline,
            ):
                return
            raise FlareSolverrSessionOwnershipError(session_id) from error

        returned_session = data.get("session")
        if returned_session != session_id:
            if await self._created_session_is_visible(
                session_id,
                deadline=reconciliation_deadline,
            ):
                return
            raise FlareSolverrSessionOwnershipError(session_id) from (
                FlareSolverrProtocolError(
                    "FlareSolverr returned a mismatched session identity"
                )
            )
        self._uncertain_create_ids.discard(session_id)
        if deadline.remaining() <= 0:
            raise FlareSolverrSessionOwnershipError(session_id)

    async def _created_session_is_visible(
        self,
        session_id: str,
        *,
        deadline: Deadline,
    ) -> bool:
        """Promote an ambiguous create only after positive inventory evidence."""
        if deadline.remaining() <= 0:
            return False
        try:
            sessions = await self._list_sessions(deadline=deadline)
        except FlareSolverrError:
            return False
        if session_id not in sessions:
            # Absence cannot release an ID while the original create may still
            # be inside get_webdriver() before publishing into the registry.
            return False
        self._uncertain_create_ids.discard(session_id)
        if deadline.remaining() <= 0:
            raise FlareSolverrSessionOwnershipError(session_id)
        return True

    async def destroy_session(
        self,
        session: FlareSolverrSession,
        *,
        deadline: Deadline | None = None,
    ) -> None:
        """Destroy one identity and release it only after an ownership receipt."""
        if self._http is None:
            raise RuntimeError("FlareSolverrClient must be used as an async context")
        parent_deadline = (
            Deadline.after(_SESSION_LIFECYCLE_TIMEOUT_SECONDS)
            if deadline is None
            else deadline
        )
        operation_deadline = parent_deadline.bounded(_SESSION_LIFECYCLE_TIMEOUT_SECONDS)
        await self._destroy_session_id(
            session._session_id,
            deadline=operation_deadline,
        )

    async def _destroy_session_id(
        self,
        session_id: str,
        *,
        deadline: Deadline,
    ) -> None:
        self._owned_session_ids.add(session_id)
        if session_id in self._uncertain_destroy_ids:
            # FlareSolverr pops the registry entry before driver.quit().  A
            # timed-out destroy may therefore be stuck after pop; neither list
            # absence nor a replay can prove that first browser process died.
            raise FlareSolverrSessionOwnershipError(session_id)
        if session_id in self._uncertain_create_ids:
            # Positive inventory evidence can promote a completed create.  A
            # negative result is not authoritative because the original create
            # may publish later; never send destroy or report release in that
            # state.
            if not await self._created_session_is_visible(
                session_id,
                deadline=deadline,
            ):
                raise FlareSolverrSessionOwnershipError(session_id)
            self._uncertain_create_ids.discard(session_id)

        last_error: BaseException | None = None
        for _ in range(_SESSION_DESTROY_ATTEMPTS):
            if deadline.remaining() <= 0:
                raise FlareSolverrSessionOwnershipError(session_id)
            # Tombstone before awaiting the mutation.  It is removed only by
            # proof that dispatch never happened or by a strict success ACK,
            # so cancellation and every other BaseException remain fail-closed.
            self._uncertain_destroy_ids.add(session_id)
            try:
                await self._post(
                    {
                        "cmd": "sessions.destroy",
                        "session": session_id,
                    },
                    # driver.quit() is an external lifecycle phase.  _post
                    # independently caps connect/write/pool at five seconds.
                    deadline=deadline,
                )
            except FlareSolverrRequestError as error:
                last_error = error
                if self._is_pre_dispatch_failure(error):
                    self._uncertain_destroy_ids.discard(session_id)
                    continue
                break
            except FlareSolverrProtocolError as error:
                last_error = error
                break
            else:
                if deadline.remaining() <= 0:
                    self._uncertain_destroy_ids.add(session_id)
                    raise FlareSolverrSessionOwnershipError(session_id)
                self._owned_session_ids.discard(session_id)
                self._uncertain_create_ids.discard(session_id)
                self._uncertain_destroy_ids.discard(session_id)
                self._uncertain_request_ids.discard(session_id)
                return

        ownership_error = FlareSolverrSessionOwnershipError(session_id)
        if last_error is not None:
            raise ownership_error from last_error
        raise ownership_error

    async def get(
        self,
        url: str,
        *,
        session_id: str,
        timeout_ms: int = 60_000,
        turnstile_tabs: int | None = None,
        deadline: Deadline | None = None,
    ) -> FlareSolverrResult:
        if session_id in self._uncertain_destroy_ids:
            raise FlareSolverrSessionOwnershipError(session_id)
        if session_id in self._uncertain_request_ids:
            raise FlareSolverrSessionOwnershipError(session_id)
        if session_id not in self._owned_session_ids:
            raise FlareSolverrSessionOwnershipError(session_id)
        operation_deadline = _bounded_solve_deadline(
            timeout_ms,
            maximum_ms=_MAX_MANAGED_TIMEOUT_MS,
            deadline=deadline,
        )
        remaining = operation_deadline.remaining()
        work_remaining = remaining - _SOLVE_RESPONSE_MARGIN_SECONDS
        if work_remaining <= 0:
            raise FlareSolverrRequestError(
                kind="transport",
                transport_type="TimeoutError",
            )
        effective_timeout_ms = max(
            1,
            min(timeout_ms, int(work_remaining * 1000)),
        )
        payload: dict[str, Any] = {
            "cmd": "request.get",
            "url": url,
            "session": session_id,
            "maxTimeout": effective_timeout_ms,
            "returnOnlyCookies": True,
        }
        if turnstile_tabs is not None:
            payload["tabs_till_verify"] = validate_turnstile_tabs(turnstile_tabs)

        # request.get navigates a persistent browser.  Tombstone before the
        # await so cancellation and every ambiguous response prevent reuse.
        self._uncertain_request_ids.add(session_id)
        try:
            data = await self._post(
                payload,
                deadline=operation_deadline,
            )
        except FlareSolverrRequestError as error:
            if self._is_pre_dispatch_failure(error):
                self._uncertain_request_ids.discard(session_id)
            raise
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
        result = FlareSolverrResult(
            cookies=cookies,
            user_agent=user_agent,
            turnstile_token=token if isinstance(token, str) and token else None,
        )
        _require_flaresolverr_deadline(operation_deadline)
        self._uncertain_request_ids.discard(session_id)
        return result

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        deadline: Deadline,
    ) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("FlareSolverrClient must be used as an async context")

        remaining = deadline.remaining()
        if remaining <= 0:
            raise FlareSolverrRequestError(
                kind="transport",
                transport_type="TimeoutError",
            )
        phase_timeout = min(_TRANSPORT_PHASE_TIMEOUT_SECONDS, remaining)
        transport_timeout = httpx.Timeout(
            remaining,
            connect=phase_timeout,
            write=phase_timeout,
            pool=phase_timeout,
            read=remaining,
        )
        try:
            response = await asyncio.wait_for(
                self._http.post(
                    self.endpoint,
                    json=payload,
                    timeout=transport_timeout,
                ),
                timeout=remaining,
            )
            _require_flaresolverr_deadline(deadline)
            response.raise_for_status()
            _require_flaresolverr_deadline(deadline)
        except TimeoutError:
            raise FlareSolverrRequestError(
                kind="transport",
                transport_type="TimeoutError",
            ) from None
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
        _require_flaresolverr_deadline(deadline)
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
        _require_flaresolverr_deadline(deadline)
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
