"""Resolution policy for CAPTCHA widgets embedded in the Forums login form."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ...exceptions import (
    BrowserIdentityApplyException,
    BrowserMutationOutcomeUnknownError,
    LoginFailedException,
    LoginTokenInjectionOutcomeUnknownError,
)
from ..browser.flaresolverr import FlareSolverrError
from ..challenge_policy import validate_turnstile_tabs
from ..utils import Deadline, wait_for_zendriver
from ..utils.mutation import wait_for_zendriver_mutation
from .constants import MAX_MANUAL_CHALLENGE_TIMEOUT_SECONDS
from .detector import CaptchaDetector
from .models import Kind

_TOKEN_READ_TIMEOUT_SECONDS = 3.0
_TOKEN_MUTATION_TIMEOUT_SECONDS = 5.0


class TurnstileSolver(Protocol):
    """Automatic solver contract used by the login flow."""

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> str | None: ...


@dataclass(frozen=True)
class _ChallengeSpec:
    label: str
    response_selector: str


_CHALLENGE_SPECS: dict[Kind, _ChallengeSpec] = {
    "turnstile_widget": _ChallengeSpec(
        label="Cloudflare Turnstile",
        response_selector='[name="cf-turnstile-response"]',
    ),
    "recaptcha_v2": _ChallengeSpec(
        label="reCAPTCHA v2",
        response_selector="#g-recaptcha-response",
    ),
}


class LoginChallengeHandler:
    """Resolve the challenge in the login form before it can be submitted."""

    def __init__(
        self,
        *,
        detector: CaptchaDetector,
        logger: logging.Logger,
        headless: bool,
        manual_timeout: float,
        turnstile_tabs: int,
        automatic_solver: TurnstileSolver | None,
    ) -> None:
        if isinstance(manual_timeout, bool) or not (
            0 < manual_timeout <= MAX_MANUAL_CHALLENGE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "manual_timeout must be in (0, "
                f"{MAX_MANUAL_CHALLENGE_TIMEOUT_SECONDS:g}]"
            )
        turnstile_tabs = validate_turnstile_tabs(turnstile_tabs)

        self._detector = detector
        self._logger = logger
        self._headless = headless
        self._manual_timeout = manual_timeout
        self._turnstile_tabs = turnstile_tabs
        self._automatic_solver = automatic_solver

    async def resolve(self, page: Any) -> None:
        detection = await self._detector.detect(page, timeout=3.0)
        spec = _CHALLENGE_SPECS.get(detection.kind)
        if spec is None:
            return

        self._logger.info("%s detected on login form", spec.label)
        if await self._read_response_token(page, spec.response_selector):
            self._logger.info("%s response token is already present", spec.label)
            return

        if (
            detection.kind == "turnstile_widget"
            and (solver := self._automatic_solver) is not None
        ):
            if await self._solve_turnstile(page, detection.url, spec, solver):
                return

        if self._headless:
            raise LoginFailedException(
                f"{spec.label} requires interaction, but the browser is headless"
            )

        await self._wait_for_manual_resolution(page, spec)

    async def _solve_turnstile(
        self,
        page: Any,
        url: str,
        spec: _ChallengeSpec,
        solver: TurnstileSolver,
    ) -> bool:
        self._logger.info("Trying FlareSolverr for login Turnstile")
        try:
            token = await solver.solve_turnstile(
                url,
                tabs=self._turnstile_tabs,
            )
        except BrowserIdentityApplyException:
            # A partially applied UA/cookie identity is unsafe to continue
            # with. Let the driver close this browser instead of falling back
            # to manual interaction on a mixed identity.
            raise
        except FlareSolverrError as error:
            self._logger.warning(
                "FlareSolverr could not solve login Turnstile (%s)",
                type(error).__name__,
            )
            return False

        if not token:
            self._logger.warning("FlareSolverr returned no login Turnstile token")
            return False

        if not await self._write_response_token(
            page,
            spec.response_selector,
            token,
        ):
            self._logger.warning(
                "Login Turnstile response field disappeared before token injection"
            )
            return False

        self._logger.info("FlareSolverr supplied the login Turnstile token")
        return True

    async def _wait_for_manual_resolution(
        self,
        page: Any,
        spec: _ChallengeSpec,
    ) -> None:
        self._logger.info(
            "Please complete %s in the browser. Waiting up to %.0f seconds...",
            spec.label,
            self._manual_timeout,
        )
        deadline = Deadline.after(self._manual_timeout)
        while not deadline.expired:
            remaining = deadline.remaining()
            if remaining <= 0:
                break
            token = await self._read_response_token(
                page,
                spec.response_selector,
                timeout=min(_TOKEN_READ_TIMEOUT_SECONDS, remaining),
            )
            remaining = deadline.remaining()
            if token:
                if remaining > 0:
                    self._logger.info("%s completed by user", spec.label)
                    return
                break
            if remaining <= 0:
                break
            await asyncio.sleep(min(1.0, remaining))

        raise LoginFailedException(
            f"{spec.label} was not completed within "
            f"{self._manual_timeout:.0f} seconds"
        )

    @staticmethod
    async def _read_response_token(
        page: Any,
        selector: str,
        *,
        timeout: float = _TOKEN_READ_TIMEOUT_SECONDS,
    ) -> str:
        if timeout <= 0:
            return ""
        selector_json = json.dumps(selector)
        value = await wait_for_zendriver(
            page.evaluate(
                "(() => {"
                f"const element = document.querySelector({selector_json});"
                "return element && typeof element.value === 'string' "
                "? element.value : '';"
                "})()"
            ),
            timeout=min(_TOKEN_READ_TIMEOUT_SECONDS, timeout),
            owner=page,
        )
        return value if isinstance(value, str) else ""

    @staticmethod
    async def _write_response_token(
        page: Any,
        selector: str,
        token: str,
    ) -> bool:
        selector_json = json.dumps(selector)
        token_json = json.dumps(token)
        result: Any = None
        injection_failed = False
        try:
            result = await wait_for_zendriver_mutation(
                page.evaluate(
                    "(() => {"
                    f"const element = document.querySelector({selector_json});"
                    "if (!element) return false;"
                    f"element.value = {token_json};"
                    "element.dispatchEvent(new Event('input', { bubbles: true }));"
                    "element.dispatchEvent(new Event('change', { bubbles: true }));"
                    f"return element.value === {token_json};"
                    "})()"
                ),
                timeout=_TOKEN_MUTATION_TIMEOUT_SECONDS,
                owner=page,
                operation="Login challenge token injection",
            )
        except BrowserMutationOutcomeUnknownError:
            # CDP exceptions can include the full evaluate expression. Do not
            # retain or chain an error that may contain the injected token.
            injection_failed = True
        if injection_failed:
            raise LoginTokenInjectionOutcomeUnknownError(
                "Login challenge token injection outcome is unknown"
            )
        return result is True
