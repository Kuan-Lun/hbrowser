"""Resolution policy for a page-level challenge in one browser generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ...exceptions import LoginFailedException
from ..browser.flaresolverr import FlareSolverrError, FlareSolverrSolveReceipt
from .models import ChallengeDetection


class PageChallengeDetector(Protocol):
    """Detect a challenge without changing browser or network state."""

    async def detect(
        self,
        page: Any,
        timeout: float = 2.0,
    ) -> ChallengeDetection: ...


class PageChallengeSolver(Protocol):
    """Solve within one route-bound FlareSolverr session scope."""

    async def solve_managed(self, url: str) -> FlareSolverrSolveReceipt: ...

    def mark_identity_applied(self, receipt: FlareSolverrSolveReceipt) -> None: ...

    async def retire(self) -> None: ...


ApplyIdentity = Callable[[FlareSolverrSolveReceipt], Awaitable[None]]
Navigate = Callable[[str], Awaitable[None]]
SaveDiagnostic = Callable[[str], Awaitable[Any]]


class PageChallengeHandler:
    """Resolve a page challenge without owning browser or route lifecycle."""

    def __init__(
        self,
        *,
        detector: PageChallengeDetector,
        logger: logging.Logger,
        headless: bool,
        manual_timeout: float,
        automatic_solver: PageChallengeSolver | None,
        apply_identity: ApplyIdentity,
        navigate: Navigate,
        save_diagnostic: SaveDiagnostic,
    ) -> None:
        if manual_timeout <= 0:
            raise ValueError("manual_timeout must be greater than zero")

        self._detector = detector
        self._logger = logger
        self._headless = headless
        self._manual_timeout = manual_timeout
        self._automatic_solver = automatic_solver
        self._apply_identity = apply_identity
        self._navigate = navigate
        self._save_diagnostic = save_diagnostic

    async def resolve(
        self,
        page: Any,
        url: str,
        *,
        detect_timeout: float = 3.0,
    ) -> None:
        detection = await self._detector.detect(page, timeout=detect_timeout)
        if detection.kind == "none":
            return

        if detection.kind == "cf_managed_challenge":
            self._logger.info("Cloudflare verification detected")
        else:
            self._logger.info(
                "Browser verification detected (kind=%s)",
                detection.kind,
            )
        solver = self._automatic_solver
        if detection.kind == "cf_managed_challenge" and solver is not None:
            self._logger.info("Solving Cloudflare verification automatically")
            try:
                receipt = await solver.solve_managed(url)
            except FlareSolverrError as error:
                self._logger.warning(
                    "Automatic Cloudflare verification failed (%s)",
                    type(error).__name__,
                )
            else:
                # Applying a solver identity is a fail-closed boundary. In
                # particular, BrowserIdentityApplyException must propagate so
                # the caller can retire a possibly mixed-identity browser.
                await self._apply_identity(receipt)
                solver.mark_identity_applied(receipt)
                await self._navigate(url)
                detection = await self._detector.detect(
                    page,
                    timeout=detect_timeout,
                )
                if detection.kind == "none":
                    self._logger.info("Cloudflare verification solved automatically")
                    return
                self._logger.warning(
                    "Automatic Cloudflare verification did not clear the "
                    "challenge (kind=%s)",
                    detection.kind,
                )
                await solver.retire()

        await self._save_diagnostic("challenge_page")
        if self._headless:
            self._logger.warning(
                "Manual browser verification is required (kind=%s), but the "
                "browser is headless",
                detection.kind,
            )
            raise LoginFailedException(
                f"Page challenge {detection.kind!r} requires interaction, "
                "but the browser is headless"
            )

        await self._wait_for_manual_resolution(
            page,
            challenge_kind=detection.kind,
            detect_timeout=detect_timeout,
        )

    async def _wait_for_manual_resolution(
        self,
        page: Any,
        *,
        challenge_kind: str,
        detect_timeout: float,
    ) -> None:
        self._logger.warning(
            "Manual browser verification is required (kind=%s); complete it "
            "within %.0f seconds",
            challenge_kind,
            self._manual_timeout,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._manual_timeout
        while loop.time() < deadline:
            detection = await self._detector.detect(
                page,
                timeout=detect_timeout,
            )
            if detection.kind == "none":
                self._logger.info("Browser verification completed manually")
                return
            await asyncio.sleep(1)

        raise LoginFailedException(
            f"Page challenge {challenge_kind!r} was not resolved within "
            f"{self._manual_timeout:.0f} seconds"
        )
