"""Browser lifecycle ownership and target-bound tab transports.

This module is intentionally independent from :class:`~hbrowser.gallery.Driver`.
It provides the lower-level ownership boundary needed by applications that keep
more than one tab alive without relying on a mutable ``Driver.page`` pointer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any, Self, cast
from uuid import uuid4

from .factory import create_browser, stop_browser

type BrowserFactory[BrowserT, TabT] = Callable[[], Awaitable[tuple[BrowserT, TabT]]]
type BrowserCloser[BrowserT] = Callable[[BrowserT], Awaitable[None]]
type TabFactory[BrowserT, TabT] = Callable[[BrowserT], Awaitable[TabT]]
type TabNavigator[TabT] = Callable[[TabT, str], Awaitable[None]]
type TargetIdGetter[TabT] = Callable[[TabT], str]


class BrowserOwnershipError(RuntimeError):
    """Base error for browser ownership and tab binding failures."""


class BrowserOwnerStateError(BrowserOwnershipError):
    """The requested operation is invalid for the owner's lifecycle state."""


class TabBindingError(BrowserOwnershipError):
    """A tab cannot be bound to a stable role and target identity."""


class TabTransportUnavailableError(BrowserOwnershipError):
    """A command was submitted after the owning browser began closing."""


class BrowserOwnerState(StrEnum):
    """Lifecycle state of a :class:`BrowserOwner`."""

    NEW = "new"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TabHandle:
    """Immutable identity assigned to one browser target for its lifetime."""

    owner_id: str
    role: str
    target_id: str


async def _open_zendriver_tab(browser: Any) -> Any:
    return await browser.get("about:blank", new_tab=True)


async def _navigate_zendriver_tab(tab: Any, url: str) -> None:
    await tab.get(url)


def _get_zendriver_target_id(tab: Any) -> str:
    target = getattr(tab, "target", None)
    target_id = getattr(target, "target_id", None)
    if target_id is None:
        target_id = getattr(tab, "target_id", None)
    if target_id is None or not str(target_id).strip():
        raise TabBindingError("browser tab does not expose a target id")
    return str(target_id)


class TabTransport[TabT]:
    """Serialize commands against one fixed tab object and target.

    A transport never consults a browser's ``main_tab`` or an application's
    current-page field.  The tab object supplied at construction remains bound
    for the complete transport lifetime.  Commands may contain multiple awaits;
    the per-tab lock remains held until the command returns.
    """

    __slots__ = (
        "_command_lock",
        "_handle",
        "_is_owner_open",
        "_navigate_tab",
        "_tab",
    )

    def __init__(
        self,
        *,
        handle: TabHandle,
        tab: TabT,
        navigate_tab: TabNavigator[TabT],
        is_owner_open: Callable[[], bool],
    ) -> None:
        self._handle = handle
        self._tab = tab
        self._navigate_tab = navigate_tab
        self._is_owner_open = is_owner_open
        self._command_lock = asyncio.Lock()

    @property
    def handle(self) -> TabHandle:
        return self._handle

    @property
    def role(self) -> str:
        return self._handle.role

    @property
    def target_id(self) -> str:
        return self._handle.target_id

    @property
    def is_available(self) -> bool:
        """Whether the owner still accepts new commands."""

        return self._is_owner_open()

    def _require_available(self) -> None:
        if not self._is_owner_open():
            raise TabTransportUnavailableError(
                f"tab role {self.role!r} is unavailable because its browser "
                "owner is not open"
            )

    async def execute[ResultT](
        self,
        command: Callable[[TabT], Awaitable[ResultT]],
    ) -> ResultT:
        """Run one atomic async command against this transport's fixed tab."""

        self._require_available()
        async with self._command_lock:
            # Closing may have started while this command waited for an earlier
            # operation on the same tab.
            self._require_available()
            return await command(self._tab)

    async def navigate(self, url: str) -> None:
        """Navigate the fixed tab through its injected navigation adapter."""

        async def navigate(tab: TabT) -> None:
            await self._navigate_tab(tab, url)

        await self.execute(navigate)

    async def _drain(self) -> None:
        """Wait until an in-flight command releases this tab's lock."""

        async with self._command_lock:
            pass


class BrowserOwner[BrowserT, TabT]:
    """Own one browser and every target-bound transport created from it.

    The default adapters launch Zendriver through the existing browser factory.
    Tests and higher layers may inject fakes without importing or starting a real
    browser.  A close operation is represented by one shared task, so concurrent
    or repeated callers can never invoke the browser closer more than once.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        main_tab_role: str = "main",
        owner_id: str | None = None,
        browser_factory: BrowserFactory[BrowserT, TabT] | None = None,
        browser_closer: BrowserCloser[BrowserT] | None = None,
        tab_factory: TabFactory[BrowserT, TabT] | None = None,
        tab_navigator: TabNavigator[TabT] | None = None,
        target_id_getter: TargetIdGetter[TabT] | None = None,
    ) -> None:
        self._validate_role(main_tab_role)
        resolved_owner_id = owner_id or uuid4().hex
        if not resolved_owner_id.strip():
            raise ValueError("owner_id must not be empty")

        default_factory = partial(create_browser, headless=headless)
        self._browser_factory = browser_factory or cast(
            BrowserFactory[BrowserT, TabT],
            default_factory,
        )
        self._browser_closer = browser_closer or cast(
            BrowserCloser[BrowserT],
            stop_browser,
        )
        self._tab_factory = tab_factory or cast(
            TabFactory[BrowserT, TabT],
            _open_zendriver_tab,
        )
        self._tab_navigator = tab_navigator or cast(
            TabNavigator[TabT],
            _navigate_zendriver_tab,
        )
        self._target_id_getter = target_id_getter or cast(
            TargetIdGetter[TabT],
            _get_zendriver_target_id,
        )

        self._owner_id = resolved_owner_id
        self._main_tab_role = main_tab_role
        self._state = BrowserOwnerState.NEW
        self._browser: BrowserT | None = None
        self._tabs_by_role: dict[str, TabTransport[TabT]] = {}
        self._tabs_by_target: dict[str, TabTransport[TabT]] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    def _validate_role(role: str) -> None:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("tab role must be a non-empty string")

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def state(self) -> BrowserOwnerState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is BrowserOwnerState.OPEN

    @property
    def main_tab(self) -> TabTransport[TabT]:
        """The transport bound to the tab returned by the browser factory."""

        return self.tab(self._main_tab_role)

    @property
    def tabs(self) -> tuple[TabTransport[TabT], ...]:
        """All transports in stable role insertion order."""

        return tuple(self._tabs_by_role.values())

    def tab(self, role: str) -> TabTransport[TabT]:
        """Look up a previously bound transport by its stable role."""

        try:
            return self._tabs_by_role[role]
        except KeyError:
            raise TabBindingError(f"unknown tab role: {role!r}") from None

    def _bind_tab(self, role: str, tab: TabT) -> TabTransport[TabT]:
        self._validate_role(role)
        if role in self._tabs_by_role:
            raise TabBindingError(f"tab role is already bound: {role!r}")

        target_id = self._target_id_getter(tab)
        if not target_id.strip():
            raise TabBindingError("target id must not be empty")
        if target_id in self._tabs_by_target:
            existing = self._tabs_by_target[target_id]
            raise TabBindingError(
                f"browser target {target_id!r} is already bound to role "
                f"{existing.role!r}"
            )

        handle = TabHandle(
            owner_id=self._owner_id,
            role=role,
            target_id=target_id,
        )
        transport = TabTransport(
            handle=handle,
            tab=tab,
            navigate_tab=self._tab_navigator,
            is_owner_open=lambda: self._state is BrowserOwnerState.OPEN,
        )
        self._tabs_by_role[role] = transport
        self._tabs_by_target[target_id] = transport
        return transport

    async def start(self) -> Self:
        """Launch and bind the browser factory's main tab exactly once."""

        async with self._lifecycle_lock:
            if self._state is BrowserOwnerState.OPEN:
                return self
            if self._state is not BrowserOwnerState.NEW:
                raise BrowserOwnerStateError(
                    f"cannot start browser owner in {self._state.value!r} state"
                )

            browser: BrowserT | None = None
            try:
                browser, main_tab = await self._browser_factory()
                self._browser = browser
                self._bind_tab(self._main_tab_role, main_tab)
            except BaseException as startup_error:
                self._state = BrowserOwnerState.CLOSING
                if browser is not None:
                    try:
                        await self._browser_closer(browser)
                    except BaseException as cleanup_error:
                        startup_error.add_note(
                            "browser cleanup after startup failure also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                self._browser = None
                self._state = BrowserOwnerState.CLOSED
                raise

            self._state = BrowserOwnerState.OPEN
            return self

    async def open_tab(
        self,
        role: str,
        *,
        url: str | None = None,
    ) -> TabTransport[TabT]:
        """Create a new tab and permanently bind it to ``role``.

        If initial navigation fails, the transport remains owned and retrievable
        by role so diagnostics or recovery can inspect that exact target.
        """

        self._validate_role(role)
        async with self._lifecycle_lock:
            if self._state is not BrowserOwnerState.OPEN:
                raise BrowserOwnerStateError(
                    f"cannot open tab while browser owner is {self._state.value!r}"
                )
            if role in self._tabs_by_role:
                raise TabBindingError(f"tab role is already bound: {role!r}")

            browser = self._browser
            if browser is None:  # Defensive invariant guard for injected fakes.
                raise BrowserOwnerStateError("open owner has no browser instance")

            tab = await self._tab_factory(browser)
            transport = self._bind_tab(role, tab)
            if url is not None:
                await transport.navigate(url)
            return transport

    async def _finish_close(self) -> None:
        try:
            if self._tabs_by_role:
                await asyncio.gather(
                    *(transport._drain() for transport in self._tabs_by_role.values())
                )
            browser = self._browser
            if browser is not None:
                await self._browser_closer(browser)
        finally:
            self._browser = None
            self._state = BrowserOwnerState.CLOSED

    async def close(self) -> None:
        """Drain tab commands and close the owned browser exactly once."""

        async with self._lifecycle_lock:
            if self._close_task is not None:
                close_task = self._close_task
            elif self._state is BrowserOwnerState.CLOSED:
                return
            else:
                self._state = BrowserOwnerState.CLOSING
                self._close_task = asyncio.create_task(
                    self._finish_close(),
                    name=f"browser-owner-close-{self._owner_id}",
                )
                close_task = self._close_task

        # One caller being cancelled must not cancel the shared browser cleanup.
        await asyncio.shield(close_task)

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        await self.close()


__all__ = [
    "BrowserCloser",
    "BrowserFactory",
    "BrowserOwner",
    "BrowserOwnerState",
    "BrowserOwnerStateError",
    "BrowserOwnershipError",
    "TabBindingError",
    "TabFactory",
    "TabHandle",
    "TabNavigator",
    "TabTransport",
    "TabTransportUnavailableError",
    "TargetIdGetter",
]
