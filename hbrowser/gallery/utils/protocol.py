"""Cancellation-safe timeouts for Zendriver protocol operations."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any

from websockets.exceptions import ConnectionClosed

_LIFECYCLE_ATTRIBUTE = "_hbrowser_zendriver_lifecycle"
_OPERATION_DRAIN_TIMEOUT_SECONDS = 2.0
# Only cancellation-resistant tasks enter this set, after retirement has made
# them ineligible for normal lifecycle ownership. asyncio keeps weak references
# to tasks, so a strong reference is required until each orphan finally exits.
_DRAINING_OPERATIONS: set[asyncio.Future[Any]] = set()


class ZendriverOperationTimeout(TimeoutError):
    """A protocol watchdog expired while the remote outcome stayed unknown."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self.timeout_seconds = _validated_timeout(timeout_seconds)
        super().__init__(
            "Zendriver operation timed out after "
            f"{self.timeout_seconds:g} seconds; remote outcome is unknown"
        )


class ZendriverOwnerRetiredError(ConnectionError):
    """An operation was submitted after its browser generation was retired."""


class _ZendriverQuiescenceChanged(RuntimeError):
    """A new or unresolved transport invalidated the current drain phase."""


@dataclass(slots=True)
class _ZendriverLifecycle:
    """Protocol operations owned by one browser generation.

    The state lives on the browser itself. There is deliberately no process-wide
    fallback registry: every operation must be tied to a concrete generation so
    shutdown can isolate it from every other browser.
    """

    retired: bool = False
    operations: dict[asyncio.Future[Any], Any | None] = field(default_factory=dict)
    operation_owners: dict[asyncio.Future[Any], Any] = field(default_factory=dict)
    shutdown_task: asyncio.Task[None] | None = None
    browser_stop_task: asyncio.Future[Any] | None = None
    tor_stop_task: asyncio.Future[Any] | None = None
    shutdown_connections: list[Any] = field(default_factory=list)
    rejected_awaitables: list[Awaitable[Any]] = field(default_factory=list)
    rejected_awaitable_connections: dict[int, Any | None] = field(default_factory=dict)
    rejected_awaitable_owners: dict[int, Any] = field(default_factory=dict)
    quiescent_connection_ids: set[int] = field(default_factory=set)
    safe_to_cancel_operations: bool = False
    protocol_retirement_complete: bool = False
    browser_cleanup_complete: bool = False
    tor_cleanup_complete: bool = False
    janitor_cleanup_complete: bool = False
    shutdown_complete: bool = False

    def begin_retirement(self) -> None:
        self.capture_connections(self.connections())
        self.retired = True

    def capture_connections(self, connections: tuple[Any, ...]) -> None:
        known = {id(connection) for connection in self.shutdown_connections}
        gained_unquiesced_connection = False
        for connection in connections:
            if connection is not None and id(connection) not in known:
                known.add(id(connection))
                self.shutdown_connections.append(connection)
                gained_unquiesced_connection = (
                    gained_unquiesced_connection
                    or id(connection) not in self.quiescent_connection_ids
                )
        if self.safe_to_cancel_operations and gained_unquiesced_connection:
            self._reopen_protocol_retirement()

    def _reopen_protocol_retirement(self) -> None:
        self.safe_to_cancel_operations = False
        self.protocol_retirement_complete = False
        self.shutdown_complete = False

    def register(
        self,
        operation: asyncio.Future[Any],
        connection: Any | None,
        owner: Any,
    ) -> None:
        if self.retired:
            raise ZendriverOwnerRetiredError(
                "Zendriver browser generation has already been retired"
            )
        self.operations[operation] = connection
        self.operation_owners[operation] = owner
        operation.add_done_callback(self._observe_completion)

    def retain_rejected_future(
        self,
        operation: asyncio.Future[Any],
        connection: Any | None,
        owner: Any,
    ) -> bool:
        """Retain already-started work until its listener cannot reply.

        ``retired`` is only a tombstone.  It does not make cancelling a
        Zendriver Transaction safe while the originating listener is alive.
        """

        connection = self._refresh_owner_connection(owner, connection)
        self.capture_connections((connection,))
        if self._connection_is_safe_to_cancel(connection, owner):
            return False
        if operation not in self.operations:
            self.operations[operation] = connection
            self.operation_owners[operation] = owner
            operation.add_done_callback(self._observe_completion)
        return True

    def retain_rejected_awaitable(
        self,
        awaitable: Awaitable[Any],
        connection: Any | None,
        owner: Any,
    ) -> bool:
        """Keep opaque, possibly-started work alive until transport quiescence."""

        connection = self._refresh_owner_connection(owner, connection)
        self.capture_connections((connection,))
        if self._connection_is_safe_to_cancel(connection, owner):
            return False
        if not any(candidate is awaitable for candidate in self.rejected_awaitables):
            self.rejected_awaitables.append(awaitable)
            self.rejected_awaitable_connections[id(awaitable)] = connection
            self.rejected_awaitable_owners[id(awaitable)] = owner
        return True

    def _connection_is_safe_to_cancel(
        self,
        connection: Any | None,
        owner: Any,
    ) -> bool:
        if not self.safe_to_cancel_operations:
            return False
        if connection is None:
            if self._owner_connection_can_appear(owner):
                self._reopen_protocol_retirement()
                return False
            return True
        if id(connection) in self.quiescent_connection_ids:
            return True

        # A caller revealed a previously unknown, already-started transport
        # after an earlier drain. Reopen protocol retirement for that exact
        # connection instead of cancelling while its listener may still live.
        self._reopen_protocol_retirement()
        return False

    def _observe_completion(self, operation: asyncio.Future[Any]) -> None:
        if self.retired:
            try:
                self.refresh_operation_connection(operation)
            except BaseException:
                # Completion callbacks must never interfere with Zendriver's
                # listener. Shutdown's synchronous snapshots retry resolution.
                pass
        self.operations.pop(operation, None)
        self.operation_owners.pop(operation, None)
        _observe_future(operation)

    @staticmethod
    def _refresh_owner_connection(owner: Any, connection: Any | None) -> Any | None:
        if connection is not None:
            return connection
        _, resolved_connection = _resolve_owner(owner)
        return resolved_connection

    @staticmethod
    def _owner_connection_can_appear(owner: Any) -> bool:
        browser, connection = _resolve_owner(owner)
        if connection is not None:
            return False
        try:
            return "connection" in vars(browser)
        except TypeError:
            return False

    def refresh_operation_connection(
        self,
        operation: asyncio.Future[Any],
    ) -> Any | None:
        if operation not in self.operations:
            return None
        connection = self.operations[operation]
        owner = self.operation_owners[operation]
        resolved_connection = self._refresh_owner_connection(owner, connection)
        if resolved_connection is not connection:
            self.operations[operation] = resolved_connection
        if self.retired:
            self.capture_connections((resolved_connection,))
        return resolved_connection

    def begin_retirement_for_operation(
        self,
        operation: asyncio.Future[Any],
        *,
        owner: Any,
        connection: Any | None,
    ) -> None:
        """Tombstone and retain an operation's latest exact transport."""

        if operation in self.operations:
            resolved_connection = self.refresh_operation_connection(operation)
        else:
            resolved_connection = self._refresh_owner_connection(owner, connection)
        self.capture_connections((resolved_connection,))
        self.begin_retirement()

    def _refresh_rejected_awaitable_connection(
        self,
        awaitable: Awaitable[Any],
    ) -> Any | None:
        key = id(awaitable)
        connection = self.rejected_awaitable_connections[key]
        owner = self.rejected_awaitable_owners[key]
        resolved_connection = self._refresh_owner_connection(owner, connection)
        if resolved_connection is not connection:
            self.rejected_awaitable_connections[key] = resolved_connection
        if self.retired:
            self.capture_connections((resolved_connection,))
        return resolved_connection

    def connections(self) -> tuple[Any, ...]:
        seen: set[int] = set()
        result: list[Any] = []
        for operation in tuple(self.operations):
            connection = self.refresh_operation_connection(operation)
            if connection is None or id(connection) in seen:
                continue
            seen.add(id(connection))
            result.append(connection)
        for awaitable in tuple(self.rejected_awaitables):
            connection = self._refresh_rejected_awaitable_connection(awaitable)
            if connection is None or id(connection) in seen:
                continue
            seen.add(id(connection))
            result.append(connection)
        return tuple(result)

    def prepare_operation_retirement(
        self,
        quiescent_connections: tuple[Any, ...],
    ) -> None:
        """Atomically enter the phase where transaction cancellation is safe."""

        self.capture_connections(self.connections())
        quiescent_ids = {id(connection) for connection in quiescent_connections}
        missing = [
            connection
            for connection in self.shutdown_connections
            if id(connection) not in quiescent_ids
        ]
        if missing:
            raise _ZendriverQuiescenceChanged(
                "Zendriver operations gained an unquiesced owning connection "
                "during shutdown"
            )
        unresolved_owners = [
            self.operation_owners[operation]
            for operation, connection in self.operations.items()
            if not operation.done()
            and connection is None
            and self._owner_connection_can_appear(self.operation_owners[operation])
        ]
        unresolved_owners.extend(
            self.rejected_awaitable_owners[id(awaitable)]
            for awaitable in self.rejected_awaitables
            if self.rejected_awaitable_connections[id(awaitable)] is None
            and self._owner_connection_can_appear(
                self.rejected_awaitable_owners[id(awaitable)]
            )
        )
        if unresolved_owners:
            raise _ZendriverQuiescenceChanged(
                "Zendriver operations still have an unresolved owning connection"
            )
        self.quiescent_connection_ids.update(quiescent_ids)
        self.safe_to_cancel_operations = True

    async def retire_operations(self) -> None:
        """Cancel operations only after every owning listener is dead."""

        if not self.safe_to_cancel_operations:
            raise RuntimeError(
                "Zendriver operations cannot be retired before their listeners stop"
            )
        errors: list[BaseException] = []
        operations = tuple(self.operations)
        rejected_awaitables = tuple(self.rejected_awaitables)

        # Nothing may suspend between the quiescence check and selecting the
        # operations it authorizes. Request cancellation for that immutable
        # snapshot now. Work revealed after the first await remains registered
        # for a later close/quiescence pass and is never included below.
        _request_operation_cancellation(operations)
        rejected_ids = {id(awaitable) for awaitable in rejected_awaitables}
        self.rejected_awaitables = [
            awaitable
            for awaitable in self.rejected_awaitables
            if id(awaitable) not in rejected_ids
        ]
        rejected_metadata = {
            id(awaitable): (
                self.rejected_awaitable_connections.pop(id(awaitable)),
                self.rejected_awaitable_owners.pop(id(awaitable)),
            )
            for awaitable in rejected_awaitables
        }
        for awaitable in rejected_awaitables:
            try:
                await _dispose_rejected_awaitable(awaitable)
            except BaseException as error:
                connection, owner = rejected_metadata[id(awaitable)]
                self.rejected_awaitables.append(awaitable)
                self.rejected_awaitable_connections[id(awaitable)] = connection
                self.rejected_awaitable_owners[id(awaitable)] = owner
                errors.append(error)
        pending = await _cancel_and_drain(operations)
        if pending:
            errors.append(
                TimeoutError(
                    "Zendriver operations did not accept cancellation within "
                    f"{_OPERATION_DRAIN_TIMEOUT_SECONDS:g} seconds"
                )
            )
        if not self.safe_to_cancel_operations:
            errors.insert(
                0,
                _ZendriverQuiescenceChanged(
                    "Zendriver operation retirement gained a new transport"
                ),
            )
        if errors:
            primary, *secondary = errors
            for secondary_error in secondary:
                primary.add_note(
                    "Additional operation-retirement error: "
                    f"{type(secondary_error).__name__}: {secondary_error}"
                )
            raise primary


@dataclass(frozen=True, slots=True)
class _ZendriverRetirement:
    """Two-phase shutdown handle for one already-tombstoned generation."""

    lifecycle: _ZendriverLifecycle

    def owned_connections(self) -> tuple[Any, ...]:
        return self.lifecycle.connections()

    async def retire_operations(
        self,
        *,
        quiescent_connections: tuple[Any, ...],
    ) -> None:
        self.lifecycle.prepare_operation_retirement(quiescent_connections)
        await self.lifecycle.retire_operations()
        self.mark_protocol_retired()

    def existing_shutdown_task(self) -> asyncio.Task[None] | None:
        return self.lifecycle.shutdown_task

    def bind_shutdown_task(self, task: asyncio.Task[None]) -> None:
        if self.lifecycle.shutdown_task is not None:
            raise RuntimeError("Zendriver browser shutdown is already bound")
        self.lifecycle.shutdown_task = task

    def replace_completed_shutdown_task(self, task: asyncio.Task[None]) -> None:
        existing = self.lifecycle.shutdown_task
        if existing is None or not existing.done():
            raise RuntimeError("Zendriver browser shutdown is not replaceable")
        if self.lifecycle.shutdown_complete:
            raise RuntimeError("Completed Zendriver shutdown cannot be reopened")
        if not existing.cancelled():
            existing.exception()
        self.lifecycle.shutdown_task = task

    def is_complete(self) -> bool:
        return self.lifecycle.shutdown_complete

    def protocol_is_retired(self) -> bool:
        return self.lifecycle.protocol_retirement_complete

    def mark_protocol_retired(self) -> None:
        if not self.lifecycle.safe_to_cancel_operations:
            raise RuntimeError("Zendriver retirement cannot complete before draining")
        self.lifecycle.protocol_retirement_complete = True

    def browser_cleanup_is_complete(self) -> bool:
        return self.lifecycle.browser_cleanup_complete

    def mark_browser_cleanup_complete(self) -> None:
        self.lifecycle.browser_cleanup_complete = True

    def tor_cleanup_is_complete(self) -> bool:
        return self.lifecycle.tor_cleanup_complete

    def mark_tor_cleanup_complete(self) -> None:
        self.lifecycle.tor_cleanup_complete = True

    def janitor_cleanup_is_complete(self) -> bool:
        return self.lifecycle.janitor_cleanup_complete

    def mark_janitor_cleanup_complete(self) -> None:
        self.lifecycle.janitor_cleanup_complete = True

    def mark_complete(self) -> None:
        if not (
            self.lifecycle.protocol_retirement_complete
            and self.lifecycle.browser_cleanup_complete
            and self.lifecycle.tor_cleanup_complete
            and self.lifecycle.janitor_cleanup_complete
        ):
            raise RuntimeError("Zendriver shutdown cannot complete before all cleanup")
        self.lifecycle.shutdown_complete = True

    def capture_connections(self, connections: tuple[Any, ...]) -> None:
        self.lifecycle.capture_connections(connections)

    def captured_connections(self) -> tuple[Any, ...]:
        return tuple(self.lifecycle.shutdown_connections)

    def existing_browser_stop_task(self) -> asyncio.Future[Any] | None:
        return self.lifecycle.browser_stop_task

    def bind_browser_stop_task(self, task: asyncio.Future[Any]) -> None:
        if self.lifecycle.browser_stop_task is not None:
            raise RuntimeError("Zendriver Browser.stop task is already bound")
        self.lifecycle.browser_stop_task = task

    def replace_browser_stop_task(self, task: asyncio.Future[Any]) -> None:
        existing = self.lifecycle.browser_stop_task
        if existing is None or not existing.done():
            raise RuntimeError("Zendriver Browser.stop task is not replaceable")
        if not existing.cancelled() and existing.exception() is None:
            raise RuntimeError("Successful Zendriver Browser.stop cannot be replaced")
        self.lifecycle.browser_stop_task = task

    def existing_tor_stop_task(self) -> asyncio.Future[Any] | None:
        return self.lifecycle.tor_stop_task

    def bind_tor_stop_task(self, task: asyncio.Future[Any]) -> None:
        if self.lifecycle.tor_stop_task is not None:
            raise RuntimeError("Tor shutdown task is already bound")
        self.lifecycle.tor_stop_task = task

    def replace_tor_stop_task(self, task: asyncio.Future[Any]) -> None:
        existing = self.lifecycle.tor_stop_task
        if existing is None or not existing.done():
            raise RuntimeError("Tor shutdown task is not replaceable")
        if not existing.cancelled() and existing.exception() is None:
            raise RuntimeError("Successful Tor shutdown cannot be replaced")
        self.lifecycle.tor_stop_task = task


def _validated_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, int | float
    ):
        raise TypeError("timeout_seconds must be a real number")
    result = float(timeout_seconds)
    if not math.isfinite(result) or result < 0:
        raise ValueError("timeout_seconds must be finite and non-negative")
    return result


def _resolve_owner(owner: Any) -> tuple[Any, Any | None]:
    """Resolve an Element/Tab/Connection to its browser and exact transport."""

    if owner is None:
        raise TypeError("owner is required for every Zendriver operation")

    current = owner
    connection = None
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        try:
            attributes = vars(current)
        except TypeError as error:
            raise TypeError(
                "Zendriver operation owner must expose browser lifecycle state"
            ) from error

        if connection is None and "websocket" in attributes and "mapper" in attributes:
            connection = current

        related = next(
            (
                candidate
                for attribute in ("_tab", "browser", "_owner")
                if (candidate := attributes.get(attribute)) is not None
                and candidate is not current
            ),
            None,
        )
        if related is None:
            if connection is None:
                root_connection = attributes.get("connection")
                try:
                    root_attributes = vars(root_connection)
                except TypeError:
                    root_attributes = {}
                if "websocket" in root_attributes and "mapper" in root_attributes:
                    connection = root_connection
            return current, connection
        current = related

    raise RuntimeError("Zendriver operation owner relationship contains a cycle")


def _lifecycle_for(browser: Any, *, create: bool) -> _ZendriverLifecycle | None:
    try:
        attributes = vars(browser)
    except TypeError as error:
        raise TypeError(
            "Zendriver browser must support attached lifecycle state"
        ) from error

    lifecycle = attributes.get(_LIFECYCLE_ATTRIBUTE)
    if lifecycle is not None:
        if not isinstance(lifecycle, _ZendriverLifecycle):
            raise RuntimeError("Zendriver browser lifecycle state is corrupted")
        return lifecycle
    if not create:
        return None

    lifecycle = _ZendriverLifecycle()
    try:
        setattr(browser, _LIFECYCLE_ATTRIBUTE, lifecycle)
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "Zendriver browser must support attached lifecycle state"
        ) from error
    return lifecycle


def _observe_future(operation: asyncio.Future[Any]) -> None:
    if operation.cancelled():
        return
    try:
        operation.exception()
    except BaseException:
        # The active caller still receives the same result or exception. This
        # observation only prevents a late detached response from becoming an
        # unhandled task exception.
        pass


def _observe_draining_operation(operation: asyncio.Future[Any]) -> None:
    _DRAINING_OPERATIONS.discard(operation)
    _observe_future(operation)


async def _cancel_and_drain(
    operations: tuple[asyncio.Future[Any], ...],
) -> tuple[asyncio.Future[Any], ...]:
    _request_operation_cancellation(operations)
    if not operations:
        return ()

    done, pending = await asyncio.wait(
        operations,
        timeout=_OPERATION_DRAIN_TIMEOUT_SECONDS,
    )
    for operation in done:
        _observe_future(operation)
    for operation in pending:
        operation.cancel()
        _DRAINING_OPERATIONS.add(operation)
        operation.add_done_callback(_observe_draining_operation)
    return tuple(pending)


def _request_operation_cancellation(
    operations: tuple[asyncio.Future[Any], ...],
) -> None:
    """Issue cancellation synchronously, before a retirement phase can suspend."""

    for operation in operations:
        if not operation.done():
            operation.cancel()


async def _dispose_rejected_awaitable(awaitable: Awaitable[Any]) -> None:
    """Dispose rejected work after its owning listener has stopped."""

    if inspect.iscoroutine(awaitable):
        awaitable.close()
        return
    if isinstance(awaitable, asyncio.Future):
        pending = await _cancel_and_drain((awaitable,))
        if pending:
            pending[0].add_done_callback(_observe_future)
            raise TimeoutError(
                "Retired-owner awaitable did not accept cancellation within "
                f"{_OPERATION_DRAIN_TIMEOUT_SECONDS:g} seconds"
            )
        return

    # Some third-party awaitables own already-started work without subclassing
    # Future. Dispose through their explicit lifecycle methods and never wrap
    # them with ensure_future, which would schedule new work after retirement.
    cancel = getattr(awaitable, "cancel", None)
    if callable(cancel):
        cancel()
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()
    done = getattr(awaitable, "done", None)
    exception = getattr(awaitable, "exception", None)
    if callable(done) and done() and callable(exception):
        try:
            exception()
        except BaseException:
            pass


async def _reject_awaitable(
    awaitable: Awaitable[Any],
    *,
    lifecycle: _ZendriverLifecycle,
    connection: Any | None,
    owner: Any,
) -> None:
    """Reject new work without cancelling a live mapped transaction."""

    if inspect.iscoroutine(awaitable):
        # A native coroutine has not started until it is scheduled, so closing
        # it cannot affect Zendriver's mapper or a remote operation.
        awaitable.close()
        return
    if isinstance(awaitable, asyncio.Future):
        if lifecycle.retain_rejected_future(awaitable, connection, owner):
            return
        await _dispose_rejected_awaitable(awaitable)
        return
    if lifecycle.retain_rejected_awaitable(awaitable, connection, owner):
        return
    await _dispose_rejected_awaitable(awaitable)


def _validate_zendriver_operation(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    owner: Any,
) -> tuple[float, Any, Any | None, _ZendriverLifecycle]:
    """Validate a strict operation before its awaitable can be scheduled."""

    try:
        timeout_seconds = _validated_timeout(timeout)
        browser, connection = _resolve_owner(owner)
        lifecycle = _lifecycle_for(browser, create=True)
        assert lifecycle is not None
    except Exception:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise
    return timeout_seconds, browser, connection, lifecycle


def _retire_validated_zendriver_owner(
    *,
    browser: Any,
    connection: Any | None,
    lifecycle: _ZendriverLifecycle,
    owner: Any,
) -> None:
    """Retire the generation captured before a mutable owner operation ran."""

    resolved_connection = connection
    if resolved_connection is None:
        resolved_browser, candidate_connection = _resolve_owner(owner)
        if resolved_browser is browser:
            resolved_connection = candidate_connection
        else:
            # The owner was rebound while its operation ran. Never transfer the
            # failure to the replacement generation; the original lifecycle is
            # the only generation that accepted this mutation.
            original_browser, candidate_connection = _resolve_owner(browser)
            if original_browser is not browser:
                raise RuntimeError(
                    "Validated Zendriver browser identity changed during mutation"
                )
            resolved_connection = candidate_connection
    lifecycle.capture_connections((resolved_connection,))
    lifecycle.begin_retirement()


async def wait_for_zendriver[T](
    awaitable: Awaitable[T],
    *,
    timeout: float,
    owner: Any,
) -> T:
    """Bound a protocol await without cancelling its live transaction.

    Zendriver 0.15 leaves cancelled Transactions in its response mapper. A late
    Chrome reply then calls ``set_result`` on the cancelled Future and kills the
    connection listener. This watchdog therefore retains timed-out work under
    its exact browser generation and transport until a response arrives or the
    shutdown sequence has stopped every listener.
    """

    timeout_seconds, _, connection, lifecycle = _validate_zendriver_operation(
        awaitable,
        timeout=timeout,
        owner=owner,
    )
    if lifecycle.retired:
        retired_error = ZendriverOwnerRetiredError(
            "Zendriver browser generation has already been retired"
        )
        try:
            await _reject_awaitable(
                awaitable,
                lifecycle=lifecycle,
                connection=connection,
                owner=owner,
            )
        except Exception as disposal_error:
            retired_error.add_note(
                "Rejected awaitable cleanup failed: "
                f"{type(disposal_error).__name__}: {disposal_error}"
            )
            raise retired_error from disposal_error
        raise retired_error

    operation = asyncio.ensure_future(awaitable)
    lifecycle.register(operation, connection, owner)
    try:
        done, _ = await asyncio.wait((operation,), timeout=timeout_seconds)
    except asyncio.CancelledError:
        # Cancellation abandons a still-live transaction just like a watchdog
        # timeout.  Keep it mapped and tombstone the generation until shutdown
        # has made cancellation safe.
        lifecycle.begin_retirement_for_operation(
            operation,
            owner=owner,
            connection=connection,
        )
        raise
    if not done:
        # A live operation with an unknown outcome makes this generation unsafe
        # for every subsequent command. Tombstone it at the boundary rather than
        # relying on every caller and detached callback to remember that policy.
        lifecycle.begin_retirement_for_operation(
            operation,
            owner=owner,
            connection=connection,
        )
        raise ZendriverOperationTimeout(timeout_seconds=timeout_seconds)
    try:
        return operation.result()
    except asyncio.CancelledError:
        lifecycle.begin_retirement_for_operation(
            operation,
            owner=owner,
            connection=connection,
        )
        raise
    except (
        ConnectionClosed,
        ZendriverOperationTimeout,
        ZendriverOwnerRetiredError,
    ):
        lifecycle.begin_retirement_for_operation(
            operation,
            owner=owner,
            connection=connection,
        )
        raise


def _begin_zendriver_retirement(owner: Any) -> _ZendriverRetirement:
    """Tombstone a browser generation synchronously before shutdown awaits."""

    browser, connection = _resolve_owner(owner)
    lifecycle = _lifecycle_for(browser, create=True)
    assert lifecycle is not None
    lifecycle.capture_connections((connection,))
    lifecycle.begin_retirement()
    return _ZendriverRetirement(lifecycle)
