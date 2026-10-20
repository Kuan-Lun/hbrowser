import asyncio
import inspect
import unittest
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from websockets.exceptions import ConnectionClosed
from zendriver import cdp
from zendriver.core.connection import Transaction

from hbrowser.exceptions import (
    ArchiveDownloadOutcomeUnknownError,
    BrowserIdentityApplyException,
    BrowserMutationOutcomeUnknownError,
    LoginTokenInjectionOutcomeUnknownError,
)
from hbrowser.gallery.utils import (
    ZendriverOperationTimeout,
    is_browser_generation_error,
)
from hbrowser.gallery.utils.protocol import (
    _DRAINING_OPERATIONS,
    _LIFECYCLE_ATTRIBUTE,
    ZendriverOwnerRetiredError,
    _begin_zendriver_retirement,
    wait_for_zendriver,
)


class _DisposableAwaitable:
    def __init__(self) -> None:
        self.cancelled = False
        self.closed = False

    def __await__(self) -> Generator[None]:
        yield from ()
        return None

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True

    def done(self) -> bool:
        return self.cancelled

    def exception(self) -> None:
        return None


class _RetryingDisposableAwaitable(_DisposableAwaitable):
    def __init__(self, *, fail_close_once: bool = False) -> None:
        super().__init__()
        self.close_calls = 0
        self.fail_close_once = fail_close_once

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError("close failed")
        self.closed = True


class _TaskAwaitable:
    """Opaque owner of existing work; awaiting it after rejection is forbidden."""

    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task
        self.cancel_calls = 0
        self.close_calls = 0
        self.observed = False

    def __await__(self) -> Generator[None]:
        yield from ()
        raise AssertionError("Rejected custom awaitable was scheduled")

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.task.cancel()

    def close(self) -> None:
        self.close_calls += 1

    def done(self) -> bool:
        return self.task.done()

    def exception(self) -> BaseException | None:
        self.observed = True
        return self.task.exception()


class ZendriverTimeoutContractTests(unittest.TestCase):
    def test_timeout_error_has_one_strict_typed_constructor(self) -> None:
        error = ZendriverOperationTimeout(timeout_seconds=2.5)

        self.assertEqual(error.timeout_seconds, 2.5)
        self.assertEqual(
            str(error),
            "Zendriver operation timed out after 2.5 seconds; "
            "remote outcome is unknown",
        )

    def test_timeout_error_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(TypeError):
            ZendriverOperationTimeout(2.5)  # type: ignore[call-arg]

        invalid_values: tuple[Any, ...] = (None, "slow", True)
        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value), self.assertRaises(TypeError):
                ZendriverOperationTimeout(
                    timeout_seconds=invalid_value,
                )
        for numeric_value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=numeric_value), self.assertRaises(ValueError):
                ZendriverOperationTimeout(timeout_seconds=numeric_value)

    def test_wait_requires_an_explicit_owner(self) -> None:
        strict_wait = cast(Any, wait_for_zendriver)
        with self.assertRaises(TypeError):
            strict_wait(object(), timeout=1)

    def test_browser_generation_error_classifier_is_strict(self) -> None:
        self.assertTrue(is_browser_generation_error(ZendriverOwnerRetiredError()))
        self.assertTrue(
            is_browser_generation_error(
                ZendriverOperationTimeout(timeout_seconds=1),
            )
        )
        self.assertTrue(is_browser_generation_error(ConnectionClosed(None, None)))
        for mutation_error in (
            BrowserMutationOutcomeUnknownError(),
            BrowserIdentityApplyException(),
            ArchiveDownloadOutcomeUnknownError(),
            LoginTokenInjectionOutcomeUnknownError(),
        ):
            with self.subTest(mutation_error=type(mutation_error).__name__):
                self.assertTrue(is_browser_generation_error(mutation_error))
        self.assertFalse(is_browser_generation_error(TimeoutError()))
        self.assertFalse(is_browser_generation_error(ConnectionError()))

    def test_browser_generation_classifier_traverses_chains_cycle_safely(
        self,
    ) -> None:
        timeout = ZendriverOperationTimeout(timeout_seconds=1)
        caused = RuntimeError("domain wrapper")
        caused.__cause__ = timeout
        self.assertTrue(is_browser_generation_error(caused))

        retired = ZendriverOwnerRetiredError()
        contextual = RuntimeError("context wrapper")
        contextual.__context__ = retired
        self.assertTrue(is_browser_generation_error(contextual))

        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__context__ = second
        second.__context__ = first
        self.assertFalse(is_browser_generation_error(first))


class ZendriverTimeoutLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_watchdog_above_five_seconds_is_rejected(self) -> None:
        browser = SimpleNamespace()
        coroutine = asyncio.sleep(0)

        with self.assertRaisesRegex(ValueError, "must not exceed 5 seconds"):
            await wait_for_zendriver(coroutine, timeout=5.01, owner=browser)

        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)

    async def test_none_is_not_an_owner(self) -> None:
        coroutine = asyncio.sleep(0)
        with self.assertRaises(TypeError):
            await wait_for_zendriver(
                coroutine,
                timeout=1,
                owner=cast(Any, None),
            )
        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)

    async def test_timeout_does_not_cancel_late_protocol_transaction(self) -> None:
        browser = SimpleNamespace()
        transaction = Transaction(cdp.runtime.evaluate("1", return_by_value=True))

        with self.assertRaises(ZendriverOperationTimeout) as raised:
            await wait_for_zendriver(transaction, timeout=0, owner=browser)

        self.assertEqual(raised.exception.timeout_seconds, 0)
        self.assertFalse(transaction.cancelled())
        lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
        self.assertIn(transaction, lifecycle.operations)

        transaction(result={"result": {"type": "number", "value": 1}})
        await asyncio.sleep(0)

        remote_object, exception_details = transaction.result()
        self.assertEqual(remote_object.value, 1)
        self.assertIsNone(exception_details)
        self.assertNotIn(transaction, lifecycle.operations)

        rejected = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(rejected, timeout=1, owner=browser)
        self.assertEqual(inspect.getcoroutinestate(rejected), inspect.CORO_CLOSED)

    async def test_connection_closed_result_tombstones_generation(self) -> None:
        browser = SimpleNamespace()
        operation: asyncio.Future[None] = asyncio.Future()
        operation.set_exception(ConnectionClosed(None, None))

        with self.assertRaises(ConnectionClosed):
            await wait_for_zendriver(operation, timeout=1, owner=browser)

        lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
        self.assertTrue(lifecycle.retired)
        rejected = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(rejected, timeout=1, owner=browser)
        self.assertEqual(inspect.getcoroutinestate(rejected), inspect.CORO_CLOSED)

    async def test_caller_cancellation_does_not_cancel_protocol_operation(self) -> None:
        browser = SimpleNamespace()
        connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        operation: asyncio.Future[str] = asyncio.Future()
        watchdog = asyncio.create_task(
            wait_for_zendriver(operation, timeout=5, owner=connection)
        )
        await asyncio.sleep(0)

        watchdog.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await watchdog

        lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
        self.assertFalse(operation.cancelled())
        self.assertIn(operation, lifecycle.operations)
        self.assertTrue(lifecycle.retired)
        self.assertEqual(lifecycle.shutdown_connections, [connection])

        rejected = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(rejected, timeout=1, owner=browser)
        self.assertEqual(inspect.getcoroutinestate(rejected), inspect.CORO_CLOSED)

        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(
            quiescent_connections=(connection,),
        )
        self.assertTrue(operation.cancelled())

    async def test_retirement_isolated_to_one_browser_generation(self) -> None:
        browser_a = SimpleNamespace()
        browser_b = SimpleNamespace()
        operation_a: asyncio.Future[str] = asyncio.Future()
        operation_b: asyncio.Future[str] = asyncio.Future()

        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation_a, timeout=0, owner=browser_a)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation_b, timeout=0, owner=browser_b)

        retirement_a = _begin_zendriver_retirement(browser_a)
        await retirement_a.retire_operations(quiescent_connections=())

        self.assertTrue(operation_a.cancelled())
        self.assertFalse(operation_b.done())
        lifecycle_b = vars(browser_b)[_LIFECYCLE_ATTRIBUTE]
        self.assertIn(operation_b, lifecycle_b.operations)

        operation_b.set_result("late response")
        await asyncio.sleep(0)
        self.assertNotIn(operation_b, lifecycle_b.operations)

    async def test_elements_track_their_exact_originating_transport(self) -> None:
        browser = SimpleNamespace()
        connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        page = SimpleNamespace(
            browser=browser,
            websocket=object(),
            mapper={},
        )
        element = SimpleNamespace(_tab=page)
        operation_from_connection: asyncio.Future[None] = asyncio.Future()
        operation_from_element: asyncio.Future[None] = asyncio.Future()

        connection_watchdog = asyncio.create_task(
            wait_for_zendriver(
                operation_from_connection,
                timeout=5,
                owner=connection,
            )
        )
        element_watchdog = asyncio.create_task(
            wait_for_zendriver(
                operation_from_element,
                timeout=5,
                owner=element,
            )
        )
        await asyncio.sleep(0)

        retirement = _begin_zendriver_retirement(browser)
        self.assertEqual(
            retirement.owned_connections(),
            (connection, page),
        )

        await retirement.retire_operations(
            quiescent_connections=(connection, page),
        )
        self.assertTrue(operation_from_connection.cancelled())
        self.assertTrue(operation_from_element.cancelled())
        await asyncio.gather(
            connection_watchdog,
            element_watchdog,
            return_exceptions=True,
        )

    async def test_browser_owner_tracks_its_root_connection_exactly(self) -> None:
        browser = SimpleNamespace()
        root_connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        browser.connection = root_connection
        operation: asyncio.Future[None] = asyncio.Future()

        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        retirement = _begin_zendriver_retirement(browser)
        self.assertEqual(retirement.owned_connections(), (root_connection,))
        self.assertEqual(retirement.captured_connections(), (root_connection,))
        await retirement.retire_operations(
            quiescent_connections=(root_connection,),
        )

    async def test_retired_owner_rejects_and_closes_a_new_coroutine(self) -> None:
        browser = SimpleNamespace()
        _begin_zendriver_retirement(browser)

        coroutine = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(coroutine, timeout=1, owner=browser)

        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)

    async def test_retired_owner_retains_an_existing_future_until_safe(self) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        future: asyncio.Future[None] = asyncio.Future()

        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(future, timeout=1, owner=browser)

        self.assertFalse(future.cancelled())
        lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
        self.assertIn(future, lifecycle.operations)

        await retirement.retire_operations(quiescent_connections=())

        self.assertTrue(future.cancelled())

    async def test_retired_future_preserves_exact_connection_until_quiescent(
        self,
    ) -> None:
        browser = SimpleNamespace()
        connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        retirement = _begin_zendriver_retirement(browser)
        future: asyncio.Future[None] = asyncio.Future()
        connection.mapper[1] = future

        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(future, timeout=1, owner=connection)

        self.assertFalse(future.cancelled())
        self.assertEqual(retirement.captured_connections(), (connection,))
        with self.assertRaisesRegex(RuntimeError, "unquiesced owning connection"):
            await retirement.retire_operations(quiescent_connections=())
        self.assertFalse(future.cancelled())

        await retirement.retire_operations(
            quiescent_connections=(connection,),
        )
        self.assertTrue(future.cancelled())

    async def test_rejected_future_after_safe_phase_is_cancelled_immediately(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(quiescent_connections=())
        future: asyncio.Future[None] = asyncio.Future()

        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(future, timeout=1, owner=browser)

        self.assertTrue(future.cancelled())

    async def test_retired_owner_disposes_custom_awaitable(self) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        awaitable = _DisposableAwaitable()

        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(awaitable, timeout=1, owner=browser)

        self.assertFalse(awaitable.cancelled)
        self.assertFalse(awaitable.closed)

        await retirement.retire_operations(quiescent_connections=())

        self.assertTrue(awaitable.cancelled)
        self.assertTrue(awaitable.closed)

    async def test_custom_pending_work_retains_authority_until_verified_done(
        self,
    ) -> None:
        for already_retired in (False, True):
            with self.subTest(already_retired=already_retired):
                browser = SimpleNamespace()
                retirement = _begin_zendriver_retirement(browser)
                if already_retired:
                    await retirement.retire_operations(quiescent_connections=())
                    retirement.mark_browser_cleanup_complete()
                    retirement.mark_tor_cleanup_complete()
                    retirement.mark_complete()
                allow_stop = asyncio.Event()

                async def stubborn(stop: asyncio.Event = allow_stop) -> None:
                    while not stop.is_set():
                        try:
                            await stop.wait()
                        except asyncio.CancelledError:
                            pass

                task = asyncio.create_task(stubborn())
                await asyncio.sleep(0)
                custom = _TaskAwaitable(task)
                try:
                    with patch(
                        "hbrowser.gallery.utils.protocol._OPERATION_DRAIN_TIMEOUT_SECONDS",
                        0.01,
                    ):
                        with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
                            await wait_for_zendriver(custom, timeout=1, owner=browser)
                        if already_retired:
                            self.assertIsInstance(
                                rejected.exception.__cause__, TimeoutError
                            )
                        else:
                            self.assertEqual(custom.cancel_calls, 0)
                            with self.assertRaises(TimeoutError):
                                await retirement.retire_operations(
                                    quiescent_connections=()
                                )
                    lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
                    self.assertEqual(lifecycle.rejected_awaitables, [custom])
                    self.assertIs(
                        lifecycle.rejected_awaitable_owners[id(custom)], browser
                    )
                    self.assertIsNone(
                        lifecycle.rejected_awaitable_connections[id(custom)]
                    )
                    self.assertFalse(task.done())
                    self.assertFalse(retirement.protocol_is_retired())
                    self.assertFalse(retirement.is_complete())
                    with self.assertRaises(RuntimeError):
                        retirement.mark_complete()
                    allow_stop.set()
                    await task
                    await retirement.retire_operations(quiescent_connections=())
                    self.assertEqual(lifecycle.rejected_awaitables, [])
                    self.assertEqual(lifecycle.rejected_awaitable_owners, {})
                    self.assertTrue(custom.observed)
                    self.assertTrue(retirement.protocol_is_retired())
                finally:
                    allow_stop.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_late_custom_disposal_is_shared_with_shutdown_retry(self) -> None:
        for cancel_rejection in (False, True):
            with self.subTest(cancel_rejection=cancel_rejection):
                browser = SimpleNamespace()
                retirement = _begin_zendriver_retirement(browser)
                await retirement.retire_operations(quiescent_connections=())
                retirement.mark_browser_cleanup_complete()
                retirement.mark_tor_cleanup_complete()
                retirement.mark_complete()
                cancelling = asyncio.Event()
                release = asyncio.Event()

                async def pending(
                    started: asyncio.Event = cancelling,
                    finish: asyncio.Event = release,
                ) -> None:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        started.set()
                        await finish.wait()

                operation = asyncio.create_task(pending())
                await asyncio.sleep(0)
                custom = _TaskAwaitable(operation)
                rejection = asyncio.create_task(
                    wait_for_zendriver(custom, timeout=1, owner=browser)
                )
                retry: asyncio.Task[None] | None = None
                try:
                    await asyncio.wait_for(cancelling.wait(), timeout=1)
                    retry = asyncio.create_task(
                        retirement.retire_operations(quiescent_connections=())
                    )
                    await asyncio.sleep(0)
                    if cancel_rejection:
                        rejection.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await rejection
                    self.assertFalse(retirement.is_complete())
                    self.assertEqual(custom.cancel_calls, 1)
                    release.set()
                    await asyncio.wait_for(retry, timeout=1)
                    if not cancel_rejection:
                        with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
                            await rejection
                        self.assertIsNone(rejected.exception.__cause__)
                    self.assertTrue(operation.done())
                    self.assertEqual(custom.close_calls, 1)
                    self.assertTrue(retirement.is_complete())
                    lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
                    self.assertEqual(lifecycle.rejected_awaitables, [])
                    self.assertEqual(lifecycle.rejected_disposal_tasks, {})
                finally:
                    release.set()
                    await asyncio.gather(operation, rejection, return_exceptions=True)
                    if retry is not None:
                        await asyncio.gather(retry, return_exceptions=True)

    async def test_late_custom_disposal_restores_completed_shutdown(self) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(quiescent_connections=())
        retirement.mark_browser_cleanup_complete()
        retirement.mark_tor_cleanup_complete()
        retirement.mark_complete()
        retirement.bind_shutdown_deadline(0.0)
        custom = _DisposableAwaitable()

        with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
            await wait_for_zendriver(custom, timeout=1, owner=browser)

        self.assertIsNone(rejected.exception.__cause__)
        self.assertTrue(custom.closed)
        self.assertTrue(retirement.protocol_is_retired())
        self.assertTrue(retirement.is_complete())

    async def test_eager_factory_preserves_disposal_registration_and_retry(
        self,
    ) -> None:
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        loop.set_task_factory(asyncio.eager_task_factory)
        try:
            for fail_once in (False, True):
                with self.subTest(fail_once=fail_once):
                    browser = SimpleNamespace()
                    retirement = _begin_zendriver_retirement(browser)
                    await retirement.retire_operations(quiescent_connections=())
                    custom = _RetryingDisposableAwaitable(fail_close_once=fail_once)
                    with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
                        await wait_for_zendriver(custom, timeout=1, owner=browser)
                    if fail_once:
                        self.assertIsInstance(
                            rejected.exception.__cause__, RuntimeError
                        )
                        await retirement.retire_operations(quiescent_connections=())
                    else:
                        self.assertIsNone(rejected.exception.__cause__)
                    self.assertEqual(custom.close_calls, 2 if fail_once else 1)
                    self.assertTrue(retirement.protocol_is_retired())
                    lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
                    self.assertEqual(lifecycle.rejected_awaitables, [])
                    self.assertEqual(lifecycle.rejected_disposal_tasks, {})
        finally:
            loop.set_task_factory(original_factory)

    async def test_finalized_late_custom_task_can_finish_after_old_deadline(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(quiescent_connections=())
        retirement.mark_browser_cleanup_complete()
        retirement.mark_tor_cleanup_complete()
        retirement.mark_complete()
        retirement.bind_shutdown_deadline(0.0)
        operation = asyncio.create_task(asyncio.sleep(100))
        await asyncio.sleep(0)
        custom = _TaskAwaitable(operation)
        try:
            with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
                await wait_for_zendriver(custom, timeout=1, owner=browser)
            self.assertIsNone(rejected.exception.__cause__)
            self.assertTrue(operation.cancelled())
            self.assertTrue(custom.observed)
            self.assertTrue(retirement.is_complete())
            lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
            self.assertEqual(lifecycle.rejected_awaitables, [])
            self.assertEqual(lifecycle.rejected_disposal_tasks, {})
            self.assertEqual(lifecycle.shutdown_deadline_expires_at, 0.0)
        finally:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    async def test_late_future_retains_authority_when_cancellation_does_not_finish(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(quiescent_connections=())
        retirement.mark_browser_cleanup_complete()
        retirement.mark_tor_cleanup_complete()
        retirement.mark_complete()
        release = asyncio.Event()

        async def stubborn() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        task = asyncio.create_task(stubborn())
        await asyncio.sleep(0)
        try:
            with patch(
                "hbrowser.gallery.utils.protocol._OPERATION_DRAIN_TIMEOUT_SECONDS", 0.01
            ):
                with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
                    await wait_for_zendriver(task, timeout=1, owner=browser)
            self.assertIsInstance(rejected.exception.__cause__, TimeoutError)
            self.assertIn(task, vars(browser)[_LIFECYCLE_ATTRIBUTE].operations)
            self.assertFalse(retirement.protocol_is_retired())
            self.assertFalse(retirement.is_complete())
            release.set()
            await task
            self.assertTrue(retirement.is_complete())
            await retirement.retire_operations(quiescent_connections=())
            self.assertTrue(retirement.protocol_is_retired())
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_custom_drain_respects_existing_shutdown_deadline(self) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        retirement.bind_shutdown_deadline(0.0)
        custom = _DisposableAwaitable()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(custom, timeout=1, owner=browser)
        with (
            patch.object(custom, "done", return_value=False),
            patch(
                "hbrowser.gallery.utils.protocol.asyncio.sleep",
                side_effect=AssertionError("No budget remains"),
            ),
            self.assertRaises(TimeoutError),
        ):
            await retirement.retire_operations(quiescent_connections=())
        self.assertFalse(retirement.protocol_is_retired())
        self.assertEqual(
            vars(browser)[_LIFECYCLE_ATTRIBUTE].rejected_awaitables, [custom]
        )
        await retirement.retire_operations(quiescent_connections=())

    async def test_custom_cooperative_cancellation_is_observed_before_release(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        task = asyncio.create_task(asyncio.sleep(100))
        await asyncio.sleep(0)
        custom = _TaskAwaitable(task)
        try:
            with self.assertRaises(ZendriverOwnerRetiredError):
                await wait_for_zendriver(custom, timeout=1, owner=browser)
            await retirement.retire_operations(quiescent_connections=())
            self.assertTrue(task.done())
            self.assertTrue(custom.observed)
            self.assertEqual(custom.cancel_calls, 1)
            self.assertEqual(custom.close_calls, 1)
            self.assertTrue(retirement.protocol_is_retired())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_custom_missing_or_nonboolean_done_cannot_complete_retirement(
        self,
    ) -> None:
        for done in (None, lambda: 1):
            with self.subTest(done=done):
                browser = SimpleNamespace()
                retirement = _begin_zendriver_retirement(browser)
                custom = _DisposableAwaitable()
                with patch.object(custom, "done", done):
                    with self.assertRaises(ZendriverOwnerRetiredError):
                        await wait_for_zendriver(custom, timeout=1, owner=browser)
                    with self.assertRaisesRegex(RuntimeError, "completion"):
                        await retirement.retire_operations(quiescent_connections=())
                    self.assertFalse(retirement.protocol_is_retired())
                    self.assertEqual(
                        vars(browser)[_LIFECYCLE_ATTRIBUTE].rejected_awaitables,
                        [custom],
                    )
                await retirement.retire_operations(quiescent_connections=())

    async def test_late_custom_work_cannot_be_hidden_by_inflight_retirement(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        first = _DisposableAwaitable()
        late = _DisposableAwaitable()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(first, timeout=1, owner=browser)
        started = asyncio.Event()
        release = asyncio.Event()

        async def dispose(awaitable: Any, **_kwargs: Any) -> None:
            if awaitable is first:
                started.set()
                await release.wait()
                awaitable.cancel()
                awaitable.close()
            else:
                raise TimeoutError("Late work still pending")

        with patch(
            "hbrowser.gallery.utils.protocol._dispose_rejected_awaitable",
            side_effect=dispose,
        ):
            task = asyncio.create_task(
                retirement.retire_operations(quiescent_connections=())
            )
            try:
                await started.wait()
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(late, timeout=1, owner=browser)
                release.set()
                with self.assertRaisesRegex(RuntimeError, "pending"):
                    await task
                self.assertFalse(retirement.protocol_is_retired())
                self.assertEqual(
                    vars(browser)[_LIFECYCLE_ATTRIBUTE].rejected_awaitables, [late]
                )
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        await retirement.retire_operations(quiescent_connections=())
        self.assertTrue(retirement.protocol_is_retired())

    async def test_custom_disposal_failure_keeps_item_and_drains_the_rest(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        failing = _RetryingDisposableAwaitable(fail_close_once=True)
        succeeding = _RetryingDisposableAwaitable()
        for awaitable in (failing, succeeding):
            with self.assertRaises(ZendriverOwnerRetiredError):
                await wait_for_zendriver(awaitable, timeout=1, owner=browser)

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            await retirement.retire_operations(quiescent_connections=())

        self.assertEqual(failing.close_calls, 1)
        self.assertEqual(succeeding.close_calls, 1)
        self.assertTrue(succeeding.closed)
        lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
        self.assertEqual(lifecycle.rejected_awaitables, [failing])

        await retirement.retire_operations(quiescent_connections=())

        self.assertEqual(failing.close_calls, 2)
        self.assertTrue(failing.closed)
        self.assertEqual(lifecycle.rejected_awaitables, [])

    async def test_new_transport_during_disposal_is_not_cancelled_by_old_phase(
        self,
    ) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        existing_awaitable = _DisposableAwaitable()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(
                existing_awaitable,
                timeout=1,
                owner=browser,
            )

        disposal_started = asyncio.Event()
        allow_disposal = asyncio.Event()

        async def blocked_disposal(awaitable: Any, **_kwargs: Any) -> None:
            disposal_started.set()
            await allow_disposal.wait()
            awaitable.cancel()
            awaitable.close()

        with patch(
            "hbrowser.gallery.utils.protocol._dispose_rejected_awaitable",
            side_effect=blocked_disposal,
        ):
            retirement_task = asyncio.create_task(
                retirement.retire_operations(quiescent_connections=())
            )
            await disposal_started.wait()

            connection = SimpleNamespace(
                _owner=browser,
                websocket=object(),
                mapper={},
            )
            late_future: asyncio.Future[None] = asyncio.Future()
            connection.mapper[1] = late_future
            with self.assertRaises(ZendriverOwnerRetiredError):
                await wait_for_zendriver(
                    late_future,
                    timeout=1,
                    owner=connection,
                )

            self.assertFalse(late_future.cancelled())
            allow_disposal.set()
            with self.assertRaisesRegex(RuntimeError, "gained a new transport"):
                await retirement_task

        self.assertFalse(late_future.cancelled())
        await retirement.retire_operations(
            quiescent_connections=(connection,),
        )
        self.assertTrue(late_future.cancelled())

    async def test_retired_owner_bounds_cancellation_resistant_task(self) -> None:
        browser = SimpleNamespace()
        retirement = _begin_zendriver_retirement(browser)
        allow_stop = False

        async def stubborn_operation() -> None:
            nonlocal allow_stop
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if allow_stop:
                        return

        operation = asyncio.create_task(stubborn_operation())
        await asyncio.sleep(0)
        try:
            with self.assertRaises(ZendriverOwnerRetiredError):
                await wait_for_zendriver(operation, timeout=1, owner=browser)

            self.assertFalse(operation.done())
            self.assertNotIn(operation, _DRAINING_OPERATIONS)
            with (
                patch(
                    "hbrowser.gallery.utils.protocol._OPERATION_DRAIN_TIMEOUT_SECONDS",
                    0.01,
                ),
                self.assertRaisesRegex(
                    TimeoutError,
                    "operations did not accept cancellation",
                ),
            ):
                await retirement.retire_operations(quiescent_connections=())

            self.assertIn(operation, _DRAINING_OPERATIONS)
        finally:
            allow_stop = True
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            await asyncio.sleep(0)
        self.assertNotIn(operation, _DRAINING_OPERATIONS)

    async def test_tombstone_blocks_registration_during_retirement(self) -> None:
        browser = SimpleNamespace()
        attempted_after_cancel = False

        async def operation() -> None:
            nonlocal attempted_after_cancel
            try:
                await asyncio.Event().wait()
            finally:
                attempted_after_cancel = True
                coroutine = asyncio.sleep(0)
                with self.assertRaises(ZendriverOwnerRetiredError):
                    await wait_for_zendriver(coroutine, timeout=1, owner=browser)

        operation_task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation_task, timeout=0, owner=browser)

        retirement = _begin_zendriver_retirement(browser)
        await retirement.retire_operations(quiescent_connections=())

        self.assertTrue(attempted_after_cancel)
        self.assertTrue(operation_task.cancelled())


if __name__ == "__main__":
    unittest.main()
