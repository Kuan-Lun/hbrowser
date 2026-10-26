import asyncio
import unittest
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import (
    BrowserOwner,
    BrowserOwnerState,
    BrowserOwnerStateError,
    TabBindingError,
    TabHandle,
    TabTransportUnavailableError,
)
from hbrowser.gallery.browser import owner as owner_module
from hbrowser.gallery.utils import Deadline, ZendriverOperationTimeout
from hbrowser.gallery.utils import deadline as deadline_module
from hbrowser.gallery.utils.page_state import PageStateTimeout


@dataclass(slots=True)
class _FakeTab:
    target_id: str


@dataclass(slots=True)
class _FakeBrowser:
    current_tab: _FakeTab


class BrowserOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_navigation_of_existing_tab_retires_generation(self) -> None:
        owner, browser, closer, _ = self._owner(
            tab_navigator=AsyncMock(
                side_effect=BrowserMutationOutcomeUnknownError("unknown")
            )
        )
        await owner.start()
        try:
            with self.assertRaises(BrowserMutationOutcomeUnknownError):
                await owner.main_tab.navigate("https://example.test/")
            self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
            with self.assertRaises(TabTransportUnavailableError):
                await owner.main_tab.execute(AsyncMock())
        finally:
            await owner.close()
        closer.assert_awaited_once_with(browser, ANY)

    async def test_existing_tab_navigation_ack_without_readiness_retires_owner(
        self,
    ) -> None:
        browser = SimpleNamespace()
        methods: list[str] = []

        async def send(command: Any) -> Any:
            payload = next(command)
            methods.append(payload["method"])
            match payload["method"]:
                case "Page.getFrameTree":
                    return SimpleNamespace(
                        frame=SimpleNamespace(id_="main", loader_id="old")
                    )
                case "Page.setLifecycleEventsEnabled":
                    return None
                case "Page.navigate":
                    return "main", "new", None, False
                case _:
                    raise AssertionError(payload)

        tab = SimpleNamespace(
            browser=browser,
            target_id="main-target",
            websocket=object(),
            mapper={},
            send=send,
            add_handler=Mock(),
            remove_handlers=Mock(),
        )
        closer = AsyncMock()
        owner: BrowserOwner[SimpleNamespace, SimpleNamespace] = BrowserOwner(
            browser_factory=AsyncMock(return_value=(browser, tab)),
            browser_closer=closer,
            target_id_getter=lambda item: item.target_id,
        )
        await owner.start()
        try:
            with (
                patch.object(owner_module, "_TAB_NAVIGATION_DEADLINE_SECONDS", 0.02),
                self.assertRaises(PageStateTimeout),
            ):
                await owner.main_tab.navigate("https://example.test/new")
            self.assertIn("Page.navigate", methods)
            self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
            mutation = AsyncMock()
            with self.assertRaises(TabTransportUnavailableError):
                await owner.main_tab.execute(mutation)
            mutation.assert_not_awaited()
        finally:
            await owner.close()
        closer.assert_awaited_once_with(browser, ANY)

    async def test_unknown_tab_creation_retires_existing_transports(self) -> None:
        owner, browser, closer, tab_factory = self._owner()
        tab_factory.side_effect = BrowserMutationOutcomeUnknownError("unknown")
        await owner.start()
        try:
            with self.assertRaises(BrowserMutationOutcomeUnknownError):
                await owner.open_tab("new")
            self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
            self.assertEqual(len(owner.tabs), 1)
            with self.assertRaises(TabTransportUnavailableError):
                await owner.main_tab.execute(AsyncMock())
        finally:
            await owner.close()
        closer.assert_awaited_once_with(browser, ANY)

    async def test_unknown_initial_navigation_retires_all_owned_transports(
        self,
    ) -> None:
        failures = (
            ZendriverOperationTimeout(timeout_seconds=1),
            PageStateTimeout("target readiness not confirmed"),
            BrowserMutationOutcomeUnknownError("navigation outcome unknown"),
            asyncio.CancelledError(),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                navigator = AsyncMock(side_effect=failure)
                owner, browser, closer, _ = self._owner(
                    extra_tabs=[_FakeTab("new-target")], tab_navigator=navigator
                )
                await owner.start()
                try:
                    with self.assertRaises(type(failure)):
                        await owner.open_tab("new", url="https://example.test/")
                    self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
                    mutation = AsyncMock()
                    for transport in owner.tabs:
                        with self.assertRaises(TabTransportUnavailableError):
                            await transport.execute(mutation)
                    mutation.assert_not_awaited()
                    with self.assertRaises(BrowserOwnerStateError):
                        await owner.open_tab("later")
                    closer.assert_not_awaited()
                finally:
                    await owner.close()
                closer.assert_awaited_once_with(browser, ANY)
                self.assertEqual(owner.state, BrowserOwnerState.CLOSED)

    async def test_created_target_missing_from_inventory_retires_owner(self) -> None:
        main = _FakeTab("main-target")
        browser = SimpleNamespace(targets=[])
        connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
            add_handler=Mock(),
            remove_handlers=Mock(),
            send=AsyncMock(return_value="created-target"),
        )
        browser.connection = connection
        closer = AsyncMock()
        owner: BrowserOwner[SimpleNamespace, _FakeTab] = BrowserOwner(
            browser_factory=AsyncMock(return_value=(browser, main)),
            browser_closer=closer,
            target_id_getter=lambda tab: tab.target_id,
        )
        await owner.start()
        try:
            with (
                patch.object(owner_module, "_TAB_OPEN_DEADLINE_SECONDS", 0.02),
                self.assertRaises(PageStateTimeout),
            ):
                await owner.open_tab("unconfirmed")
            connection.send.assert_awaited_once()
            self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
            mutation = AsyncMock()
            with self.assertRaises(TabTransportUnavailableError):
                await owner.main_tab.execute(mutation)
            with self.assertRaises(BrowserOwnerStateError):
                await owner.open_tab("unconfirmed")
            mutation.assert_not_awaited()
            self.assertEqual(len(owner.tabs), 1)
        finally:
            await owner.close()
        closer.assert_awaited_once_with(browser, ANY)

    def _owner(
        self,
        *,
        main_tab: _FakeTab | None = None,
        extra_tabs: list[_FakeTab] | None = None,
        tab_navigator: Callable[[_FakeTab, str], Awaitable[None]] | None = None,
    ) -> tuple[
        BrowserOwner[_FakeBrowser, _FakeTab],
        _FakeBrowser,
        AsyncMock,
        AsyncMock,
    ]:
        main_tab = main_tab or _FakeTab("main-target")
        browser = _FakeBrowser(main_tab)
        remaining_tabs = iter(extra_tabs or [])
        browser_factory = AsyncMock(return_value=(browser, main_tab))
        browser_closer = AsyncMock()

        async def open_tab(_: _FakeBrowser) -> _FakeTab:
            return next(remaining_tabs)

        tab_factory = AsyncMock(side_effect=open_tab)
        owner = BrowserOwner(
            owner_id="test-owner",
            main_tab_role="persistent",
            browser_factory=browser_factory,
            browser_closer=browser_closer,
            tab_factory=tab_factory,
            tab_navigator=tab_navigator or AsyncMock(),
            target_id_getter=lambda tab: tab.target_id,
        )
        return owner, browser, browser_closer, tab_factory

    async def test_context_owns_browser_and_closes_it_only_once(self) -> None:
        owner, browser, browser_closer, _ = self._owner()

        async with owner as opened:
            self.assertIs(opened, owner)
            self.assertEqual(owner.state, BrowserOwnerState.OPEN)
            self.assertEqual(owner.main_tab.role, "persistent")
            self.assertEqual(owner.main_tab.target_id, "main-target")

        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
        browser_closer.assert_awaited_once_with(browser, ANY)

        await asyncio.gather(owner.close(), owner.close())
        browser_closer.assert_awaited_once_with(browser, ANY)

    async def test_tab_handle_is_immutable(self) -> None:
        handle = TabHandle("owner", "isekai", "target")

        with self.assertRaises(FrozenInstanceError):
            handle.role = "persistent"  # type: ignore[misc]

    async def test_transport_stays_bound_when_browser_current_tab_changes(
        self,
    ) -> None:
        main = _FakeTab("persistent-target")
        isekai = _FakeTab("isekai-target")
        owner, browser, _, _ = self._owner(main_tab=main, extra_tabs=[isekai])
        await owner.start()
        self.addAsyncCleanup(owner.close)

        isekai_transport = await owner.open_tab("isekai")
        browser.current_tab = main

        seen = await isekai_transport.execute(lambda tab: _return(tab.target_id))

        self.assertEqual(seen, "isekai-target")
        self.assertIs(owner.tab("isekai"), isekai_transport)

    async def test_commands_are_serialized_within_one_tab(self) -> None:
        owner, _, _, _ = self._owner()
        await owner.start()
        self.addAsyncCleanup(owner.close)
        transport = owner.main_tab
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_entered = asyncio.Event()
        order: list[str] = []

        async def first(_: _FakeTab) -> None:
            order.append("first-enter")
            first_entered.set()
            await release_first.wait()
            order.append("first-exit")

        async def second(_: _FakeTab) -> None:
            order.append("second-enter")
            second_entered.set()

        first_task = asyncio.create_task(transport.execute(first))
        await first_entered.wait()
        second_task = asyncio.create_task(transport.execute(second))
        await asyncio.sleep(0)

        self.assertFalse(second_entered.is_set())
        release_first.set()
        await asyncio.gather(first_task, second_task)
        self.assertEqual(order, ["first-enter", "first-exit", "second-enter"])

    async def test_different_tabs_have_independent_command_locks(self) -> None:
        owner, _, _, _ = self._owner(extra_tabs=[_FakeTab("isekai-target")])
        await owner.start()
        self.addAsyncCleanup(owner.close)
        isekai = await owner.open_tab("isekai")
        both_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def command(_: _FakeTab) -> None:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()

        persistent_task = asyncio.create_task(owner.main_tab.execute(command))
        isekai_task = asyncio.create_task(isekai.execute(command))
        await asyncio.wait_for(both_entered.wait(), timeout=1)

        release.set()
        await asyncio.gather(persistent_task, isekai_task)

    async def test_navigation_uses_bound_tab_and_its_command_lock(self) -> None:
        main = _FakeTab("persistent-target")
        navigations: list[tuple[str, str]] = []
        blocker_entered = asyncio.Event()
        release_blocker = asyncio.Event()

        async def navigate(tab: _FakeTab, url: str) -> None:
            navigations.append((tab.target_id, url))

        owner, _, _, _ = self._owner(main_tab=main, tab_navigator=navigate)
        await owner.start()
        self.addAsyncCleanup(owner.close)

        async def blocker(_: _FakeTab) -> None:
            blocker_entered.set()
            await release_blocker.wait()

        blocker_task = asyncio.create_task(owner.main_tab.execute(blocker))
        await blocker_entered.wait()
        navigation_task = asyncio.create_task(
            owner.main_tab.navigate("https://example.test/persistent")
        )
        await asyncio.sleep(0)
        self.assertEqual(navigations, [])

        release_blocker.set()
        await asyncio.gather(blocker_task, navigation_task)
        self.assertEqual(
            navigations,
            [("persistent-target", "https://example.test/persistent")],
        )

    async def test_close_drains_commands_rejects_new_work_and_is_shared(
        self,
    ) -> None:
        owner, browser, browser_closer, _ = self._owner()
        await owner.start()
        command_entered = asyncio.Event()
        release_command = asyncio.Event()

        async def command(_: _FakeTab) -> None:
            command_entered.set()
            await release_command.wait()

        command_task = asyncio.create_task(owner.main_tab.execute(command))
        await command_entered.wait()
        first_close = asyncio.create_task(owner.close())
        second_close = asyncio.create_task(owner.close())
        while owner.state is BrowserOwnerState.OPEN:
            await asyncio.sleep(0)
        while browser_closer.await_count == 0:
            await asyncio.sleep(0)

        self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
        browser_closer.assert_awaited_once_with(browser, ANY)
        self.assertFalse(first_close.done())
        with self.assertRaises(TabTransportUnavailableError):
            await owner.main_tab.execute(_return)

        release_command.set()
        await asyncio.gather(command_task, first_close, second_close)
        browser_closer.assert_awaited_once_with(browser, ANY)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)

    async def test_close_is_bounded_after_browser_closes_with_stuck_command(
        self,
    ) -> None:
        owner, browser, browser_closer, _ = self._owner()
        await owner.start()
        command_entered = asyncio.Event()
        release_command = asyncio.Event()

        async def stuck_command(_: _FakeTab) -> None:
            command_entered.set()
            await release_command.wait()

        command_task = asyncio.create_task(owner.main_tab.execute(stuck_command))
        await command_entered.wait()
        try:
            with patch.object(owner_module, "_COMMAND_DRAIN_TIMEOUT_SECONDS", 0.01):
                with self.assertRaisesRegex(TimeoutError, "did not drain"):
                    await asyncio.wait_for(owner.close(), timeout=1)
                first_deadline = owner._close_deadline
                self.assertFalse(command_task.done())
                with self.assertRaisesRegex(TimeoutError, "did not drain"):
                    await asyncio.wait_for(owner.close(), timeout=1)
                self.assertIsNot(owner._close_deadline, first_deadline)

            self.assertFalse(command_task.done())
            self.assertTrue(owner.main_tab._command_lock.locked())
            self.assertIsNone(owner._browser)
            self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
            with self.assertRaises(TabTransportUnavailableError):
                await owner.main_tab.execute(_return)
            browser_closer.assert_awaited_once_with(browser, ANY)
            release_command.set()
            await command_task
            await owner.close()
            self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
            browser_closer.assert_awaited_once_with(browser, ANY)
        finally:
            release_command.set()
            await command_task

    async def test_cancelled_drain_retains_release_and_shared_caller_ownership(
        self,
    ) -> None:
        for cancel_attempt in (False, True):
            with self.subTest(cancel_attempt=cancel_attempt):
                await self._assert_cancelled_drain(cancel_attempt=cancel_attempt)

    async def _assert_cancelled_drain(self, *, cancel_attempt: bool) -> None:
        owner, browser, closer, _ = self._owner()
        await owner.start()
        command_entered = asyncio.Event()
        release_command = asyncio.Event()
        drain_entered = asyncio.Event()
        original_drain = owner_module.TabTransport._drain

        async def command(_: _FakeTab) -> None:
            command_entered.set()
            await release_command.wait()

        async def observe_drain(
            transport: owner_module.TabTransport[_FakeTab],
        ) -> None:
            drain_entered.set()
            await original_drain(transport)

        command_task = asyncio.create_task(owner.main_tab.execute(command))
        await command_entered.wait()
        callers: list[asyncio.Task[None]] = []
        try:
            with patch.object(owner_module.TabTransport, "_drain", observe_drain):
                callers.append(asyncio.create_task(owner.close()))
                await asyncio.wait_for(drain_entered.wait(), timeout=1)
                shared_task = owner._close_task
                assert shared_task is not None
                shared_deadline = owner._close_deadline
                if not cancel_attempt:
                    callers.append(asyncio.create_task(owner.close()))
                cancelled = shared_task if cancel_attempt else callers[0]
                cancelled.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(callers[0], timeout=1)
                self.assertFalse(command_task.done())
                self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
                self.assertIsNone(owner._browser)
                closer.assert_awaited_once_with(browser, ANY)
                if not cancel_attempt:
                    self.assertIs(owner._close_task, shared_task)
                    self.assertIs(owner._close_deadline, shared_deadline)
                    self.assertFalse(shared_task.done())
                    self.assertFalse(callers[1].done())
                release_command.set()
                await command_task
                if not cancel_attempt:
                    await asyncio.wait_for(callers[1], timeout=1)
                await asyncio.wait_for(owner.close(), timeout=1)
                self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
                closer.assert_awaited_once_with(browser, ANY)
        finally:
            release_command.set()
            await command_task
            await asyncio.wait_for(
                asyncio.gather(*callers, return_exceptions=True), timeout=1
            )

    async def test_late_positive_closer_retains_only_undrained_commands(self) -> None:
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                await self._assert_late_closer(blocked=blocked)

    async def _assert_late_closer(self, *, blocked: bool) -> None:
        owner, browser, closer, _ = self._owner()
        await owner.start()
        command_entered = asyncio.Event()
        release_command = asyncio.Event()
        now = 0.0

        async def command(_: _FakeTab) -> None:
            command_entered.set()
            await release_command.wait()

        async def late_close(
            observed_browser: _FakeBrowser, deadline: Deadline
        ) -> None:
            nonlocal now
            self.assertIs(observed_browser, browser)
            now = deadline.expires_at

        closer.side_effect = late_close
        command_task = asyncio.create_task(owner.main_tab.execute(command))
        await command_entered.wait()
        try:
            if not blocked:
                release_command.set()
                await command_task
            with patch.object(
                deadline_module, "time", SimpleNamespace(monotonic=lambda: now)
            ):
                with self.assertRaisesRegex(
                    owner_module.BrowserOwnershipError, "completed after"
                ):
                    await asyncio.wait_for(owner.close(), timeout=1)
                self.assertIsNone(owner._browser)
                expected = (
                    BrowserOwnerState.CLOSING if blocked else BrowserOwnerState.CLOSED
                )
                self.assertEqual(owner.state, expected)
                if blocked:
                    self.assertFalse(command_task.done())
                    with (
                        patch.object(
                            owner_module, "_COMMAND_DRAIN_TIMEOUT_SECONDS", 0.01
                        ),
                        self.assertRaisesRegex(TimeoutError, "did not drain"),
                    ):
                        await asyncio.wait_for(owner.close(), timeout=1)
                    closer.assert_awaited_once_with(browser, ANY)
                release_command.set()
                await command_task
                await asyncio.wait_for(owner.close(), timeout=1)
                self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
                closer.assert_awaited_once_with(browser, ANY)
        finally:
            release_command.set()
            await command_task

    async def test_startup_late_positive_closer_is_not_repeated(self) -> None:
        owner, browser, closer, _ = self._owner(main_tab=_FakeTab(""))
        now = 0.0

        async def late_close(
            observed_browser: _FakeBrowser, deadline: Deadline
        ) -> None:
            nonlocal now
            self.assertIs(observed_browser, browser)
            now = deadline.expires_at

        closer.side_effect = late_close
        with patch.object(
            deadline_module, "time", SimpleNamespace(monotonic=lambda: now)
        ):
            with self.assertRaises(TabBindingError) as raised:
                await asyncio.wait_for(owner.start(), timeout=1)
            self.assertIn(
                "_BrowserCloserCompletedAfterDeadline",
                " ".join(raised.exception.__notes__),
            )
            self.assertIsNone(owner._browser)
            self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
            await asyncio.wait_for(owner.close(), timeout=1)
            closer.assert_awaited_once_with(browser, ANY)

    async def test_hung_injected_closer_uses_one_non_resetting_deadline(
        self,
    ) -> None:
        owner, browser, browser_closer, _ = self._owner()
        await owner.start()
        closer_started = asyncio.Event()
        release_closer = asyncio.Event()
        expire_closer = asyncio.Event()
        overlapping_callers = asyncio.Event()
        now = 0.0
        caller_wait_count = 0
        callers: list[asyncio.Task[None]] = []
        original_wait = asyncio.wait

        async def ordered_wait(
            tasks: tuple[asyncio.Task[None], ...],
            *,
            timeout: float,
        ) -> tuple[set[asyncio.Task[None]], set[asyncio.Task[None]]]:
            nonlocal now, caller_wait_count
            (task,) = tasks
            if task is owner._browser_closer_task:
                close_deadline = owner._close_deadline
                self.assertIsNotNone(close_deadline)
                assert close_deadline is not None
                self.assertEqual(timeout, close_deadline.remaining())
                await closer_started.wait()
                await expire_closer.wait()
                now = close_deadline.expires_at
                return set(), {task}
            self.assertIs(task, owner._close_task)
            caller_wait_count += 1
            if caller_wait_count == 2:
                overlapping_callers.set()
            # Both waits share one expiry. Deliver the closer's timeout first
            # so host scheduling cannot select the outer caller timeout instead.
            return await original_wait(tasks)

        async def ignore_cancellation(
            observed_browser: _FakeBrowser,
            deadline: Deadline,
        ) -> None:
            self.assertIs(observed_browser, browser)
            self.assertIs(deadline, owner._close_deadline)
            closer_started.set()
            while not release_closer.is_set():
                try:
                    await release_closer.wait()
                except asyncio.CancelledError:
                    continue

        browser_closer.side_effect = ignore_cancellation
        with (
            patch.object(
                deadline_module, "time", SimpleNamespace(monotonic=lambda: now)
            ),
            patch.object(
                owner_module,
                "asyncio",
                SimpleNamespace(**(vars(asyncio) | {"wait": ordered_wait})),
            ),
        ):
            try:
                callers.append(asyncio.create_task(owner.close()))
                await asyncio.wait_for(closer_started.wait(), timeout=1)
                shared_deadline = owner._close_deadline
                now += 1.0
                callers.append(asyncio.create_task(owner.close()))
                await asyncio.wait_for(overlapping_callers.wait(), timeout=1)
                self.assertIs(owner._close_deadline, shared_deadline)
                expire_closer.set()
                results = await asyncio.wait_for(
                    asyncio.gather(*callers, return_exceptions=True), timeout=1
                )
                for result in results:
                    self.assertIsInstance(result, owner_module.BrowserOwnershipError)
                    self.assertIn("shared ownership deadline", str(result))
                self.assertIs(results[0], results[1])

                with self.assertRaisesRegex(
                    owner_module.BrowserOwnershipError, "shared ownership deadline"
                ):
                    await asyncio.wait_for(owner.close(), timeout=1)
                browser_closer.assert_awaited_once()

                release_closer.set()
                closer_task = owner._browser_closer_task
                assert closer_task is not None
                await asyncio.wait_for(asyncio.shield(closer_task), timeout=1)
                # A completed failed attempt permits a fresh deadline on retry,
                # while reconciliation must not invoke the closer again.
                await asyncio.wait_for(owner.close(), timeout=1)
                self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
                browser_closer.assert_awaited_once()
            finally:
                # Release every fixture-owned task even when an assertion fails.
                expire_closer.set()
                release_closer.set()
                await asyncio.wait_for(
                    asyncio.gather(*callers, return_exceptions=True), timeout=1
                )
                closer_task = owner._browser_closer_task
                if closer_task is not None and not closer_task.done():
                    await asyncio.wait_for(asyncio.shield(closer_task), timeout=1)

    async def test_first_close_uses_supplied_absolute_deadline(self) -> None:
        owner, browser, browser_closer, _ = self._owner()
        await owner.start()
        supplied_deadline = Deadline.after(1)

        await owner.close(deadline=supplied_deadline)

        browser_closer.assert_awaited_once()
        assert browser_closer.await_args is not None
        observed_browser, observed_deadline = browser_closer.await_args.args
        self.assertIs(observed_browser, browser)
        self.assertEqual(observed_deadline.expires_at, supplied_deadline.expires_at)
        assert owner._close_deadline is not None
        self.assertEqual(
            owner._close_deadline.expires_at,
            supplied_deadline.expires_at,
        )

    async def test_expired_first_close_never_starts_browser_closer(self) -> None:
        owner, _, browser_closer, _ = self._owner()
        await owner.start()

        with self.assertRaisesRegex(
            owner_module.BrowserOwnershipError,
            "caller deadline expired",
        ):
            await owner.close(deadline=Deadline.after(0))

        browser_closer.assert_not_awaited()
        self.assertEqual(owner.state, BrowserOwnerState.OPEN)

        await owner.close(deadline=Deadline.after(1))
        browser_closer.assert_awaited_once()
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)

    async def test_close_deadline_includes_lifecycle_lock_wait(self) -> None:
        owner, _, browser_closer, _ = self._owner()
        await owner.start()
        await owner._lifecycle_lock.acquire()
        try:
            with self.assertRaisesRegex(
                owner_module.BrowserOwnershipError,
                "waiting for lifecycle lock",
            ):
                await owner.close(deadline=Deadline.after(0.01))
        finally:
            owner._lifecycle_lock.release()

        browser_closer.assert_not_awaited()
        self.assertEqual(owner.state, BrowserOwnerState.OPEN)
        await owner.close()

    async def test_failed_close_is_single_flight_and_can_be_retried(self) -> None:
        owner, browser, browser_closer, _ = self._owner()
        await owner.start()
        failure = RuntimeError("transport still live")
        browser_closer.side_effect = failure

        first, second = await asyncio.gather(
            owner.close(),
            owner.close(),
            return_exceptions=True,
        )

        self.assertIs(first, failure)
        self.assertIs(second, failure)
        browser_closer.assert_awaited_once_with(browser, ANY)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSING)

        browser_closer.side_effect = None
        await owner.close()

        self.assertEqual(browser_closer.await_count, 2)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)

    async def test_default_tab_adapters_bound_hung_zendriver_operations(self) -> None:
        release_open = asyncio.Event()
        release_navigation = asyncio.Event()

        async def open_tab(*_: object, **__: object) -> object:
            await release_open.wait()
            return object()

        async def navigate(*_: object, **__: object) -> None:
            await release_navigation.wait()

        browser = SimpleNamespace()
        connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        browser.connection = connection
        browser.get = AsyncMock(side_effect=open_tab)
        navigation_browser = SimpleNamespace()
        tab = SimpleNamespace(
            browser=navigation_browser,
            websocket=object(),
            mapper={},
            get=AsyncMock(side_effect=navigate),
        )

        with (
            patch.object(owner_module, "_TAB_OPEN_DEADLINE_SECONDS", 0.05),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await owner_module._open_zendriver_tab(browser)
        with (
            patch.object(
                owner_module,
                "_TAB_NAVIGATION_DEADLINE_SECONDS",
                0.05,
            ),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await owner_module._navigate_zendriver_tab(tab, "https://example.test")

        release_open.set()
        release_navigation.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_default_tab_open_generic_failure_is_outcome_unknown(self) -> None:
        browser = SimpleNamespace()
        browser.connection = SimpleNamespace(
            _owner=browser,
            websocket=object(),
            mapper={},
        )
        browser.get = AsyncMock(side_effect=RuntimeError("sensitive open failure"))

        with self.assertRaises(BrowserMutationOutcomeUnknownError) as raised:
            await owner_module._open_zendriver_tab(browser)

        browser.get.assert_awaited_once_with("about:blank", new_tab=True)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    async def test_default_tab_navigation_generic_failure_is_outcome_unknown(
        self,
    ) -> None:
        browser = SimpleNamespace()
        tab = SimpleNamespace(
            browser=browser,
            websocket=object(),
            mapper={},
            get=AsyncMock(side_effect=RuntimeError("sensitive navigation failure")),
        )

        with self.assertRaises(BrowserMutationOutcomeUnknownError) as raised:
            await owner_module._navigate_zendriver_tab(
                tab,
                "https://example.test/private",
            )

        tab.get.assert_awaited_once_with("https://example.test/private")
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    async def test_open_tab_rejects_duplicate_role_before_creating_target(
        self,
    ) -> None:
        owner, _, _, tab_factory = self._owner(extra_tabs=[_FakeTab("unused")])
        await owner.start()
        self.addAsyncCleanup(owner.close)

        with self.assertRaisesRegex(TabBindingError, "already bound"):
            await owner.open_tab("persistent")

        tab_factory.assert_not_awaited()

    async def test_duplicate_target_cannot_be_bound_to_a_second_role(self) -> None:
        owner, _, _, _ = self._owner(
            extra_tabs=[_FakeTab("main-target")],
        )
        await owner.start()
        self.addAsyncCleanup(owner.close)

        with self.assertRaisesRegex(TabBindingError, "already bound to role"):
            await owner.open_tab("isekai")
        self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
        with self.assertRaises(TabTransportUnavailableError):
            await owner.main_tab.execute(AsyncMock())

    async def test_failed_main_tab_binding_closes_launched_browser(self) -> None:
        main = _FakeTab("")
        owner, browser, browser_closer, _ = self._owner(main_tab=main)

        with self.assertRaisesRegex(TabBindingError, "target id must not be empty"):
            await owner.start()

        browser_closer.assert_awaited_once_with(browser, ANY)
        assert browser_closer.await_args is not None
        passed_deadline = browser_closer.await_args.args[1]
        self.assertIs(passed_deadline, owner._close_deadline)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
        with self.assertRaises(BrowserOwnerStateError):
            await owner.start()


async def _return[ResultT](value: ResultT) -> ResultT:
    return value


if __name__ == "__main__":
    unittest.main()
