import asyncio
import unittest
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError, dataclass
from unittest.mock import AsyncMock

from hbrowser.gallery.browser import (
    BrowserOwner,
    BrowserOwnerState,
    BrowserOwnerStateError,
    TabBindingError,
    TabHandle,
    TabTransportUnavailableError,
)


@dataclass(slots=True)
class _FakeTab:
    target_id: str


@dataclass(slots=True)
class _FakeBrowser:
    current_tab: _FakeTab


class BrowserOwnerTests(unittest.IsolatedAsyncioTestCase):
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
        browser_closer.assert_awaited_once_with(browser)

        await asyncio.gather(owner.close(), owner.close())
        browser_closer.assert_awaited_once_with(browser)

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

        self.assertEqual(owner.state, BrowserOwnerState.CLOSING)
        self.assertFalse(browser_closer.await_count)
        with self.assertRaises(TabTransportUnavailableError):
            await owner.main_tab.execute(_return)

        release_command.set()
        await asyncio.gather(command_task, first_close, second_close)
        browser_closer.assert_awaited_once_with(browser)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)

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

    async def test_failed_main_tab_binding_closes_launched_browser(self) -> None:
        main = _FakeTab("")
        owner, browser, browser_closer, _ = self._owner(main_tab=main)

        with self.assertRaisesRegex(TabBindingError, "target id must not be empty"):
            await owner.start()

        browser_closer.assert_awaited_once_with(browser)
        self.assertEqual(owner.state, BrowserOwnerState.CLOSED)
        with self.assertRaises(BrowserOwnerStateError):
            await owner.start()


async def _return[ResultT](value: ResultT) -> ResultT:
    return value


if __name__ == "__main__":
    unittest.main()
