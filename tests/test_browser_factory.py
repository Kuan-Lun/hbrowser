import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, call, patch

import zendriver as zd
from zendriver import cdp

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import factory
from hbrowser.gallery.browser import process as process_module
from hbrowser.gallery.browser import tor as tor_module
from hbrowser.gallery.browser.process import (
    OwnedProcess,
    ProcessOwnershipError,
)
from hbrowser.gallery.browser.tor import terminate_tor_process
from hbrowser.gallery.utils import (
    Deadline,
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
from hbrowser.gallery.utils import deadline as deadline_module
from hbrowser.gallery.utils.protocol import (
    ZendriverOwnerRetiredError,
    _begin_zendriver_retirement,
)


def _tab(target_id: str = "page-1", *, type_: str = "page") -> zd.Tab:
    target = cdp.target.TargetInfo(
        target_id=cdp.target.TargetID(target_id),
        type_=type_,
        title="",
        url="about:blank",
        attached=False,
        can_access_opener=False,
    )
    return zd.Tab(f"ws://127.0.0.1/devtools/page/{target_id}", target)


def _bind_browser_process_owner(
    browser: object,
    owner: Mock | None = None,
    *,
    returncode: int | None = None,
) -> Mock:
    if owner is None:
        owner = Mock(spec=OwnedProcess)
        owner.poll.return_value = returncode
    setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)
    return owner


@contextmanager
def _release_mock_owner_directories() -> Iterator[None]:
    """Complete the real directory obligations that mocked owners cannot own."""

    # Enter after TemporaryDirectory and before the operation's mocks: teardown
    # uses the original guards with real APIs before the enclosing tree is removed.
    original_capture = process_module._PrivateDirectory.capture
    with ExitStack() as cleanup:

        def capture(
            path: Path, *, platform_name: str | None = None
        ) -> process_module._PrivateDirectory:
            guard = original_capture(path, platform_name=platform_name)
            cleanup.callback(
                lambda: guard.remove(deadline=Deadline.after(5).expires_at)
            )
            return guard

        with patch.object(
            process_module._PrivateDirectory, "capture", side_effect=capture
        ):
            yield


class OwnedZendriverContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_browser_stopped_requires_a_valid_process_owner(self) -> None:
        config = zd.Config(user_data_dir="/private/profile")
        browser = factory._OwnedZendriverBrowser(config)
        raw_process = Mock()
        browser._process = raw_process

        with self.assertRaisesRegex(
            ProcessOwnershipError,
            "valid OwnedProcess owner.*missing",
        ):
            _ = browser.stopped

        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, object())
        with self.assertRaisesRegex(
            ProcessOwnershipError,
            r"valid OwnedProcess owner.*invalid \(object\)",
        ):
            _ = browser.stopped

        raw_process.poll.assert_not_called()

    async def test_owned_browser_stopped_uses_only_the_durable_owner(self) -> None:
        config = zd.Config(user_data_dir="/private/profile")
        browser = factory._OwnedZendriverBrowser(config)
        owner = Mock(spec=OwnedProcess)
        owner.poll.side_effect = [None, 17]
        _bind_browser_process_owner(browser, owner)

        self.assertFalse(browser.stopped)
        self.assertTrue(browser.stopped)

        self.assertEqual(owner.poll.call_count, 2)

    async def test_devtools_discovery_rejects_an_expired_absolute_deadline(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(
                TimeoutError,
                "shared deadline",
            ),
        ):
            await factory._wait_for_devtools_active_port_async(
                Mock(spec=OwnedProcess),
                Path(directory),
                deadline=Deadline(0.0),
            )

    async def test_devtools_late_file_read_cannot_publish_a_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active_port = Path(directory) / "DevToolsActivePort"
            active_port.write_text(
                "43123\n/devtools/browser/test\n",
                encoding="utf-8",
            )
            now = [100.0]

            def late_read(*_: object, **__: object) -> str:
                now[0] = 101.0
                return "43123\n/devtools/browser/test\n"

            with (
                patch(
                    "hbrowser.gallery.utils.deadline.time.monotonic",
                    side_effect=lambda: now[0],
                ),
                patch.object(Path, "read_text", side_effect=late_read),
                self.assertRaisesRegex(TimeoutError, "shared deadline"),
            ):
                await factory._wait_for_devtools_active_port_async(
                    Mock(spec=OwnedProcess),
                    Path(directory),
                    deadline=Deadline(101.0),
                )

    async def test_late_browser_client_construction_cleans_owner_without_publication(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="hbrowser-late-client-") as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            executable = root / "browser"
            executable.touch()
            profile = root / "profile"
            profile.mkdir()
            config = zd.Config(
                user_data_dir=profile,
                browser_executable_path=executable,
            )
            owner = Mock(spec=OwnedProcess)
            browser = Mock()
            now = [100.0]

            def late_construct(_: object) -> Mock:
                now[0] = 100.5
                return browser

            with (
                patch(
                    "hbrowser.gallery.utils.deadline.time.monotonic",
                    side_effect=lambda: now[0],
                ),
                patch.object(
                    factory,
                    "start_owned_browser_process",
                    return_value=owner,
                ),
                patch.object(
                    factory,
                    "_wait_for_devtools_active_port_async",
                    new=AsyncMock(return_value=43123),
                ),
                patch.object(
                    factory,
                    "_OwnedZendriverBrowser",
                    side_effect=late_construct,
                ),
                patch.object(factory, "_terminate_unbound_owner") as terminate,
                self.assertRaisesRegex(TimeoutError, "ownership publication"),
            ):
                await factory._construct_owned_browser_async(
                    config,
                    cleanup_paths=(process_module._PrivateDirectory.capture(profile),),
                    startup_deadline=Deadline(101.0),
                )

            terminate.assert_called_once_with(
                owner,
                deadline=unittest.mock.ANY,
            )

    def test_config_uses_an_explicit_hbrowser_owned_profile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-profile-test-") as profile:
            config = factory._build_config(
                True,
                None,
                profile,
                chrome_path="/browser",
            )

            self.assertTrue(config.uses_custom_data_dir)
            self.assertEqual(config.user_data_dir, profile)

    async def test_connect_existing_never_calls_zendriver_process_launcher(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-profile-test-") as profile:
            config = zd.Config(
                user_data_dir=profile,
                browser_executable_path="/browser",
            )
            config.host = "127.0.0.1"
            config.port = 9222
            config.autodiscover_targets = False
            browser = factory._OwnedZendriverBrowser(config)
            owner = Mock(spec=OwnedProcess)
            owner.poll.return_value = None
            setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)

            async def connected() -> bool:
                cast(Any, browser).info = SimpleNamespace(
                    webSocketDebuggerUrl="ws://127.0.0.1:9222/devtools/browser/test"
                )
                return True

            browser.test_connection = AsyncMock(side_effect=connected)  # type: ignore[method-assign]
            browser.update_targets = AsyncMock()  # type: ignore[method-assign]
            with (
                patch.object(zd.util, "_start_process") as zendriver_launcher,
                patch("zendriver.core.browser.Connection", return_value=Mock()),
            ):
                observed = await browser.start()

            self.assertIs(observed, browser)
            zendriver_launcher.assert_not_called()
            browser.update_targets.assert_awaited_once_with()
            self.assertIsNone(browser._process)

    async def test_prelaunch_discovers_port_and_binds_owner_without_global_hook(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="hbrowser-contract-test-") as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            executable = root / "browser"
            executable.touch()
            profile = root / "profile"
            profile.mkdir()
            config = zd.Config(
                user_data_dir=profile,
                browser_executable_path=executable,
            )
            owner = Mock(spec=OwnedProcess)
            browser = Mock()
            original_launcher = zd.util._start_process

            with (
                patch.object(
                    factory,
                    "start_owned_browser_process",
                    return_value=owner,
                ) as start_process,
                patch.object(
                    factory,
                    "_wait_for_devtools_active_port_async",
                    new=AsyncMock(return_value=43123),
                ),
                patch.object(
                    factory,
                    "_OwnedZendriverBrowser",
                    return_value=browser,
                ),
            ):
                observed = await factory._construct_owned_browser_async(
                    config,
                    cleanup_paths=(process_module._PrivateDirectory.capture(profile),),
                    startup_deadline=Deadline.after(25.0),
                )

            self.assertIs(observed, browser)
            self.assertIs(zd.util._start_process, original_launcher)
            self.assertEqual(config.host, "127.0.0.1")
            self.assertEqual(config.port, 43123)
            parameters = start_process.call_args.args[1]
            self.assertIn("--remote-debugging-port=0", parameters)
            self.assertEqual(parameters[-1], "about:blank")
            self.assertIs(
                getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE),
                owner,
            )


class _Target:
    type_ = "page"
    url = "about:blank"


class _LateDeadline:
    def remaining(self) -> float:
        return 1.0

    @property
    def expired(self) -> bool:
        return True


class _Browser:
    def __init__(
        self,
        main_tab: zd.Tab | None,
        tabs: list[object] | None = None,
        returncode: int | None = None,
    ) -> None:
        self._main_tab = main_tab
        self.tabs = tabs or []
        self.targets = self.tabs
        self.process_owner = _bind_browser_process_owner(
            self,
            returncode=returncode,
        )

    @property
    def main_tab(self) -> zd.Tab | None:
        return self._main_tab

    @property
    def stopped(self) -> bool:
        return self.process_owner.poll() is not None


class PostCreateSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_geolocation_failure_is_terminal_and_skips_proxy_probe(
        self,
    ) -> None:
        page = SimpleNamespace(
            send=AsyncMock(side_effect=RuntimeError("sensitive CDP payload")),
        )

        with (
            patch.object(factory, "verify_proxy_ip", new=AsyncMock()) as verify,
            self.assertRaises(BrowserMutationOutcomeUnknownError) as raised,
        ):
            await factory._post_create_setup(
                cast(zd.Browser, Mock()),
                cast(zd.Tab, page),
                use_tor=True,
                deadline=Deadline.after(1),
            )

        page.send.assert_awaited_once()
        verify.assert_not_awaited()
        self.assertNotIn("sensitive CDP payload", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)


class _OwnedBrowser:
    """Hashable browser fake for durable ownership/registry tests."""

    def __init__(self) -> None:
        self.connection = None
        self.targets: list[object] = []
        self._tor_process = None
        self.stop = AsyncMock()
        self.process_owner = _bind_browser_process_owner(self)


class BrowserProcessShutdownTests(unittest.TestCase):
    def test_owned_browser_delegates_to_the_bounded_shutdown_state_machine(
        self,
    ) -> None:
        owner = Mock(spec=OwnedProcess)
        browser = SimpleNamespace()
        _bind_browser_process_owner(browser, owner)

        factory._terminate_browser_process(browser, deadline=Deadline.after(15))

        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=unittest.mock.ANY,
        )
        owner.terminate.assert_not_called()
        owner.kill.assert_not_called()
        self.assertIs(
            getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE),
            owner,
        )

    def test_failed_private_release_retains_exact_browser_owner(self) -> None:
        owner = Mock(spec=OwnedProcess)
        owner.shutdown.side_effect = PermissionError("profile locked")
        browser = SimpleNamespace()
        _bind_browser_process_owner(browser, owner)

        with self.assertRaises(ProcessOwnershipError) as raised:
            factory._terminate_browser_process(
                browser,
                deadline=Deadline.after(15),
            )

        self.assertIsInstance(raised.exception.__cause__, PermissionError)
        self.assertIs(
            getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE),
            owner,
        )
        owner.terminate.assert_not_called()
        owner.kill.assert_not_called()

    def test_missing_owner_fails_closed_without_using_raw_process(self) -> None:
        raw_process = Mock()
        browser = SimpleNamespace(_process=raw_process)

        with self.assertRaisesRegex(
            ProcessOwnershipError,
            "valid OwnedProcess owner.*missing",
        ):
            factory._terminate_browser_process(
                browser,
                deadline=Deadline.after(15),
            )

        raw_process.terminate.assert_not_called()
        raw_process.wait.assert_not_called()
        raw_process.kill.assert_not_called()

    def test_corrupt_owner_fails_closed_without_using_raw_process(self) -> None:
        raw_process = Mock()
        browser = SimpleNamespace(_process=raw_process)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, object())

        with self.assertRaisesRegex(
            ProcessOwnershipError,
            r"valid OwnedProcess owner.*invalid \(object\)",
        ):
            factory._terminate_browser_process(
                browser,
                deadline=Deadline.after(15),
            )

        raw_process.terminate.assert_not_called()
        raw_process.wait.assert_not_called()
        raw_process.kill.assert_not_called()


class ExpiredShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_deadline_does_not_invoke_browser_stop(self) -> None:
        browser = _OwnedBrowser()
        retirement = _begin_zendriver_retirement(browser)
        runner = AsyncMock()

        errors = await factory._ensure_browser_cleanup(
            browser,
            retirement,
            allow_fallback=True,
            run_blocking_cleanup=runner,
            deadline=Deadline(0.0),
        )

        browser.stop.assert_not_awaited()
        runner.assert_not_awaited()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ProcessOwnershipError)

    async def test_expired_deadline_does_not_invoke_connection_close(self) -> None:
        close = AsyncMock()
        connection = SimpleNamespace(websocket=object(), aclose=close)

        closed, errors = await factory._close_and_wait_for_connection(
            connection,
            deadline=Deadline(0.0),
        )

        self.assertFalse(closed)
        close.assert_not_awaited()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ProcessOwnershipError)


class WaitForMainTabTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_timeout_above_target_discovery_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 5\]"):
            await factory._wait_for_main_tab(
                cast(zd.Browser, _Browser(None)),
                timeout=5.01,
            )

    async def test_returns_immediate_main_tab_without_sleeping(self) -> None:
        page = _tab()
        browser = _Browser(page)

        with patch(
            "hbrowser.gallery.browser.factory.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)
        sleep.assert_not_awaited()

    async def test_waits_for_delayed_target_created_event(self) -> None:
        page = _tab()
        browser = _Browser(None, tabs=[_Target()])

        async def expose_tab(_: float) -> None:
            browser.targets.append(page)

        logger = Mock()
        with (
            patch(
                "hbrowser.gallery.browser.factory.asyncio.sleep",
                new=AsyncMock(side_effect=expose_tab),
            ) as sleep,
            patch.object(factory, "logger", logger),
        ):
            result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)
        sleep.assert_awaited_once()
        logger.warning.assert_called_once()
        logger.debug.assert_called_once_with(
            "Browser main tab became available after startup delay"
        )
        logger.info.assert_not_called()

    async def test_recovers_tab_when_main_tab_is_blocked_by_connection(self) -> None:
        page = _tab()
        browser = _Browser(None, tabs=[_Target(), page])

        result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)

    async def test_timeout_includes_browser_owner_and_target_state(self) -> None:
        browser = _Browser(None, tabs=[_Target()])

        with self.assertRaises(RuntimeError) as raised:
            await factory._wait_for_main_tab(
                cast(zd.Browser, browser),
                timeout=0,
            )

        message = str(raised.exception)
        self.assertIn("within 0.0 seconds", message)
        self.assertIn("stopped=False", message)
        self.assertIn("process_owner_returncode=None", message)
        self.assertIn("_Target(type='page', url='about:blank')", message)

    async def test_fails_immediately_when_browser_process_exits(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page], returncode=17)

        with (
            patch(
                "hbrowser.gallery.browser.factory.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "stopped=True, process_owner_returncode=17",
            ),
        ):
            await factory._wait_for_main_tab(cast(zd.Browser, browser))

        sleep.assert_not_awaited()

    async def test_rejects_non_page_tab_returned_by_main_tab(self) -> None:
        worker = _tab("worker-1", type_="service_worker")
        browser = _Browser(worker, tabs=[worker])

        with self.assertRaises(RuntimeError):
            await factory._wait_for_main_tab(
                cast(zd.Browser, browser),
                timeout=0,
            )


class CreateBrowserCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_handoff_never_recaptures_a_replaced_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            extension = parent / "extension"
            extension.mkdir()
            (extension / "credentials").write_text("synthetic", encoding="utf-8")
            guard = process_module._PrivateDirectory.capture(extension)
            original = parent / "original"
            replacement = parent / "replacement"
            replacement.mkdir()
            (replacement / "keep").write_text("unrelated", encoding="utf-8")
            extension.rename(original)
            replacement.rename(extension)
            try:
                with (
                    patch.object(
                        factory, "_configured_chrome_executable", return_value="/chrome"
                    ),
                    patch.object(factory, "configure_proxy", return_value=guard),
                    patch.object(
                        factory, "_construct_owned_browser_async", new=AsyncMock()
                    ) as construct,
                    self.assertRaises(ProcessOwnershipError),
                ):
                    await factory._create_browser(True)
                construct.assert_not_awaited()
                self.assertEqual((extension / "keep").read_text(), "unrelated")
                self.assertEqual((original / "credentials").read_text(), "synthetic")
                self.assertEqual(
                    (guard.device, guard.inode),
                    (original.stat().st_dev, original.stat().st_ino),
                )
            finally:
                extension.rename(replacement)
                original.rename(extension)
                guard.remove(deadline=Deadline.after(5).expires_at)

    async def test_cancelled_failed_launch_preserves_owned_private_paths(self) -> None:
        for settlement in (1, 2):
            with self.subTest(settlement=settlement):
                await self._assert_failed_launch_retains_paths(settlement)

    async def _assert_failed_launch_retains_paths(self, settlement: int) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            profile, extension = root / "profile", root / "extension"
            profile.mkdir()
            extension.mkdir()
            (extension / "credentials").write_text("synthetic", encoding="utf-8")
            started, release = threading.Event(), threading.Event()
            failure = ProcessOwnershipError("startup cleanup could not prove exit")

            def fail_launch(*_: object, **__: object) -> None:
                started.set()
                if not release.wait(2):
                    raise AssertionError("test launch barrier was not released")
                raise failure

            real_settle = factory._settle_owned_startup_task
            settlements = 0

            async def cancel_at_settlement(
                task: asyncio.Task[Any],
                cancellation: asyncio.CancelledError | None = None,
            ) -> tuple[Any, asyncio.CancelledError | None]:
                nonlocal settlements
                settlements += 1
                if settlements == settlement:
                    waiter = asyncio.current_task()
                    assert waiter is not None

                    def cancel_waiter(completed: asyncio.Task[Any]) -> None:
                        if settlement == 1:
                            self.assertIs(completed.exception(), failure)
                        else:
                            self.assertIsNone(completed.exception())
                        waiter.cancel()

                    # Register before shield's done callback: either the real
                    # launch thread has failed, or independent cleanup finished,
                    # but its waiting caller has not resumed yet.
                    task.add_done_callback(cancel_waiter)
                return await real_settle(task, cancellation)

            with (
                patch.object(
                    factory, "_configured_chrome_executable", return_value="/chrome"
                ),
                patch.object(factory, "should_use_tor", return_value=False),
                patch.object(
                    factory,
                    "configure_proxy",
                    return_value=process_module._PrivateDirectory.capture(
                        Path(extension)
                    ),
                ),
                patch.object(
                    factory,
                    "tempfile",
                    SimpleNamespace(mkdtemp=lambda **_: str(profile)),
                ),
                patch.object(factory, "_build_config", return_value=object()),
                patch.object(
                    factory,
                    "_prepare_owned_browser_launch",
                    return_value=(Path("/chrome"), []),
                ),
                patch.object(
                    factory, "start_owned_browser_process", side_effect=fail_launch
                ),
                patch.object(
                    factory,
                    "_settle_owned_startup_task",
                    side_effect=cancel_at_settlement,
                ),
            ):
                creation = asyncio.create_task(factory._create_browser(True))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 1))
                    release.set()
                    try:
                        await creation
                    except BaseException as error:
                        observed_error = error
                    else:
                        self.fail("failed launch was published")
                finally:
                    release.set()
                    if not creation.done():
                        creation.cancel()
                        await asyncio.gather(creation, return_exceptions=True)
            self.assertTrue(profile.is_dir(), "unresolved owner lost its profile")
            self.assertEqual((extension / "credentials").read_text(), "synthetic")
            self.assertIs(observed_error, failure)
            self.assertTrue(any("cancel" in note.lower() for note in failure.__notes__))

    async def test_settled_nonownership_failure_retains_cancellation(self) -> None:
        async def fail() -> None:
            raise ValueError("startup was rejected before owning resources")

        task = asyncio.create_task(fail())
        await asyncio.gather(task, return_exceptions=True)
        cancellation = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError) as raised:
            await factory._settle_owned_startup_task(task, cancellation)
        self.assertIs(raised.exception, cancellation)
        self.assertTrue(any("ValueError" in note for note in cancellation.__notes__))

    async def test_uncertain_chrome_constructor_retains_transferred_private_paths(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            _release_mock_owner_directories(),
        ):
            profile = Path(directory) / "profile"
            extension = Path(directory) / "extension"
            profile.mkdir()
            extension.mkdir()
            failure = ProcessOwnershipError("Chrome ownership unresolved")
            extension_guard = process_module._PrivateDirectory.capture(extension)
            with (
                patch.object(
                    factory, "_configured_chrome_executable", return_value="/chrome"
                ),
                patch.object(factory, "should_use_tor", return_value=False),
                patch.object(
                    factory,
                    "configure_proxy",
                    return_value=extension_guard,
                ),
                patch.object(tempfile, "mkdtemp", return_value=str(profile)),
                patch.object(factory, "_build_config", return_value=object()),
                patch.object(
                    factory,
                    "_construct_owned_browser_async",
                    new=AsyncMock(side_effect=failure),
                ) as construct,
                patch.object(
                    factory,
                    "_cleanup_unowned_private_paths",
                    new=AsyncMock(return_value=[]),
                ) as cleanup,
                self.assertRaises(ProcessOwnershipError) as raised,
            ):
                await factory._create_browser(True)
            self.assertIs(raised.exception, failure)
            self.assertIs(
                construct.call_args.kwargs["cleanup_paths"][-1], extension_guard
            )
            self.assertEqual(
                tuple(
                    guard.path for guard in construct.call_args.kwargs["cleanup_paths"]
                ),
                (profile, extension),
            )
            cleanup.assert_awaited_once_with((), deadline=unittest.mock.ANY)
            self.assertTrue(profile.is_dir())
            self.assertTrue(extension.is_dir())

    async def test_failed_discovery_preserves_replacement_profile(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            profile = root / "profile"
            profile.mkdir()
            owner = Mock(spec=OwnedProcess)
            captured: list[process_module._PrivateDirectory] = []

            def start(*_args: object, **kwargs: Any) -> Mock:
                captured.extend(kwargs["cleanup_paths"])
                return owner

            def shutdown(**_kwargs: object) -> None:
                for guard in captured:
                    guard.path.rename(root / "claimed")
                    guard.path.mkdir()
                    (guard.path / "foreign").write_text("preserve", encoding="utf-8")

            owner.shutdown.side_effect = shutdown
            try:
                with (
                    patch.object(
                        factory, "_configured_chrome_executable", return_value="/chrome"
                    ),
                    patch.object(factory, "configure_proxy", return_value=None),
                    patch.object(factory, "should_use_tor", return_value=False),
                    patch.object(tempfile, "mkdtemp", return_value=str(profile)),
                    patch.object(
                        factory,
                        "_build_config",
                        return_value=SimpleNamespace(user_data_dir=str(profile)),
                    ),
                    patch.object(
                        factory,
                        "_prepare_owned_browser_launch",
                        return_value=(Path("/chrome"), []),
                    ),
                    patch.object(
                        factory, "start_owned_browser_process", side_effect=start
                    ),
                    patch.object(
                        factory,
                        "_wait_for_devtools_active_port_async",
                        new=AsyncMock(side_effect=RuntimeError("discovery failed")),
                    ),
                    self.assertRaises(ProcessOwnershipError),
                ):
                    await factory._create_browser(True)
                self.assertEqual((profile / "foreign").read_text(), "preserve")
                self.assertEqual(captured[0].path, profile)
            finally:
                # Restore only the original generation after the preservation oracle.
                if (root / "claimed").exists():
                    profile.rename(root / "replacement")
                    (root / "claimed").rename(profile)

    async def test_installer_wait_preserves_replacement_staging(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            chrome = root / "chrome"
            chrome.write_text("synthetic", encoding="utf-8")
            owner = Mock(spec=OwnedProcess)
            owner.poll.return_value = 0

            def start(
                _executable: object, parameters: tuple[str, ...], **kwargs: Any
            ) -> Mock:
                guard = kwargs["cleanup_paths"][0]

                def wait(**_kwargs: object) -> int:
                    guard.path.rename(root / "claimed")
                    guard.path.mkdir()
                    (guard.path / "foreign").write_text("preserve", encoding="utf-8")
                    return 0

                owner.wait.side_effect = wait
                Path(parameters[2]).write_text(
                    json.dumps(
                        {
                            "schema": 1,
                            "nonce": parameters[3],
                            "chrome": str(chrome),
                            "version": "synthetic",
                        }
                    ),
                    encoding="utf-8",
                )
                return owner

            try:
                with (
                    patch.object(
                        factory,
                        "create_chrome_install_staging_root",
                        return_value=staging,
                    ),
                    patch.object(factory, "start_owned_process", side_effect=start),
                    self.assertRaises(ProcessOwnershipError),
                ):
                    await factory._install_chrome_in_owned_worker(
                        work_deadline=Deadline.after(5),
                        cleanup_deadline=Deadline.after(10),
                    )
                self.assertEqual((staging / "foreign").read_text(), "preserve")
                owner.shutdown.assert_not_called()
            finally:
                # Restore only the original generation after the preservation oracle.
                if (root / "claimed").exists():
                    staging.rename(root / "replacement")
                    (root / "claimed").rename(staging)

    async def test_installer_ownership_failure_cleans_untransferred_proxy_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extension = Path(directory) / "proxy-extension"
            extension.mkdir()
            (extension / "credentials").write_text("synthetic", encoding="utf-8")
            failure = ProcessOwnershipError("installer ownership unresolved")
            with (
                patch.object(
                    factory, "_configured_chrome_executable", return_value=None
                ),
                patch.object(factory, "should_use_tor", return_value=False),
                patch.object(
                    factory,
                    "configure_proxy",
                    return_value=process_module._PrivateDirectory.capture(
                        Path(extension)
                    ),
                ),
                patch.object(
                    factory,
                    "_install_chrome_in_owned_worker",
                    new=AsyncMock(side_effect=failure),
                ),
                patch.object(factory, "_construct_owned_browser_async") as construct,
                self.assertRaises(ProcessOwnershipError) as raised,
            ):
                await factory._create_browser(True)
            self.assertIs(raised.exception, failure)
            construct.assert_not_called()
            self.assertFalse(extension.exists())

    async def test_stop_rejects_missing_owner_before_zendriver_cleanup(self) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )

        with self.assertRaisesRegex(
            ProcessOwnershipError,
            "valid OwnedProcess owner.*missing",
        ):
            await factory.stop_browser(browser)

        browser.stop.assert_not_awaited()

    async def test_chrome_install_worker_receipt_is_nonce_bound_and_reaped(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory(
                prefix="hbrowser-install-receipt-"
            ) as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            receipt_directory = root / "receipt"
            receipt_directory.mkdir()
            staging_directory = root / "staging"
            staging_directory.mkdir()
            chrome = root / "chrome"
            chrome.write_text("binary", encoding="utf-8")
            owner = Mock(spec=OwnedProcess)
            owner.poll.return_value = 0
            owner.wait.return_value = 0

            def start_worker(
                _: object,
                parameters: tuple[str, ...],
                **__: object,
            ) -> Mock:
                receipt_path = Path(parameters[2])
                nonce = parameters[3]
                receipt_path.write_text(
                    json.dumps(
                        {
                            "schema": 1,
                            "nonce": nonce,
                            "chrome": str(chrome),
                            "version": "test-version",
                        }
                    ),
                    encoding="utf-8",
                )
                return owner

            with (
                patch(
                    "hbrowser.gallery.browser.factory.tempfile.mkdtemp",
                    return_value=str(receipt_directory),
                ),
                patch.object(
                    factory,
                    "create_chrome_install_staging_root",
                    return_value=staging_directory,
                ),
                patch.object(
                    factory,
                    "start_owned_process",
                    side_effect=start_worker,
                ) as start,
                patch.object(
                    factory,
                    "_cleanup_unowned_private_paths",
                    new=AsyncMock(return_value=[]),
                ) as cleanup_receipt,
            ):
                work_deadline = Deadline.after(1)
                cleanup_deadline = Deadline.after(2)
                result = await factory._install_chrome_in_owned_worker(
                    work_deadline=work_deadline,
                    cleanup_deadline=cleanup_deadline,
                )

        self.assertEqual(result.chrome, str(chrome))
        self.assertEqual(result.version, "test-version")
        owner.wait.assert_called_once_with(timeout=unittest.mock.ANY)
        owner.shutdown.assert_not_called()
        self.assertEqual(
            start.call_args.kwargs["deadline"], cleanup_deadline.expires_at
        )
        self.assertEqual(
            float(start.call_args.args[1][-2]),
            work_deadline.expires_at,
        )
        self.assertEqual(
            tuple(guard.path for guard in start.call_args.kwargs["cleanup_paths"]),
            (staging_directory,),
        )
        self.assertEqual(
            tuple(guard.path for guard in cleanup_receipt.call_args.args[0]),
            (receipt_directory, staging_directory),
        )
        self.assertEqual(cleanup_receipt.call_args.kwargs["deadline"], cleanup_deadline)

    async def test_chrome_install_cancellation_reaps_worker_before_return(self) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="hbrowser-install-cancel-") as directory,
            _release_mock_owner_directories(),
        ):
            receipt_directory = Path(directory) / "receipt"
            receipt_directory.mkdir()
            staging_directory = Path(directory) / "staging"
            staging_directory.mkdir()
            owner = Mock(spec=OwnedProcess)
            owner.poll.return_value = None
            owner.shutdown.return_value = 0

            with (
                patch(
                    "hbrowser.gallery.browser.factory.tempfile.mkdtemp",
                    return_value=str(receipt_directory),
                ),
                patch.object(
                    factory,
                    "create_chrome_install_staging_root",
                    return_value=staging_directory,
                ),
                patch.object(factory, "start_owned_process", return_value=owner),
                patch.object(
                    factory,
                    "_cleanup_unowned_private_paths",
                    new=AsyncMock(return_value=[]),
                ),
            ):
                cleanup_deadline = Deadline.after(2)
                install = asyncio.create_task(
                    factory._install_chrome_in_owned_worker(
                        work_deadline=Deadline.after(1),
                        cleanup_deadline=cleanup_deadline,
                    )
                )
                while not owner.poll.called:
                    await asyncio.sleep(0)
                install.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await install

        owner.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=unittest.mock.ANY,
            kill_timeout=unittest.mock.ANY,
            cleanup_timeout=unittest.mock.ANY,
            deadline=cleanup_deadline.expires_at,
        )

    @unittest.skipUnless(os.name == "posix", "POSIX process-group regression")
    async def test_interrupted_chrome_worker_cannot_leave_staging_orphan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-install-interrupt-"
        ) as directory:
            root = Path(directory)
            receipt_directory = root / "receipt"
            receipt_directory.mkdir()
            staging_directory = root / "staging"
            staging_directory.mkdir()
            worker = root / "blocked-installer.py"
            worker.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys, time\n"
                "staging = pathlib.Path(sys.argv[1])\n"
                "(staging / 'worker-started').write_text('started')\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            worker.chmod(0o700)

            def worker_command(
                _: Path,
                __: str,
                *,
                work_deadline: Deadline,
                staging_directory: Path,
            ) -> tuple[str, tuple[str, ...]]:
                self.assertFalse(work_deadline.expired)
                return str(worker), (str(staging_directory),)

            with (
                patch(
                    "hbrowser.gallery.browser.factory.tempfile.mkdtemp",
                    return_value=str(receipt_directory),
                ),
                patch.object(
                    factory,
                    "create_chrome_install_staging_root",
                    return_value=staging_directory,
                ),
                patch.object(
                    factory,
                    "_chrome_install_worker_command",
                    side_effect=worker_command,
                ),
            ):
                install = asyncio.create_task(
                    factory._install_chrome_in_owned_worker(
                        work_deadline=Deadline.after(5),
                        cleanup_deadline=Deadline.after(10),
                    )
                )
                started_marker = staging_directory / "worker-started"
                marker_deadline = asyncio.get_running_loop().time() + 3
                while not started_marker.exists():
                    if asyncio.get_running_loop().time() >= marker_deadline:
                        self.fail("Chrome installer fixture did not start")
                    await asyncio.sleep(0.01)

                install.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await install

            self.assertFalse(staging_directory.exists())
            self.assertFalse(receipt_directory.exists())

    async def test_private_path_cleanup_cancellation_waits_for_owned_worker(
        self,
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        guard = Mock()

        def remove(*, deadline: float) -> None:
            self.assertGreater(deadline, time.monotonic())
            started.set()
            self.assertTrue(release.wait(timeout=1))

        guard.remove.side_effect = remove
        cleanup = asyncio.create_task(
            factory._cleanup_unowned_private_paths(
                (guard,),
                deadline=Deadline.after(1),
            )
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 0.5))
        cleanup.cancel()
        await asyncio.sleep(0)
        self.assertFalse(cleanup.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await cleanup

        guard.remove.assert_called_once_with(deadline=unittest.mock.ANY)

    async def test_parallel_private_path_cleanup_shares_one_absolute_deadline(
        self,
    ) -> None:
        guards = (Mock(), Mock())
        observed_deadlines: list[float] = []
        for guard in guards:
            guard.remove.side_effect = lambda *, deadline: observed_deadlines.append(
                deadline
            )

        errors = await factory._cleanup_unowned_private_paths(
            guards,
            deadline=Deadline.after(1),
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(observed_deadlines), 2)
        self.assertEqual(observed_deadlines[0], observed_deadlines[1])

    async def test_late_browser_start_receipt_retires_generation(self) -> None:
        browser = SimpleNamespace(start=AsyncMock(return_value=object()))
        lifecycle = Mock()

        with (
            patch.object(
                factory,
                "_validate_zendriver_operation",
                return_value=(None, browser, None, lifecycle),
            ),
            self.assertRaisesRegex(TimeoutError, "completed after"),
        ):
            await factory._await_browser_start(
                cast(zd.Browser, browser),
                deadline=cast(Deadline, _LateDeadline()),
            )

        lifecycle.begin_retirement_for_operation.assert_called_once()

    async def test_success_completes_post_setup_before_return(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page])
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        events: list[str] = []

        async def post_setup(*_: object, **__: object) -> None:
            events.append("post-setup")

        with (
            _release_mock_owner_directories(),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ),
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(factory, "_register_browser_atexit") as register_atexit,
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(side_effect=post_setup),
            ),
        ):
            result = await factory.create_browser()

        self.assertEqual(result, (browser, page))
        self.assertEqual(events, ["post-setup"])
        register_atexit.assert_called_once_with(browser)

    async def test_direct_startup_checks_geolocation_completion_deadline(self) -> None:
        for offset in (-0.001, 0.0, 0.001):
            with self.subTest(completion_offset=offset):
                page = _tab()
                browser = _Browser(page, tabs=[page])
                page.browser = cast(zd.Browser, browser)
                clock = [time.monotonic()]
                start = AsyncMock(return_value=browser)

                async def acknowledge(
                    *_: object,
                    startup: AsyncMock = start,
                    current_clock: list[float] = clock,
                    completion_offset: float = offset,
                ) -> None:
                    assert startup.await_args is not None
                    deadline = startup.await_args.kwargs["deadline"]
                    # Deliver a successful CDP acknowledgement in the same loop
                    # turn that advances the clock to the completion boundary.
                    current_clock[0] = deadline.expires_at + completion_offset

                with (
                    tempfile.TemporaryDirectory() as profile,
                    _release_mock_owner_directories(),
                    patch.object(factory, "should_use_tor", return_value=False),
                    patch.object(factory, "configure_proxy", return_value=None),
                    patch.object(
                        factory, "_configured_chrome_executable", return_value="/chrome"
                    ),
                    patch.object(
                        factory,
                        "tempfile",
                        SimpleNamespace(mkdtemp=Mock(return_value=profile)),
                    ),
                    patch.object(factory, "_build_config", return_value=object()),
                    patch.object(
                        factory,
                        "_construct_owned_browser_async",
                        new=AsyncMock(return_value=browser),
                    ),
                    patch.object(factory, "_register_browser_atexit"),
                    patch.object(factory, "_await_browser_start", new=start),
                    patch.object(page, "send", new=AsyncMock(side_effect=acknowledge)),
                    patch.object(factory, "stop_browser", new=AsyncMock()) as stop,
                    patch.object(factory, "logger") as logger,
                    patch(
                        "hbrowser.gallery.utils.deadline.time.monotonic",
                        side_effect=lambda current_clock=clock: current_clock[0],
                    ),
                ):
                    if offset < 0:
                        self.assertEqual(
                            await factory.create_browser(),
                            (browser, page),
                        )
                        stop.assert_not_awaited()
                        logger.info.assert_any_call(
                            "Browser ready (%s, %s)", "headless", "direct"
                        )
                    else:
                        with self.assertRaises(BrowserMutationOutcomeUnknownError):
                            await factory.create_browser()
                        stop.assert_awaited_once_with(browser, unittest.mock.ANY)
                        assert stop.await_args is not None
                        assert start.await_args is not None
                        self.assertGreater(
                            stop.await_args.args[1].expires_at,
                            start.await_args.kwargs["deadline"].expires_at,
                        )
                        self.assertNotIn(
                            call("Browser ready (%s, %s)", "headless", "direct"),
                            logger.info.call_args_list,
                        )
                        with self.assertRaises(ZendriverOwnerRetiredError):
                            await wait_for_zendriver(
                                asyncio.sleep(0), timeout=1, owner=page
                            )

    async def test_only_the_selected_proxy_route_can_start_tor(self) -> None:
        from hbrowser.gallery.browser import proxy as proxy_module

        cases = (
            (True, None, False),
            (True, "1", False),
            (True, "0", False),
            (False, None, True),
            (False, "0", False),
        )
        for residential, use_tor_setting, expected_tor in cases:
            with self.subTest(residential=residential, use_tor=use_tor_setting):
                page = _tab()
                browser = _Browser(page, tabs=[page])
                browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
                environment: dict[str, str] = {}
                if residential:
                    environment.update(
                        RP_USERNAME="synthetic",
                        RP_PASSWORD="probe",
                        RP_DNS="proxy:8080",
                    )
                if use_tor_setting is not None:
                    environment["USE_TOR"] = use_tor_setting
                tor_process = Mock(spec=OwnedProcess)
                with (
                    tempfile.TemporaryDirectory() as extension,
                    tempfile.TemporaryDirectory() as profile,
                    _release_mock_owner_directories(),
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(
                        tor_module, "_find_tor_binary", return_value="/tor"
                    ) as find_tor,
                    patch.object(
                        factory, "_configured_chrome_executable", return_value="/chrome"
                    ),
                    patch.object(
                        proxy_module,
                        "_create_proxy_extension",
                        return_value=process_module._PrivateDirectory.capture(
                            Path(extension)
                        ),
                    ),
                    patch.object(
                        factory,
                        "tempfile",
                        SimpleNamespace(mkdtemp=Mock(return_value=profile)),
                    ),
                    patch.object(
                        factory, "find_available_port", return_value=9150
                    ) as find_port,
                    patch.object(
                        factory,
                        "start_tor_with_retry",
                        return_value=tor_process,
                        side_effect=(
                            None
                            if expected_tor
                            else RuntimeError("unused Tor bootstrap must not run")
                        ),
                    ) as start_tor,
                    patch.object(
                        factory, "_build_config", return_value=object()
                    ) as build_config,
                    patch.object(
                        factory,
                        "_construct_owned_browser_async",
                        new=AsyncMock(return_value=browser),
                    ),
                    patch.object(factory, "_register_browser_atexit"),
                    patch.object(
                        factory, "_post_create_setup", new=AsyncMock()
                    ) as setup,
                ):
                    result = await factory.create_browser()
                self.assertEqual(result, (browser, page))
                expected_extension = extension if residential else None
                build_config.assert_called_once_with(
                    True,
                    expected_extension,
                    profile,
                    expected_tor,
                    9150 if expected_tor else None,
                    "/chrome",
                )
                setup.assert_awaited_once_with(
                    browser,
                    page,
                    expected_tor,
                    proxy_extension=expected_extension,
                    deadline=unittest.mock.ANY,
                )
                self.assertIs(
                    getattr(browser, "_tor_process", None),
                    tor_process if expected_tor else None,
                )
                if residential:
                    find_tor.assert_not_called()
                if expected_tor:
                    start_tor.assert_called_once_with(9150, deadline=unittest.mock.ANY)
                    find_port.assert_called_once()
                else:
                    start_tor.assert_not_called()
                    find_port.assert_not_called()

    async def test_invalid_residential_proxy_rejects_before_tor_admission(self) -> None:
        for environment in (
            {"RP_DNS": "proxy:8080"},
            {
                "RP_DNS": "proxy:invalid",
                "RP_USERNAME": "probe",
                "RP_PASSWORD": "synthetic",
            },
        ):
            with self.subTest(environment_keys=tuple(environment)):
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(
                        factory, "_configured_chrome_executable", return_value="/chrome"
                    ),
                    patch.object(
                        tor_module, "_find_tor_binary", return_value="/tor"
                    ) as find_tor,
                    patch.object(factory, "find_available_port", return_value=9150),
                    patch.object(
                        factory,
                        "start_tor_with_retry",
                        side_effect=RuntimeError("unused Tor bootstrap must not run"),
                    ) as start_tor,
                    patch.object(
                        factory, "_construct_owned_browser_async", new=AsyncMock()
                    ) as construct,
                    self.assertRaises(ValueError),
                ):
                    await factory.create_browser()
                find_tor.assert_not_called()
                start_tor.assert_not_called()
                construct.assert_not_awaited()

    async def test_residential_proxy_is_verified_before_browser_publication(
        self,
    ) -> None:
        from hbrowser.gallery.browser.proxy import ProxyVerificationError

        for verification_error in (None, ProxyVerificationError("proxy not applied")):
            with self.subTest(verification_error=verification_error):
                page = _tab()
                browser = _Browser(page, tabs=[page])
                browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
                page.browser = cast(zd.Browser, browser)
                verify = AsyncMock(side_effect=verification_error)
                with (
                    tempfile.TemporaryDirectory() as extension,
                    _release_mock_owner_directories(),
                    patch.object(factory, "should_use_tor", return_value=False),
                    patch.object(
                        factory,
                        "configure_proxy",
                        return_value=process_module._PrivateDirectory.capture(
                            Path(extension)
                        ),
                    ),
                    patch.object(
                        factory,
                        "_install_chrome_in_owned_worker",
                        new=AsyncMock(
                            return_value=SimpleNamespace(chrome="/test/chrome")
                        ),
                    ),
                    patch.object(factory, "_build_config", return_value=object()),
                    patch.object(
                        factory,
                        "_construct_owned_browser_async",
                        new=AsyncMock(return_value=browser),
                    ),
                    patch.object(factory, "_register_browser_atexit"),
                    patch.object(page, "send", new=AsyncMock()),
                    patch.object(
                        factory, "verify_proxy_ip", new=AsyncMock()
                    ) as ip_probe,
                    patch.object(factory, "verify_residential_proxy", new=verify),
                    patch.object(factory, "stop_browser", new=AsyncMock()) as stop,
                ):
                    if verification_error is None:
                        result = await factory.create_browser()
                        self.assertEqual(result, (browser, page))
                        stop.assert_not_awaited()
                    else:
                        with self.assertRaises(ProxyVerificationError) as raised:
                            await factory.create_browser()
                        self.assertIs(raised.exception, verification_error)
                        stop.assert_awaited_once_with(browser, unittest.mock.ANY)

                verify.assert_awaited_once_with(
                    browser, extension, deadline=unittest.mock.ANY
                )
                if verification_error is None:
                    ip_probe.assert_awaited_once_with(
                        browser, page, deadline=unittest.mock.ANY
                    )
                else:
                    ip_probe.assert_not_awaited()

    async def test_main_tab_failure_stops_browser_without_post_setup(self) -> None:
        browser = _Browser(None)
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        failure = RuntimeError("main tab unavailable")

        with (
            _release_mock_owner_directories(),
            patch.object(factory, "_configured_chrome_executable", return_value=None),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ) as install_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ) as construct_browser,
            patch.object(factory, "_register_browser_atexit"),
            patch.object(
                factory,
                "_wait_for_main_tab",
                new=AsyncMock(side_effect=failure),
            ),
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(),
            ) as post_setup,
            patch.object(
                factory,
                "stop_browser",
                new=AsyncMock(),
            ) as stop_browser,
            self.assertRaisesRegex(RuntimeError, "main tab unavailable"),
        ):
            await factory.create_browser()

        stop_browser.assert_awaited_once_with(browser, unittest.mock.ANY)
        work_deadline = install_chrome.call_args.kwargs["work_deadline"]
        self.assertIs(
            construct_browser.call_args.kwargs["startup_deadline"],
            work_deadline,
        )
        assert stop_browser.await_args is not None
        cleanup_deadline = stop_browser.await_args.args[1]
        self.assertAlmostEqual(
            cleanup_deadline.expires_at - work_deadline.expires_at,
            factory._BROWSER_STARTUP_CLEANUP_RESERVE_SECONDS,
        )
        post_setup.assert_not_awaited()

    async def test_tor_start_cancellation_waits_for_result_then_reaps(self) -> None:
        tor_started = threading.Event()
        allow_tor_start = threading.Event()
        tor_process = Mock()

        def delayed_tor_start(_: int, **__: object) -> Mock:
            tor_started.set()
            allow_tor_start.wait(timeout=1)
            return tor_process

        with (
            patch.object(factory, "should_use_tor", return_value=True),
            patch.object(factory, "find_available_port", return_value=9150),
            patch.object(
                factory,
                "start_tor_with_retry",
                side_effect=delayed_tor_start,
            ) as start_tor,
            patch.object(factory, "terminate_tor_process") as terminate_tor,
            patch.object(
                factory, "configure_proxy", return_value=None
            ) as configure_proxy,
        ):
            create_task = asyncio.create_task(factory._create_browser(True))
            while not tor_started.is_set():
                await asyncio.sleep(0)
            create_task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(create_task.done())

            allow_tor_start.set()
            with self.assertRaises(asyncio.CancelledError):
                await create_task

        terminate_tor.assert_called_once_with(
            tor_process,
            deadline=unittest.mock.ANY,
        )
        work_deadline = start_tor.call_args.kwargs["deadline"]
        cleanup_deadline = terminate_tor.call_args.kwargs["deadline"]
        self.assertAlmostEqual(
            cleanup_deadline.expires_at - work_deadline.expires_at,
            factory._BROWSER_STARTUP_CLEANUP_RESERVE_SECONDS,
        )
        configure_proxy.assert_called_once_with(cleanup_deadline=cleanup_deadline)

    async def test_constructor_cancellation_reaps_process_before_returning(
        self,
    ) -> None:
        launch_started = threading.Event()
        allow_launch = threading.Event()
        owner = Mock(spec=OwnedProcess)

        def delayed_launch(*_: object, **__: object) -> OwnedProcess:
            launch_started.set()
            allow_launch.wait(timeout=1)
            return owner

        with (
            tempfile.TemporaryDirectory(
                prefix="hbrowser-construction-test-"
            ) as directory,
            _release_mock_owner_directories(),
        ):
            root = Path(directory)
            executable = root / "browser"
            executable.touch()
            profile = root / "profile"
            profile.mkdir()
            config = zd.Config(
                user_data_dir=profile,
                browser_executable_path=executable,
            )
            with (
                patch.object(
                    factory,
                    "start_owned_browser_process",
                    side_effect=delayed_launch,
                ),
                patch.object(
                    factory,
                    "_wait_for_devtools_active_port_async",
                    new=AsyncMock(),
                ) as wait_for_devtools,
                patch.object(
                    factory,
                    "_terminate_unbound_owner",
                ) as terminate_owner,
            ):
                construction_task = asyncio.create_task(
                    factory._construct_owned_browser_async(
                        config,
                        cleanup_paths=(
                            process_module._PrivateDirectory.capture(profile),
                        ),
                        startup_deadline=Deadline.after(25.0),
                    )
                )
                while not launch_started.is_set():
                    await asyncio.sleep(0)

                # Only the bounded process-ownership handshake runs in a
                # thread; unrelated event-loop work remains responsive.
                marker = AsyncMock()
                await marker()
                marker.assert_awaited_once_with()

                construction_task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(construction_task.done())
                allow_launch.set()
                with self.assertRaises(asyncio.CancelledError):
                    await construction_task

        terminate_owner.assert_called_once_with(
            owner,
            deadline=unittest.mock.ANY,
        )
        wait_for_devtools.assert_not_awaited()

    async def test_browser_launch_cancellation_stops_returned_browser(self) -> None:
        launch_started = asyncio.Event()
        allow_launch = asyncio.Event()
        browser = SimpleNamespace(connection=None)
        _bind_browser_process_owner(browser)

        async def delayed_launch() -> object:
            launch_started.set()
            await allow_launch.wait()
            return browser

        browser.start = AsyncMock(side_effect=delayed_launch)

        async def cleanup_started_browser(
            actual_browser: object,
            _deadline: Deadline,
        ) -> None:
            self.assertIs(actual_browser, browser)
            allow_launch.set()
            await asyncio.sleep(0)

        with (
            _release_mock_owner_directories(),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ),
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(factory, "_register_browser_atexit"),
            patch.object(
                factory,
                "stop_browser",
                new=AsyncMock(side_effect=cleanup_started_browser),
            ) as stop_browser,
        ):
            create_task = asyncio.create_task(factory._create_browser(True))
            await launch_started.wait()
            create_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await create_task

        stop_browser.assert_awaited_once_with(browser, unittest.mock.ANY)

    async def test_browser_start_failure_cleans_exact_owned_process(
        self,
    ) -> None:
        owner = Mock(spec=OwnedProcess)
        startup_error = RuntimeError("target discovery failed")
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
        )
        _bind_browser_process_owner(browser, owner)

        async def failed_start() -> object:
            raise startup_error

        browser.start = AsyncMock(side_effect=failed_start)
        browser.stop = AsyncMock()
        with (
            _release_mock_owner_directories(),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ),
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(factory, "_register_browser_atexit"),
            self.assertRaises(RuntimeError) as raised,
        ):
            await factory._create_browser(True)

        self.assertIs(raised.exception, startup_error)
        browser.stop.assert_awaited_once_with()
        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=unittest.mock.ANY,
        )
        self.assertIs(
            getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE),
            owner,
        )

    async def test_start_failure_runs_explicit_fallback_before_returning(
        self,
    ) -> None:
        browser = _OwnedBrowser()
        startup_error = RuntimeError("target discovery failed")
        cleanup_error = RuntimeError("first cleanup failed")
        registered_instances: set[object] = set()
        registered_cleanups: list[factory._BrowserAtexitCleanup] = []

        async def failed_start() -> object:
            self.assertIsNotNone(
                getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            )
            registered_instances.add(browser)
            raise startup_error

        browser.start = AsyncMock(side_effect=failed_start)  # type: ignore[attr-defined]
        browser.stop.side_effect = [cleanup_error, None]

        with (
            _release_mock_owner_directories(),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ),
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.register",
                side_effect=lambda cleanup, **_: registered_cleanups.append(cleanup),
            ) as register_cleanup,
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await factory._create_browser(True)

            self.assertIs(raised.exception, startup_error)
            self.assertEqual(len(registered_cleanups), 1)
            cleanup = registered_cleanups[0]
            self.assertIsNone(cleanup.browser)
            self.assertIsNone(
                getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            )
            self.assertNotIn(browser, registered_instances)
            unregister_cleanup.assert_called_once_with(cleanup, loop=cleanup.loop)

        register_cleanup.assert_called_once()
        browser.stop.assert_awaited_once_with()

    async def test_normal_shutdown_releases_each_restart_generation(self) -> None:
        browsers = [_OwnedBrowser(), _OwnedBrowser()]
        registered_instances: set[object] = set(browsers)

        with (
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.register"
            ) as register_cleanup,
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
        ):
            cleanups: list[factory._BrowserAtexitCleanup] = []
            for browser in browsers:
                factory._register_browser_atexit(cast(zd.Browser, browser))
                cleanups.append(
                    getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
                )
                await factory.stop_browser(browser)

        self.assertEqual(register_cleanup.call_count, 2)
        self.assertEqual(unregister_cleanup.call_count, 2)
        self.assertEqual(registered_instances, set())
        self.assertTrue(all(cleanup.browser is None for cleanup in cleanups))
        self.assertTrue(
            all(
                getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE) is None
                for browser in browsers
            )
        )

    async def test_caller_deadline_is_the_generation_shutdown_deadline(self) -> None:
        browser = _OwnedBrowser()
        caller_deadline = Deadline.after(1.0)

        await factory.stop_browser(browser, caller_deadline)

        lifecycle = cast(Any, browser)._hbrowser_zendriver_lifecycle
        self.assertEqual(
            lifecycle.shutdown_deadline_expires_at,
            caller_deadline.expires_at,
        )

    async def test_ownership_release_failures_remain_retryable(self) -> None:
        unregister_browser = _OwnedBrowser()
        registered_instances: set[object] = {unregister_browser}
        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister",
                side_effect=[RuntimeError("unregister failed"), None],
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
        ):
            factory._register_browser_atexit(cast(zd.Browser, unregister_browser))
            with self.assertRaisesRegex(RuntimeError, "unregister failed"):
                await factory.stop_browser(unregister_browser)

            cleanup = getattr(
                unregister_browser,
                factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE,
            )
            self.assertIs(cleanup.browser, unregister_browser)
            await factory.stop_browser(unregister_browser)

        self.assertEqual(unregister_cleanup.call_count, 2)
        unregister_browser.stop.assert_awaited_once_with()
        self.assertNotIn(unregister_browser, registered_instances)

        discard_browser = _OwnedBrowser()
        registered_instances_mock = Mock()
        registered_instances_mock.discard.side_effect = [
            RuntimeError("discard failed"),
            None,
        ]
        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances_mock,
            ),
        ):
            factory._register_browser_atexit(cast(zd.Browser, discard_browser))
            with self.assertRaisesRegex(RuntimeError, "discard failed"):
                await factory.stop_browser(discard_browser)
            await factory.stop_browser(discard_browser)

        unregister_cleanup.assert_called_once()
        self.assertEqual(registered_instances_mock.discard.call_count, 2)
        discard_browser.stop.assert_awaited_once_with()

    async def test_atexit_callback_renews_an_expired_ownership_deadline(
        self,
    ) -> None:
        browser = _OwnedBrowser()
        owner = Mock(spec=OwnedProcess)
        owner.shutdown.side_effect = [
            RuntimeError("initial owner failure"),
            RuntimeError("fallback owner failure"),
            None,
        ]
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)
        registered_instances: set[object] = {browser}
        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
        ):
            factory._register_browser_atexit(cast(zd.Browser, browser))
            cleanup = getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)

            with self.assertRaises(ProcessOwnershipError):
                await factory.stop_browser(browser)
            lifecycle = cast(Any, browser)._hbrowser_zendriver_lifecycle
            first_deadline = lifecycle.shutdown_deadline_expires_at
            lifecycle.shutdown_deadline_expires_at = 0.0

            await cleanup()

        browser.stop.assert_awaited_once_with()
        self.assertEqual(owner.shutdown.call_count, 3)
        self.assertGreater(lifecycle.shutdown_deadline_expires_at, first_deadline)
        self.assertIsNone(cleanup.browser)
        self.assertNotIn(browser, registered_instances)
        # Avoid mutating asyncio-atexit's live callback iteration. The package
        # clears every callback only after all registered browsers have run.
        unregister_cleanup.assert_not_called()

    async def test_atexit_recovers_shutdown_that_expired_before_task_creation(
        self,
    ) -> None:
        browser = _OwnedBrowser()
        owner = _bind_browser_process_owner(browser)
        registered_instances: set[object] = {browser}
        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.unregister"),
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
        ):
            factory._register_browser_atexit(cast(zd.Browser, browser))
            cleanup = getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            with self.assertRaises(ProcessOwnershipError):
                await factory.stop_browser(browser, Deadline(0.0))
            lifecycle = cast(Any, browser)._hbrowser_zendriver_lifecycle
            self.assertIsNone(lifecycle.shutdown_task)
            # An ordinary caller still cannot reset the generation's budget.
            with self.assertRaises(ProcessOwnershipError):
                await factory.stop_browser(browser, Deadline.after(1))
            self.assertEqual(lifecycle.shutdown_deadline_expires_at, 0.0)
            owner.shutdown.assert_not_called()

            await cleanup()

        browser.stop.assert_awaited_once_with()
        owner.shutdown.assert_called_once()
        self.assertIsNone(cleanup.browser)
        self.assertNotIn(browser, registered_instances)
        self.assertTrue(_begin_zendriver_retirement(browser).is_complete())

    async def test_atexit_cannot_renew_an_active_cleanup_task(self) -> None:
        browser = _OwnedBrowser()
        owner = _bind_browser_process_owner(browser)
        retirement = _begin_zendriver_retirement(browser)
        retirement.bind_shutdown_deadline(0.0)
        release = asyncio.Event()

        async def active_cleanup() -> None:
            await release.wait()

        task = asyncio.create_task(active_cleanup())
        retirement.bind_shutdown_task(task)
        cleanup = factory._BrowserAtexitCleanup(
            cast(zd.Browser, browser), asyncio.get_running_loop()
        )
        try:
            with self.assertRaises(ProcessOwnershipError):
                await cleanup()
            self.assertIs(retirement.existing_shutdown_task(), task)
            self.assertFalse(task.done())
            self.assertEqual(retirement.bind_shutdown_deadline(1.0), 0.0)
            owner.shutdown.assert_not_called()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_atexit_cleanup_does_not_require_the_default_executor(self) -> None:
        browser = _OwnedBrowser()
        owner = Mock(spec=OwnedProcess)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)
        registered_instances: set[object] = {browser}

        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister_cleanup,
            patch.object(
                zd.util,
                "get_registered_instances",
                return_value=registered_instances,
            ),
            patch.object(
                asyncio,
                "to_thread",
                new=AsyncMock(
                    side_effect=RuntimeError("Executor shutdown has been called")
                ),
            ) as to_thread,
        ):
            factory._register_browser_atexit(cast(zd.Browser, browser))
            cleanup = getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            await asyncio.get_running_loop().shutdown_default_executor()

            await cleanup()

        browser.stop.assert_awaited_once_with()
        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=unittest.mock.ANY,
        )
        to_thread.assert_not_awaited()
        self.assertIsNone(cleanup.browser)
        self.assertNotIn(browser, registered_instances)
        unregister_cleanup.assert_not_called()

    async def test_post_setup_cleanup_failure_overrides_cancellation(
        self,
    ) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page])
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        setup_started = asyncio.Event()

        async def blocked_setup(*_: object, **__: object) -> None:
            setup_started.set()
            await asyncio.Event().wait()

        cleanup_error = RuntimeError("cleanup failed")
        with (
            _release_mock_owner_directories(),
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(return_value=SimpleNamespace(chrome="/test/chrome")),
            ),
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser_async",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(factory, "_register_browser_atexit"),
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(side_effect=blocked_setup),
            ),
            patch.object(
                factory,
                "stop_browser",
                new=AsyncMock(side_effect=cleanup_error),
            ) as stop_browser,
        ):
            create_task = asyncio.create_task(factory._create_browser(True))
            await setup_started.wait()
            create_task.cancel()
            with self.assertRaises(ProcessOwnershipError) as raised:
                await create_task

        stop_browser.assert_awaited_once_with(browser, unittest.mock.ANY)
        self.assertIs(raised.exception.__cause__, cleanup_error)
        self.assertIn(
            "Startup failure type: CancelledError",
            raised.exception.__notes__,
        )

    async def test_browser_stops_before_protocol_operation_cancellation(self) -> None:
        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        events: list[str] = []
        operation = asyncio.create_task(asyncio.Event().wait())
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        async def stop_process() -> None:
            self.assertFalse(operation.cancelled())
            events.append("stop-browser")

        browser.stop.side_effect = stop_process
        await factory.stop_browser(browser)

        self.assertEqual(events, ["stop-browser"])
        self.assertTrue(operation.cancelled())
        browser.stop.assert_awaited_once_with()

    async def test_browser_stop_failure_uses_explicit_fallback_after_retirement(
        self,
    ) -> None:
        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=RuntimeError("browser stop failed")),
        )
        owner = _bind_browser_process_owner(browser)
        operation: asyncio.Future[None] = asyncio.Future()
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        await factory.stop_browser(browser)

        self.assertTrue(operation.cancelled())
        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=unittest.mock.ANY,
        )

    async def test_browser_stop_failure_preserves_operation_on_live_transport(
        self,
    ) -> None:
        browser = SimpleNamespace()
        connection = SimpleNamespace(
            websocket=object(),
            mapper={},
            listener=None,
            _owner=browser,
            aclose=AsyncMock(side_effect=RuntimeError("connection close failed")),
        )
        browser.connection = connection
        browser.targets = []
        browser._tor_process = None
        browser.stop = AsyncMock(side_effect=RuntimeError("browser stop failed"))
        _bind_browser_process_owner(browser)
        operation: asyncio.Future[None] = asyncio.Future()
        connection.mapper[1] = operation
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=connection)

        with self.assertRaises(ProcessOwnershipError) as raised:
            await factory.stop_browser(browser)

        self.assertIn("connection close failed", str(raised.exception.__cause__))
        self.assertFalse(operation.done())
        self.assertIn(1, connection.mapper)
        operation.set_result(None)
        await asyncio.sleep(0)

    async def test_failed_close_retries_detached_transport_then_finalizes(
        self,
    ) -> None:
        browser = SimpleNamespace()
        operation: asyncio.Future[None] = asyncio.Future()
        allow_connection_close = False

        async def close_connection() -> None:
            if not allow_connection_close:
                raise RuntimeError("connection close failed")
            target.websocket = None

        target = SimpleNamespace(
            websocket=object(),
            mapper={1: operation},
            listener=None,
            browser=browser,
            aclose=AsyncMock(side_effect=close_connection),
        )

        async def stop_process() -> None:
            browser.targets.clear()
            operation.set_result(None)
            await asyncio.sleep(0)

        browser.connection = None
        browser.targets = [target]
        browser._tor_process = None
        browser.stop = AsyncMock(side_effect=stop_process)
        _bind_browser_process_owner(browser)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=target)

        with self.assertRaises(ProcessOwnershipError) as raised:
            await factory.stop_browser(browser)

        self.assertIn("connection close failed", str(raised.exception.__cause__))
        self.assertEqual(browser.targets, [])
        self.assertTrue(operation.done())
        self.assertIn(1, target.mapper)
        target.aclose.assert_awaited_once_with()

        allow_connection_close = True
        await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        self.assertEqual(target.aclose.await_count, 2)
        self.assertIsNone(target.websocket)
        self.assertEqual(target.mapper, {})

    async def test_late_timeout_completion_keeps_detached_transport_for_stop(
        self,
    ) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        operation: asyncio.Future[str] = asyncio.Future()
        target = SimpleNamespace(
            websocket=object(),
            mapper={1: operation},
            listener=None,
            browser=browser,
        )

        async def close_target() -> None:
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=target)

        operation.set_result("late response")
        await asyncio.sleep(0)
        await factory.stop_browser(browser)

        target.aclose.assert_awaited_once_with()
        self.assertEqual(target.mapper, {})

    async def test_startup_late_root_is_closed_before_operation_cancellation(
        self,
    ) -> None:
        reveal_root = asyncio.Event()
        root_revealed = asyncio.Event()
        cancelled_after_root_close = False
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        root = SimpleNamespace(
            websocket=object(),
            mapper={},
            listener=None,
            _owner=browser,
        )

        async def close_root() -> None:
            root.websocket = None

        root.aclose = AsyncMock(side_effect=close_root)
        trigger = SimpleNamespace(
            websocket=object(),
            mapper={},
            listener=None,
            browser=browser,
        )

        async def close_trigger() -> None:
            trigger.websocket = None
            reveal_root.set()
            await root_revealed.wait()

        trigger.aclose = AsyncMock(side_effect=close_trigger)
        browser.targets = [trigger]

        async def startup_operation() -> None:
            nonlocal cancelled_after_root_close
            await reveal_root.wait()
            browser.connection = root
            current_task = asyncio.current_task()
            assert current_task is not None
            root.mapper[1] = current_task
            root_revealed.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled_after_root_close = root.websocket is None

        operation = asyncio.create_task(startup_operation())
        await asyncio.sleep(0)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        await factory.stop_browser(browser)

        trigger.aclose.assert_awaited_once_with()
        root.aclose.assert_awaited_once_with()
        self.assertTrue(operation.cancelled())
        self.assertTrue(cancelled_after_root_close)
        self.assertEqual(root.mapper, {})

    async def test_retired_rejected_transaction_can_receive_before_shutdown(
        self,
    ) -> None:
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()
        release_listener = asyncio.Event()

        async def block_stop() -> None:
            stop_started.set()
            await allow_stop.wait()

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=block_stop),
        )
        _bind_browser_process_owner(browser)
        transaction: asyncio.Future[str] = asyncio.Future()
        target = SimpleNamespace(
            websocket=object(),
            mapper={1: transaction},
            browser=browser,
        )

        async def deliver_response() -> None:
            await release_listener.wait()
            target.mapper.pop(1).set_result("late response")

        listener_task = asyncio.create_task(deliver_response())
        target.listener = SimpleNamespace(
            task=listener_task,
            cancel=Mock(side_effect=listener_task.cancel),
        )

        async def close_target() -> None:
            release_listener.set()
            await listener_task
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)
        stop_task = asyncio.create_task(factory.stop_browser(browser))
        await stop_started.wait()

        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(transaction, timeout=1, owner=target)
        self.assertFalse(transaction.cancelled())

        allow_stop.set()
        await stop_task

        self.assertEqual(transaction.result(), "late response")
        target.aclose.assert_awaited_once_with()
        target.listener.cancel.assert_not_called()
        self.assertEqual(target.mapper, {})

    async def test_completed_late_future_preserves_idempotent_shutdown(self) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        await factory.stop_browser(browser)
        retirement = _begin_zendriver_retirement(browser)
        completed: asyncio.Future[None] = asyncio.Future()
        completed.set_result(None)

        with self.assertRaises(ZendriverOwnerRetiredError) as rejected:
            await wait_for_zendriver(completed, timeout=1, owner=browser)

        self.assertIsNone(rejected.exception.__cause__)
        self.assertTrue(retirement.is_complete())
        with patch.object(Deadline, "expired", new=True):
            await factory.stop_browser(browser)
        browser.stop.assert_awaited_once_with()

    async def test_completed_shutdown_reopens_for_new_exact_connection(
        self,
    ) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        await factory.stop_browser(browser)

        operation: asyncio.Future[None] = asyncio.Future()
        target = SimpleNamespace(
            websocket=object(),
            mapper={1: operation},
            listener=None,
            browser=browser,
        )

        async def close_target() -> None:
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(operation, timeout=1, owner=target)

        self.assertFalse(operation.cancelled())
        await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        target.aclose.assert_awaited_once_with()
        self.assertTrue(operation.cancelled())
        self.assertEqual(target.mapper, {})

    async def test_detached_target_can_deliver_late_response_before_retire(
        self,
    ) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        operation: asyncio.Future[str] = asyncio.Future()
        release_listener = asyncio.Event()
        response_delivered = asyncio.Event()
        target = SimpleNamespace(
            websocket=object(),
            mapper={1: operation},
            browser=browser,
        )

        async def listener_loop() -> None:
            await release_listener.wait()
            transaction = target.mapper.pop(1)
            transaction.set_result("late response")
            response_delivered.set()

        listener_task = asyncio.create_task(listener_loop())
        listener = SimpleNamespace(
            task=listener_task,
            cancel=Mock(side_effect=listener_task.cancel),
        )
        target.listener = listener

        async def close_target() -> None:
            release_listener.set()
            await response_delivered.wait()
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=target)

        await factory.stop_browser(browser)

        target.aclose.assert_awaited_once_with()
        listener.cancel.assert_not_called()
        self.assertTrue(listener_task.done())
        self.assertEqual(operation.result(), "late response")
        self.assertEqual(target.mapper, {})

    async def test_stop_tombstones_generation_before_first_await(self) -> None:
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()

        async def blocking_stop() -> None:
            stop_started.set()
            await allow_stop.wait()

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=blocking_stop),
        )
        _bind_browser_process_owner(browser)
        stop_task = asyncio.create_task(factory.stop_browser(browser))
        await stop_started.wait()

        coroutine = asyncio.sleep(0)
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(coroutine, timeout=1, owner=browser)

        allow_stop.set()
        await stop_task

    async def test_caller_cancellation_waits_for_complete_browser_cleanup(self) -> None:
        browser_stop_started = asyncio.Event()
        allow_browser_stop = asyncio.Event()

        async def blocking_stop() -> None:
            browser_stop_started.set()
            await allow_browser_stop.wait()

        tor_process = Mock()
        browser = SimpleNamespace(
            targets=[],
            _tor_process=tor_process,
            stop=AsyncMock(side_effect=blocking_stop),
        )
        _bind_browser_process_owner(browser)
        operation: asyncio.Future[None] = asyncio.Future()
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        with patch.object(factory, "terminate_tor_process") as terminate_tor:
            stop_task = asyncio.create_task(factory.stop_browser(browser))
            await browser_stop_started.wait()
            stop_task.cancel()
            await asyncio.sleep(0)

            self.assertFalse(stop_task.done())
            self.assertFalse(operation.cancelled())
            terminate_tor.assert_not_called()

            allow_browser_stop.set()
            with self.assertRaises(asyncio.CancelledError):
                await stop_task

            terminate_tor.assert_called_once_with(
                tor_process,
                deadline=unittest.mock.ANY,
            )

        browser.stop.assert_awaited_once_with()
        self.assertTrue(operation.cancelled())

    async def test_later_stop_caller_keeps_its_shorter_deadline(self) -> None:
        for seconds in (0.0, 0.01):
            with self.subTest(seconds=seconds):
                started = asyncio.Event()
                release = asyncio.Event()

                async def blocking_stop(
                    started: asyncio.Event = started,
                    release: asyncio.Event = release,
                ) -> None:
                    started.set()
                    await release.wait()

                browser = SimpleNamespace(
                    connection=None,
                    targets=[],
                    stop=AsyncMock(side_effect=blocking_stop),
                )
                _bind_browser_process_owner(browser)
                first_deadline = Deadline.after(1)
                first = asyncio.create_task(
                    factory.stop_browser(browser, first_deadline)
                )
                try:
                    await asyncio.wait_for(started.wait(), 1)
                    with self.assertRaises(ProcessOwnershipError):
                        await asyncio.wait_for(
                            factory.stop_browser(browser, Deadline.after(seconds)), 0.2
                        )
                    self.assertFalse(first.done())
                    lifecycle = browser._hbrowser_zendriver_lifecycle
                    self.assertEqual(
                        lifecycle.shutdown_deadline_expires_at,
                        first_deadline.expires_at,
                    )
                    self.assertFalse(lifecycle.shutdown_task.cancelled())
                finally:
                    release.set()
                    await first
                browser.stop.assert_awaited_once()

    async def test_late_waiter_rejects_completed_cleanup_without_reopening_it(
        self,
    ) -> None:
        now = 0.0
        started = asyncio.Event()
        release = asyncio.Event()
        original_wait = asyncio.wait
        later: asyncio.Task[None] | None = None

        async def blocking_stop() -> None:
            started.set()
            await release.wait()

        browser = _OwnedBrowser()
        browser.stop.side_effect = blocking_stop

        async def delayed_wait(
            tasks: tuple[asyncio.Task[None], ...], *, timeout: float
        ) -> tuple[set[asyncio.Task[None]], set[asyncio.Task[None]]]:
            nonlocal now
            if asyncio.current_task() is later:
                self.assertIs(
                    tasks[0],
                    _begin_zendriver_retirement(browser).existing_shutdown_task(),
                )
                release.set()
                result = await original_wait(tasks, timeout=timeout)
                self.assertTrue(tasks[0].done())
                # Advance only the injected deadline clock after real cleanup
                # completed, before this waiter can inspect that completion.
                now = 1.0
                return result
            return await original_wait(tasks, timeout=timeout)

        with (
            patch.object(
                deadline_module, "time", SimpleNamespace(monotonic=lambda: now)
            ),
            patch.object(
                factory,
                "asyncio",
                SimpleNamespace(**(vars(asyncio) | {"wait": delayed_wait})),
            ),
        ):
            first = asyncio.create_task(factory.stop_browser(browser, Deadline(5.0)))
            try:
                await asyncio.wait_for(started.wait(), 1)
                later = asyncio.create_task(
                    factory.stop_browser(browser, Deadline(1.0))
                )
                with self.assertRaisesRegex(
                    ProcessOwnershipError, "completed after its caller deadline"
                ):
                    await asyncio.wait_for(later, 1)
                await asyncio.wait_for(first, 1)
                retirement = _begin_zendriver_retirement(browser)
                self.assertTrue(retirement.is_complete())
                self.assertTrue(retirement.browser_cleanup_is_complete())
                self.assertEqual(
                    retirement.lifecycle.shutdown_deadline_expires_at,
                    5.0,
                )
                completed_task = retirement.existing_shutdown_task()
                await factory.stop_browser(browser, Deadline(2.0))
                self.assertIs(retirement.existing_shutdown_task(), completed_task)
                browser.stop.assert_awaited_once()
                browser.process_owner.shutdown.assert_called_once()
            finally:
                release.set()
                await asyncio.gather(
                    first, *(() if later is None else (later,)), return_exceptions=True
                )

    async def test_late_cleanup_failure_keeps_its_original_precedence(self) -> None:
        for original in (ValueError("cleanup failed"), ProcessOwnershipError("owned")):
            with self.subTest(error=type(original).__name__):

                async def failed_cleanup(error: BaseException = original) -> None:
                    raise error

                task = asyncio.create_task(failed_cleanup())
                await asyncio.gather(task, return_exceptions=True)
                with self.assertRaises(type(original)) as raised:
                    await factory._await_cleanup_deferring_caller_cancellation(
                        task, Deadline(0.0)
                    )
                self.assertIs(raised.exception, original)

    async def test_concurrent_stop_callers_share_one_cleanup(self) -> None:
        browser_stop_started = asyncio.Event()
        allow_browser_stop = asyncio.Event()

        async def blocking_stop() -> None:
            browser_stop_started.set()
            await allow_browser_stop.wait()

        tor_process = Mock()
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=tor_process,
            stop=AsyncMock(side_effect=blocking_stop),
        )
        _bind_browser_process_owner(browser)

        with patch.object(factory, "terminate_tor_process") as terminate_tor:
            first = asyncio.create_task(factory.stop_browser(browser))
            await browser_stop_started.wait()
            second = asyncio.create_task(factory.stop_browser(browser))
            await asyncio.sleep(0)

            browser.stop.assert_awaited_once_with()
            terminate_tor.assert_not_called()

            allow_browser_stop.set()
            await asyncio.gather(first, second)

            browser.stop.assert_awaited_once_with()
            terminate_tor.assert_called_once_with(
                tor_process,
                deadline=unittest.mock.ANY,
            )

    async def test_stuck_target_listener_blocks_retirement_without_hanging(
        self,
    ) -> None:
        allow_listener_stop = False

        async def stubborn_listener() -> None:
            nonlocal allow_listener_stop
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if allow_listener_stop:
                        return

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        listener_task = asyncio.create_task(stubborn_listener())
        target = SimpleNamespace(
            websocket=object(),
            mapper={},
            browser=browser,
            listener=SimpleNamespace(
                task=listener_task,
                cancel=Mock(side_effect=listener_task.cancel),
            ),
        )

        async def close_target() -> None:
            target.websocket = None

        target.aclose = AsyncMock(side_effect=close_target)
        browser.targets = [target]
        operation: asyncio.Future[None] = asyncio.Future()
        target.mapper[1] = operation
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=target)

        with (
            patch.object(
                factory,
                "_CONNECTION_LISTENER_STOP_TIMEOUT_SECONDS",
                0.01,
            ),
            self.assertRaises(ProcessOwnershipError) as raised,
        ):
            await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

        self.assertIn("listener did not stop", str(raised.exception.__cause__))
        self.assertFalse(operation.done())
        self.assertIn(1, target.mapper)
        browser.stop.assert_awaited_once_with()

        allow_listener_stop = True
        listener_task.cancel()
        await listener_task

        await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        self.assertTrue(operation.cancelled())
        self.assertEqual(target.mapper, {})

    async def test_browser_stop_watchdog_runs_owned_shutdown_in_same_call(
        self,
    ) -> None:
        async def blocking_stop() -> None:
            await asyncio.Event().wait()

        owner = Mock(spec=OwnedProcess)
        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=blocking_stop),
        )
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)
        operation: asyncio.Future[None] = asyncio.Future()
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        with patch.object(factory, "_BROWSER_STOP_TIMEOUT_SECONDS", 0.01):
            await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

        browser.stop.assert_awaited_once_with()
        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=unittest.mock.ANY,
        )
        self.assertTrue(operation.cancelled())

    async def test_pending_browser_stop_retains_ownership_until_retry_settles(
        self,
    ) -> None:
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        original_wait = asyncio.wait
        browser = _OwnedBrowser()
        retirement = _begin_zendriver_retirement(browser)

        async def stubborn_stop() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()

        browser.stop.side_effect = stubborn_stop

        async def expired_stop_watchdog(
            tasks: tuple[asyncio.Task[None], ...], *, timeout: float
        ) -> tuple[set[asyncio.Task[None]], set[asyncio.Task[None]]]:
            (task,) = tasks
            if task is retirement.existing_browser_stop_task() and not task.done():
                await started.wait()
                if task.cancelling():
                    await cancellation_seen.wait()
                return set(), {task}
            return await original_wait(tasks, timeout=timeout)

        registered_instances: set[object] = {browser}
        with (
            patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
            patch(
                "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
            ) as unregister,
            patch.object(
                zd.util, "get_registered_instances", return_value=registered_instances
            ),
            patch.object(
                factory,
                "asyncio",
                SimpleNamespace(**(vars(asyncio) | {"wait": expired_stop_watchdog})),
            ),
        ):
            factory._register_browser_atexit(cast(zd.Browser, browser))
            cleanup = getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            deadline = Deadline.after(5)
            try:
                for attempt in range(2):
                    with self.assertRaisesRegex(
                        ProcessOwnershipError, "cleanup remains unresolved"
                    ) as raised:
                        await asyncio.wait_for(
                            factory.stop_browser(browser, deadline), 1
                        )
                    self.assertIsInstance(raised.exception.__cause__, TimeoutError)
                    stop_task = retirement.existing_browser_stop_task()
                    assert stop_task is not None
                    self.assertFalse(stop_task.done())
                    self.assertFalse(retirement.browser_cleanup_is_complete())
                    self.assertFalse(retirement.is_complete())
                    self.assertIs(cleanup.browser, browser)
                    self.assertIn(browser, registered_instances)
                    unregister.assert_not_called()
                    browser.stop.assert_awaited_once()
                    self.assertEqual(
                        browser.process_owner.shutdown.call_count, attempt + 1
                    )
                    self.assertEqual(
                        retirement.lifecycle.shutdown_deadline_expires_at,
                        deadline.expires_at,
                    )
                # The assertions above precede the fixture's cooperative release.
                # A retry may finalize only after the exact retained task exits.
                release.set()
                assert stop_task is not None
                await asyncio.wait_for(asyncio.shield(stop_task), 1)
                await factory.stop_browser(browser, deadline)
                self.assertTrue(retirement.is_complete())
                self.assertIs(retirement.existing_browser_stop_task(), stop_task)
                self.assertIsNone(cleanup.browser)
                self.assertNotIn(browser, registered_instances)
                unregister.assert_called_once_with(cleanup, loop=cleanup.loop)
                browser.stop.assert_awaited_once()
            finally:
                release.set()
                retained = retirement.existing_browser_stop_task()
                if retained is not None:
                    await asyncio.wait_for(asyncio.shield(retained), 1)
                await factory.stop_browser(browser, deadline)

    async def test_browser_stop_settling_during_fallback_can_finalize(self) -> None:
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        original_wait = asyncio.wait
        browser = _OwnedBrowser()
        retirement = _begin_zendriver_retirement(browser)

        async def stubborn_stop() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()

        browser.stop.side_effect = stubborn_stop

        async def expired_stop_watchdog(
            tasks: tuple[asyncio.Task[None], ...], *, timeout: float
        ) -> tuple[set[asyncio.Task[None]], set[asyncio.Task[None]]]:
            (task,) = tasks
            if task is retirement.existing_browser_stop_task() and not task.done():
                await started.wait()
                if task.cancelling():
                    await cancellation_seen.wait()
                return set(), {task}
            return await original_wait(tasks, timeout=timeout)

        async def cleanup_while_stop_settles(operation: Callable[[], None]) -> None:
            stop_task = retirement.existing_browser_stop_task()
            assert stop_task is not None
            self.assertFalse(stop_task.done())
            self.assertTrue(cancellation_seen.is_set())
            operation()
            release.set()
            await asyncio.wait_for(asyncio.shield(stop_task), 1)

        with (
            patch.object(
                factory,
                "asyncio",
                SimpleNamespace(**(vars(asyncio) | {"wait": expired_stop_watchdog})),
            ),
            patch.object(
                factory, "_run_blocking_cleanup_in_thread", cleanup_while_stop_settles
            ),
        ):
            try:
                await asyncio.wait_for(factory.stop_browser(browser), 1)
                self.assertTrue(retirement.is_complete())
                retained = retirement.existing_browser_stop_task()
                assert retained is not None
                self.assertTrue(retained.done())
                browser.stop.assert_awaited_once()
                browser.process_owner.shutdown.assert_called_once()
            finally:
                release.set()
                retained = retirement.existing_browser_stop_task()
                if retained is not None:
                    await asyncio.wait_for(asyncio.shield(retained), 1)
                await factory.stop_browser(browser)

    async def test_tor_termination_failure_is_retried_without_browser_replay(
        self,
    ) -> None:
        tor_process = Mock()
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=tor_process,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        with patch.object(
            factory,
            "terminate_tor_process",
            side_effect=[RuntimeError("tor reap failed"), None],
        ) as terminate_tor:
            with self.assertRaises(ProcessOwnershipError) as raised:
                await factory.stop_browser(browser)
            self.assertIn("tor reap failed", str(raised.exception.__cause__))

            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        self.assertEqual(
            terminate_tor.call_args_list,
            [
                call(tor_process, deadline=unittest.mock.ANY),
                call(tor_process, deadline=unittest.mock.ANY),
            ],
        )

    async def test_retirement_drain_bounds_cancellation_resistant_operation(
        self,
    ) -> None:
        allow_stop = False

        async def stubborn_operation() -> None:
            nonlocal allow_stop
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if allow_stop:
                        return

        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        _bind_browser_process_owner(browser)
        operation = asyncio.create_task(stubborn_operation())
        await asyncio.sleep(0)
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        try:
            with (
                patch(
                    "hbrowser.gallery.utils.protocol._OPERATION_DRAIN_TIMEOUT_SECONDS",
                    0.01,
                ),
                self.assertRaises(ProcessOwnershipError),
            ):
                await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

            self.assertFalse(operation.done())
        finally:
            allow_stop = True
            operation.cancel()
            await operation

    async def test_late_protocol_completion_still_requires_atexit_release(self) -> None:
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        try:
            for eager in (False, True):
                with self.subTest(eager=eager):
                    loop.set_task_factory(asyncio.eager_task_factory if eager else None)
                    await self._assert_late_completion_releases_atexit()
        finally:
            loop.set_task_factory(previous_factory)

    async def _assert_late_completion_releases_atexit(self) -> None:
        browser = _OwnedBrowser()
        # This synthetic operation has no transport or pending Browser.start.
        del browser.connection
        release = asyncio.Event()

        async def stubborn_operation() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        operation = asyncio.create_task(stubborn_operation())
        await asyncio.sleep(0)
        registered_instances: set[object] = {browser}
        try:
            with (
                patch("hbrowser.gallery.browser.factory.asyncio_atexit.register"),
                patch(
                    "hbrowser.gallery.browser.factory.asyncio_atexit.unregister"
                ) as unregister,
                patch.object(
                    zd.util,
                    "get_registered_instances",
                    return_value=registered_instances,
                ),
            ):
                factory._register_browser_atexit(cast(zd.Browser, browser))
                cleanup = getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
                with self.assertRaises(ZendriverOperationTimeout):
                    await wait_for_zendriver(operation, timeout=0, owner=browser)
                with (
                    patch(
                        "hbrowser.gallery.utils.protocol._OPERATION_DRAIN_TIMEOUT_SECONDS",
                        0.01,
                    ),
                    self.assertRaises(ProcessOwnershipError),
                ):
                    await factory.stop_browser(browser)
                retirement = _begin_zendriver_retirement(browser)
                self.assertTrue(retirement.browser_cleanup_is_complete())
                self.assertTrue(retirement.tor_cleanup_is_complete())
                release.set()
                await operation
                self.assertTrue(retirement.protocol_is_retired())
                self.assertFalse(retirement.is_complete())
                self.assertIs(cleanup.browser, browser)
                unregister.assert_not_called()
                failed_task = retirement.existing_shutdown_task()

                await factory.stop_browser(browser)

                self.assertTrue(retirement.is_complete())
                successful_task = retirement.existing_shutdown_task()
                self.assertIsNot(successful_task, failed_task)
                assert successful_task is not None
                self.assertTrue(successful_task.done())
                self.assertIsNone(successful_task.exception())
                self.assertIsNone(cleanup.browser)
                self.assertNotIn(browser, registered_instances)
                unregister.assert_called_once_with(cleanup, loop=cleanup.loop)
                browser.stop.assert_awaited_once_with()
        finally:
            release.set()
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)


class TorRealDeadlineTests(unittest.TestCase):
    def test_bootstrap_exhaustion_reaps_process_before_returning(self) -> None:
        owners: list[OwnedProcess] = []
        data_directories: list[Path] = []

        def start_fake_tor(*_args: Any, **kwargs: Any) -> OwnedProcess:
            def parameters(data: Path) -> list[str]:
                data_directories.append(data)
                return ["-c", "import time; time.sleep(60)"]

            owner = process_module._start_owned_process_with_private_directory(
                sys.executable,
                parameters,
                **kwargs,
            )
            owners.append(owner)
            return owner

        deadline = Deadline.after(4)
        try:
            with (
                patch.object(tor_module, "_find_tor_binary", return_value="tor"),
                patch.object(
                    tor_module,
                    "_start_owned_process_with_private_directory",
                    new=start_fake_tor,
                ),
                self.assertRaisesRegex(RuntimeError, "failed to bootstrap") as raised,
            ):
                tor_module._start_tor_process(9150, deadline=deadline)
            self.assertNotIsInstance(raised.exception, ProcessOwnershipError)
            self.assertEqual(len(owners), 1)
            self.assertIsNotNone(owners[0].poll())
            self.assertEqual(len(data_directories), 1)
            self.assertFalse(data_directories[0].exists())
            self.assertGreater(deadline.remaining(), 0)
        finally:
            for owner in owners:
                # Negative controls must not leave a live synthetic Tor process.
                terminate_tor_process(owner)


class TorProcessCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        tor_binary = patch.object(
            tor_module,
            "_find_tor_binary",
            return_value="/tor",
        )
        tor_binary.start()
        self.addCleanup(tor_binary.stop)

    @staticmethod
    def _owned_process(*, stdout: object) -> Mock:
        process = Mock(spec=OwnedProcess)
        process.stdout = stdout
        return process

    def test_tor_process_is_launched_outside_the_terminal_group(self) -> None:
        process = self._owned_process(stdout=[b"Bootstrapped 100%: Done\n"])
        process.poll.return_value = None
        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                return_value=process,
            ) as start_process,
            patch("hbrowser.gallery.browser.tor.atexit.register"),
        ):
            observed = tor_module._start_tor_process(9150)

        self.assertIs(observed, process)
        start_process.assert_called_once()
        self.assertEqual(start_process.call_args.args[0], "/tor")
        self.assertEqual(start_process.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(start_process.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(
            start_process.call_args.kwargs["directory_prefix"], "tor_data_"
        )
        parameters = start_process.call_args.args[1]
        data = Path("owned-tor-data")
        self.assertEqual(
            parameters(data),
            ["--SocksPort", "9150", "--DataDirectory", str(data)],
        )
        terminate_tor_process(process)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )

    def test_owned_tor_uses_bounded_shutdown_policy(self) -> None:
        process = self._owned_process(stdout=None)

        terminate_tor_process(process)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=None,
        )

    def test_cleanup_registration_failure_releases_exact_owner(self) -> None:
        process = self._owned_process(stdout=[])
        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                return_value=process,
            ),
            patch(
                "hbrowser.gallery.browser.tor.atexit.register",
                side_effect=RuntimeError("registration failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "registration failed"),
        ):
            tor_module._start_tor_process(9150)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )
        process_cleanup = getattr(
            process,
            tor_module._TOR_PROCESS_CLEANUP_ATTRIBUTE,
        )
        self.assertIsNone(process_cleanup._tor_process)

    def test_missing_stdout_reaps_unowned_bootstrap_process(self) -> None:
        process = self._owned_process(stdout=None)
        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                return_value=process,
            ),
            self.assertRaisesRegex(RuntimeError, "capture Tor process output"),
        ):
            tor_module._start_tor_process(9150)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )

    def test_early_bootstrap_exit_is_reaped_before_error(self) -> None:
        process = self._owned_process(stdout=[])
        process.poll.return_value = 7
        process.returncode = 7
        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                return_value=process,
            ),
            self.assertRaisesRegex(RuntimeError, "exited unexpectedly with code 7"),
        ):
            tor_module._start_tor_process(9150)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )

    def test_bootstrap_timeout_terminates_and_reaps_process(self) -> None:
        process = self._owned_process(stdout=[])
        process.poll.return_value = None
        now = [99.0]
        deadline = Deadline(100.0)

        def start_process(*_args: object, **_kwargs: object) -> Mock:
            # Bootstrap's reserved deadline expires while cleanup still has time.
            now[0] = 99.75
            return process

        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                side_effect=start_process,
            ),
            patch.object(time, "monotonic", side_effect=lambda: now[0]),
            self.assertRaisesRegex(RuntimeError, "failed to bootstrap"),
        ):
            tor_module._start_tor_process(9150, deadline=deadline)

        process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=deadline.expires_at,
        )

    def test_expired_shared_deadline_does_not_start_tor(self) -> None:
        with (
            patch.object(
                tor_module, "_start_owned_process_with_private_directory"
            ) as start_process,
            self.assertRaisesRegex(RuntimeError, "shared browser deadline"),
        ):
            tor_module._start_tor_process(9150, deadline=Deadline(0.0))

        start_process.assert_not_called()

    def test_unreaped_bootstrap_process_is_never_retried(self) -> None:
        cleanup_failure = tor_module._TorProcessCleanupError(
            "Tor process ownership unresolved"
        )
        self.assertIsInstance(cleanup_failure, ProcessOwnershipError)
        with (
            patch.object(
                tor_module,
                "_start_tor_process",
                side_effect=cleanup_failure,
            ) as start_process,
            self.assertRaises(tor_module._TorProcessCleanupError),
        ):
            tor_module.start_tor_with_retry(9150, max_retries=3, retry_wait=0)

        start_process.assert_called_once_with(9150, deadline=unittest.mock.ANY)

    def test_retry_policy_is_capped_and_uses_one_absolute_deadline(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_retries"):
            tor_module.start_tor_with_retry(9150, max_retries=4)
        with self.assertRaisesRegex(ValueError, "retry_wait"):
            tor_module.start_tor_with_retry(9150, retry_wait=5.01)

        observed_deadlines: list[Deadline] = []
        process = Mock()

        def start(_: int, *, deadline: Deadline) -> Mock:
            observed_deadlines.append(deadline)
            if len(observed_deadlines) < 3:
                raise RuntimeError("retry")
            return process

        with (
            patch.object(tor_module, "_start_tor_process", side_effect=start),
            patch("hbrowser.gallery.browser.tor.time.sleep"),
        ):
            result = tor_module.start_tor_with_retry(
                9150,
                max_retries=3,
                retry_wait=0,
            )

        self.assertIs(result, process)
        self.assertEqual(len(observed_deadlines), 3)
        self.assertTrue(
            all(deadline is observed_deadlines[0] for deadline in observed_deadlines)
        )

    def test_failed_bootstrap_retains_exact_process_cleanup_for_atexit(
        self,
    ) -> None:
        process = self._owned_process(stdout=None)
        retained_cleanups: list[tor_module._TorProcessAtexitCleanup] = []

        def retain_cleanup(
            cleanup: tor_module._TorProcessAtexitCleanup,
        ) -> tor_module._TorProcessAtexitCleanup:
            retained_cleanups.append(cleanup)
            return cleanup

        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                return_value=process,
            ),
            patch(
                "hbrowser.gallery.browser.tor.atexit.register",
                side_effect=retain_cleanup,
            ) as register_cleanup,
            patch.object(
                tor_module,
                "_terminate_tor_process",
                side_effect=[RuntimeError("reap failed"), None],
            ) as terminate_process,
            patch(
                "hbrowser.gallery.browser.tor.atexit.unregister"
            ) as unregister_cleanup,
        ):
            with self.assertRaises(tor_module._TorProcessCleanupError):
                tor_module._start_tor_process(9150)

            register_cleanup.assert_called_once_with(retained_cleanups[0])
            retained_cleanup = retained_cleanups[0]
            self.assertIsInstance(
                retained_cleanup,
                tor_module._TorProcessAtexitCleanup,
            )
            self.assertIs(retained_cleanup._tor_process, process)
            unregister_cleanup.assert_not_called()

            retained_cleanup()
            retained_cleanup()
            self.assertIsNone(retained_cleanup._tor_process)
            unregister_cleanup.assert_not_called()

            tor_module.terminate_tor_process(process)
            unregister_cleanup.assert_called_once_with(retained_cleanup)

            self.assertEqual(
                terminate_process.call_args_list,
                [
                    call(process, deadline=unittest.mock.ANY),
                    call(process, deadline=unittest.mock.ANY),
                ],
            )

    def test_two_normal_generations_each_release_their_atexit_cleanup(self) -> None:
        first_process = self._owned_process(stdout=[b"Bootstrapped 100%: Done\n"])
        second_process = self._owned_process(stdout=[b"Bootstrapped 100%: Done\n"])
        first_process.poll.return_value = None
        second_process.poll.return_value = None

        def immediate_thread(*, target: object, daemon: bool) -> Mock:
            del daemon
            thread = Mock()
            thread.start.side_effect = target
            return thread

        with (
            patch.object(
                tor_module,
                "_start_owned_process_with_private_directory",
                side_effect=[first_process, second_process],
            ),
            patch(
                "hbrowser.gallery.browser.tor.threading.Thread",
                side_effect=immediate_thread,
            ),
            patch("hbrowser.gallery.browser.tor.atexit.register") as register_cleanup,
            patch(
                "hbrowser.gallery.browser.tor.atexit.unregister"
            ) as unregister_cleanup,
        ):
            first_result = tor_module.start_tor_with_retry(
                9150,
                max_retries=3,
                retry_wait=0,
            )
            first_cleanup = register_cleanup.call_args.args[0]
            tor_module.terminate_tor_process(first_process)
            tor_module.terminate_tor_process(first_process)

            second_result = tor_module.start_tor_with_retry(
                9151,
                max_retries=3,
                retry_wait=0,
            )
            second_cleanup = register_cleanup.call_args.args[0]
            tor_module.terminate_tor_process(second_process)
            tor_module.terminate_tor_process(second_process)

        self.assertIs(first_result, first_process)
        self.assertIs(second_result, second_process)
        self.assertIsInstance(first_cleanup, tor_module._TorProcessAtexitCleanup)
        self.assertIsInstance(second_cleanup, tor_module._TorProcessAtexitCleanup)
        self.assertIsNot(first_cleanup, second_cleanup)
        self.assertIsNone(first_cleanup._tor_process)
        self.assertIsNone(second_cleanup._tor_process)
        self.assertEqual(
            register_cleanup.call_args_list,
            [call(first_cleanup), call(second_cleanup)],
        )
        self.assertEqual(
            unregister_cleanup.call_args_list,
            [call(first_cleanup), call(second_cleanup)],
        )
        first_process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )
        second_process.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=unittest.mock.ANY,
        )
