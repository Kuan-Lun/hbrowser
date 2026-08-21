import asyncio
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, call, patch

import zendriver as zd
from zendriver import cdp

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import factory
from hbrowser.gallery.browser import tor as tor_module
from hbrowser.gallery.browser.mapper import start_zendriver_mapper_janitor
from hbrowser.gallery.browser.process import OwnedProcess, ProcessOwnershipError
from hbrowser.gallery.browser.tor import terminate_tor_process
from hbrowser.gallery.utils import (
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
from hbrowser.gallery.utils.protocol import ZendriverOwnerRetiredError


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


class OwnedZendriverContractTests(unittest.IsolatedAsyncioTestCase):
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
                setattr(
                    browser,
                    "info",
                    SimpleNamespace(
                        webSocketDebuggerUrl=(
                            "ws://127.0.0.1:9222/devtools/browser/test"
                        )
                    ),
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

    def test_prelaunch_discovers_port_and_binds_owner_without_global_hook(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-contract-test-") as directory:
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
                    "_wait_for_devtools_active_port",
                    return_value=43123,
                ),
                patch.object(
                    factory,
                    "_OwnedZendriverBrowser",
                    return_value=browser,
                ),
            ):
                observed = factory._construct_owned_browser(
                    config,
                    cleanup_paths=(str(profile),),
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


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _Target:
    type_ = "page"
    url = "about:blank"


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
        self._process = _Process(returncode)

    @property
    def main_tab(self) -> zd.Tab | None:
        return self._main_tab

    @property
    def stopped(self) -> bool:
        return self._process.poll() is not None


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


class BrowserProcessShutdownTests(unittest.TestCase):
    def test_owned_browser_delegates_to_the_bounded_shutdown_state_machine(
        self,
    ) -> None:
        owner = Mock(spec=OwnedProcess)
        browser = SimpleNamespace(_process=None)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)

        factory._terminate_browser_process(browser)

        owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
        )
        owner.terminate.assert_not_called()
        owner.kill.assert_not_called()
        self.assertIsNone(getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE))

    def test_failed_private_release_retains_exact_browser_owner(self) -> None:
        owner = Mock(spec=OwnedProcess)
        owner.shutdown.side_effect = PermissionError("profile locked")
        browser = SimpleNamespace(_process=None)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)

        with self.assertRaises(ProcessOwnershipError) as raised:
            factory._terminate_browser_process(browser)

        self.assertIsInstance(raised.exception.__cause__, PermissionError)
        self.assertIs(
            getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE),
            owner,
        )
        owner.terminate.assert_not_called()
        owner.kill.assert_not_called()


class WaitForMainTabTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_timeout_includes_browser_process_and_target_state(self) -> None:
        browser = _Browser(None, tabs=[_Target()])

        with self.assertRaises(RuntimeError) as raised:
            await factory._wait_for_main_tab(
                cast(zd.Browser, browser),
                timeout=0,
            )

        message = str(raised.exception)
        self.assertIn("within 0.0 seconds", message)
        self.assertIn("stopped=False", message)
        self.assertIn("process_returncode=None", message)
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
                "stopped=True, process_returncode=17",
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
    async def test_success_starts_mapper_janitor_after_post_setup(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page])
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        events: list[str] = []

        async def post_setup(*_: object) -> None:
            events.append("post-setup")

        def start_janitor(_: object) -> None:
            events.append("start-janitor")

        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser",
                return_value=browser,
            ),
            patch.object(factory, "_register_browser_atexit") as register_atexit,
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(side_effect=post_setup),
            ),
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
                side_effect=start_janitor,
            ) as start_mapper_janitor,
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            result = await factory.create_browser()

        self.assertEqual(result, (browser, page))
        self.assertEqual(events, ["post-setup", "start-janitor"])
        start_mapper_janitor.assert_called_once_with(browser)
        register_atexit.assert_called_once_with(browser)

    async def test_main_tab_failure_stops_browser_without_post_setup(self) -> None:
        browser = _Browser(None)
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        failure = RuntimeError("main tab unavailable")

        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                factory,
                "_construct_owned_browser",
                return_value=browser,
            ),
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
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
            ) as start_mapper_janitor,
            self.assertRaisesRegex(RuntimeError, "main tab unavailable"),
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            await factory.create_browser()

        stop_browser.assert_awaited_once_with(browser)
        post_setup.assert_not_awaited()
        start_mapper_janitor.assert_not_called()

    async def test_tor_start_cancellation_waits_for_result_then_reaps(self) -> None:
        tor_started = threading.Event()
        allow_tor_start = threading.Event()
        tor_process = Mock()

        def delayed_tor_start(_: int) -> Mock:
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
            ),
            patch.object(factory, "terminate_tor_process") as terminate_tor,
            patch.object(factory, "configure_proxy") as configure_proxy,
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

        terminate_tor.assert_called_once_with(tor_process)
        configure_proxy.assert_not_called()

    async def test_browser_launch_cancellation_stops_returned_browser(self) -> None:
        launch_started = asyncio.Event()
        allow_launch = asyncio.Event()
        browser = SimpleNamespace(connection=None)

        async def delayed_launch() -> object:
            launch_started.set()
            await allow_launch.wait()
            return browser

        browser.start = AsyncMock(side_effect=delayed_launch)

        async def cleanup_started_browser(actual_browser: object) -> None:
            self.assertIs(actual_browser, browser)
            allow_launch.set()
            await asyncio.sleep(0)

        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(factory, "_construct_owned_browser", return_value=browser),
            patch.object(factory, "_register_browser_atexit"),
            patch.object(
                factory,
                "stop_browser",
                new=AsyncMock(side_effect=cleanup_started_browser),
            ) as stop_browser,
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            create_task = asyncio.create_task(factory._create_browser(True))
            await launch_started.wait()
            create_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await create_task

        stop_browser.assert_awaited_once_with(browser)

    async def test_browser_start_failure_cleans_exact_process_and_profile(
        self,
    ) -> None:
        process = Mock()
        profile_cleanup = AsyncMock()
        startup_error = RuntimeError("target discovery failed")
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            _process=None,
            _process_pid=None,
            _cleanup_temporary_profile=profile_cleanup,
        )

        async def failed_start() -> object:
            browser._process = process
            browser._process_pid = 123
            raise startup_error

        async def stop_started_browser() -> None:
            process.terminate()
            process.wait(timeout=3)
            browser._process = None
            browser._process_pid = None
            await profile_cleanup()

        browser.start = AsyncMock(side_effect=failed_start)
        browser.stop = AsyncMock(side_effect=stop_started_browser)
        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(factory, "_construct_owned_browser", return_value=browser),
            patch.object(factory, "_register_browser_atexit"),
            self.assertRaises(RuntimeError) as raised,
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            await factory._create_browser(True)

        self.assertIs(raised.exception, startup_error)
        browser.stop.assert_awaited_once_with()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)
        profile_cleanup.assert_awaited_once_with()
        self.assertIsNone(browser._process)
        self.assertIsNone(browser._process_pid)

    async def test_start_failure_keeps_atexit_fallback_until_retry_succeeds(
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
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(factory, "_construct_owned_browser", return_value=browser),
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
            ensure_chrome.return_value.chrome = "/test/chrome"
            with self.assertRaises(ProcessOwnershipError) as raised:
                await factory._create_browser(True)

            self.assertIs(raised.exception.__cause__, cleanup_error)
            self.assertIn(
                "Startup failure type: RuntimeError",
                raised.exception.__notes__,
            )
            self.assertEqual(len(registered_cleanups), 1)
            cleanup = registered_cleanups[0]
            self.assertIs(cleanup.browser, browser)
            self.assertIn(browser, registered_instances)
            unregister_cleanup.assert_not_called()

            await cleanup()

            self.assertIsNone(cleanup.browser)
            self.assertIsNone(
                getattr(browser, factory._BROWSER_ATEXIT_CLEANUP_ATTRIBUTE)
            )
            self.assertNotIn(browser, registered_instances)
            # The running callback is left in the registry's current iteration;
            # asyncio-atexit clears the list after all browser callbacks run.
            unregister_cleanup.assert_not_called()

        register_cleanup.assert_called_once()
        self.assertEqual(browser.stop.await_count, 2)

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

    async def test_atexit_callback_retries_before_the_loop_closes(self) -> None:
        browser = _OwnedBrowser()
        browser.stop.side_effect = [RuntimeError("transient stop failure"), None]
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

            await cleanup()

        self.assertEqual(browser.stop.await_count, 2)
        self.assertIsNone(cleanup.browser)
        self.assertNotIn(browser, registered_instances)
        # Avoid mutating asyncio-atexit's live callback iteration. The package
        # clears every callback only after all registered browsers have run.
        unregister_cleanup.assert_not_called()

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

        async def blocked_setup(*_: object) -> None:
            setup_started.set()
            await asyncio.Event().wait()

        cleanup_error = RuntimeError("cleanup failed")
        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(factory, "_construct_owned_browser", return_value=browser),
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
            ensure_chrome.return_value.chrome = "/test/chrome"
            create_task = asyncio.create_task(factory._create_browser(True))
            await setup_started.wait()
            create_task.cancel()
            with self.assertRaises(ProcessOwnershipError) as raised:
                await create_task

        stop_browser.assert_awaited_once_with(browser)
        self.assertIs(raised.exception.__cause__, cleanup_error)
        self.assertIn(
            "Startup failure type: CancelledError",
            raised.exception.__notes__,
        )

    async def test_stop_waits_for_mapper_janitor_before_browser(self) -> None:
        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        events: list[str] = []
        janitor = start_zendriver_mapper_janitor(browser)
        operation = asyncio.create_task(asyncio.Event().wait())
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        async def stop_process() -> None:
            self.assertTrue(janitor.done())
            self.assertFalse(operation.cancelled())
            events.append("stop-browser")

        browser.stop.side_effect = stop_process
        await factory.stop_browser(browser)

        self.assertEqual(events, ["stop-browser"])
        self.assertTrue(janitor.cancelled())
        self.assertTrue(operation.cancelled())
        browser.stop.assert_awaited_once_with()

    async def test_browser_stop_failure_retires_when_no_connection_is_live(
        self,
    ) -> None:
        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=RuntimeError("browser stop failed")),
        )
        janitor = start_zendriver_mapper_janitor(browser)
        operation: asyncio.Future[None] = asyncio.Future()
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        with self.assertRaisesRegex(RuntimeError, "browser stop failed"):
            await factory.stop_browser(browser)

        self.assertTrue(janitor.done())
        self.assertTrue(janitor.cancelled())
        self.assertTrue(operation.cancelled())

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
        operation: asyncio.Future[None] = asyncio.Future()
        connection.mapper[1] = operation
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=connection)

        with self.assertRaisesRegex(RuntimeError, "browser stop failed"):
            await factory.stop_browser(browser)

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
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=target)

        with self.assertRaisesRegex(RuntimeError, "connection close failed"):
            await factory.stop_browser(browser)

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

    async def test_completed_shutdown_reopens_for_new_exact_connection(
        self,
    ) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
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

            terminate_tor.assert_called_once_with(tor_process)

        browser.stop.assert_awaited_once_with()
        self.assertTrue(operation.cancelled())

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
            terminate_tor.assert_called_once_with(tor_process)

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
            self.assertRaisesRegex(TimeoutError, "listener did not stop"),
        ):
            await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

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

    async def test_browser_stop_watchdog_bounds_a_hung_stop(self) -> None:
        async def blocking_stop() -> None:
            await asyncio.Event().wait()

        browser = SimpleNamespace(
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=blocking_stop),
        )
        operation: asyncio.Future[None] = asyncio.Future()
        with self.assertRaises(ZendriverOperationTimeout):
            await wait_for_zendriver(operation, timeout=0, owner=browser)

        with (
            patch.object(factory, "_BROWSER_STOP_TIMEOUT_SECONDS", 0.01),
            self.assertRaisesRegex(TimeoutError, "browser stop exceeded"),
        ):
            await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

        self.assertTrue(operation.cancelled())

    async def test_timed_out_browser_stop_settles_and_is_reused_on_retry(
        self,
    ) -> None:
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()
        stop_finished = asyncio.Event()
        process = Mock()

        async def delayed_stop() -> None:
            stop_started.set()
            await allow_stop.wait()
            process.terminate()
            process.wait(timeout=3)
            browser._process = None
            stop_finished.set()

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            _process=process,
            stop=AsyncMock(side_effect=delayed_stop),
        )
        with (
            patch.object(factory, "_BROWSER_STOP_TIMEOUT_SECONDS", 0.01),
            self.assertRaisesRegex(TimeoutError, "browser stop exceeded"),
        ):
            await factory.stop_browser(browser)

        await stop_started.wait()
        process.terminate.assert_not_called()
        allow_stop.set()
        await stop_finished.wait()

        with patch.object(factory, "_BROWSER_STOP_TIMEOUT_SECONDS", 0.01):
            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)

    async def test_browser_stop_failure_retries_remaining_process_cleanup(
        self,
    ) -> None:
        process = Mock()
        attempts = 0

        async def stop_process() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("stop interrupted")
            process.terminate()
            process.wait(timeout=3)
            browser._process = None

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            _process=process,
            stop=AsyncMock(side_effect=stop_process),
        )

        with self.assertRaisesRegex(RuntimeError, "stop interrupted"):
            await factory.stop_browser(browser)
        self.assertIs(browser._process, process)

        await factory.stop_browser(browser)

        self.assertEqual(browser.stop.await_count, 2)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(browser._process)

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
        with patch.object(
            factory,
            "terminate_tor_process",
            side_effect=[RuntimeError("tor reap failed"), None],
        ) as terminate_tor:
            with self.assertRaisesRegex(RuntimeError, "tor reap failed"):
                await factory.stop_browser(browser)

            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
        self.assertEqual(
            terminate_tor.call_args_list,
            [call(tor_process), call(tor_process)],
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
                self.assertRaisesRegex(
                    TimeoutError,
                    "operations did not accept cancellation",
                ),
            ):
                await asyncio.wait_for(factory.stop_browser(browser), timeout=1)

            self.assertFalse(operation.done())
        finally:
            allow_stop = True
            operation.cancel()
            await operation

    async def test_janitor_stop_failure_still_stops_browser(self) -> None:
        browser = Mock()
        browser._tor_process = None
        browser.stop = AsyncMock()
        failure = RuntimeError("janitor stop failed")

        with (
            patch.object(
                factory,
                "stop_zendriver_mapper_janitor",
                new=AsyncMock(side_effect=failure),
            ),
            self.assertRaisesRegex(RuntimeError, "janitor stop failed"),
        ):
            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()

    async def test_stubborn_janitor_keeps_shutdown_retryable_until_terminal(
        self,
    ) -> None:
        allow_stop = False

        async def stubborn_janitor() -> None:
            nonlocal allow_stop
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if allow_stop:
                        return

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        janitor = asyncio.create_task(stubborn_janitor())
        setattr(browser, "_hbrowser_zendriver_mapper_janitor_task", janitor)
        await asyncio.sleep(0)
        with patch.object(factory, "_JANITOR_STOP_TIMEOUT_SECONDS", 0.01):
            for _ in range(2):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "mapper janitor stop exceeded",
                ):
                    await factory.stop_browser(browser)
                self.assertFalse(janitor.done())

            allow_stop = True
            janitor.cancel()
            await janitor
            await asyncio.sleep(0)
            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()


class TorProcessCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        data_directory = patch.object(
            tempfile,
            "mkdtemp",
            return_value="/private/tmp/hbrowser-tor-unit-data",
        )
        data_directory.start()
        self.addCleanup(data_directory.stop)

    def test_tor_process_is_launched_outside_the_terminal_group(self) -> None:
        process = Mock(stdout=[b"Bootstrapped 100%: Done\n"])
        process.poll.return_value = None
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ) as start_process,
            patch.object(tor_module, "_find_tor_binary", return_value="/tor"),
            patch("hbrowser.gallery.browser.tor.atexit.register"),
        ):
            observed = tor_module._start_tor_process(9150)

        self.assertIs(observed, process)
        start_process.assert_called_once()
        self.assertEqual(start_process.call_args.args[0], "/tor")
        self.assertEqual(start_process.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(start_process.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(len(start_process.call_args.kwargs["cleanup_paths"]), 1)
        terminate_tor_process(process)

    def test_cleanup_attachment_failure_reaps_exact_process_and_preserves_error(
        self,
    ) -> None:
        class _UnattachableProcess:
            __slots__ = ("stdout", "terminate", "wait")

            def __init__(self) -> None:
                self.stdout: list[bytes] = []
                self.terminate = Mock()
                self.wait = Mock()

        process = _UnattachableProcess()
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ),
            patch(
                "hbrowser.gallery.browser.tor.atexit.register",
            ) as register_cleanup,
            self.assertRaises(AttributeError) as raised,
        ):
            tor_module._start_tor_process(9150)

        self.assertIn("_hbrowser_tor_process_cleanup", str(raised.exception))
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        register_cleanup.assert_not_called()

    def test_force_killed_tor_is_always_reaped(self) -> None:
        process = Mock()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="tor", timeout=5),
            0,
        ]

        terminate_tor_process(process)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=5), call(timeout=5)],
        )

    def test_missing_stdout_reaps_unowned_bootstrap_process(self) -> None:
        process = Mock(stdout=None)
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ),
            self.assertRaisesRegex(RuntimeError, "capture Tor process output"),
        ):
            tor_module._start_tor_process(9150)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_early_bootstrap_exit_is_reaped_before_error(self) -> None:
        process = Mock(stdout=[])
        process.poll.return_value = 7
        process.returncode = 7
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ),
            self.assertRaisesRegex(RuntimeError, "exited unexpectedly with code 7"),
        ):
            tor_module._start_tor_process(9150)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_bootstrap_timeout_terminates_and_reaps_process(self) -> None:
        process = Mock(stdout=[])
        process.poll.return_value = None
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ),
            patch(
                "hbrowser.gallery.browser.tor.time.time",
                side_effect=[0, 121],
            ),
            self.assertRaisesRegex(RuntimeError, "failed to bootstrap"),
        ):
            tor_module._start_tor_process(9150)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

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

        start_process.assert_called_once_with(9150)

    def test_failed_bootstrap_retains_exact_process_cleanup_for_atexit(
        self,
    ) -> None:
        process = Mock(stdout=None)
        retained_cleanups: list[tor_module._TorProcessAtexitCleanup] = []

        def retain_cleanup(
            cleanup: tor_module._TorProcessAtexitCleanup,
        ) -> tor_module._TorProcessAtexitCleanup:
            retained_cleanups.append(cleanup)
            return cleanup

        with (
            patch.object(
                tor_module,
                "start_owned_process",
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
                [call(process), call(process)],
            )

    def test_two_normal_generations_each_release_their_atexit_cleanup(self) -> None:
        first_process = Mock(stdout=[b"Bootstrapped 100%: Done\n"])
        second_process = Mock(stdout=[b"Bootstrapped 100%: Done\n"])
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
                "start_owned_process",
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
        first_process.terminate.assert_called_once_with()
        first_process.wait.assert_called_once_with(timeout=5)
        second_process.terminate.assert_called_once_with()
        second_process.wait.assert_called_once_with(timeout=5)
