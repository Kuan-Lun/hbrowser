import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
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
    Deadline,
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
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


class OwnedZendriverContractTests(unittest.IsolatedAsyncioTestCase):
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
        with tempfile.TemporaryDirectory(prefix="hbrowser-late-client-") as directory:
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
                    cleanup_paths=(str(profile),),
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

    async def test_prelaunch_discovers_port_and_binds_owner_without_global_hook(
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
                    cleanup_paths=(str(profile),),
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


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


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


class BrowserProcessShutdownTests(unittest.TestCase):
    def test_owned_browser_delegates_to_the_bounded_shutdown_state_machine(
        self,
    ) -> None:
        owner = Mock(spec=OwnedProcess)
        browser = SimpleNamespace(_process=None)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)

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
        self.assertIsNone(getattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE))

    def test_failed_private_release_retains_exact_browser_owner(self) -> None:
        owner = Mock(spec=OwnedProcess)
        owner.shutdown.side_effect = PermissionError("profile locked")
        browser = SimpleNamespace(_process=None)
        setattr(browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE, owner)

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
    async def test_chrome_install_worker_receipt_is_nonce_bound_and_reaped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-install-receipt-"
        ) as directory:
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
            start.call_args.kwargs["cleanup_paths"],
            (staging_directory,),
        )
        cleanup_receipt.assert_awaited_once_with(
            (str(receipt_directory), str(staging_directory)),
            deadline=cleanup_deadline,
        )

    async def test_chrome_install_cancellation_reaps_worker_before_return(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-install-cancel-"
        ) as directory:
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
        with patch(
            "hbrowser.gallery.browser.factory._PrivateDirectory.capture",
            return_value=guard,
        ):
            cleanup = asyncio.create_task(
                factory._cleanup_unowned_private_paths(
                    ("/private/profile",),
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

        with patch(
            "hbrowser.gallery.browser.factory._PrivateDirectory.capture",
            side_effect=guards,
        ):
            errors = await factory._cleanup_unowned_private_paths(
                ("/private/profile", "/private/extension"),
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

    async def test_success_starts_mapper_janitor_after_post_setup(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page])
        browser.start = AsyncMock(return_value=browser)  # type: ignore[attr-defined]
        events: list[str] = []

        async def post_setup(*_: object, **__: object) -> None:
            events.append("post-setup")

        def start_janitor(_: object) -> None:
            events.append("start-janitor")

        with (
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
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
                side_effect=start_janitor,
            ) as start_mapper_janitor,
        ):
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
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
            ) as start_mapper_janitor,
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
        start_mapper_janitor.assert_not_called()

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
        configure_proxy.assert_not_called()

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

        with tempfile.TemporaryDirectory(
            prefix="hbrowser-construction-test-"
        ) as directory:
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
                        cleanup_paths=(str(profile),),
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
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3)
        profile_cleanup.assert_awaited_once_with()
        self.assertIsNone(browser._process)
        self.assertIsNone(browser._process_pid)

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

        lifecycle = getattr(browser, "_hbrowser_zendriver_lifecycle")
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
            lifecycle = getattr(browser, "_hbrowser_zendriver_lifecycle")
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

    async def test_browser_stop_failure_uses_explicit_fallback_after_retirement(
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

            terminate_tor.assert_called_once_with(
                tor_process,
                deadline=unittest.mock.ANY,
            )

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

    async def test_browser_stop_watchdog_runs_process_fallback_in_same_call(
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

    async def test_timed_out_browser_stop_does_not_reset_on_retry(
        self,
    ) -> None:
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()
        process = Mock()

        async def delayed_stop() -> None:
            stop_started.set()
            await allow_stop.wait()
            process.terminate()
            process.wait(timeout=3)
            browser._process = None

        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            _process=process,
            stop=AsyncMock(side_effect=delayed_stop),
        )
        with patch.object(factory, "_BROWSER_STOP_TIMEOUT_SECONDS", 0.01):
            await factory.stop_browser(browser)

        await stop_started.wait()
        process.terminate.assert_called_once_with()

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

        await factory.stop_browser(browser)

        self.assertEqual(browser.stop.await_count, 1)
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
            self.assertRaises(ProcessOwnershipError) as raised,
        ):
            await factory.stop_browser(browser)

        self.assertIn("janitor stop failed", str(raised.exception.__cause__))
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
                with self.assertRaises(ProcessOwnershipError):
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
        tor_binary = patch.object(
            tor_module,
            "_find_tor_binary",
            return_value="/tor",
        )
        tor_binary.start()
        self.addCleanup(tor_binary.stop)

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
        deadline = Mock(
            expires_at=time.monotonic() + 1.0,
            expired=True,
        )
        deadline.remaining.return_value = 1.0
        with (
            patch.object(
                tor_module,
                "start_owned_process",
                return_value=process,
            ),
            patch(
                "hbrowser.gallery.browser.tor.Deadline.after",
                return_value=deadline,
            ),
            self.assertRaisesRegex(RuntimeError, "failed to bootstrap"),
        ):
            tor_module._start_tor_process(9150)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=1.0)

    def test_expired_shared_deadline_does_not_start_tor(self) -> None:
        with (
            patch.object(tor_module, "start_owned_process") as start_process,
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
                [
                    call(process, deadline=unittest.mock.ANY),
                    call(process, deadline=None),
                ],
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
