from __future__ import annotations

import asyncio
import sys
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from hbrowser.gallery.browser import factory
from hbrowser.gallery.browser.process import ProcessOwnershipError
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.utils.deadline import Deadline


class _Driver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


def _driver(*, owned: bool) -> _Driver:
    driver = _Driver()
    driver.logger = Mock()
    driver.browser = SimpleNamespace()
    driver.page = SimpleNamespace()
    driver._owns_browser = owned
    return driver


class DriverShutdownPriorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_ownership_survives_cancelled_body_in_task_group(
        self,
    ) -> None:
        for owned in (False, True):
            with self.subTest(owned=owned):
                driver = _driver(owned=owned)
                failure = ProcessOwnershipError("diagnostic worker unproven")
                failure.add_note("original ownership evidence")
                cancellation = asyncio.CancelledError("body cancelled")
                stop = AsyncMock()

                async def body(
                    driver: _Driver = driver,
                    cancellation: asyncio.CancelledError = cancellation,
                ) -> None:
                    await driver.__aexit__(type(cancellation), cancellation, None)
                    raise cancellation

                with (
                    patch.object(
                        driver, "save_page_diagnostic", AsyncMock(side_effect=failure)
                    ),
                    patch("hbrowser.gallery.driver_base.stop_browser", stop),
                    self.assertRaises(ExceptionGroup) as raised,
                ):
                    async with asyncio.TaskGroup() as group:
                        group.create_task(body())
                self.assertEqual(raised.exception.exceptions, (failure,))
                self.assertIn("original ownership evidence", failure.__notes__)
                self.assertTrue(
                    any("CancelledError" in note for note in failure.__notes__)
                )
                self.assertEqual(stop.await_count, int(owned))

    async def test_cleanup_ownership_survives_diagnostic_cancellation(self) -> None:
        driver = _driver(owned=True)
        failure = ProcessOwnershipError("browser cleanup unproven")
        cancellation = asyncio.CancelledError("diagnostic cancelled")
        primary = ValueError("body failed")

        async def body() -> None:
            await driver.__aexit__(type(primary), primary, None)
            raise primary

        with (
            patch.object(
                driver, "save_page_diagnostic", AsyncMock(side_effect=cancellation)
            ),
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                AsyncMock(side_effect=failure),
            ),
            self.assertRaises(ExceptionGroup) as raised,
        ):
            async with asyncio.TaskGroup() as group:
                group.create_task(body())
        self.assertEqual(raised.exception.exceptions, (failure,))
        self.assertTrue(any("CancelledError" in note for note in failure.__notes__))
        self.assertTrue(any("ValueError" in note for note in failure.__notes__))

    async def test_existing_ownership_error_survives_later_cancellation(self) -> None:
        driver = _driver(owned=True)
        failure = ProcessOwnershipError("body already lost process ownership")

        async def body() -> None:
            await driver.__aexit__(type(failure), failure, None)
            raise failure

        with (
            patch.object(
                driver,
                "save_page_diagnostic",
                AsyncMock(side_effect=asyncio.CancelledError()),
            ),
            patch("hbrowser.gallery.driver_base.stop_browser", AsyncMock()),
            self.assertRaises(ExceptionGroup) as raised,
        ):
            async with asyncio.TaskGroup() as group:
                group.create_task(body())
        self.assertEqual(raised.exception.exceptions, (failure,))

    async def test_ordinary_failures_keep_existing_body_error(self) -> None:
        for owned in (False, True):
            with self.subTest(owned=owned):
                driver = _driver(owned=owned)
                primary = ValueError("body failed")
                with (
                    patch.object(
                        driver,
                        "save_page_diagnostic",
                        AsyncMock(side_effect=OSError("diagnostic failed")),
                    ),
                    patch("hbrowser.gallery.driver_base.stop_browser", AsyncMock()),
                ):
                    await driver.__aexit__(type(primary), primary, None)
                self.assertTrue(any("OSError" in note for note in primary.__notes__))

    async def test_clean_shutdown_preserves_cancellation(self) -> None:
        driver = _driver(owned=True)
        cancellation = asyncio.CancelledError("diagnostic cancelled")
        with (
            patch.object(
                driver, "save_page_diagnostic", AsyncMock(side_effect=cancellation)
            ),
            patch("hbrowser.gallery.driver_base.stop_browser", AsyncMock()),
            self.assertRaises(asyncio.CancelledError) as raised,
        ):
            await driver.__aexit__(ValueError, ValueError("body failed"), None)
        self.assertIs(raised.exception, cancellation)


class BrowserCleanupPriorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_startup_routes_preserve_cancellation_and_ownership(
        self,
    ) -> None:
        for route in ("install", "chrome", "tor"):
            for cancel, ownership in ((False, False), (True, False), (True, True)):
                with self.subTest(route=route, cancel=cancel, ownership=ownership):
                    await self._check_public_startup(route, cancel, ownership)

    async def _check_public_startup(
        self, route: str, cancel: bool, ownership: bool
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        failure = (
            ProcessOwnershipError("synthetic startup ownership unresolved")
            if ownership
            else TimeoutError("synthetic startup failed without starting a process")
        )

        def fail_startup(*_args: object, **_kwargs: object) -> None:
            started.set()
            if not release.wait(3):
                raise AssertionError("test did not release the startup thread")
            raise failure

        with TemporaryDirectory() as directory:
            root = Path(directory)

            def new_directory(*, prefix: str) -> str:
                destination = root / prefix.rstrip("-")
                destination.mkdir()
                return str(destination)

            staging = root / "install-staging"
            staging.mkdir()
            with ExitStack() as patches:
                patches.enter_context(
                    patch.object(
                        factory,
                        "_configured_chrome_executable",
                        return_value=None
                        if route == "install"
                        else "/synthetic/chrome",
                    )
                )
                patches.enter_context(
                    patch.object(factory, "configure_proxy", return_value=None)
                )
                patches.enter_context(
                    patch.object(factory, "should_use_tor", return_value=route == "tor")
                )
                patches.enter_context(
                    patch.object(factory, "find_available_port", return_value=12345)
                )
                patches.enter_context(
                    patch.object(
                        factory, "tempfile", SimpleNamespace(mkdtemp=new_directory)
                    )
                )
                patches.enter_context(
                    patch.object(factory, "_build_config", return_value=object())
                )
                patches.enter_context(
                    patch.object(
                        factory,
                        "_prepare_owned_browser_launch",
                        return_value=(Path("/synthetic/chrome"), []),
                    )
                )
                patches.enter_context(
                    patch.object(
                        factory,
                        "create_chrome_install_staging_root",
                        return_value=staging,
                    )
                )
                patches.enter_context(
                    patch.object(
                        factory,
                        "_chrome_install_worker_command",
                        return_value=(sys.executable, ()),
                    )
                )
                launch = patches.enter_context(
                    patch.object(
                        factory,
                        {
                            "install": "start_owned_process",
                            "chrome": "start_owned_browser_process",
                            "tor": "start_tor_with_retry",
                        }[route],
                        side_effect=fail_startup,
                    )
                )
                creation = asyncio.create_task(factory.create_browser())
                try:
                    async with asyncio.timeout(1):
                        while not started.is_set():
                            await asyncio.sleep(0)
                    if cancel:
                        creation.cancel("caller cancelled during startup")
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
                        self.assertFalse(creation.done())
                    release.set()
                    with self.assertRaises(BaseException) as raised:
                        await creation
                    if ownership:
                        self.assertIs(raised.exception, failure)
                        self.assertTrue(
                            any("cancel" in note.lower() for note in failure.__notes__)
                        )
                    elif cancel:
                        self.assertIsInstance(raised.exception, asyncio.CancelledError)
                        self.assertTrue(creation.cancelled())
                    else:
                        self.assertIs(raised.exception, failure)
                    self.assertEqual(launch.call_count, 1)
                finally:
                    release.set()
                    await asyncio.gather(creation, return_exceptions=True)

    async def test_startup_settlement_preserves_outcome_priority_for_all_timings(
        self,
    ) -> None:
        for completed in (False, True):
            for ownership in (False, True):
                with self.subTest(completed=completed, ownership=ownership):
                    release = asyncio.Event()
                    failure = (
                        ProcessOwnershipError("startup ownership unproven")
                        if ownership
                        else OSError("ordinary startup failure")
                    )
                    cancellation = asyncio.CancelledError("earlier caller cancel")

                    async def startup(
                        release: asyncio.Event = release,
                        failure: BaseException = failure,
                    ) -> None:
                        await release.wait()
                        raise failure

                    target = asyncio.create_task(startup())
                    if completed:
                        release.set()
                        await asyncio.gather(target, return_exceptions=True)
                    waiter = asyncio.create_task(
                        factory._settle_owned_startup_task(target, cancellation)
                    )
                    try:
                        await asyncio.sleep(0)
                        release.set()
                        with self.assertRaises(BaseException) as raised:
                            await waiter
                        expected = failure if ownership else cancellation
                        self.assertIs(raised.exception, expected)
                        self.assertTrue(expected.__notes__)
                        self.assertFalse(target.cancelled())
                    finally:
                        release.set()
                        await asyncio.gather(target, waiter, return_exceptions=True)

    async def test_successful_startup_keeps_first_repeated_cancellation(self) -> None:
        release = asyncio.Event()

        async def startup() -> object:
            await release.wait()
            return release

        target = asyncio.create_task(startup())
        waiter = asyncio.create_task(factory._settle_owned_startup_task(target))
        try:
            await asyncio.sleep(0)
            waiter.cancel("first")
            await asyncio.sleep(0)
            waiter.cancel("second")
            await asyncio.sleep(0)
            self.assertFalse(target.cancelled())
            self.assertFalse(waiter.done())
            release.set()
            result, cancellation = await waiter
            self.assertIs(result, release)
            self.assertIsInstance(cancellation, asyncio.CancelledError)
            assert cancellation is not None
            self.assertEqual(cancellation.args, ("first",))
        finally:
            release.set()
            await asyncio.gather(target, waiter, return_exceptions=True)

    async def test_unknown_cleanup_survives_caller_cancel_in_task_group(self) -> None:
        for expires in (False, True):
            with self.subTest(expires=expires):
                release = asyncio.Event()
                started = asyncio.Event()
                failure = ProcessOwnershipError("underlying cleanup unproven")

                async def cleanup(
                    release: asyncio.Event = release,
                    expires: bool = expires,
                    failure: ProcessOwnershipError = failure,
                ) -> None:
                    await release.wait()
                    if not expires:
                        raise failure

                cleanup_task = asyncio.create_task(cleanup())

                async def waiter(
                    started: asyncio.Event = started,
                    cleanup_task: asyncio.Task[None] = cleanup_task,
                    expires: bool = expires,
                ) -> None:
                    started.set()
                    await factory._await_cleanup_deferring_caller_cancellation(
                        cleanup_task, Deadline.after(0.05 if expires else 1)
                    )

                try:
                    with self.assertRaises(ExceptionGroup) as raised:
                        async with asyncio.TaskGroup() as group:
                            task = group.create_task(waiter())
                            await started.wait()
                            task.cancel("caller cancelled")
                            await asyncio.sleep(0)
                            if not expires:
                                release.set()
                    error = raised.exception.exceptions[0]
                    self.assertIsInstance(error, ProcessOwnershipError)
                    if not expires:
                        self.assertIs(error, failure)
                    self.assertTrue(
                        any("cancel" in note.lower() for note in error.__notes__)
                    )
                    self.assertFalse(cleanup_task.cancelled())
                finally:
                    release.set()
                    await asyncio.gather(cleanup_task, return_exceptions=True)

    async def test_settled_ordinary_failure_retains_caller_cancellation(self) -> None:
        release = asyncio.Event()

        async def cleanup() -> None:
            await release.wait()
            raise OSError("ordinary cleanup report failure")

        target = asyncio.create_task(cleanup())
        waiter = asyncio.create_task(
            factory._await_cleanup_deferring_caller_cancellation(
                target, Deadline.after(1)
            )
        )
        try:
            await asyncio.sleep(0)
            waiter.cancel("caller cancelled")
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
        finally:
            release.set()
            await asyncio.gather(target, waiter, return_exceptions=True)
