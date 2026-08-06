from __future__ import annotations

import asyncio
import os
import stat
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.utils import diagnostic as diagnostic_module
from hbrowser.gallery.utils.diagnostic import write_page_diagnostic


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class PageDiagnosticTests(unittest.TestCase):
    def test_diagnostics_are_unique_and_private(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = write_page_diagnostic(directory, "driver_error", "first")
            second = write_page_diagnostic(directory, "driver_error", "second")

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(), "first")
            self.assertEqual(second.read_text(), "second")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(second.stat().st_mode), 0o600)

    def test_diagnostics_keep_only_the_newest_files(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_FILE_LIMIT",
                    2,
                ),
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_TOTAL_BYTES",
                    1024,
                ),
            ):
                first = write_page_diagnostic(directory, "driver_error", "first")
                second = write_page_diagnostic(directory, "challenge_page", "second")
                third = write_page_diagnostic(directory, "download_timeout", "third")

            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())

    def test_oversized_diagnostic_is_truncated(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_MAX_FILE_BYTES",
                    128,
                ),
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_TOTAL_BYTES",
                    256,
                ),
            ):
                path = write_page_diagnostic(directory, "driver_error", "x" * 1000)

            content = path.read_bytes()
            self.assertLessEqual(len(content), 128)
            self.assertIn(b"hbrowser page diagnostic truncated", content)

    def test_diagnostics_obey_the_total_byte_budget(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_MAX_FILE_BYTES",
                    10,
                ),
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_TOTAL_BYTES",
                    10,
                ),
            ):
                first = write_page_diagnostic(directory, "driver_error", "1111")
                second = write_page_diagnostic(directory, "challenge_page", "2222")
                third = write_page_diagnostic(
                    directory,
                    "download_timeout",
                    "333333",
                )

            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())
            self.assertLessEqual(
                sum(path.stat().st_size for path in (second, third)),
                10,
            )

    def test_random_encounter_query_values_are_redacted_from_html(self) -> None:
        first_secret = "FIRST-SHORT-LIVED-SECRET="
        second_secret = "SECOND-SHORT-LIVED-SECRET"
        html = (
            '<html><body><div id="eventpane">'
            '<a class="trusted" href="https://hentaiverse.org/?s=Battle&amp;'
            f'ss=ba&amp;encounter={first_secret}">Fight</a>'
            '<script>const retry = "https://hentaiverse.org/?encounter='
            f'{second_secret}&ss=ba&s=Battle";</script>'
            '<a href="https://example.org/?notencounter=VISIBLE">Context</a>'
            "</div></body></html>"
        )

        with TemporaryDirectory() as directory_name:
            path = write_page_diagnostic(
                Path(directory_name),
                "driver_error",
                html,
            )
            diagnostic = path.read_text()

        self.assertNotIn(first_secret, diagnostic)
        self.assertNotIn(second_secret, diagnostic)
        self.assertIn("s=Battle&amp;ss=ba&amp;encounter=REDACTED", diagnostic)
        self.assertIn("?encounter=REDACTED&ss=ba&s=Battle", diagnostic)
        self.assertIn('id="eventpane"', diagnostic)
        self.assertIn("?notencounter=VISIBLE", diagnostic)

    def test_kind_cannot_escape_the_diagnostic_directory(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)

            with self.assertRaisesRegex(ValueError, "Invalid page diagnostic kind"):
                write_page_diagnostic(directory, "../outside", "content")


class DriverExitDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.driver = _TestDriver(headless=True)
        self.logger = Mock()
        self.driver.logger = self.logger
        self.driver.page = Mock()
        self.driver.page.get_content = AsyncMock(return_value="<html>failure</html>")
        self.driver.browser = Mock()

    async def test_exit_keeps_each_error_page_and_logs_safe_error_types(self) -> None:
        try:
            raise ValueError("broken")
        except ValueError as error:
            traceback = error.__traceback__

            with (
                TemporaryDirectory() as directory_name,
                patch(
                    "hbrowser.gallery.driver_base.get_log_dir",
                    return_value=Path(directory_name),
                ),
                patch(
                    "hbrowser.gallery.driver_base.stop_browser",
                    new=AsyncMock(),
                ) as stop_browser,
            ):
                await self.driver.__aexit__(ValueError, error, traceback)
                await self.driver.__aexit__(ValueError, error, traceback)
                diagnostics = sorted(Path(directory_name).glob("driver_error_*.html"))
                diagnostic_contents = [path.read_text() for path in diagnostics]

        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(
            diagnostic_contents,
            ["<html>failure</html>", "<html>failure</html>"],
        )
        self.logger.error.assert_any_call(
            "Browser session exiting after error: error_type=%s cause_type=%s",
            "ValueError",
            "none",
        )
        self.assertEqual(stop_browser.await_count, 2)

    async def test_exit_diagnostic_never_persists_encounter_secret(self) -> None:
        secret = "EXIT-DIAGNOSTIC-ENCOUNTER-SECRET="
        self.driver.page.get_content.return_value = (
            '<html><div id="eventpane"><a href="https://hentaiverse.org/'
            f'?s=Battle&amp;ss=ba&amp;encounter={secret}">Fight</a></div></html>'
        )
        error = RuntimeError("campaign interrupted")

        with (
            TemporaryDirectory() as directory_name,
            patch(
                "hbrowser.gallery.driver_base.get_log_dir",
                return_value=Path(directory_name),
            ),
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                new=AsyncMock(),
            ),
        ):
            await self.driver.__aexit__(RuntimeError, error, error.__traceback__)
            diagnostics = list(Path(directory_name).glob("driver_error_*.html"))
            self.assertEqual(len(diagnostics), 1)
            diagnostic = diagnostics[0].read_text()

        self.assertNotIn(secret, diagnostic)
        self.assertIn("encounter=REDACTED", diagnostic)
        self.assertIn('id="eventpane"', diagnostic)

    async def test_exit_log_does_not_render_chained_exception_detail(self) -> None:
        sentinel = "SENSITIVE-BROWSER-CAUSE\nSECOND-LINE"
        cause = RuntimeError(sentinel)
        error = ValueError("safe outer error")
        error.__cause__ = cause

        with patch(
            "hbrowser.gallery.driver_base.stop_browser",
            new=AsyncMock(),
        ):
            await self.driver.__aexit__(ValueError, error, error.__traceback__)

        self.logger.error.assert_called_once_with(
            "Browser session exiting after error: error_type=%s cause_type=%s",
            "ValueError",
            "RuntimeError",
        )
        self.assertNotIn(sentinel, repr(self.logger.method_calls))

    async def test_page_write_runs_outside_the_event_loop_thread(self) -> None:
        event_loop_thread = threading.get_ident()
        writer_thread: int | None = None

        def write_diagnostic(kind: str, content: str) -> Path:
            nonlocal writer_thread
            writer_thread = threading.get_ident()
            self.assertEqual(kind, "driver_error")
            self.assertEqual(content, "content")
            return Path("diagnostic.html")

        with patch.object(
            self.driver,
            "_write_page_diagnostic",
            side_effect=write_diagnostic,
        ):
            path = await self.driver._save_page_diagnostic(
                "driver_error",
                "content",
            )

        self.assertEqual(path, Path("diagnostic.html"))
        self.assertIsNotNone(writer_thread)
        self.assertNotEqual(writer_thread, event_loop_thread)

    async def test_save_failure_is_visible_and_browser_still_closes(self) -> None:
        write_error = OSError("disk full")
        stop_browser = AsyncMock()
        with (
            patch.object(
                self.driver,
                "_write_page_diagnostic",
                side_effect=write_error,
            ),
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                new=stop_browser,
            ),
        ):
            await self.driver.__aexit__(RuntimeError, RuntimeError("battle"), None)

        self.logger.warning.assert_any_call(
            "Failed to save %s page diagnostic: error_type=%s",
            "driver_error",
            "OSError",
        )
        stop_browser.assert_awaited_once_with(self.driver.browser)

    async def test_capture_failure_reports_only_the_safe_error_type(self) -> None:
        capture_error = RuntimeError("session detached")
        self.driver.page.get_content.side_effect = capture_error

        path = await self.driver._save_page_diagnostic("driver_error")

        self.assertIsNone(path)
        self.logger.warning.assert_called_once_with(
            "Failed to capture %s page diagnostic: error_type=%s",
            "driver_error",
            "RuntimeError",
        )

    async def test_page_capture_has_a_timeout(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        cancelled = False

        async def returns_late() -> str:
            nonlocal cancelled
            started.set()
            try:
                await release.wait()
                return "<html>late</html>"
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                finished.set()

        self.driver.page.get_content.side_effect = returns_late
        with patch(
            "hbrowser.gallery.driver_base._PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS",
            0.001,
        ):
            path = await self.driver._save_page_diagnostic("driver_error")

        self.assertTrue(started.is_set())
        self.assertIsNone(path)
        self.assertFalse(cancelled)
        self.logger.warning.assert_called_once_with(
            "Failed to capture %s page diagnostic: error_type=%s",
            "driver_error",
            "TimeoutError",
        )

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertFalse(cancelled)

    async def test_late_capture_exception_is_observed(self) -> None:
        release = asyncio.Event()
        finished = asyncio.Event()
        loop_errors: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()

        async def fails_late() -> str:
            try:
                await release.wait()
                raise RuntimeError("late protocol failure")
            finally:
                finished.set()

        self.driver.page.get_content.side_effect = fails_late
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            with patch(
                "hbrowser.gallery.driver_base._PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS",
                0.001,
            ):
                path = await self.driver._save_page_diagnostic("driver_error")

            self.assertIsNone(path)
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertEqual(loop_errors, [])

    async def test_browser_close_failure_is_visible(self) -> None:
        close_error = RuntimeError("process did not exit")
        with patch(
            "hbrowser.gallery.driver_base.stop_browser",
            new=AsyncMock(side_effect=close_error),
        ):
            await self.driver.__aexit__(None, None, None)

        self.logger.warning.assert_called_once_with(
            "Failed to close browser cleanly: error_type=%s",
            "RuntimeError",
        )
