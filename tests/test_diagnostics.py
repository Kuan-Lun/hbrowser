from __future__ import annotations

import asyncio
import inspect
import io
import json
import multiprocessing
import os
import stat
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, AsyncMock, Mock, patch

from hbrowser import EHDriver
from hbrowser.gallery import utils as gallery_utils
from hbrowser.gallery.browser import page_diagnostic as page_diagnostic_module
from hbrowser.gallery.browser.process import ProcessOwnershipError
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.utils import (
    Deadline,
    ZendriverOperationTimeout,
)
from hbrowser.gallery.utils import diagnostic as diagnostic_module
from hbrowser.gallery.utils.protocol import ZendriverOwnerRetiredError


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


def _managed_page_diagnostic_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if diagnostic_module._PAGE_DIAGNOSTIC_FILENAME_PATTERN.fullmatch(path.name)
        and path.is_file()
        and not path.is_symlink()
    )


def write_page_diagnostic(
    directory: Path,
    kind: str,
    content: str,
    *,
    deadline: float | None = None,
) -> Path:
    """Exercise the worker-only synchronous core with a finite test deadline."""

    effective_deadline = time.monotonic() + 5.0 if deadline is None else deadline
    return diagnostic_module._write_prepared_page_diagnostic(
        directory,
        kind,
        diagnostic_module._bounded_page_diagnostic_content(content),
        deadline=effective_deadline,
    )


class PageDiagnosticTests(unittest.TestCase):
    def test_public_driver_exposes_async_page_diagnostic_capture(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(EHDriver.save_page_diagnostic))
        self.assertFalse(hasattr(gallery_utils, "write_page_diagnostic"))

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
                lock_path = directory / diagnostic_module._PAGE_DIAGNOSTIC_LOCK_FILENAME
                self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_concurrent_writers_keep_both_diagnostics(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(
                    write_page_diagnostic,
                    directory,
                    "search_error",
                    "first",
                )
                second_future = executor.submit(
                    write_page_diagnostic,
                    directory,
                    "search_error",
                    "second",
                )
                first = first_future.result(timeout=5)
                second = second_future.result(timeout=5)

            self.assertNotEqual(first, second)
            self.assertEqual(
                {first.read_text(), second.read_text()},
                {"first", "second"},
            )
            self.assertEqual(
                set(_managed_page_diagnostic_paths(directory)),
                {first, second},
            )

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_child_does_not_inherit_a_stuck_thread_lock(self) -> None:
        context = multiprocessing.get_context("fork")
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            receive_connection, send_connection = context.Pipe(duplex=False)

            def write_in_child() -> None:
                receive_connection.close()
                try:
                    path = write_page_diagnostic(
                        directory,
                        "search_error",
                        "child",
                    )
                except BaseException as error:
                    send_connection.send((False, repr(error)))
                else:
                    send_connection.send((True, str(path)))
                finally:
                    send_connection.close()

            with diagnostic_module._locked_page_diagnostic_directory(
                directory,
                deadline=None,
            ):
                process = context.Process(target=write_in_child)
                process.start()

            send_connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                self.fail("fork child remained blocked by the inherited lock")

            self.assertEqual(process.exitcode, 0)
            succeeded, detail = receive_connection.recv()
            receive_connection.close()
            process.close()
            self.assertTrue(succeeded, detail)
            self.assertTrue(Path(detail).exists())

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

    def test_retention_ignores_unmanaged_entries_and_symlinks(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manual_path = directory / "search_error_manual_notes.html"
            manual_path.write_text("keep me")
            matching_directory = directory / f"search_error_{9:016x}_{'d' * 32}.html"
            matching_directory.mkdir()
            matching_symlink = directory / f"search_error_{10:016x}_{'e' * 32}.html"
            matching_symlink.symlink_to(manual_path)

            with patch.object(
                diagnostic_module,
                "_PAGE_DIAGNOSTIC_FILE_LIMIT",
                1,
            ):
                created = write_page_diagnostic(
                    directory,
                    "search_error",
                    "new",
                )

            self.assertEqual(_managed_page_diagnostic_paths(directory), [created])
            self.assertTrue(manual_path.exists())
            self.assertTrue(matching_directory.is_dir())
            self.assertTrue(matching_symlink.is_symlink())

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

    def test_parent_preparation_caps_input_before_redaction_and_encoding(self) -> None:
        with (
            patch.object(
                diagnostic_module,
                "_PAGE_DIAGNOSTIC_MAX_INPUT_CHARACTERS",
                8,
            ),
            patch.object(
                diagnostic_module,
                "_PAGE_DIAGNOSTIC_MAX_FILE_BYTES",
                128,
            ),
            patch.object(
                diagnostic_module,
                "_PAGE_DIAGNOSTIC_TOTAL_BYTES",
                128,
            ),
            patch.object(
                diagnostic_module,
                "_redact_page_diagnostic_secrets",
                wraps=diagnostic_module._redact_page_diagnostic_secrets,
            ) as redact,
        ):
            prepared = diagnostic_module._bounded_page_diagnostic_content(
                "abcdefghignored-tail"
            )

        redact.assert_called_once_with("abcdefgh")
        self.assertLessEqual(len(prepared), 128)
        self.assertIn(b"hbrowser page diagnostic truncated", prepared)

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

    def test_diagnostic_deadline_must_be_finite(self) -> None:
        with TemporaryDirectory() as directory_name:
            for invalid in (True, float("inf"), float("nan")):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(
                        ValueError,
                        "deadline must be finite",
                    ),
                ):
                    write_page_diagnostic(
                        Path(directory_name),
                        "driver_error",
                        "content",
                        deadline=invalid,
                    )

    def test_diagnostic_completed_after_deadline_is_removed(self) -> None:
        clock = 0.0
        original_write = diagnostic_module._write_private_file

        def complete_late(path: Path, content: bytes) -> None:
            nonlocal clock
            original_write(path, content)
            clock = 11.0

        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch(
                    "hbrowser.gallery.utils.diagnostic.time.monotonic",
                    side_effect=lambda: clock,
                ),
                patch.object(
                    diagnostic_module,
                    "_write_private_file",
                    side_effect=complete_late,
                ),
                self.assertRaisesRegex(TimeoutError, "diagnostic publication"),
            ):
                write_page_diagnostic(
                    directory,
                    "driver_error",
                    "content",
                    deadline=10.0,
                )

            self.assertEqual(_managed_page_diagnostic_paths(directory), [])

    def test_next_write_removes_bounded_orphan_partial(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            partial_path = (
                directory / diagnostic_module._PAGE_DIAGNOSTIC_PARTIAL_FILENAME
            )
            partial_path.write_bytes(
                b"x" * diagnostic_module._PAGE_DIAGNOSTIC_MAX_FILE_BYTES
            )

            path = write_page_diagnostic(
                directory,
                "driver_error",
                "next worker",
            )

            self.assertFalse(partial_path.exists())
            self.assertEqual(path.read_text(), "next worker")

    def test_partial_write_is_removed(self) -> None:
        class _FailingWriter:
            def __init__(self, descriptor: int) -> None:
                self.descriptor = descriptor

            def __enter__(self) -> _FailingWriter:
                return self

            def __exit__(self, *args: object) -> None:
                del args
                os.close(self.descriptor)

            def write(self, content: bytes) -> int:
                os.write(self.descriptor, content[:1])
                raise OSError("simulated partial write")

        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch(
                    "hbrowser.gallery.utils.diagnostic.os.fdopen",
                    side_effect=lambda descriptor, mode: _FailingWriter(descriptor),
                ),
                self.assertRaisesRegex(OSError, "simulated partial write"),
            ):
                write_page_diagnostic(directory, "search_error", "content")

            self.assertEqual(_managed_page_diagnostic_paths(directory), [])

    def test_prune_failure_does_not_add_another_file(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            existing_paths = [
                directory / f"search_error_{index:016x}_{index:032x}.html"
                for index in range(2)
            ]
            for path in existing_paths:
                path.write_text("old")

            original_unlink = Path.unlink

            def fail_oldest_unlink(path: Path, missing_ok: bool = False) -> None:
                if path == existing_paths[0]:
                    raise PermissionError("simulated retention failure")
                original_unlink(path, missing_ok=missing_ok)

            with (
                patch.object(
                    diagnostic_module,
                    "_PAGE_DIAGNOSTIC_FILE_LIMIT",
                    2,
                ),
                patch.object(
                    Path,
                    "unlink",
                    autospec=True,
                    side_effect=fail_oldest_unlink,
                ),
                self.assertRaisesRegex(OSError, "retention"),
            ):
                write_page_diagnostic(directory, "search_error", "new")

            self.assertEqual(
                _managed_page_diagnostic_paths(directory),
                existing_paths,
            )


class PageDiagnosticOwnerTests(unittest.IsolatedAsyncioTestCase):
    _NONCE = "0123456789abcdef0123456789abcdef"

    def _receipt(self) -> bytes:
        return (
            json.dumps(
                {
                    "schema": 1,
                    "nonce": self._NONCE,
                    "filename": (f"driver_error_0000000000000000_{self._NONCE}.html"),
                }
            ).encode()
            + b"\n"
        )

    def _owner(self, receipt: bytes = b"") -> Mock:
        self.enterContext(
            patch.object(page_diagnostic_module, "_transfer_payload", new=AsyncMock())
        )
        self.enterContext(
            patch.object(
                page_diagnostic_module,
                "_read_frame",
                new=AsyncMock(return_value=receipt),
            )
        )
        owner = Mock()
        owner.stdout = io.BytesIO(receipt)
        owner.poll = Mock(return_value=0)
        owner.kill = Mock()
        owner.shutdown = Mock(return_value=0)
        return owner

    async def test_real_owned_worker_creates_and_writes_target_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name) / "missing" / "log"
            with patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=directory,
            ):
                path = await page_diagnostic_module.write_page_diagnostic_owned(
                    "driver_error",
                    "owned content",
                    deadline=Deadline.after(5),
                )

            self.assertEqual(path.parent, directory)
            self.assertEqual(path.read_text(), "owned content")

    async def test_short_deadline_never_starts_or_stages_a_worker(self) -> None:
        with (
            TemporaryDirectory() as directory_name,
            patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=Path(directory_name),
            ),
            patch.object(page_diagnostic_module, "_transfer_payload") as transfer,
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
            ) as start_owned_process,
            self.assertRaisesRegex(TimeoutError, "safe worker ownership budget"),
        ):
            await page_diagnostic_module.write_page_diagnostic_owned(
                "driver_error",
                "content",
                deadline=Deadline.after(0.01),
            )

        transfer.assert_not_called()
        start_owned_process.assert_not_called()

    async def test_timeout_kills_reaps_and_closes_pipe(self) -> None:
        owner = self._owner()
        owner.poll.return_value = None
        started_at = time.monotonic()

        with (
            TemporaryDirectory() as directory_name,
            patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=Path(directory_name),
            ),
            patch(
                "hbrowser.gallery.browser.page_diagnostic.secrets.token_hex",
                return_value=self._NONCE,
            ),
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
                return_value=owner,
            ),
            patch.object(
                page_diagnostic_module,
                "_DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS",
                0.02,
            ),
            self.assertRaisesRegex(TimeoutError, "work deadline"),
        ):
            await page_diagnostic_module.write_page_diagnostic_owned(
                "driver_error",
                "content",
                deadline=Deadline.after(0.05),
            )

        self.assertLess(time.monotonic() - started_at, 0.5)
        owner.kill.assert_called_once_with()
        owner.shutdown.assert_called_once()
        self.assertTrue(owner.stdout.closed)

    async def test_cancellation_waits_for_force_reap_and_pipe_close(self) -> None:
        owner = self._owner()
        owner.poll.return_value = None
        shutdown_started = threading.Event()
        allow_shutdown = threading.Event()

        def settle_shutdown(**_kwargs: object) -> int:
            shutdown_started.set()
            allow_shutdown.wait(timeout=1)
            return 0

        owner.shutdown.side_effect = settle_shutdown
        with (
            TemporaryDirectory() as directory_name,
            patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=Path(directory_name),
            ),
            patch(
                "hbrowser.gallery.browser.page_diagnostic.secrets.token_hex",
                return_value=self._NONCE,
            ),
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
                return_value=owner,
            ),
            patch.object(
                page_diagnostic_module,
                "_DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS",
                0.2,
            ),
        ):
            task = asyncio.create_task(
                page_diagnostic_module.write_page_diagnostic_owned(
                    "driver_error",
                    "content",
                    deadline=Deadline.after(1),
                )
            )
            while owner.poll.call_count == 0:
                await asyncio.sleep(0)
            task.cancel()
            while not shutdown_started.is_set():
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            allow_shutdown.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        owner.kill.assert_called_once_with()
        owner.shutdown.assert_called_once()
        self.assertTrue(owner.stdout.closed)

    async def test_late_receipt_is_rejected_after_natural_reap(self) -> None:
        owner = self._owner(self._receipt())

        def late_shutdown(**_kwargs: object) -> int:
            time.sleep(0.03)
            return 0

        owner.shutdown.side_effect = late_shutdown

        with (
            TemporaryDirectory() as directory_name,
            patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=Path(directory_name),
            ),
            patch(
                "hbrowser.gallery.browser.page_diagnostic.secrets.token_hex",
                return_value=self._NONCE,
            ),
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
                return_value=owner,
            ),
            patch.object(
                page_diagnostic_module,
                "_DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS",
                0.005,
            ),
            self.assertRaisesRegex(TimeoutError, "receipt arrived"),
        ):
            await page_diagnostic_module.write_page_diagnostic_owned(
                "driver_error",
                "content",
                deadline=Deadline.after(0.02),
            )

        owner.kill.assert_not_called()
        owner.shutdown.assert_called_once()
        self.assertTrue(owner.stdout.closed)

    async def test_malformed_receipt_is_rejected_after_natural_reap(self) -> None:
        owner = self._owner(b"not-json\n")

        with (
            TemporaryDirectory() as directory_name,
            patch.object(
                page_diagnostic_module,
                "_page_diagnostic_directory_hint",
                return_value=Path(directory_name),
            ),
            patch(
                "hbrowser.gallery.browser.page_diagnostic.secrets.token_hex",
                return_value=self._NONCE,
            ),
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
                return_value=owner,
            ),
            patch.object(
                page_diagnostic_module,
                "_DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS",
                0.2,
            ),
            patch.object(Path, "resolve", autospec=True) as resolve_path,
            patch.object(Path, "mkdir", autospec=True) as make_directory,
            self.assertRaisesRegex(RuntimeError, "invalid JSON"),
        ):
            await page_diagnostic_module.write_page_diagnostic_owned(
                "driver_error",
                "content",
                deadline=Deadline.after(1),
            )

        owner.kill.assert_not_called()
        owner.shutdown.assert_called_once()
        resolve_path.assert_not_called()
        make_directory.assert_not_called()
        self.assertTrue(owner.stdout.closed)


class DriverExitDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = self.enterContext(TemporaryDirectory())
        self.enterContext(
            patch(
                "hbrowser.gallery.browser.page_diagnostic."
                "_page_diagnostic_directory_hint",
                return_value=Path(directory),
            )
        )
        self.driver = _TestDriver(headless=True)
        self.logger = Mock()
        self.driver.logger = self.logger
        page = Mock()
        browser = Mock()
        page.browser = browser
        page.get_content = AsyncMock(return_value="<html>failure</html>")
        self.driver.bind_existing_browser(browser, page, owns_browser=True)

    async def test_exit_keeps_each_error_page_and_logs_safe_error_types(self) -> None:
        try:
            raise ValueError("broken")
        except ValueError as error:
            traceback = error.__traceback__

            with (
                TemporaryDirectory() as directory_name,
                patch(
                    "hbrowser.gallery.browser.page_diagnostic."
                    "_page_diagnostic_directory_hint",
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
                "hbrowser.gallery.browser.page_diagnostic."
                "_page_diagnostic_directory_hint",
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

    async def test_page_write_uses_async_owned_worker_contract(self) -> None:
        async def write_diagnostic(
            kind: str,
            content: str,
            *,
            deadline: Deadline,
        ) -> Path:
            self.assertEqual(kind, "driver_error")
            self.assertEqual(content, "content")
            self.assertGreater(deadline.remaining(), 0)
            self.assertLessEqual(deadline.remaining(), 5)
            return Path("diagnostic.html")

        with patch.object(
            self.driver,
            "_write_page_diagnostic",
            side_effect=write_diagnostic,
        ):
            path = await self.driver.save_page_diagnostic(
                "driver_error",
                "content",
            )

        self.assertEqual(path, Path("diagnostic.html"))

    async def test_driver_propagates_settled_worker_cancellation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def write_diagnostic(
            _kind: str,
            _content: str,
            *,
            deadline: Deadline,
        ) -> Path:
            self.assertGreater(deadline.remaining(), 0)
            started.set()
            try:
                await release.wait()
                return Path("diagnostic.html")
            finally:
                finished.set()

        with patch.object(
            self.driver,
            "_write_page_diagnostic",
            side_effect=write_diagnostic,
        ):
            save_task = asyncio.create_task(
                self.driver.save_page_diagnostic("driver_error", "content")
            )
            await started.wait()
            save_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await save_task

        self.assertTrue(finished.is_set())

    async def test_unresolved_diagnostic_worker_ownership_is_not_best_effort(
        self,
    ) -> None:
        failure = ProcessOwnershipError("diagnostic worker ownership unresolved")
        next_action = AsyncMock()
        with patch.object(
            self.driver, "_write_page_diagnostic", new=AsyncMock(side_effect=failure)
        ):
            with self.assertRaises(ProcessOwnershipError) as raised:
                await self.driver.save_page_diagnostic("driver_error", "content")
                await next_action()

        self.assertIs(raised.exception, failure)
        next_action.assert_not_awaited()
        self.logger.warning.assert_not_called()

    async def test_owned_worker_timeout_is_reported(self) -> None:
        with patch.object(
            self.driver,
            "_write_page_diagnostic",
            new=AsyncMock(side_effect=TimeoutError("worker deadline")),
        ):
            result = await self.driver.save_page_diagnostic(
                "driver_error",
                "content",
            )

        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with(
            "Failed to save %s page diagnostic: error_type=%s",
            "driver_error",
            "TimeoutError",
        )

    def test_sync_diagnostic_lock_wait_obeys_deadline(self) -> None:
        with TemporaryDirectory() as directory_name:
            diagnostic_module._PAGE_DIAGNOSTIC_THREAD_LOCK.acquire()
            try:
                with self.assertRaisesRegex(TimeoutError, "thread lock"):
                    write_page_diagnostic(
                        Path(directory_name),
                        "driver_error",
                        "content",
                        deadline=time.monotonic() + 0.001,
                    )
            finally:
                diagnostic_module._PAGE_DIAGNOSTIC_THREAD_LOCK.release()

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
        stop_browser.assert_awaited_once_with(self.driver.browser, deadline=ANY)

    async def test_capture_failure_reports_only_the_safe_error_type(self) -> None:
        capture_error = RuntimeError("session detached")
        self.driver.page.get_content.side_effect = capture_error

        path = await self.driver.save_page_diagnostic("driver_error")

        self.assertIsNone(path)
        self.logger.warning.assert_called_once_with(
            "Failed to capture %s page diagnostic: error_type=%s",
            "driver_error",
            "RuntimeError",
        )

    async def test_retired_capture_error_is_not_swallowed(self) -> None:
        retired = ZendriverOwnerRetiredError("generation retired")
        self.driver.page.get_content.side_effect = retired

        with self.assertRaises(ZendriverOwnerRetiredError) as raised:
            await self.driver.save_page_diagnostic("driver_error")

        self.assertIs(raised.exception, retired)
        self.logger.warning.assert_not_called()

    async def test_generation_failure_exit_skips_same_browser_diagnostic(
        self,
    ) -> None:
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        with patch(
            "hbrowser.gallery.driver_base.stop_browser",
            new=AsyncMock(),
        ) as stop_browser:
            await self.driver.__aexit__(
                ZendriverOperationTimeout,
                timeout,
                timeout.__traceback__,
            )

        self.driver.page.get_content.assert_not_awaited()
        stop_browser.assert_awaited_once_with(self.driver.browser, deadline=ANY)

    async def test_plain_timeout_skips_overdue_read_and_uses_exit_deadline(
        self,
    ) -> None:
        timeout = TimeoutError("semantic state expired")
        with (
            patch.object(
                self.driver,
                "save_page_diagnostic",
                new=AsyncMock(),
            ) as save_diagnostic,
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                new=AsyncMock(),
            ) as stop_browser,
        ):
            await self.driver.__aexit__(TimeoutError, timeout, None)

        save_diagnostic.assert_not_awaited()
        self.driver.page.get_content.assert_not_awaited()
        stop_browser.assert_awaited_once_with(self.driver.browser, deadline=ANY)
        await_args = stop_browser.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        deadline = await_args.kwargs["deadline"]
        self.assertGreater(deadline.remaining(), 0)
        self.assertLessEqual(deadline.remaining(), 20.0)

    async def test_page_capture_protocol_timeout_is_terminal(self) -> None:
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
        with (
            patch(
                "hbrowser.gallery.driver_base._PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS",
                0.001,
            ),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await self.driver.save_page_diagnostic("driver_error")

        self.assertTrue(started.is_set())
        self.assertFalse(cancelled)
        self.logger.warning.assert_not_called()

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
            with (
                patch(
                    "hbrowser.gallery.driver_base._PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS",
                    0.001,
                ),
                self.assertRaises(ZendriverOperationTimeout),
            ):
                await self.driver.save_page_diagnostic("driver_error")

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertEqual(loop_errors, [])

    async def test_browser_close_failure_is_visible(self) -> None:
        close_error = RuntimeError("process did not exit")
        with (
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                new=AsyncMock(side_effect=close_error),
            ),
            self.assertRaisesRegex(RuntimeError, "process did not exit"),
        ):
            await self.driver.__aexit__(None, None, None)

        self.logger.warning.assert_called_once_with(
            "Failed to close browser cleanly: error_type=%s",
            "RuntimeError",
        )

    async def test_cancellation_during_error_exit_propagates_after_cleanup(
        self,
    ) -> None:
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()

        async def cancellation_deferring_stop(
            _: object,
            **_kwargs: object,
        ) -> None:
            stop_started.set()
            cancellation: asyncio.CancelledError | None = None
            try:
                await allow_stop.wait()
            except asyncio.CancelledError as error:
                cancellation = error
                await allow_stop.wait()
            if cancellation is not None:
                raise cancellation

        primary = ValueError("primary browser failure")
        with patch(
            "hbrowser.gallery.driver_base.stop_browser",
            new=AsyncMock(side_effect=cancellation_deferring_stop),
        ):
            exit_task = asyncio.create_task(
                self.driver.__aexit__(ValueError, primary, primary.__traceback__)
            )
            await stop_started.wait()
            exit_task.cancel()
            await asyncio.sleep(0)

            self.assertFalse(exit_task.done())
            allow_stop.set()
            with self.assertRaises(asyncio.CancelledError) as raised:
                await exit_task

        self.assertIn(
            "Browser session was already exiting after: ValueError",
            raised.exception.__notes__,
        )


class DiagnosticPipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_frame_reads_partial_raw_pipe_without_blocking(self) -> None:
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb") as stream:
            owner = Mock(stdout=stream)

            async def publish() -> None:
                os.write(writer, b'{"schema":')
                await asyncio.sleep(0.01)
                os.write(writer, b"1}\n")

            publisher = asyncio.create_task(publish())
            try:
                frame = await page_diagnostic_module._read_frame(
                    owner, deadline=Deadline.after(1)
                )
                await publisher
                self.assertEqual(frame, b'{"schema":1}\n')
            finally:
                os.close(writer)
                publisher.cancel()
                await asyncio.gather(publisher, return_exceptions=True)

    async def test_raw_pipe_rejects_truncated_oversize_and_excess_frames(self) -> None:
        for content, message in (
            (b'{"schema":', "truncated"),
            (b"x" * 4097, "size limit"),
            (b"{}\n{}\n", "excess frame"),
        ):
            with self.subTest(message=message):
                reader, writer = os.pipe()
                with os.fdopen(reader, "rb") as stream:
                    os.set_blocking(writer, False)

                    async def publish(
                        content: bytes = content, writer: int = writer
                    ) -> None:
                        try:
                            offset = 0
                            while offset < len(content):
                                try:
                                    written = os.write(
                                        writer, content[offset : offset + 1024]
                                    )
                                except BlockingIOError:
                                    await asyncio.sleep(0)
                                    continue
                                offset += written
                                await asyncio.sleep(0)
                        finally:
                            os.close(writer)

                    publisher = asyncio.create_task(publish())
                    try:
                        with self.assertRaisesRegex(RuntimeError, message):
                            await page_diagnostic_module._read_frame(
                                Mock(stdout=stream), deadline=Deadline.after(1)
                            )
                    finally:
                        publisher.cancel()
                        await asyncio.gather(publisher, return_exceptions=True)

    async def test_open_pipe_timeout_and_cancellation_do_not_start_reader_thread(
        self,
    ) -> None:
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb") as stream:
            try:
                with self.assertRaises(TimeoutError):
                    await page_diagnostic_module._read_frame(
                        Mock(stdout=stream), deadline=Deadline.after(0.02)
                    )
                task = asyncio.create_task(
                    page_diagnostic_module._read_frame(
                        Mock(stdout=stream), deadline=Deadline.after(1)
                    )
                )
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                os.close(writer)

    async def test_final_receipt_rejects_trailing_bytes_in_a_separate_raw_read(
        self,
    ) -> None:
        reader, writer = os.pipe()
        frame = b'{"schema":1}\n'
        real_read = os.read
        with os.fdopen(reader, "rb") as stream:
            os.write(writer, frame + b"extra")
            os.close(writer)
            with (
                patch.object(
                    os,
                    "read",
                    side_effect=lambda fd, size: real_read(fd, min(size, len(frame))),
                ),
                self.assertRaisesRegex(RuntimeError, "trailing bytes"),
            ):
                await page_diagnostic_module._read_frame(
                    Mock(stdout=stream), deadline=Deadline.after(1), require_eof=True
                )

    async def test_final_receipt_requires_eof_but_ready_does_not(self) -> None:
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb") as stream:
            try:
                os.write(writer, b"{}\n")
                with self.assertRaisesRegex(RuntimeError, "remains open"):
                    await page_diagnostic_module._read_frame(
                        Mock(stdout=stream),
                        deadline=Deadline.after(1),
                        require_eof=True,
                    )
            finally:
                os.close(writer)

    async def test_ready_authority_mismatch_never_opens_payload_socket(self) -> None:
        nonce = "1" * 32
        valid: dict[str, object] = {
            "schema": 1,
            "phase": "ready",
            "nonce": nonce,
            "size": 4,
            "port": 12345,
            "capability": "2" * 64,
        }
        invalid = [
            valid | {"schema": True},
            valid | {"phase": "complete"},
            valid | {"nonce": "3" * 32},
            valid | {"size": True},
            valid | {"size": 5},
            valid | {"port": True},
            valid | {"port": 0},
            valid | {"port": 65536},
            valid | {"capability": "z" * 64},
            valid | {"capability": "2" * 62},
            valid | {"host": "127.0.0.1"},
        ]
        for frame in invalid:
            with (
                self.subTest(frame=frame),
                patch.object(
                    page_diagnostic_module,
                    "_read_frame",
                    new=AsyncMock(return_value=json.dumps(frame).encode() + b"\n"),
                ),
                patch.object(
                    page_diagnostic_module, "send_payload", new=AsyncMock()
                ) as send,
                self.assertRaisesRegex(RuntimeError, "invalid READY authority"),
            ):
                await page_diagnostic_module._transfer_payload(
                    Mock(), b"page", nonce=nonce, deadline=Deadline.after(1)
                )
            send.assert_not_awaited()


def test_first_diagnostic_does_not_start_a_resource_tracker(tmp_path: Path) -> None:
    """A fresh interpreter exercises real worker cleanup, including on OLD failure."""
    import subprocess
    import sys
    from textwrap import dedent

    root = Path(__file__).resolve().parents[1]
    entry = root / "scripts/pytest-source.py"
    source = dedent(
        """
        import sys
        assert sys.pycache_prefix is not None, (
            'diagnostic probe lost source cache isolation')
        assert sys.dont_write_bytecode, 'diagnostic probe may write bytecode'
        import asyncio
        from multiprocessing import resource_tracker
        from pathlib import Path
        from unittest.mock import patch
        from hbrowser.gallery.browser import page_diagnostic as diagnostic
        from hbrowser.gallery.utils.deadline import Deadline
        tracker = resource_tracker._resource_tracker
        assert tracker._pid is None
        owners = []
        real_start = diagnostic.start_owned_process
        def record_start(*args, **kwargs):
            owner = real_start(*args, **kwargs)
            owners.append(owner)
            return owner
        async def check():
            with patch.object(diagnostic, 'start_owned_process', record_start):
                path = await diagnostic.write_page_diagnostic_owned(
                    'driver_error', 'synthetic page', deadline=Deadline.after(5))
            assert path.read_text() == 'synthetic page'
            assert owners and all(
                owner._closed and owner._supervisor.poll() is not None
                for owner in owners)
            assert tracker._pid is None, 'diagnostic started an unowned tracker'
        try:
            boundary['adapter'].run(lambda: asyncio.run(check()))
        finally:
            tracker._stop()
        """
    )
    environment = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PYTHONPATH": str(root),
        "PYTHONPYCACHEPREFIX": str(tmp_path / "bytecode"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HBROWSER_LOG_DIR": str(tmp_path / "diagnostics"),
    }
    for name in ("SystemRoot", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT"):
        if name in os.environ:
            environment[name] = os.environ[name]
    scope = "H2H_PRIVATE_POSIX_PROCESS_SCOPE"
    if scope in os.environ:
        environment[scope] = os.environ[scope]
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            f"import runpy\nboundary=runpy.run_path({str(entry)!r})\n" + source,
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class DiagnosticCancellationOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_started(self, started: threading.Event) -> None:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0)

    async def test_settle_normalizes_pending_and_completed_task_results(self) -> None:
        for completed in (False, True):
            for failure in (None, TimeoutError("ordinary startup failure")):
                with self.subTest(completed=completed, failure=failure):
                    release = asyncio.Event()

                    async def operation(
                        release: asyncio.Event = release,
                        failure: TimeoutError | None = failure,
                    ) -> int:
                        await release.wait()
                        if failure is not None:
                            raise failure
                        return 7

                    target = asyncio.create_task(operation())
                    if completed:
                        release.set()
                        await asyncio.gather(target, return_exceptions=True)
                    waiter = asyncio.create_task(
                        page_diagnostic_module._settle_task(target)
                    )
                    await asyncio.sleep(0)
                    release.set()
                    try:
                        value, error, cancellation = await waiter
                        self.assertEqual(value, 7 if failure is None else None)
                        self.assertIs(error, failure)
                        self.assertIsNone(cancellation)
                    finally:
                        release.set()
                        await asyncio.gather(target, waiter, return_exceptions=True)

    async def test_inner_task_cancellation_is_a_task_result_not_caller_cancellation(
        self,
    ) -> None:
        async def operation() -> None:
            raise asyncio.CancelledError("inner operation stopped")

        target = asyncio.create_task(operation())
        value, error, cancellation = await page_diagnostic_module._settle_task(target)
        self.assertIsNone(value)
        self.assertIsInstance(error, asyncio.CancelledError)
        self.assertIsNone(cancellation)

    async def test_driver_preserves_cancellation_before_clean_startup_failure(
        self,
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        failure = TimeoutError("startup deadline expired after successful cleanup")

        def start(*_args: object, **_kwargs: object) -> None:
            started.set()
            if not release.wait(1):
                raise AssertionError("test did not release startup")
            raise failure

        driver = _TestDriver(headless=True)
        driver.logger = Mock()
        with patch.object(page_diagnostic_module, "start_owned_process", start):
            task = asyncio.create_task(
                driver.save_page_diagnostic("driver_error", "synthetic page")
            )
            try:
                await self._wait_started(started)
                task.cancel("caller stopped")
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_completed_startup_ownership_error_remains_primary_on_cancellation(
        self,
    ) -> None:
        failure = ProcessOwnershipError("startup cleanup is unproven")
        settle = page_diagnostic_module._settle_task

        async def cancel_on_completion(
            task: asyncio.Task[object],
        ) -> tuple[object | None, BaseException | None, asyncio.CancelledError | None]:
            waiter = asyncio.current_task()
            assert waiter is not None
            task.add_done_callback(lambda _done: waiter.cancel("late caller cancel"))
            return await settle(task)

        with (
            patch.object(
                page_diagnostic_module,
                "start_owned_process",
                side_effect=failure,
            ),
            patch.object(
                page_diagnostic_module, "_settle_task", side_effect=cancel_on_completion
            ),
        ):
            task = asyncio.create_task(
                page_diagnostic_module.write_page_diagnostic_owned(
                    "driver_error", "synthetic page", deadline=Deadline.after(5)
                )
            )
            with self.assertRaises(ProcessOwnershipError) as raised:
                await task
        self.assertIs(raised.exception, failure)
        self.assertTrue(
            any("cancellation" in note for note in getattr(failure, "__notes__", ()))
        )

    async def test_shutdown_settles_success_or_ownership_failure_after_cancellation(
        self,
    ) -> None:
        for failure in (None, ProcessOwnershipError("shutdown cleanup is unproven")):
            with self.subTest(failure=failure):
                started = threading.Event()
                release = threading.Event()

                def shutdown(
                    *_args: object,
                    failure: ProcessOwnershipError | None = failure,
                    started: threading.Event = started,
                    release: threading.Event = release,
                    **_kwargs: object,
                ) -> int:
                    started.set()
                    if not release.wait(1):
                        raise AssertionError("test did not release shutdown")
                    if failure is not None:
                        raise failure
                    return 0

                with patch.object(
                    page_diagnostic_module, "_shutdown_owned_worker", shutdown
                ):
                    task = asyncio.create_task(
                        page_diagnostic_module._settle_owned_worker(
                            Mock(), deadline=Deadline.after(5), force=True
                        )
                    )
                    try:
                        await self._wait_started(started)
                        task.cancel("caller stopped during shutdown")
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        release.set()
                        if failure is None:
                            self.assertIsInstance(await task, asyncio.CancelledError)
                        else:
                            with self.assertRaises(ProcessOwnershipError) as raised:
                                await task
                            self.assertIs(raised.exception, failure)
                            self.assertTrue(
                                any(
                                    "cancellation" in note
                                    for note in getattr(failure, "__notes__", ())
                                )
                            )
                    finally:
                        release.set()
                        await asyncio.gather(task, return_exceptions=True)
