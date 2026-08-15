from __future__ import annotations

import asyncio
import io
import logging
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from hbrowser.gallery.utils import log as log_module
from hbrowser.gallery.utils.log import (
    _isolated_process_log_handlers_for_testing,
    get_log_dir,
    log_context,
    setup_logger,
)


def _log_record(
    message: str,
    *,
    name: str = "hbrowser.tests.semantic",
    extra: dict[str, object] | None = None,
) -> logging.LogRecord:
    logger = logging.getLogger(name)
    return logger.makeRecord(
        name,
        logging.INFO,
        __file__,
        1,
        message,
        (),
        None,
        extra=extra,
    )


class SemanticLogContextTests(unittest.TestCase):
    def test_default_and_nested_context_labels_restore_safely(self) -> None:
        formatter = log_module._formatter()
        rendered = [formatter.format(_log_record("outside"))]

        with log_context(account="main", realm="persistent", tab_role="persistent"):
            rendered.append(formatter.format(_log_record("realm")))
            with log_context(activity="Check-in"):
                rendered.append(formatter.format(_log_record("nested")))
                with log_context(scope="Browser"):
                    rendered.append(formatter.format(_log_record("scoped")))
            rendered.append(formatter.format(_log_record("restored realm")))

        rendered.append(formatter.format(_log_record("restored system")))

        self.assertIn(" - INFO - [System] outside", rendered[0])
        self.assertIn(" - INFO - [Persistent] realm", rendered[1])
        self.assertIn(" - INFO - [Persistent · Check-in] nested", rendered[2])
        self.assertIn(" - INFO - [Browser] scoped", rendered[3])
        self.assertIn(" - INFO - [Persistent] restored realm", rendered[4])
        self.assertIn(" - INFO - [System] restored system", rendered[5])

    def test_context_fields_are_injected_without_changing_record_name(self) -> None:
        record = _log_record("diagnostic", name="diagnostic.module")

        with log_context(
            account="main",
            realm="isekai",
            tab_role="isekai",
            activity="Battle",
        ):
            self.assertTrue(log_module._LOG_CONTEXT_FILTER.filter(record))

        rendered = log_module._formatter().format(record)

        self.assertEqual(record.name, "diagnostic.module")
        self.assertEqual(record.__dict__["account"], "main")
        self.assertEqual(record.__dict__["realm"], "isekai")
        self.assertEqual(record.__dict__["tab_role"], "isekai")
        self.assertEqual(record.__dict__["activity"], "Battle")
        self.assertIsNone(record.__dict__["scope"])
        self.assertIn("[Isekai · Battle] diagnostic", rendered)
        self.assertNotIn("diagnostic.module", rendered)

    def test_record_activity_and_scope_override_inherited_context(self) -> None:
        formatter = log_module._formatter()
        with log_context(realm="persistent", activity="Battle"):
            activity_record = _log_record(
                "activity override",
                extra={"activity": "Maintenance"},
            )
            scope_record = _log_record(
                "scope override",
                extra={"scope": "Browser", "activity": "Maintenance"},
            )
            invalid_override = _log_record(
                "invalid override",
                extra={"activity": "   ", "scope": 42},
            )
            activity_rendered = formatter.format(activity_record)
            scope_rendered = formatter.format(scope_record)
            invalid_rendered = formatter.format(invalid_override)

        self.assertIn("[Persistent · Maintenance] activity override", activity_rendered)
        self.assertIn("[Browser] scope override", scope_rendered)
        self.assertIn("[Persistent · Battle] invalid override", invalid_rendered)

    def test_tab_role_is_used_when_realm_is_unspecified(self) -> None:
        with log_context(tab_role="persistent", activity="Login"):
            rendered = log_module._formatter().format(_log_record("role fallback"))

        self.assertIn("[Persistent · Login] role fallback", rendered)

    def test_invalid_context_values_are_rejected_without_leaking_context(self) -> None:
        for field, value, error_type in (
            ("realm", "", ValueError),
            ("activity", "   ", ValueError),
            ("scope", 5, TypeError),
        ):
            with self.subTest(field=field), self.assertRaises(error_type):
                with log_context(**{field: value}):  # type: ignore[arg-type]
                    self.fail("invalid context unexpectedly entered")

        rendered = log_module._formatter().format(_log_record("after errors"))
        self.assertIn("[System] after errors", rendered)


class AsyncSemanticLogContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_isolated_between_concurrent_tasks(self) -> None:
        formatter = log_module._formatter()
        release = asyncio.Event()
        ready = [asyncio.Event(), asyncio.Event()]

        async def render(index: int, realm: str) -> str:
            with log_context(realm=realm, activity="Battle"):
                ready[index].set()
                await release.wait()
                return formatter.format(_log_record(f"task {index}"))

        tasks = [
            asyncio.create_task(render(0, "isekai")),
            asyncio.create_task(render(1, "persistent")),
        ]
        await asyncio.gather(*(event.wait() for event in ready))
        release.set()
        rendered = await asyncio.gather(*tasks)

        self.assertIn("[Isekai · Battle] task 0", rendered[0])
        self.assertIn("[Persistent · Battle] task 1", rendered[1])
        parent = formatter.format(_log_record("parent"))
        self.assertIn("[System] parent", parent)


class LoggerSetupTests(unittest.TestCase):
    def test_logger_keeps_stdout_handler_and_non_propagating_contract(self) -> None:
        logger_name = f"hbrowser.tests.compatibility.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate
        output = io.StringIO()
        try:
            logger.handlers = []
            with (
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
                patch.dict(
                    "os.environ",
                    {
                        "HBROWSER_LOG_LEVEL": "DEBUG",
                        "HBROWSER_PROCESS_LOG_FILE": "",
                    },
                    clear=False,
                ),
            ):
                configured = setup_logger(logger_name)
                configured.debug("compatibility record")
                configured_handlers = configured.handlers[:]
                configured_level = configured.level
                handler_level = configured.handlers[0].level
                configured_propagate = configured.propagate
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertIs(configured, logger)
        self.assertEqual(len(configured_handlers), 1)
        self.assertIsInstance(configured_handlers[0], logging.StreamHandler)
        self.assertEqual(configured_level, logging.DEBUG)
        self.assertEqual(handler_level, logging.DEBUG)
        self.assertFalse(configured_propagate)
        self.assertIn("compatibility record", output.getvalue())

    def test_repeated_setup_does_not_duplicate_handlers(self) -> None:
        logger_name = f"hbrowser.tests.idempotent.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with patch.dict(
                os.environ,
                {"HBROWSER_PROCESS_LOG_FILE": ""},
                clear=False,
            ):
                first = setup_logger(logger_name)
                handler = first.handlers[0]
                second = setup_logger(logger_name)
                configured_handlers = second.handlers[:]
        finally:
            logger.handlers = original_handlers

        self.assertIs(second, first)
        self.assertEqual(configured_handlers, [handler])

    def test_repeated_setup_refreshes_only_managed_stdout_level(self) -> None:
        logger_name = f"hbrowser.tests.level_refresh.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        output = io.StringIO()
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
                patch.dict(
                    os.environ,
                    {
                        "HBROWSER_LOG_LEVEL": "INFO",
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "process.log"
                        ),
                    },
                    clear=False,
                ),
            ):
                first = setup_logger(logger_name)
                stdout_handler = next(
                    handler
                    for handler in first.handlers
                    if not isinstance(handler, logging.FileHandler)
                )
                process_handler = next(
                    handler
                    for handler in first.handlers
                    if isinstance(handler, logging.FileHandler)
                )

                os.environ["HBROWSER_LOG_LEVEL"] = "DEBUG"
                second = setup_logger(logger_name)
                second.debug("refreshed debug record")

                self.assertIs(second, first)
                self.assertEqual(second.level, logging.DEBUG)
                self.assertEqual(stdout_handler.level, logging.DEBUG)
                self.assertEqual(process_handler.level, logging.NOTSET)
                self.assertIn("refreshed debug record", output.getvalue())
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)

    def test_configured_process_log_is_shared_and_written_once(self) -> None:
        first_name = f"hbrowser.tests.process.first.{id(self)}"
        second_name = f"hbrowser.tests.process.second.{id(self)}"
        first = logging.getLogger(first_name)
        second = logging.getLogger(second_name)
        original_first_handlers = first.handlers[:]
        original_second_handlers = second.handlers[:]
        try:
            first.handlers = []
            second.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
                patch.dict(
                    os.environ,
                    {
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "nested" / "battle.log"
                        )
                    },
                    clear=False,
                ),
            ):
                configured_first = setup_logger(first_name)
                configured_second = setup_logger(second_name)
                configured_first.info("first durable record")
                configured_second.error("second durable record")
                process_path = Path(directory_name) / "nested" / "battle.log"
                for handler in configured_first.handlers:
                    handler.flush()
                contents = process_path.read_text(encoding="utf-8")
                first_file_handlers = [
                    handler
                    for handler in configured_first.handlers
                    if isinstance(handler, logging.FileHandler)
                ]
                second_file_handlers = [
                    handler
                    for handler in configured_second.handlers
                    if isinstance(handler, logging.FileHandler)
                ]

                self.assertEqual(len(first_file_handlers), 1)
                self.assertEqual(len(second_file_handlers), 1)
                self.assertIs(first_file_handlers[0], second_file_handlers[0])
                self.assertEqual(contents.count("first durable record"), 1)
                self.assertEqual(contents.count("second durable record"), 1)
                if os.name == "posix":
                    self.assertEqual(
                        stat.S_IMODE(process_path.stat().st_mode),
                        0o600,
                    )
        finally:
            first.handlers = original_first_handlers
            second.handlers = original_second_handlers

    def test_process_log_rejects_a_symbolic_link_target(self) -> None:
        logger_name = f"hbrowser.tests.process.symlink.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                directory = Path(directory_name)
                real_path = directory / "real.log"
                real_path.touch()
                link_path = directory / "battle.log"
                try:
                    link_path.symlink_to(real_path)
                except OSError as error:
                    self.skipTest(f"symbolic links are unavailable: {error}")
                with (
                    patch.dict(
                        os.environ,
                        {"HBROWSER_PROCESS_LOG_FILE": str(link_path)},
                        clear=False,
                    ),
                    self.assertRaisesRegex(OSError, "regular file"),
                ):
                    setup_logger(logger_name)
        finally:
            logger.handlers = original_handlers

    def test_secure_open_rejects_a_symlink_created_after_prevalidation(self) -> None:
        logger_name = f"hbrowser.tests.process.symlink_race.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                directory = Path(directory_name)
                real_path = directory / "real.log"
                real_path.write_text("must remain unchanged", encoding="utf-8")
                link_path = directory / "battle.log"
                probe_path = directory / "symlink-probe"
                try:
                    probe_path.symlink_to(real_path)
                    probe_path.unlink()
                except OSError as error:
                    self.skipTest(f"symbolic links are unavailable: {error}")

                def replace_after_validation(path: Path) -> None:
                    self.assertEqual(path, link_path)
                    path.symlink_to(real_path)

                with (
                    patch.dict(
                        os.environ,
                        {"HBROWSER_PROCESS_LOG_FILE": str(link_path)},
                        clear=False,
                    ),
                    patch.object(
                        log_module,
                        "_validate_process_log_target",
                        side_effect=replace_after_validation,
                    ),
                    self.assertRaises(OSError),
                ):
                    setup_logger(logger_name)

                self.assertTrue(link_path.is_symlink())
                self.assertEqual(
                    real_path.read_text(encoding="utf-8"),
                    "must remain unchanged",
                )
        finally:
            logger.handlers = original_handlers

    def test_process_log_detects_path_replacement_before_writing(self) -> None:
        if os.name != "posix":
            self.skipTest("Replacing an open file is not portable off POSIX")

        logger_name = f"hbrowser.tests.process.replaced.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                process_path = Path(directory_name) / "battle.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configured = setup_logger(logger_name)
                    configured.info("record before replacement")
                    moved_path = process_path.with_suffix(".moved")
                    process_path.rename(moved_path)
                    process_path.write_text("replacement\n", encoding="utf-8")

                    with self.assertRaisesRegex(
                        OSError,
                        "no longer names the opened file",
                    ):
                        configured.info("must not reach either file")

                    self.assertNotIn(
                        "must not reach either file",
                        moved_path.read_text(encoding="utf-8"),
                    )
                    self.assertEqual(
                        process_path.read_text(encoding="utf-8"),
                        "replacement\n",
                    )
        finally:
            logger.handlers = original_handlers

    def test_process_log_performs_real_rotation_and_retains_backups(self) -> None:
        logger_name = f"hbrowser.tests.process.rotation.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
                patch.object(log_module, "_PROCESS_LOG_MAX_BYTES", 1),
                patch.object(log_module, "_PROCESS_LOG_BACKUP_COUNT", 2),
            ):
                process_path = Path(directory_name) / "battle.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configured = setup_logger(logger_name)
                    for sequence in range(1, 5):
                        configured.info("rotation-record-%d", sequence)

                    process_handler = next(
                        handler
                        for handler in configured.handlers
                        if isinstance(handler, logging.FileHandler)
                    )
                    process_handler.flush()
                    active = process_path.read_text(encoding="utf-8")
                    first_backup = Path(f"{process_path}.1").read_text(encoding="utf-8")
                    second_backup = Path(f"{process_path}.2").read_text(
                        encoding="utf-8"
                    )
                    combined = active + first_backup + second_backup

                    self.assertEqual(
                        {path.name for path in process_path.parent.glob("battle.log*")},
                        {"battle.log", "battle.log.1", "battle.log.2"},
                    )
                    self.assertIn("rotation-record-4", active)
                    self.assertIn("rotation-record-3", first_backup)
                    self.assertIn("rotation-record-2", second_backup)
                    self.assertNotIn("rotation-record-1", combined)
                    for sequence in range(2, 5):
                        self.assertEqual(
                            combined.count(f"rotation-record-{sequence}"),
                            1,
                        )
                    if os.name == "posix":
                        for path in process_path.parent.glob("battle.log*"):
                            self.assertEqual(
                                stat.S_IMODE(path.stat().st_mode),
                                0o600,
                            )
        finally:
            logger.handlers = original_handlers

    def test_process_log_write_failure_is_raised(self) -> None:
        logger_name = f"hbrowser.tests.process.write_failure.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                process_path = Path(directory_name) / "battle.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configured = setup_logger(logger_name)
                    process_handler = next(
                        handler
                        for handler in configured.handlers
                        if isinstance(handler, logging.FileHandler)
                    )
                    failing_stream = Mock(wraps=process_handler.stream)
                    failing_stream.write.side_effect = OSError(
                        "simulated process-log write failure"
                    )

                    with (
                        patch.object(
                            process_handler,
                            "shouldRollover",
                            return_value=False,
                        ),
                        patch.object(process_handler, "stream", failing_stream),
                        patch.object(logging, "raiseExceptions", False),
                        self.assertRaisesRegex(
                            OSError,
                            "simulated process-log write failure",
                        ),
                    ):
                        configured.error("write must fail closed")
        finally:
            logger.handlers = original_handlers

    def test_process_log_rollover_failure_is_raised(self) -> None:
        logger_name = f"hbrowser.tests.process.rollover_failure.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                process_path = Path(directory_name) / "battle.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configured = setup_logger(logger_name)
                    process_handler = next(
                        handler
                        for handler in configured.handlers
                        if isinstance(handler, logging.FileHandler)
                    )

                    with (
                        patch.object(
                            process_handler,
                            "shouldRollover",
                            return_value=True,
                        ),
                        patch.object(
                            process_handler,
                            "doRollover",
                            side_effect=OSError(
                                "simulated process-log rollover failure"
                            ),
                        ),
                        patch.object(logging, "raiseExceptions", False),
                        self.assertRaisesRegex(
                            OSError,
                            "simulated process-log rollover failure",
                        ),
                    ):
                        configured.error("rollover must fail closed")
        finally:
            logger.handlers = original_handlers


class LogDirectoryTests(unittest.TestCase):
    def test_environment_override_creates_nested_absolute_directory(self) -> None:
        with TemporaryDirectory() as directory_name:
            configured = Path(directory_name) / "nested" / "diagnostics"
            with patch.dict(
                "os.environ",
                {"HBROWSER_LOG_DIR": str(configured)},
                clear=False,
            ):
                result = get_log_dir()

            self.assertEqual(result, configured.resolve())
            self.assertTrue(result.is_dir())

    def test_default_directory_stays_next_to_main_script(self) -> None:
        with TemporaryDirectory() as directory_name:
            script = Path(directory_name) / "application" / "run.py"
            with (
                patch.dict("os.environ", {}, clear=False),
                patch("sys.argv", [str(script)]),
                patch.dict("os.environ", {"HBROWSER_LOG_DIR": ""}, clear=False),
            ):
                result = get_log_dir()

            self.assertEqual(result, script.parent.resolve() / "log")
            self.assertTrue(result.is_dir())
