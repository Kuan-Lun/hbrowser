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
    LogLevel,
    LogPersistenceError,
    _isolated_logging_state_for_testing,
    _isolated_process_log_handlers_for_testing,
    configure_logging,
    get_log_dir,
    log_context,
    log_to_process_file,
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
    def setUp(self) -> None:
        self.logging_state = _isolated_logging_state_for_testing()
        self.logging_state.__enter__()

    def tearDown(self) -> None:
        self.logging_state.__exit__(None, None, None)

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
                    {"HBROWSER_PROCESS_LOG_FILE": ""},
                    clear=False,
                ),
            ):
                configured = setup_logger(logger_name)
                configured.info("compatibility record")
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
        self.assertEqual(configured_level, logging.INFO)
        self.assertEqual(handler_level, logging.INFO)
        self.assertFalse(configured_propagate)
        self.assertIn("compatibility record", output.getvalue())

    def test_defaults_write_debug_to_file_without_printing_it(self) -> None:
        logger_name = f"hbrowser.tests.split_defaults.{id(self)}"
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
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "battle.log"
                        )
                    },
                    clear=False,
                ),
            ):
                configured = setup_logger(logger_name)
                process_handler = next(
                    handler
                    for handler in configured.handlers
                    if isinstance(handler, logging.FileHandler)
                )
                stdout_handler = next(
                    handler
                    for handler in configured.handlers
                    if not isinstance(handler, logging.FileHandler)
                )
                with patch.object(
                    process_handler,
                    "flush",
                    wraps=process_handler.flush,
                ) as flush:
                    configured.debug("durable debug record")
                    configured.info("visible info record")
                self.assertEqual(flush.call_count, 2)
                contents = Path(process_handler.baseFilename).read_text(
                    encoding="utf-8"
                )

                self.assertEqual(configured.level, logging.DEBUG)
                self.assertEqual(stdout_handler.level, logging.INFO)
                self.assertEqual(process_handler.level, logging.DEBUG)
                self.assertNotIn("durable debug record", output.getvalue())
                self.assertIn("visible info record", output.getvalue())
                self.assertIn("durable debug record", contents)
                self.assertIn("visible info record", contents)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)

    def test_configuration_is_strict_and_atomic(self) -> None:
        for keyword, value, error_type in (
            ("console_level", "INFO", TypeError),
            ("file_level", logging.DEBUG, TypeError),
            ("max_bytes", True, TypeError),
            ("max_bytes", 0, ValueError),
            ("backup_count", 1.5, TypeError),
            ("backup_count", -1, ValueError),
        ):
            with self.subTest(keyword=keyword), self.assertRaises(error_type):
                configure_logging(**{keyword: value})  # type: ignore[arg-type]

        logger_name = f"hbrowser.tests.strict_defaults.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        try:
            logger.handlers = []
            with patch.dict(
                os.environ,
                {"HBROWSER_PROCESS_LOG_FILE": ""},
                clear=False,
            ):
                configured = setup_logger(logger_name)
            self.assertEqual(configured.level, logging.INFO)
            self.assertEqual(configured.handlers[0].level, logging.INFO)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)

    def test_setup_rejects_unmanaged_stream_handler_without_mutation(self) -> None:
        logger_name = f"hbrowser.tests.unmanaged_stream.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate
        unmanaged = logging.StreamHandler(io.StringIO())
        try:
            logger.handlers = [unmanaged]
            logger.setLevel(logging.ERROR)
            logger.propagate = True
            with (
                patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": ""},
                    clear=False,
                ),
                self.assertRaisesRegex(ValueError, "unmanaged handlers"),
            ):
                setup_logger(logger_name)

            self.assertEqual(logger.handlers, [unmanaged])
            self.assertEqual(logger.level, logging.ERROR)
            self.assertTrue(logger.propagate)
            self.assertNotIn(logger_name, log_module._MANAGED_LOGGER_NAMES)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate
            unmanaged.close()

    def test_setup_rejects_unmanaged_file_handler_without_mutation(self) -> None:
        logger_name = f"hbrowser.tests.unmanaged_file.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate
        with TemporaryDirectory() as directory_name:
            unmanaged = logging.FileHandler(
                Path(directory_name) / "unmanaged.log",
                encoding="utf-8",
            )
            try:
                logger.handlers = [unmanaged]
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                with (
                    patch.dict(
                        os.environ,
                        {"HBROWSER_PROCESS_LOG_FILE": ""},
                        clear=False,
                    ),
                    self.assertRaisesRegex(ValueError, "unmanaged handlers"),
                ):
                    setup_logger(logger_name)

                self.assertEqual(logger.handlers, [unmanaged])
                self.assertEqual(logger.level, logging.CRITICAL)
                self.assertFalse(logger.propagate)
                self.assertIsNotNone(unmanaged.stream)
                self.assertNotIn(logger_name, log_module._MANAGED_LOGGER_NAMES)
            finally:
                logger.handlers = original_handlers
                logger.setLevel(original_level)
                logger.propagate = original_propagate
                unmanaged.close()

    def test_reconfiguration_preserves_post_registration_instrumentation(self) -> None:
        logger_name = f"hbrowser.tests.instrumented.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        instrumentation = logging.StreamHandler(io.StringIO())
        instrumentation.setLevel(logging.CRITICAL)
        try:
            logger.handlers = []
            with patch.dict(
                os.environ,
                {"HBROWSER_PROCESS_LOG_FILE": ""},
                clear=False,
            ):
                configured = setup_logger(logger_name)
                managed_handler = configured.handlers[0]
                configured.addHandler(instrumentation)

                configure_logging(console_level=LogLevel.WARNING)
                repeated = setup_logger(logger_name)

            self.assertIs(repeated, configured)
            self.assertEqual(
                configured.handlers,
                [managed_handler, instrumentation],
            )
            self.assertEqual(managed_handler.level, logging.WARNING)
            self.assertEqual(instrumentation.level, logging.CRITICAL)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            instrumentation.close()

    def test_file_only_record_never_reaches_console(self) -> None:
        logger_name = f"hbrowser.tests.file_only.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
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
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "battle.log"
                        )
                    },
                    clear=False,
                ),
            ):
                configured = setup_logger(logger_name)
                configure_logging(
                    console_level=LogLevel.DEBUG,
                    file_level=LogLevel.DEBUG,
                )
                with log_context(realm="isekai", activity="Worker"):
                    log_to_process_file(
                        configured,
                        LogLevel.ERROR,
                        '{"result":"error"}',
                    )
                process_handler = next(
                    handler
                    for handler in configured.handlers
                    if isinstance(handler, logging.FileHandler)
                )
                process_handler.flush()
                contents = Path(process_handler.baseFilename).read_text(
                    encoding="utf-8"
                )

                self.assertEqual(output.getvalue(), "")
                self.assertIn(" - ERROR - [Isekai · Worker] ", contents)
                self.assertIn('{"result":"error"}', contents)
        finally:
            logger.handlers = original_handlers

    def test_file_only_record_respects_sink_threshold_and_absence(self) -> None:
        logger_name = f"hbrowser.tests.file_only_threshold.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with patch.dict(
                os.environ,
                {"HBROWSER_PROCESS_LOG_FILE": ""},
                clear=False,
            ):
                configured = setup_logger(logger_name)
                log_to_process_file(configured, LogLevel.ERROR, "no sink")

            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
                patch.dict(
                    os.environ,
                    {
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "battle.log"
                        )
                    },
                    clear=False,
                ),
            ):
                configure_logging(file_level=LogLevel.CRITICAL)
                log_to_process_file(configured, LogLevel.ERROR, "filtered")
                process_handler = next(
                    handler
                    for handler in configured.handlers
                    if isinstance(handler, logging.FileHandler)
                )
                process_handler.flush()
                self.assertNotIn(
                    "filtered",
                    Path(process_handler.baseFilename).read_text(encoding="utf-8"),
                )

            unmanaged = logging.getLogger(f"hbrowser.tests.unmanaged.{id(self)}")
            with self.assertRaisesRegex(ValueError, "setup_logger"):
                log_to_process_file(unmanaged, LogLevel.ERROR, "rejected")
        finally:
            logger.handlers = original_handlers

    def test_file_only_call_detaches_sink_after_path_is_unset(self) -> None:
        logger_name = f"hbrowser.tests.file_only_unset.{id(self)}"
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

                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": ""},
                    clear=False,
                ):
                    log_to_process_file(
                        configured,
                        LogLevel.ERROR,
                        "must not use stale sink",
                    )

                self.assertNotIn(process_handler, configured.handlers)
                self.assertEqual(log_module._PROCESS_LOG_HANDLERS, {})
                self.assertIsNone(process_handler.stream)
                self.assertNotIn(
                    "must not use stale sink",
                    process_path.read_text(encoding="utf-8"),
                )
        finally:
            logger.handlers = original_handlers

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

    def test_reconfiguration_attaches_file_sink_to_all_existing_loggers(self) -> None:
        first_name = f"hbrowser.tests.reconfigure.first.{id(self)}"
        second_name = f"hbrowser.tests.reconfigure.second.{id(self)}"
        first = logging.getLogger(first_name)
        second = logging.getLogger(second_name)
        original_first_handlers = first.handlers[:]
        original_second_handlers = second.handlers[:]
        try:
            first.handlers = []
            second.handlers = []
            with patch.dict(
                os.environ,
                {"HBROWSER_PROCESS_LOG_FILE": ""},
                clear=False,
            ):
                setup_logger(first_name)
                setup_logger(second_name)
                self.assertFalse(
                    any(
                        isinstance(handler, logging.FileHandler)
                        for handler in first.handlers + second.handlers
                    )
                )

            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
                patch.dict(
                    os.environ,
                    {
                        "HBROWSER_PROCESS_LOG_FILE": str(
                            Path(directory_name) / "battle.log"
                        )
                    },
                    clear=False,
                ),
            ):
                configure_logging(
                    console_level=LogLevel.ERROR,
                    file_level=LogLevel.DEBUG,
                )
                first_file_handler = next(
                    handler
                    for handler in first.handlers
                    if isinstance(handler, logging.FileHandler)
                )
                second_file_handler = next(
                    handler
                    for handler in second.handlers
                    if isinstance(handler, logging.FileHandler)
                )

                self.assertIs(first_file_handler, second_file_handler)
                self.assertEqual(first.level, logging.DEBUG)
                self.assertEqual(second.level, logging.DEBUG)
        finally:
            first.handlers = original_first_handlers
            second.handlers = original_second_handlers

    def test_reconfiguration_updates_existing_managed_logger_levels(self) -> None:
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
                    if isinstance(handler, logging.handlers.RotatingFileHandler)
                )

                configure_logging(
                    console_level=LogLevel.WARNING,
                    file_level=LogLevel.DEBUG,
                    max_bytes=2048,
                    backup_count=2,
                )
                second = setup_logger(logger_name)
                second.debug("file-only debug record")
                second.warning("shared warning record")
                process_handler.flush()

                self.assertIs(second, first)
                self.assertEqual(second.level, logging.DEBUG)
                self.assertEqual(stdout_handler.level, logging.WARNING)
                self.assertEqual(process_handler.level, logging.DEBUG)
                self.assertEqual(process_handler.maxBytes, 2048)
                self.assertEqual(process_handler.backupCount, 2)
                self.assertNotIn("file-only debug record", output.getvalue())
                self.assertIn("shared warning record", output.getvalue())
                contents = Path(process_handler.baseFilename).read_text(
                    encoding="utf-8"
                )
                self.assertIn("file-only debug record", contents)
                self.assertIn("shared warning record", contents)
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
                    self.assertRaises(LogPersistenceError) as error_info,
                ):
                    setup_logger(logger_name)
                self.assertEqual(error_info.exception.operation, "configure")
                self.assertIsInstance(error_info.exception.__cause__, OSError)
                self.assertIn("regular file", str(error_info.exception.__cause__))
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
                    self.assertRaises(LogPersistenceError) as error_info,
                ):
                    setup_logger(logger_name)

                self.assertEqual(error_info.exception.operation, "configure")
                self.assertIsInstance(error_info.exception.__cause__, OSError)
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

                    with self.assertRaises(LogPersistenceError) as error_info:
                        configured.info("must not reach either file")

                    self.assertEqual(error_info.exception.operation, "emit")
                    self.assertIsInstance(error_info.exception.__cause__, OSError)
                    self.assertIn(
                        "no longer names the opened file",
                        str(error_info.exception.__cause__),
                    )
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

    def test_failed_old_sink_close_remains_owned_and_can_be_retried(self) -> None:
        logger_name = f"hbrowser.tests.process.close_retry.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                directory = Path(directory_name)
                first_path = directory / "first.log"
                second_path = directory / "second.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(first_path)},
                    clear=False,
                ):
                    configured = setup_logger(logger_name)
                    first_handler = next(
                        handler
                        for handler in configured.handlers
                        if isinstance(handler, logging.FileHandler)
                    )

                with (
                    patch.dict(
                        os.environ,
                        {"HBROWSER_PROCESS_LOG_FILE": str(second_path)},
                        clear=False,
                    ),
                    patch.object(
                        first_handler,
                        "close",
                        side_effect=OSError("simulated close failure"),
                    ),
                    self.assertRaises(LogPersistenceError) as error_info,
                ):
                    configure_logging(
                        console_level=LogLevel.WARNING,
                        file_level=LogLevel.ERROR,
                        max_bytes=2048,
                        backup_count=2,
                    )

                self.assertEqual(error_info.exception.operation, "close")
                self.assertIs(
                    log_module._PROCESS_LOG_HANDLERS[first_path],
                    first_handler,
                )
                self.assertNotIn(first_handler, configured.handlers)
                second_handler = log_module._PROCESS_LOG_HANDLERS[second_path]
                self.assertIn(second_handler, configured.handlers)
                self.assertEqual(configured.level, logging.WARNING)
                self.assertEqual(second_handler.level, logging.ERROR)
                self.assertEqual(second_handler.maxBytes, 2048)
                self.assertEqual(second_handler.backupCount, 2)

                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(second_path)},
                    clear=False,
                ):
                    configure_logging(
                        console_level=LogLevel.WARNING,
                        file_level=LogLevel.ERROR,
                        max_bytes=2048,
                        backup_count=2,
                    )
                self.assertNotIn(first_path, log_module._PROCESS_LOG_HANDLERS)
                self.assertIs(
                    log_module._PROCESS_LOG_HANDLERS[second_path],
                    second_handler,
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
            ):
                process_path = Path(directory_name) / "battle.log"
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configure_logging(max_bytes=1, backup_count=2)
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

    def test_rollover_secures_existing_backup_before_shifting_it(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX file modes are required")

        logger_name = f"hbrowser.tests.process.rotation_mode.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                process_path = Path(directory_name) / "battle.log"
                first_backup = Path(f"{process_path}.1")
                second_backup = Path(f"{process_path}.2")
                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configure_logging(max_bytes=1, backup_count=2)
                    configured = setup_logger(logger_name)
                    configured.info("seed active log")
                    first_backup.write_text("legacy backup\n", encoding="utf-8")
                    first_backup.chmod(0o644)

                    configured.info("trigger secure rollover")

                    self.assertEqual(
                        second_backup.read_text(encoding="utf-8"),
                        "legacy backup\n",
                    )
                    for path in (process_path, first_backup, second_backup):
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        finally:
            logger.handlers = original_handlers

    def test_rollover_rejects_symlink_backup_without_touching_target(self) -> None:
        logger_name = f"hbrowser.tests.process.rotation_symlink.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                directory = Path(directory_name)
                process_path = directory / "battle.log"
                target_path = directory / "target.log"
                target_path.write_text("must remain unchanged\n", encoding="utf-8")
                backup_path = Path(f"{process_path}.1")

                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configure_logging(max_bytes=1, backup_count=2)
                    configured = setup_logger(logger_name)
                    configured.info("seed active log")
                    try:
                        backup_path.symlink_to(target_path)
                    except OSError as error:
                        self.skipTest(f"symbolic links are unavailable: {error}")
                    with self.assertRaises(LogPersistenceError) as error_info:
                        configured.info("must fail before rollover")

                self.assertEqual(error_info.exception.operation, "rollover")
                self.assertTrue(backup_path.is_symlink())
                self.assertEqual(
                    target_path.read_text(encoding="utf-8"),
                    "must remain unchanged\n",
                )
        finally:
            logger.handlers = original_handlers

    def test_rollover_rejects_hardlink_backup_without_touching_target(self) -> None:
        logger_name = f"hbrowser.tests.process.rotation_hardlink.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            with (
                TemporaryDirectory() as directory_name,
                _isolated_process_log_handlers_for_testing(),
            ):
                directory = Path(directory_name)
                process_path = directory / "battle.log"
                target_path = directory / "target.log"
                target_path.write_text("must remain unchanged\n", encoding="utf-8")
                backup_path = Path(f"{process_path}.1")

                with patch.dict(
                    os.environ,
                    {"HBROWSER_PROCESS_LOG_FILE": str(process_path)},
                    clear=False,
                ):
                    configure_logging(max_bytes=1, backup_count=2)
                    configured = setup_logger(logger_name)
                    configured.info("seed active log")
                    try:
                        os.link(target_path, backup_path)
                    except OSError as error:
                        self.skipTest(f"hard links are unavailable: {error}")
                    with self.assertRaises(LogPersistenceError) as error_info:
                        configured.info("must fail before rollover")

                self.assertEqual(error_info.exception.operation, "rollover")
                self.assertEqual(target_path.stat().st_nlink, 2)
                self.assertEqual(backup_path.stat().st_nlink, 2)
                self.assertEqual(
                    target_path.read_text(encoding="utf-8"),
                    "must remain unchanged\n",
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
                        self.assertRaises(LogPersistenceError) as error_info,
                    ):
                        configured.error("write must fail closed")
                    self.assertEqual(error_info.exception.operation, "emit")
                    self.assertIsInstance(error_info.exception.__cause__, OSError)
                    self.assertEqual(
                        str(error_info.exception.__cause__),
                        "simulated process-log write failure",
                    )
                    self.assertNotIn("simulated", str(error_info.exception))
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
                        self.assertRaises(LogPersistenceError) as error_info,
                    ):
                        configured.error("rollover must fail closed")
                    self.assertEqual(error_info.exception.operation, "emit")
                    self.assertIsInstance(error_info.exception.__cause__, OSError)
                    self.assertEqual(
                        str(error_info.exception.__cause__),
                        "simulated process-log rollover failure",
                    )
                    self.assertNotIn("simulated", str(error_info.exception))
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
