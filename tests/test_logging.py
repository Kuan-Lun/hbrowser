from __future__ import annotations

import asyncio
import errno
import io
import json
import logging
import os
import socket
import stat
import struct
import subprocess
import sys
import time
import unittest
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory as _SystemTemporaryDirectory
from threading import Event, Thread
from unittest.mock import Mock, patch

from hbrowser.gallery.browser.process import start_owned_process
from hbrowser.gallery.utils import log as log_module
from hbrowser.gallery.utils._log_forwarding import (
    ForwardingDrainError,
    ForwardingHandler,
)
from hbrowser.gallery.utils.log import (
    LogForwardingReceiver,
    LoggingHealth,
    LogLevel,
    LogPersistenceError,
    _isolated_logging_state_for_testing,
    close_forwarded_logging,
    close_logging,
    configure_forwarded_logging,
    configure_logging,
    get_log_dir,
    log_context,
    log_to_process_file,
    logging_health,
    raise_for_log_persistence_failure,
    setup_logger,
    start_log_forwarding_receiver,
)


@contextmanager
def TemporaryDirectory() -> Iterator[str]:  # noqa: N802
    """Close the process sink before Windows removes the temporary directory."""
    with _SystemTemporaryDirectory() as directory_name:
        try:
            yield directory_name
        finally:
            if log_module._PROCESS_LOG_HANDLER is not None:
                previous_failure = log_module._LOG_PERSISTENCE_FAILURE
                try:
                    close_logging()
                except LogPersistenceError:
                    if previous_failure is None:
                        raise


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


def _read_events(directory: Path) -> list[dict[str, object]]:
    handler = log_module._PROCESS_LOG_HANDLER
    if handler is not None and handler.directory == directory:
        handler.wait_until_idle(timeout=3.0)
    events: list[dict[str, object]] = []
    for path in sorted(directory.glob("events-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _send_raw_forwarding_frame(endpoint: str, payload: bytes) -> None:
    host, raw_port = endpoint.split(":", 1)
    with socket.create_connection((host, int(raw_port)), timeout=2) as connection:
        connection.sendall(struct.pack(">I", len(payload)) + payload)


def _raw_forwarding_frame(value: dict[str, object]) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


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

    def test_record_fields_override_context_and_invalid_fields_inherit(self) -> None:
        formatter = log_module._formatter()
        with log_context(realm="persistent", activity="Battle"):
            activity = formatter.format(
                _log_record("activity", extra={"activity": "Maintenance"})
            )
            scoped = formatter.format(
                _log_record(
                    "scope",
                    extra={"scope": "Browser", "activity": "Maintenance"},
                )
            )
            inherited = formatter.format(
                _log_record("inherit", extra={"activity": "   ", "scope": 42})
            )

        self.assertIn("[Persistent · Maintenance] activity", activity)
        self.assertIn("[Browser] scope", scoped)
        self.assertIn("[Persistent · Battle] inherit", inherited)

    def test_invalid_context_values_do_not_leak(self) -> None:
        for field, value, error_type in (
            ("realm", "", ValueError),
            ("activity", "   ", ValueError),
            ("scope", 5, TypeError),
        ):
            with self.subTest(field=field), self.assertRaises(error_type):
                with log_context(**{field: value}):  # type: ignore[arg-type]
                    self.fail("invalid context unexpectedly entered")

        self.assertIn(
            "[System] after errors",
            log_module._formatter().format(_log_record("after errors")),
        )


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
        self.assertIn("[System] parent", formatter.format(_log_record("parent")))


class IsolatedLoggingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.logging_state = _isolated_logging_state_for_testing()
        self.logging_state.__enter__()

    def tearDown(self) -> None:
        self.logging_state.__exit__(None, None, None)


class LoggerConfigurationTests(IsolatedLoggingTestCase):
    def test_setup_only_registers_without_handlers_or_files(self) -> None:
        logger_name = f"hbrowser.tests.import_only.{id(self)}"
        with TemporaryDirectory() as parent_name:
            log_dir = Path(parent_name) / "not-created"
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": str(log_dir)}):
                logger = setup_logger(logger_name)
                logger.warning("suppressed before configuration")

            self.assertEqual(logger.handlers, [])
            self.assertFalse(logger.propagate)
            self.assertFalse(log_dir.exists())

    def test_explicit_configuration_attaches_console_and_file_sinks(self) -> None:
        logger_name = f"hbrowser.tests.explicit.{id(self)}"
        output = io.StringIO()
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
            ):
                logger = setup_logger(logger_name)
                configure_logging()
                logger.debug("private debug")
                logger.info("visible info")

            handlers = logger.handlers[:]
            events = _read_events(directory)

        self.assertEqual(len(handlers), 2)
        self.assertIn("visible info", output.getvalue())
        self.assertNotIn("private debug", output.getvalue())
        self.assertEqual(
            [event["message"] for event in events],
            ["private debug", "visible info"],
        )

    def test_logger_registered_after_configuration_uses_shared_handler(self) -> None:
        first_name = f"hbrowser.tests.shared.first.{id(self)}"
        second_name = f"hbrowser.tests.shared.second.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                first = setup_logger(first_name)
                configure_logging(console_level=LogLevel.CRITICAL)
                second = setup_logger(second_name)
                first.info("first")
                second.info("second")
                shared_handlers = set(first.handlers) & set(second.handlers)
                events = _read_events(Path(directory_name))

        file_handlers = [
            handler
            for handler in shared_handlers
            if getattr(handler, "_hbrowser_managed_process_handler", False)
        ]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual([event["message"] for event in events], ["first", "second"])

    def test_repeated_setup_and_reconfigure_do_not_duplicate_handlers(self) -> None:
        logger_name = f"hbrowser.tests.repeated.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                first = setup_logger(logger_name)
                configure_logging(console_level=LogLevel.WARNING)
                process_handler = log_module._PROCESS_LOG_HANDLER
                configure_logging(
                    console_level=LogLevel.ERROR,
                    file_level=LogLevel.INFO,
                    segment_bytes=4096,
                )
                second = setup_logger(logger_name)
                handler_was_reused = process_handler is log_module._PROCESS_LOG_HANDLER
                handler_count = len(first.handlers)
                logger_level = first.level

        self.assertIs(first, second)
        self.assertTrue(handler_was_reused)
        self.assertEqual(handler_count, 2)
        self.assertEqual(logger_level, logging.INFO)

    def test_reconfigure_rejects_directory_change_and_keeps_owner(self) -> None:
        with TemporaryDirectory() as first_name, TemporaryDirectory() as second_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": first_name}):
                configure_logging()
                handler = log_module._PROCESS_LOG_HANDLER
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": second_name}),
                self.assertRaisesRegex(ValueError, "cannot change"),
            ):
                configure_logging(file_level=LogLevel.INFO)
            handler_was_retained = handler is log_module._PROCESS_LOG_HANDLER

        self.assertTrue(handler_was_retained)

    def test_setup_rejects_unmanaged_handler_without_mutation(self) -> None:
        logger_name = f"hbrowser.tests.unmanaged.{id(self)}"
        logger = logging.getLogger(logger_name)
        unmanaged = logging.StreamHandler(io.StringIO())
        logger.addHandler(unmanaged)
        original_level = logger.level
        original_propagate = logger.propagate
        try:
            with self.assertRaisesRegex(ValueError, "unmanaged handlers"):
                setup_logger(logger_name)
            self.assertEqual(logger.handlers, [unmanaged])
            self.assertEqual(logger.level, original_level)
            self.assertEqual(logger.propagate, original_propagate)
        finally:
            logger.removeHandler(unmanaged)
            unmanaged.close()

    def test_configuration_arguments_are_strict(self) -> None:
        for keyword, value, error_type in (
            ("console_level", "INFO", TypeError),
            ("file_level", logging.DEBUG, TypeError),
            ("segment_bytes", True, TypeError),
            ("segment_bytes", 1.5, TypeError),
            ("segment_bytes", 0, ValueError),
            ("segment_bytes", -1, ValueError),
            ("require_file_sink", 1, TypeError),
        ):
            with self.subTest(keyword=keyword), self.assertRaises(error_type):
                configure_logging(**{keyword: value})  # type: ignore[arg-type]

        self.assertFalse(log_module._LOGGING_CONFIGURED)
        self.assertIsNone(log_module._PROCESS_LOG_HANDLER)

    def test_optional_file_sink_failure_keeps_console_and_latches_health(
        self,
    ) -> None:
        output = io.StringIO()
        logger_name = f"hbrowser.tests.optional_sink.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
                patch(
                    "hbrowser.gallery.utils.log.os.open",
                    side_effect=OSError(errno.EACCES, "private path"),
                ) as open_file,
            ):
                logger = setup_logger(logger_name)
                configure_logging(require_file_sink=False)
                logger.error("console survives")
                first_open_count = open_file.call_count

            configure_logging(
                console_level=LogLevel.INFO,
                require_file_sink=False,
            )
            health = logging_health()
            with self.assertRaises(LogPersistenceError) as direct_info:
                log_to_process_file(logger, LogLevel.ERROR, "must surface")
            with self.assertRaises(LogPersistenceError) as close_info:
                close_logging()

        self.assertGreater(first_open_count, 0)
        self.assertIn("console survives", output.getvalue())
        self.assertIsInstance(health, LoggingHealth)
        self.assertTrue(health.trace_degraded)
        self.assertEqual(health.operation, "segment-open")
        self.assertEqual(health.error_type, "PermissionError")
        self.assertEqual(health.errno, errno.EACCES)
        self.assertNotIn("private path", repr(health))
        self.assertIs(direct_info.exception, close_info.exception)

    def test_structured_jsonl_preserves_private_fields_and_console_semantics(
        self,
    ) -> None:
        output = io.StringIO()
        logger_name = f"hbrowser.tests.structured.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
            ):
                logger = setup_logger(logger_name)
                configure_logging()
                with log_context(
                    account="main",
                    realm="isekai",
                    tab_role="isekai",
                    activity="Battle",
                ):
                    logger.info("戰鬥 %s", "complete")
                event = _read_events(Path(directory_name))[0]

        self.assertIn("[Isekai · Battle] 戰鬥 complete", output.getvalue())
        self.assertEqual(event["level"], "INFO")
        self.assertEqual(event["logger"], logger_name)
        self.assertEqual(event["message"], "戰鬥 complete")
        self.assertEqual(event["account"], "main")
        self.assertEqual(event["realm"], "isekai")
        self.assertEqual(event["tab_role"], "isekai")
        self.assertEqual(event["activity"], "Battle")
        self.assertIsNone(event["scope"])
        self.assertEqual(event["semantic_label"], "Isekai · Battle")
        self.assertIsInstance(event["process_id"], int)
        self.assertIsInstance(event["thread_id"], int)
        self.assertRegex(str(event["timestamp"]), r"\+00:00\Z")

    def test_log_to_process_file_skips_console_and_respects_threshold(self) -> None:
        output = io.StringIO()
        logger_name = f"hbrowser.tests.file_only.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
            ):
                logger = setup_logger(logger_name)
                log_to_process_file(logger, LogLevel.ERROR, "before configure")
                configure_logging(file_level=LogLevel.ERROR)
                log_to_process_file(logger, LogLevel.WARNING, "filtered")
                log_to_process_file(logger, LogLevel.ERROR, "file only")
                events = _read_events(Path(directory_name))

        self.assertEqual(output.getvalue(), "")
        self.assertEqual([event["message"] for event in events], ["file only"])

    def test_log_to_process_file_rejects_unregistered_logger(self) -> None:
        logger = logging.getLogger(f"third_party.tests.unregistered.{id(self)}")
        with self.assertRaisesRegex(ValueError, "configured namespace logger"):
            log_to_process_file(logger, LogLevel.ERROR, "rejected")

    def test_log_to_process_file_rejects_noncanonical_namespace_logger(
        self,
    ) -> None:
        logger = logging.Logger(f"battle.tests.noncanonical.{id(self)}")
        with self.assertRaisesRegex(ValueError, "configured namespace logger"):
            log_to_process_file(logger, LogLevel.ERROR, "rejected")


class NamespaceAndForwardingTests(IsolatedLoggingTestCase):
    @staticmethod
    def _child_environment(receiver: LogForwardingReceiver) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(receiver.child_environment())
        return environment

    def test_ordinary_namespace_logger_uses_parent_handlers_once(self) -> None:
        output = io.StringIO()
        logger = logging.getLogger(f"hbrowser.business.{id(self)}")
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
            ):
                self.assertEqual(logger.handlers, [])
                configure_logging()
                logger.info("ordinary namespace record")
                events = _read_events(Path(directory_name))

        self.assertEqual(logger.handlers, [])
        self.assertEqual(
            [event["message"] for event in events],
            ["ordinary namespace record"],
        )
        self.assertEqual(output.getvalue().count("ordinary namespace record"), 1)

    def test_direct_file_write_accepts_namespace_roots_and_descendants(self) -> None:
        output = io.StringIO()
        loggers = [
            logging.getLogger(name)
            for namespace in ("battle", "hbrowser", "hvbrowser", "hvbattle")
            for name in (namespace, f"{namespace}.business.{id(self)}")
        ]
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
            ):
                for logger in loggers:
                    log_to_process_file(logger, LogLevel.ERROR, "before configure")
                configure_logging(file_level=LogLevel.ERROR)
                for logger in loggers:
                    log_to_process_file(logger, LogLevel.ERROR, logger.name)
                events = _read_events(Path(directory_name))

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(
            [(event["logger"], event["message"]) for event in events],
            [(logger.name, logger.name) for logger in loggers],
        )

    def test_child_send_and_close_failures_never_escape_handler_boundary(
        self,
    ) -> None:
        ready_payload = b'{"type":"ready"}'

        def connected_socket() -> Mock:
            connection = Mock()
            connection.recv.side_effect = [
                struct.pack(">I", len(ready_payload)),
                ready_payload,
            ]
            return connection

        for method_name, expected_stage in (
            ("emit", "forward-send"),
            ("close", "forward-close"),
        ):
            with self.subTest(method_name=method_name):
                connection = connected_socket()
                failure_callback = Mock()
                with patch(
                    "hbrowser.gallery.utils._log_forwarding.socket.create_connection",
                    return_value=connection,
                ):
                    handler = ForwardingHandler(
                        "127.0.0.1:43210",
                        "a" * 64,
                        failure_callback=failure_callback,
                    )
                connection.sendall.side_effect = OSError(
                    errno.EPIPE,
                    "private transport detail",
                )
                if method_name == "emit":
                    handler.handle(_log_record("business continues"))
                else:
                    handler.close_forwarding()

                self.assertTrue(_wait_until(lambda: failure_callback.call_count == 1))
                failure_callback.assert_called_once()
                self.assertEqual(failure_callback.call_args.args[0], expected_stage)
                handler.close_forwarding()

    def test_forwarding_close_fences_an_in_progress_record_encode(self) -> None:
        delivered: list[object] = []
        receiver_failure = Mock()

        def deliver(record: object) -> bool:
            delivered.append(record)
            return True

        receiver = LogForwardingReceiver(
            delivery_callback=deliver,
            failure_callback=receiver_failure,
        )
        capability = receiver.child_environment()
        handler = ForwardingHandler(
            capability["HBROWSER_LOG_FORWARD_ENDPOINT"],
            capability["HBROWSER_LOG_FORWARD_TOKEN"],
            failure_callback=Mock(),
        )
        encode_entered = Event()
        encode_release = Event()
        original_payload = handler._record_payload  # noqa: SLF001

        def blocked_payload(record: logging.LogRecord) -> dict[str, object]:
            encode_entered.set()
            encode_release.wait()
            return original_payload(record)

        with patch.object(handler, "_record_payload", side_effect=blocked_payload):
            emitter = Thread(
                target=handler.handle,
                args=(_log_record("late record"),),
            )
            emitter.start()
            self.assertTrue(encode_entered.wait(timeout=2))
            handler.close_forwarding()
            encode_release.set()
            emitter.join(timeout=2)

        self.assertFalse(emitter.is_alive())
        self.assertTrue(handler._queue.empty())  # noqa: SLF001
        receiver.close()
        self.assertEqual(delivered, [])
        receiver_failure.assert_not_called()

    def test_authenticated_child_forwards_only_bounded_whitelisted_data(
        self,
    ) -> None:
        child = """
import logging
import os
from hbrowser import close_forwarded_logging, configure_forwarded_logging
from hbrowser.gallery.utils import log_context

configure_forwarded_logging()
if "HBROWSER_LOG_FORWARD_ENDPOINT" in os.environ:
    raise SystemExit(7)
if "HBROWSER_LOG_FORWARD_TOKEN" in os.environ:
    raise SystemExit(8)
with log_context(account="child", realm="isekai", activity="Battle"):
    logging.getLogger("hbrowser.child.worker").info(
        "forwarded %s", "record", extra={"password": "must-not-cross"}
    )
    logging.getLogger("hbrowser.child.worker").warning("界" * 5000)
close_forwarded_logging()
"""
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                self.assertIsNotNone(receiver)
                assert receiver is not None
                result = subprocess.run(
                    [sys.executable, "-c", child],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=self._child_environment(receiver),
                    timeout=10,
                )
                self.assertTrue(_wait_until(lambda: len(_read_events(directory)) == 2))
                close_logging()
                events = _read_events(directory)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(events), 2)
        event, bounded_event = events
        self.assertEqual(event["logger"], "hbrowser.child.worker")
        self.assertEqual(event["message"], "forwarded record")
        self.assertEqual(event["account"], "child")
        self.assertEqual(event["realm"], "isekai")
        self.assertNotIn("password", event)
        self.assertNotIn("must-not-cross", json.dumps(event))
        self.assertLessEqual(
            len(str(bounded_event["message"]).encode("utf-8")),
            8 * 1024,
        )
        self.assertFalse(logging_health().forwarding_degraded)

    def test_multiple_children_are_serialized_into_parent_sink(self) -> None:
        child = """
import logging
import sys
from hbrowser import close_forwarded_logging, configure_forwarded_logging

configure_forwarded_logging()
logging.getLogger("hbrowser.child.concurrent").info("child-%s", sys.argv[1])
close_forwarded_logging()
"""
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                self.assertIsNotNone(receiver)
                assert receiver is not None
                children = [
                    subprocess.Popen(
                        [sys.executable, "-c", child, str(index)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=self._child_environment(receiver),
                    )
                    for index in range(4)
                ]
                results = [
                    child_process.communicate(timeout=10) for child_process in children
                ]
                self.assertTrue(_wait_until(lambda: len(_read_events(directory)) == 4))
                close_logging()
                events = _read_events(directory)

        self.assertTrue(
            all(child_process.returncode == 0 for child_process in children)
        )
        self.assertTrue(all(not stderr for _, stderr in results))
        self.assertEqual(
            {event["message"] for event in events},
            {"child-0", "child-1", "child-2", "child-3"},
        )

    def test_unauthenticated_and_oversized_noise_does_not_degrade_parent(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                self.assertIsNotNone(receiver)
                assert receiver is not None
                capability = receiver.child_environment()
                endpoint = capability["HBROWSER_LOG_FORWARD_ENDPOINT"]
                wrong_token = "0" * 64
                if wrong_token == capability["HBROWSER_LOG_FORWARD_TOKEN"]:
                    wrong_token = "1" * 64
                _send_raw_forwarding_frame(
                    endpoint,
                    json.dumps(
                        {"token": wrong_token, "type": "close"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
                host, raw_port = endpoint.split(":", 1)
                with socket.create_connection(
                    (host, int(raw_port)), timeout=2
                ) as connection:
                    connection.sendall(struct.pack(">I", 1_000_000))
                time.sleep(0.1)
                health = logging_health()

        self.assertFalse(health.trace_degraded)
        self.assertFalse(health.forwarding_degraded)

    def test_authenticated_protocol_failure_is_forwarding_only_degradation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                self.assertIsNotNone(receiver)
                assert receiver is not None
                capability = receiver.child_environment()
                malformed = json.dumps(
                    {
                        "extra": "rejected",
                        "token": capability["HBROWSER_LOG_FORWARD_TOKEN"],
                        "type": "close",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                _send_raw_forwarding_frame(
                    capability["HBROWSER_LOG_FORWARD_ENDPOINT"],
                    malformed,
                )
                self.assertTrue(
                    _wait_until(lambda: logging_health().forwarding_degraded)
                )
                health = logging_health()

        self.assertFalse(health.trace_degraded)
        self.assertTrue(health.forwarding_degraded)
        self.assertEqual(health.forwarding_operation, "forward-protocol")

    def test_close_drains_authenticated_client_before_segment_close(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                self.assertIsNotNone(receiver)
                assert receiver is not None
                capability = receiver.child_environment()
                host, raw_port = capability["HBROWSER_LOG_FORWARD_ENDPOINT"].split(
                    ":", 1
                )
                record = {
                    "account": None,
                    "activity": None,
                    "created": time.time(),
                    "exception": None,
                    "level": "INFO",
                    "logger": "hbrowser.child.drain",
                    "message": "drained before close",
                    "process_id": 123,
                    "realm": None,
                    "scope": None,
                    "stack": None,
                    "tab_role": None,
                    "thread_id": 456,
                }
                with socket.create_connection(
                    (host, int(raw_port)), timeout=2
                ) as connection:
                    connection.sendall(
                        _raw_forwarding_frame(
                            {
                                "record": record,
                                "token": capability["HBROWSER_LOG_FORWARD_TOKEN"],
                                "type": "record",
                            }
                        )
                        + _raw_forwarding_frame(
                            {
                                "token": capability["HBROWSER_LOG_FORWARD_TOKEN"],
                                "type": "close",
                            }
                        )
                    )
                    close_logging()
                events = _read_events(directory)

        self.assertEqual(
            [event["message"] for event in events], ["drained before close"]
        )
        self.assertFalse(logging_health().forwarding_degraded)

    def test_receiver_close_is_bounded_when_delivery_callback_never_returns(
        self,
    ) -> None:
        entered = Event()
        release = Event()
        failure_entered = Event()
        failure_release = Event()
        failure_stages: list[str] = []

        def blocked_delivery(_record: object) -> bool:
            entered.set()
            release.wait()
            return True

        def blocked_failure(stage: str, _error: BaseException) -> None:
            failure_stages.append(stage)
            failure_entered.set()
            failure_release.wait()

        receiver = LogForwardingReceiver(
            delivery_callback=blocked_delivery,
            failure_callback=blocked_failure,
        )
        capability = receiver.child_environment()
        host, raw_port = capability["HBROWSER_LOG_FORWARD_ENDPOINT"].split(":", 1)
        connection = socket.create_connection((host, int(raw_port)), timeout=2)
        try:
            connection.sendall(
                _raw_forwarding_frame(
                    {
                        "token": capability["HBROWSER_LOG_FORWARD_TOKEN"],
                        "type": "hello",
                    }
                )
            )
            header = connection.recv(4)
            self.assertEqual(len(header), 4)
            ready_size = struct.unpack(">I", header)[0]
            self.assertEqual(connection.recv(ready_size), b'{"type":"ready"}')
            connection.sendall(
                _raw_forwarding_frame(
                    {
                        "record": {
                            "account": None,
                            "activity": None,
                            "created": time.time(),
                            "exception": None,
                            "level": "INFO",
                            "logger": "hbrowser.child.blocked",
                            "message": "blocked delivery",
                            "process_id": 123,
                            "realm": None,
                            "scope": None,
                            "stack": None,
                            "tab_role": None,
                            "thread_id": 456,
                        },
                        "token": capability["HBROWSER_LOG_FORWARD_TOKEN"],
                        "type": "record",
                    }
                )
            )
            self.assertTrue(entered.wait(timeout=2))

            started = time.monotonic()
            with self.assertRaises(ForwardingDrainError):
                receiver.close(drain_timeout=0.05)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(failure_entered.wait(timeout=0.5))
            self.assertEqual(failure_stages, ["forward-drain"])
        finally:
            release.set()
            failure_release.set()
            connection.close()
        self.assertTrue(_wait_until(lambda: not receiver._clients))  # noqa: SLF001

    def test_receiver_drain_failure_is_secondary_to_segment_close(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(receiver)
                self.assertIsNotNone(handler)
                assert receiver is not None
                assert handler is not None
                with patch.object(
                    receiver,
                    "close",
                    side_effect=OSError(errno.EIO, "private receiver detail"),
                ):
                    close_logging()
                health = logging_health()
                self.assertTrue(handler._abandoned_until_process_exit)  # noqa: SLF001
                self.assertIsNotNone(handler._stream)  # noqa: SLF001
                handler.close_sink()

        self.assertIsNone(log_module._PROCESS_LOG_HANDLER)
        self.assertFalse(health.trace_degraded)
        self.assertTrue(health.forwarding_degraded)
        self.assertEqual(health.forwarding_operation, "forward-drain")
        raise_for_log_persistence_failure()

    def test_receiver_sink_failure_latches_parent_sink_health(self) -> None:
        child = """
import logging
from hbrowser import close_forwarded_logging, configure_forwarded_logging
configure_forwarded_logging()
logging.getLogger("hbrowser.child.failure").error("cannot persist")
close_forwarded_logging()
"""
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                receiver = start_log_forwarding_receiver()
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(receiver)
                self.assertIsNotNone(handler)
                assert receiver is not None
                assert handler is not None
                with patch.object(
                    handler,
                    "_write_bytes",
                    side_effect=OSError(errno.ENOSPC, "private sink path"),
                ):
                    result = subprocess.run(
                        [sys.executable, "-c", child],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=self._child_environment(receiver),
                        timeout=10,
                    )
                    self.assertTrue(
                        _wait_until(lambda: logging_health().trace_degraded)
                    )
                health = logging_health()

        self.assertEqual(result.returncode, 0)
        self.assertTrue(health.trace_degraded)
        self.assertFalse(health.forwarding_degraded)

    def test_optional_sink_and_receiver_bind_failure_are_non_blocking(self) -> None:
        with TemporaryDirectory() as directory_name:
            business_result = Path(directory_name) / "business-result"
            with (
                patch(
                    "hbrowser.gallery.utils.log.get_log_dir",
                    side_effect=OSError(errno.EACCES, "private path"),
                ),
                patch("hbrowser.gallery.utils.log.os.write"),
            ):
                configure_logging(require_file_sink=False)
                self.assertIsNone(start_log_forwarding_receiver())
                self.assertEqual(log_module._log_forwarding_environment_for_child(), {})
                process = start_owned_process(
                    sys.executable,
                    [
                        "-c",
                        (
                            "import pathlib,sys,time;"
                            "pathlib.Path(sys.argv[1]).write_text("
                            "'completed',encoding='utf-8');time.sleep(0.2)"
                        ),
                        str(business_result),
                    ],
                    forward_logging=True,
                )
                self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(
                business_result.read_text(encoding="utf-8"),
                "completed",
            )
        health = logging_health()
        self.assertTrue(health.trace_degraded)
        self.assertTrue(health.forwarding_degraded)

    def test_receiver_bind_failure_is_sticky_and_non_raising(self) -> None:
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch(
                    "hbrowser.gallery.utils.log.LogForwardingReceiver",
                    side_effect=OSError(errno.EADDRINUSE, "private endpoint"),
                ) as receiver_type,
                patch("hbrowser.gallery.utils.log.os.write"),
            ):
                configure_logging(console_level=LogLevel.CRITICAL)
                self.assertIsNone(start_log_forwarding_receiver())
                self.assertIsNone(start_log_forwarding_receiver())
                self.assertEqual(log_module._log_forwarding_environment_for_child(), {})
                health = logging_health()

        receiver_type.assert_called_once()
        self.assertTrue(health.forwarding_degraded)
        self.assertEqual(health.forwarding_errno, errno.EADDRINUSE)

    def test_missing_receiver_and_connection_failure_do_not_raise_from_logs(
        self,
    ) -> None:
        emergency_write = Mock(return_value=0)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("hbrowser.gallery.utils.log.os.write", emergency_write),
        ):
            configure_forwarded_logging()
            logging.getLogger("hbrowser.child.no_receiver").error("business continues")
            close_forwarded_logging()
            self.assertTrue(_wait_until(lambda: emergency_write.call_count == 1))

        health = logging_health()
        self.assertTrue(health.forwarding_degraded)
        self.assertEqual(health.forwarding_operation, "forward-configure")
        emergency_write.assert_called_once()

    def test_child_connection_refusal_does_not_change_business_result(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        emergency_write = Mock(return_value=0)
        with (
            patch.dict(
                os.environ,
                {
                    "HBROWSER_LOG_FORWARD_ENDPOINT": f"127.0.0.1:{port}",
                    "HBROWSER_LOG_FORWARD_TOKEN": "a" * 64,
                },
                clear=True,
            ),
            patch("hbrowser.gallery.utils.log.os.write", emergency_write),
        ):
            configure_forwarded_logging()
            logging.getLogger("hbrowser.child.refused").critical("business survives")
            close_forwarded_logging()
            self.assertTrue(_wait_until(lambda: emergency_write.call_count == 1))

        health = logging_health()
        self.assertTrue(health.forwarding_degraded)
        self.assertEqual(health.forwarding_operation, "forward-configure")
        emergency_write.assert_called_once()

    def test_forwarding_emergency_record_is_first_only_and_bounded(self) -> None:
        adversarial_type = type("Bad/型" + ("X" * 256), (OSError,), {})
        cause = adversarial_type()
        setattr(cause, "errno", 1 << 200)
        setattr(cause, "winerror", -1)
        emergency_write = Mock(return_value=0)
        with (
            patch("hbrowser.gallery.utils.log.os.write", emergency_write),
            patch("hbrowser.gallery.utils.log.os.getpid", return_value=1 << 200),
        ):
            log_module._record_forwarding_failure("forward-send", cause)
            log_module._record_forwarding_failure(
                "forward-close", OSError(errno.EIO, "second")
            )
            self.assertTrue(_wait_until(lambda: emergency_write.call_count == 1))

        emergency_write.assert_called_once()
        _, encoded = emergency_write.call_args.args
        payload = json.loads(encoded.decode("ascii"))
        self.assertLess(len(encoded), 512)
        self.assertEqual(
            set(payload),
            {
                "event",
                "stage",
                "error_type",
                "errno",
                "winerror",
                "writer_pid",
            },
        )
        self.assertEqual(payload["stage"], "forward-send")
        self.assertEqual(payload["error_type"], "UnknownError")
        self.assertIsNone(payload["errno"])
        self.assertIsNone(payload["winerror"])
        self.assertIsNone(payload["writer_pid"])


class SegmentLifecycleTests(IsolatedLoggingTestCase):
    def test_initial_allocation_validates_existing_segments_and_uses_max_plus_one(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "events-000001.jsonl"
            seventh = directory / "events-000007.jsonl"
            first.write_text('{"old":1}\n', encoding="utf-8")
            seventh.write_text('{"old":7}\n', encoding="utf-8")
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                setup_logger(f"hbrowser.tests.max_plus_one.{id(self)}").info("new")

            self.assertEqual(first.read_text(encoding="utf-8"), '{"old":1}\n')
            self.assertEqual(seventh.read_text(encoding="utf-8"), '{"old":7}\n')
            self.assertTrue((directory / "events-000008.jsonl").is_file())

    def test_exact_byte_boundary_rolls_before_the_next_record(self) -> None:
        logger_name = f"hbrowser.tests.exact_boundary.{id(self)}"
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(logger_name)
                configure_logging(
                    console_level=LogLevel.CRITICAL,
                    segment_bytes=1024 * 1024,
                )
                logger.info("exact first")
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                self.assertTrue(handler.wait_until_idle(timeout=3.0))
                first = directory / "events-000001.jsonl"
                exact_size = first.stat().st_size
                configure_logging(
                    console_level=LogLevel.CRITICAL,
                    segment_bytes=exact_size,
                )
                logger.info("second segment")
                self.assertTrue(handler.wait_until_idle(timeout=3.0))

            first_events = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            second_events = [
                json.loads(line)
                for line in (directory / "events-000002.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_size_after_rollover = first.stat().st_size

        self.assertEqual(first_size_after_rollover, exact_size)
        self.assertEqual([event["message"] for event in first_events], ["exact first"])
        self.assertEqual(
            [event["message"] for event in second_events], ["second segment"]
        )

    def test_oversized_first_record_does_not_create_an_empty_segment(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(f"hbrowser.tests.oversized.{id(self)}")
                configure_logging(
                    console_level=LogLevel.CRITICAL,
                    segment_bytes=1,
                )
                logger.info("oversized")
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                self.assertTrue(handler.wait_until_idle(timeout=3.0))

            paths = sorted(directory.glob("events-*.jsonl"))
            self.assertEqual([path.name for path in paths], ["events-000001.jsonl"])
            self.assertGreater(paths[0].stat().st_size, 1)

    def test_rollover_never_renames_replaces_or_unlinks_segments(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch("hbrowser.gallery.utils.log.os.rename") as rename,
                patch("hbrowser.gallery.utils.log.os.replace") as replace,
                patch("hbrowser.gallery.utils.log.os.unlink") as unlink,
            ):
                logger = setup_logger(f"hbrowser.tests.no_rename.{id(self)}")
                configure_logging(
                    console_level=LogLevel.CRITICAL,
                    segment_bytes=1,
                )
                logger.info("first")
                logger.info("second")
                logger.info("third")
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                self.assertTrue(handler.wait_until_idle(timeout=3.0))

            names = [path.name for path in sorted(directory.glob("events-*"))]

        self.assertEqual(
            names,
            [
                "events-000001.jsonl",
                "events-000002.jsonl",
                "events-000003.jsonl",
            ],
        )
        rename.assert_not_called()
        replace.assert_not_called()
        unlink.assert_not_called()

    def test_exclusive_create_race_fails_instead_of_skipping_sequence(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raced = directory / "events-000001.jsonl"

            def race() -> int:
                raced.write_text("competitor", encoding="utf-8")
                return 1

            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch.object(
                    log_module._AppendOnlyJsonlHandler,
                    "_discover_next_segment_index",
                    side_effect=race,
                ),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

            self.assertEqual(raced.read_text(encoding="utf-8"), "competitor")
            self.assertFalse((directory / "events-000002.jsonl").exists())

        self.assertEqual(error_info.exception.operation, "segment-open")
        self.assertEqual(error_info.exception.segment_index, 1)

    def test_directory_owner_lock_rejects_concurrent_process_and_allows_next(
        self,
    ) -> None:
        child = """
from hbrowser import LogPersistenceError, close_logging, configure_logging

try:
    configure_logging()
except LogPersistenceError as error:
    raise SystemExit(0 if error.operation == "configure" else 2)
else:
    close_logging()
    raise SystemExit(3)
"""
        successor = """
from hbrowser import close_logging, configure_logging

configure_logging()
close_logging()
"""
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            environment = os.environ.copy()
            environment["HBROWSER_LOG_DIR"] = directory_name
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging(console_level=LogLevel.CRITICAL)
                competing = subprocess.run(
                    [sys.executable, "-c", child],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                close_logging()
            sequential = subprocess.run(
                [sys.executable, "-c", successor],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            segment_names = [
                path.name for path in sorted(directory.glob("events-*.jsonl"))
            ]

        self.assertEqual(competing.returncode, 0, competing.stderr)
        self.assertEqual(sequential.returncode, 0, sequential.stderr)
        self.assertEqual(
            segment_names,
            ["events-000001.jsonl", "events-000002.jsonl"],
        )

    def test_existing_symlink_segment_is_rejected(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            target = directory / "target"
            target.write_text("private", encoding="utf-8")
            link = directory / "events-000001.jsonl"
            try:
                link.symlink_to(target)
            except NotImplementedError, OSError:
                self.skipTest("symbolic links are unavailable")
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

            self.assertEqual(target.read_text(encoding="utf-8"), "private")

        self.assertEqual(error_info.exception.operation, "segment-open")

    def test_existing_hardlink_segment_is_rejected(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            target = directory / "target"
            target.write_text("private", encoding="utf-8")
            link = directory / "events-000001.jsonl"
            try:
                os.link(target, link)
            except OSError:
                self.skipTest("hard links are unavailable")
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

            self.assertEqual(target.read_text(encoding="utf-8"), "private")

        self.assertEqual(error_info.exception.operation, "segment-open")

    def test_existing_non_regular_segment_is_rejected_without_opening_it(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            candidate = directory / "events-000001.jsonl"
            candidate.mkdir()
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

        self.assertEqual(error_info.exception.operation, "segment-open")

    def test_symlink_log_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as parent_name:
            parent = Path(parent_name)
            target = parent / "target"
            target.mkdir()
            link = parent / "linked-log"
            try:
                link.symlink_to(target, target_is_directory=True)
            except NotImplementedError, OSError:
                self.skipTest("directory symbolic links are unavailable")
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": str(link)}),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

            self.assertEqual(tuple(target.iterdir()), ())

        self.assertEqual(error_info.exception.operation, "segment-open")

    def test_windows_reparse_directory_is_rejected(self) -> None:
        directory_stat = Mock(
            st_mode=stat.S_IFDIR | 0o700,
            st_file_attributes=0x400,
        )
        with (
            patch(
                "hbrowser.gallery.utils.log.stat.FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
                create=True,
            ),
            patch.object(Path, "lstat", return_value=directory_stat),
            self.assertRaisesRegex(OSError, "non-reparse"),
        ):
            log_module._AppendOnlyJsonlHandler._validate_directory(
                Path("windows-junction")
            )

    def test_directory_replacement_is_latched_before_writing(self) -> None:
        with TemporaryDirectory() as parent_name:
            parent = Path(parent_name)
            directory = parent / "log"
            directory.mkdir()
            moved = parent / "moved-log"
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": str(directory)}):
                logger = setup_logger(f"hbrowser.tests.dir_identity.{id(self)}")
                configure_logging(console_level=LogLevel.CRITICAL)
                directory.rename(moved)
                directory.mkdir()
                logger.error("must not enter replacement")
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                self.assertTrue(handler.wait_until_idle(timeout=3.0))
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

            self.assertEqual(tuple(directory.iterdir()), ())

        self.assertEqual(error_info.exception.operation, "segment-write")

    def test_segment_replacement_is_latched_before_writing(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            active = directory / "events-000001.jsonl"
            moved = directory / "moved-original.jsonl"
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(f"hbrowser.tests.segment_identity.{id(self)}")
                configure_logging(console_level=LogLevel.CRITICAL)
                active.rename(moved)
                active.write_text("replacement", encoding="utf-8")
                logger.error("must not enter replacement")
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                self.assertTrue(handler.wait_until_idle(timeout=3.0))
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

            self.assertEqual(active.read_text(encoding="utf-8"), "replacement")
            self.assertEqual(moved.read_bytes(), b"")

        self.assertEqual(error_info.exception.operation, "segment-write")

    def test_forked_process_cannot_use_inherited_handler(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(f"hbrowser.tests.fork_owner.{id(self)}")
                configure_logging(console_level=LogLevel.CRITICAL)
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(handler)
                assert handler is not None
                owner_pid = handler._writer_pid
                with patch("hbrowser.gallery.utils.log.os.getpid", return_value=999999):
                    logger.error("forked child")
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        self.assertEqual(error_info.exception.writer_pid, owner_pid)
        self.assertEqual(error_info.exception.operation, "segment-write")

    @unittest.skipUnless(os.name == "posix", "POSIX permissions only")
    def test_segments_are_owner_only(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                configure_logging()
            mode = stat.S_IMODE(
                (Path(directory_name) / "events-000001.jsonl").stat().st_mode
            )

        self.assertEqual(mode, 0o600)

    def test_concurrent_emit_and_reconfigure_produces_valid_json_once(self) -> None:
        logger_name = f"hbrowser.tests.concurrent.{id(self)}"
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(logger_name)
                configure_logging(
                    console_level=LogLevel.CRITICAL,
                    segment_bytes=4096,
                )

                def emit(worker: int) -> None:
                    for index in range(25):
                        logger.info("worker=%d index=%d", worker, index)

                def reconfigure() -> None:
                    for segment_bytes in (2048, 8192, 1024, 4096):
                        configure_logging(
                            console_level=LogLevel.CRITICAL,
                            segment_bytes=segment_bytes,
                        )

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(emit, worker) for worker in range(4)]
                    futures.append(executor.submit(reconfigure))
                    for future in futures:
                        future.result()

                events = _read_events(Path(directory_name))

        self.assertEqual(len(events), 100)
        self.assertEqual(len({event["message"] for event in events}), 100)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_child_resets_custom_locks_before_latching_owner_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = setup_logger(f"hbrowser.tests.real_fork.{id(self)}")
                configure_logging(console_level=LogLevel.CRITICAL)
                self.assertIsNotNone(start_log_forwarding_receiver())
                acquired = Event()
                release = Event()

                def hold_process_locks() -> None:
                    with log_module._LOGGING_CONFIGURATION_LOCK:
                        with log_module._LOG_PERSISTENCE_HEALTH_LOCK:
                            acquired.set()
                            release.wait()

                holder = Thread(target=hold_process_locks)
                holder.start()
                acquired.wait(timeout=5)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"This process .* is multi-threaded.*",
                        category=DeprecationWarning,
                    )
                    child_pid = os.fork()
                if child_pid == 0:
                    logger.error("child must fail closed without deadlock")
                    os._exit(0 if logging_health().trace_degraded else 4)
                try:
                    _, wait_status = os.waitpid(child_pid, 0)
                finally:
                    release.set()
                    holder.join(timeout=5)

        self.assertEqual(os.waitstatus_to_exitcode(wait_status), 0)
        self.assertFalse(logging_health().trace_degraded)


class PersistenceHealthTests(IsolatedLoggingTestCase):
    def _configure(self, directory_name: str) -> logging.Logger:
        logger = setup_logger(f"hbrowser.tests.health.{id(self)}")
        configure_logging(
            console_level=LogLevel.CRITICAL,
            segment_bytes=1024 * 1024,
        )
        return logger

    def _wait_for_sink(self, handler: log_module._AppendOnlyJsonlHandler) -> None:
        self.assertTrue(handler.wait_until_idle(timeout=3.0))

    def test_ordinary_log_call_does_not_wait_for_a_blocked_file_write(
        self,
    ) -> None:
        entered = Event()
        release = Event()
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                original_write = handler._write_bytes

                def blocked_write(data: bytes) -> None:
                    entered.set()
                    release.wait()
                    original_write(data)

                with patch.object(
                    handler,
                    "_write_bytes",
                    side_effect=blocked_write,
                ):
                    logger.error("writer blocks")
                    self.assertTrue(entered.wait(timeout=2))
                    started = time.monotonic()
                    logger.error("business continues")
                    elapsed = time.monotonic() - started
                    release.set()
                    self._wait_for_sink(handler)
                    events = _read_events(directory)

        self.assertLess(elapsed, 0.25)
        self.assertEqual(
            [event["message"] for event in events],
            ["writer blocks", "business continues"],
        )

    def test_idle_receipt_cannot_overtake_a_concurrent_enqueue(self) -> None:
        set_entered = Event()
        set_release = Event()
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                original_set = handler._idle.set  # noqa: SLF001
                set_calls = 0

                def blocked_first_set() -> None:
                    nonlocal set_calls
                    set_calls += 1
                    if set_calls == 1:
                        set_entered.set()
                        set_release.wait()
                    original_set()

                with patch.object(
                    handler._idle,  # noqa: SLF001
                    "set",
                    side_effect=blocked_first_set,
                ):
                    logger.error("first")
                    self.assertTrue(set_entered.wait(timeout=2))
                    second = Thread(target=logger.error, args=("second",))
                    second.start()
                    time.sleep(0.05)
                    self.assertTrue(second.is_alive())
                    set_release.set()
                    second.join(timeout=2)
                    self.assertFalse(second.is_alive())
                    self._wait_for_sink(handler)
                    events = _read_events(directory)

        self.assertEqual(
            [event["message"] for event in events],
            ["first", "second"],
        )

    def test_close_logging_abandons_a_blocked_writer_within_deadline(
        self,
    ) -> None:
        entered = Event()
        release = Event()
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None

                def blocked_write(_data: bytes) -> None:
                    entered.set()
                    release.wait()

                with patch.object(
                    handler,
                    "_write_bytes",
                    side_effect=blocked_write,
                ):
                    logger.error("stuck writer")
                    self.assertTrue(entered.wait(timeout=2))
                    started = time.monotonic()
                    with self.assertRaises(LogPersistenceError) as error_info:
                        close_logging()
                    elapsed = time.monotonic() - started
                    release.set()
                    self.assertTrue(handler._stopped.wait(timeout=2))  # noqa: SLF001

        self.assertLess(elapsed, 3.0)
        self.assertEqual(error_info.exception.operation, "segment-close")
        self.assertTrue(handler._abandoned_until_process_exit)  # noqa: SLF001

    def test_ordinary_write_failure_is_sticky_sanitized_and_non_raising(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(handler)
                assert handler is not None
                failure = PermissionError(errno.EACCES, "secret", "/private/path")
                setattr(failure, "winerror", 32)
                with patch.object(
                    handler,
                    "_write_bytes",
                    side_effect=failure,
                ) as write:
                    logger.error("first failure")
                    logger.error("disabled write")
                    self._wait_for_sink(handler)

                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        error = error_info.exception
        self.assertEqual(write.call_count, 1)
        self.assertEqual(error.operation, "segment-write")
        self.assertEqual(error.error_type, "PermissionError")
        self.assertEqual(error.errno, errno.EACCES)
        self.assertEqual(error.winerror, 32)
        self.assertEqual(error.writer_pid, os.getpid())
        self.assertEqual(error.segment_index, 1)
        self.assertEqual(error.segment_name, "events-000001.jsonl")
        self.assertIs(error.__cause__, failure)
        self.assertNotIn("secret", str(error))
        self.assertNotIn("private", str(error))

    def test_first_failure_writes_one_allowlisted_emergency_json_record(self) -> None:
        emergency_write = Mock(return_value=0)
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                cause = PermissionError(
                    errno.EACCES,
                    "sensitive operating-system message",
                    "/private/secret/path",
                )
                with (
                    patch(
                        "hbrowser.gallery.utils.log.os.write",
                        emergency_write,
                    ),
                    patch.object(handler, "_write_bytes", side_effect=cause),
                ):
                    logger.error("first")
                    logger.error("second")
                    self._wait_for_sink(handler)
                    log_module._record_persistence_failure(
                        LogPersistenceError(
                            "segment-flush",
                            OSError(errno.EIO, "later failure"),
                            segment_index=1,
                        )
                    )
                    self.assertTrue(
                        _wait_until(lambda: emergency_write.call_count == 1)
                    )

        self.assertEqual(emergency_write.call_count, 1)
        descriptor, encoded = emergency_write.call_args.args
        self.assertEqual(descriptor, 2)
        line = encoded.decode("ascii")
        self.assertTrue(line.endswith("\n"))
        payload = json.loads(line)
        self.assertEqual(
            set(payload),
            {
                "event",
                "stage",
                "error_type",
                "errno",
                "winerror",
                "writer_pid",
                "segment_index",
            },
        )
        self.assertEqual(payload["event"], "hbrowser.trace_degraded")
        self.assertEqual(payload["stage"], "segment-write")
        self.assertEqual(payload["error_type"], "PermissionError")
        self.assertEqual(payload["errno"], errno.EACCES)
        self.assertNotIn("sensitive", line)
        self.assertNotIn("private", line)

    def test_emergency_stderr_failure_is_swallowed_without_retry_or_recursion(
        self,
    ) -> None:
        emergency_write = Mock(side_effect=OSError(errno.EIO, "stderr unavailable"))
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                with (
                    patch(
                        "hbrowser.gallery.utils.log.os.write",
                        emergency_write,
                    ),
                    patch.object(
                        handler,
                        "_write_bytes",
                        side_effect=OSError(errno.ENOSPC, "trace full"),
                    ),
                ):
                    logger.error("ordinary call must return")
                    logger.error("disabled call must return")
                    self._wait_for_sink(handler)
                    self.assertTrue(
                        _wait_until(lambda: emergency_write.call_count == 1)
                    )

        self.assertEqual(emergency_write.call_count, 1)
        self.assertTrue(logging_health().trace_degraded)

    def test_first_failure_remains_latched_across_reconfigure(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                first = OSError(errno.ENOSPC, "disk detail")
                with patch.object(handler, "_write_bytes", side_effect=first):
                    logger.error("failure")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError) as configure_info:
                    configure_logging(file_level=LogLevel.INFO)
                with self.assertRaises(LogPersistenceError) as first_info:
                    raise_for_log_persistence_failure()
                with self.assertRaises(LogPersistenceError) as second_info:
                    raise_for_log_persistence_failure()

        self.assertIs(configure_info.exception, first_info.exception)
        self.assertIs(first_info.exception, second_info.exception)
        self.assertIs(first_info.exception.__cause__, first)

    def test_partial_write_failure_is_latched(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(handler)
                assert handler is not None
                stream = handler._stream
                self.assertIsNotNone(stream)
                assert stream is not None

                def partial(data: bytes) -> None:
                    stream.write(data[:8])
                    raise OSError(errno.ENOSPC, "after partial write")

                with patch.object(handler, "_write_bytes", side_effect=partial):
                    logger.error("partially persisted")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        self.assertEqual(error_info.exception.operation, "segment-write")
        self.assertEqual(error_info.exception.errno, errno.ENOSPC)

    def test_flush_failure_is_latched_without_escaping_logger_call(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                with patch.object(
                    handler,
                    "_flush_stream",
                    side_effect=OSError(errno.EIO, "flush"),
                ):
                    logger.error("flush failure")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        self.assertEqual(error_info.exception.operation, "segment-flush")

    def test_rollover_open_failure_is_latched(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                logger.info("first")
                self._wait_for_sink(handler)
                handler.segment_bytes = 1
                cause = OSError(errno.EACCES, "open next")
                failure = LogPersistenceError(
                    "segment-open",
                    cause,
                    segment_index=2,
                )
                with patch.object(
                    handler,
                    "_open_next_segment",
                    side_effect=failure,
                ):
                    logger.info("roll")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        self.assertIs(error_info.exception, failure)

    def test_rollover_close_failure_is_latched(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                logger.info("first")
                self._wait_for_sink(handler)
                handler.segment_bytes = 1
                cause = OSError(errno.EIO, "close")
                failure = LogPersistenceError(
                    "segment-close",
                    cause,
                    segment_index=1,
                )
                with patch.object(
                    handler,
                    "_close_stream",
                    side_effect=failure,
                ):
                    logger.info("roll")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError) as error_info:
                    raise_for_log_persistence_failure()

        self.assertIs(error_info.exception, failure)

    def test_log_to_process_file_surfaces_failure_from_its_write(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                with (
                    patch.object(
                        handler,
                        "_write_bytes",
                        side_effect=OSError(errno.ENOSPC, "full"),
                    ),
                    self.assertRaises(LogPersistenceError) as error_info,
                ):
                    log_to_process_file(logger, LogLevel.ERROR, "terminal")

        self.assertEqual(error_info.exception.operation, "segment-write")

    def test_log_to_process_file_surfaces_preexisting_failure(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                assert handler is not None
                with patch.object(
                    handler,
                    "_write_bytes",
                    side_effect=OSError(errno.ENOSPC, "full"),
                ):
                    logger.error("ordinary")
                    self._wait_for_sink(handler)
                with self.assertRaises(LogPersistenceError):
                    log_to_process_file(logger, LogLevel.ERROR, "terminal")

    def test_close_logging_is_idempotent_and_permanently_ends_lifecycle(self) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                logger = self._configure(directory_name)
                logger.info("before close")
                self.assertEqual(logging_health(), LoggingHealth(trace_degraded=False))
                close_logging()
                close_logging()
                with self.assertRaisesRegex(RuntimeError, "already closed"):
                    setup_logger(f"hbrowser.tests.after_close.{id(self)}")
                with self.assertRaisesRegex(RuntimeError, "already closed"):
                    configure_logging()
                with self.assertRaisesRegex(RuntimeError, "already closed"):
                    log_to_process_file(logger, LogLevel.ERROR, "after close")

        self.assertIsNone(log_module._PROCESS_LOG_HANDLER)
        raise_for_log_persistence_failure()

    def test_close_failure_is_latched_and_repeated_close_raises_same_error(
        self,
    ) -> None:
        with TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}):
                self._configure(directory_name)
                handler = log_module._PROCESS_LOG_HANDLER
                self.assertIsNotNone(handler)
                assert handler is not None
                original_close = handler._close_stream
                cause = OSError(errno.EIO, "close detail")
                failure = LogPersistenceError(
                    "segment-close",
                    cause,
                    segment_index=1,
                )

                def close_then_fail() -> None:
                    original_close()
                    raise failure

                with (
                    patch.object(
                        handler,
                        "_close_stream",
                        side_effect=close_then_fail,
                    ),
                    self.assertRaises(LogPersistenceError) as first_info,
                ):
                    close_logging()
                with self.assertRaises(LogPersistenceError) as second_info:
                    close_logging()

        self.assertIs(first_info.exception, failure)
        self.assertIs(second_info.exception, failure)

    def test_initial_directory_failure_raises_immediately(self) -> None:
        cause = OSError(errno.EACCES, "secret directory", "/private/log")
        with (
            patch("hbrowser.gallery.utils.log.get_log_dir", side_effect=cause),
            self.assertRaises(LogPersistenceError) as error_info,
        ):
            configure_logging()

        error = error_info.exception
        self.assertEqual(error.operation, "configure")
        self.assertEqual(error.errno, errno.EACCES)
        self.assertIs(error.__cause__, cause)
        self.assertNotIn("secret", str(error))
        self.assertIsNone(log_module._PROCESS_LOG_HANDLER)

    def test_open_failure_raises_immediately(self) -> None:
        with TemporaryDirectory() as directory_name:
            with (
                patch.dict(os.environ, {"HBROWSER_LOG_DIR": directory_name}),
                patch(
                    "hbrowser.gallery.utils.log.os.open",
                    side_effect=OSError(errno.EACCES, "secret open"),
                ),
                self.assertRaises(LogPersistenceError) as error_info,
            ):
                configure_logging()

        self.assertEqual(error_info.exception.operation, "segment-open")
        self.assertEqual(error_info.exception.errno, errno.EACCES)

    def test_operation_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            LogPersistenceError("arbitrary", OSError())  # type: ignore[arg-type]

    def test_public_and_emergency_health_fields_are_strictly_bounded(self) -> None:
        adversarial_type = type("Bad/型" + ("X" * 256), (OSError,), {})
        cause = adversarial_type()
        setattr(cause, "errno", 1 << 200)
        setattr(cause, "winerror", -1)
        failure = LogPersistenceError(
            "segment-write",
            cause,
            segment_index=1,
        )
        emergency_write = Mock(return_value=0)
        with patch(
            "hbrowser.gallery.utils.log.os.write",
            emergency_write,
        ):
            log_module._record_persistence_failure(failure)
            self.assertTrue(_wait_until(lambda: emergency_write.call_count == 1))

        health = logging_health()
        _, encoded = emergency_write.call_args.args
        payload = json.loads(encoded.decode("ascii"))
        self.assertLess(len(encoded), 512)
        self.assertEqual(failure.error_type, "UnknownError")
        self.assertIsNone(failure.errno)
        self.assertIsNone(failure.winerror)
        self.assertEqual(health.error_type, "UnknownError")
        self.assertEqual(payload["error_type"], "UnknownError")
        self.assertIsNone(payload["errno"])
        self.assertIsNone(payload["winerror"])

        for keyword in ("segment_index", "writer_pid"):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                LogPersistenceError(
                    "segment-write",
                    OSError(),
                    **{keyword: 1 << 200},
                )


class LogDirectoryTests(unittest.TestCase):
    def test_environment_override_creates_nested_absolute_directory(self) -> None:
        with TemporaryDirectory() as directory_name:
            expected = Path(directory_name) / "nested" / "diagnostics"
            with patch.dict(os.environ, {"HBROWSER_LOG_DIR": str(expected)}):
                actual = get_log_dir()

            self.assertEqual(actual, expected)
            self.assertTrue(actual.is_dir())

    def test_default_directory_stays_next_to_main_script(self) -> None:
        with TemporaryDirectory() as directory_name:
            script = Path(directory_name) / "bin" / "application.py"
            expected = script.parent.resolve() / "log"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("hbrowser.gallery.utils.log.sys.argv", [str(script)]),
            ):
                actual = get_log_dir()

            self.assertEqual(actual, expected)
            self.assertTrue(actual.is_dir())


if __name__ == "__main__":
    unittest.main()
