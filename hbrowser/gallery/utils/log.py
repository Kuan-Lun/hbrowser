"""Logging configuration and semantic context helpers."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import BinaryIO, Literal, cast

from ._log_forwarding import (
    FORWARD_ENDPOINT_ENVIRONMENT_VARIABLE,
    FORWARD_TOKEN_ENVIRONMENT_VARIABLE,
    ForwardedRecord,
    ForwardingConfigurationError,
    ForwardingHandler,
)
from ._log_forwarding import (
    LogForwardingReceiver as LogForwardingReceiver,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_LOG_DIR_ENVIRONMENT_VARIABLE = "HBROWSER_LOG_DIR"
_DEFAULT_SEGMENT_BYTES = 10 * 1024 * 1024
_MAX_QUEUED_PROCESS_RECORDS = 512
_PROCESS_SINK_CLOSE_TIMEOUT_SECONDS = 2.0
_PROCESS_SINK_POLL_SECONDS = 0.05
_LOGGING_CONFIGURATION_LOCK = RLock()
_LOG_PERSISTENCE_HEALTH_LOCK = Lock()
_MANAGED_LOGGER_NAMES: set[str] = set()
_NAMESPACE_LOGGER_NAMES = ("battle", "hbrowser", "hvbrowser", "hvbattle")
_MANAGED_STDOUT_HANDLER_ATTRIBUTE = "_hbrowser_managed_stdout_handler"
_MANAGED_PROCESS_HANDLER_ATTRIBUTE = "_hbrowser_managed_process_handler"
_MANAGED_FORWARDING_HANDLER_ATTRIBUTE = "_hbrowser_managed_forwarding_handler"
_LOG_CONTEXT_FIELDS = ("account", "realm", "tab_role", "activity", "scope")
_SEGMENT_NAME_PATTERN = re.compile(r"events-(\d{6,})\.jsonl\Z")
_OWNER_LOCK_NAME = ".hbrowser-log-writer.lock"
_MAX_ERROR_TYPE_LENGTH = 128
_MAX_SAFE_INTEGER = 0xFFFFFFFF
_SAFE_ERROR_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")

_PersistenceOperation = Literal[
    "configure",
    "segment-open",
    "segment-write",
    "segment-flush",
    "segment-close",
]
_PERSISTENCE_OPERATIONS = frozenset(
    {
        "configure",
        "segment-open",
        "segment-write",
        "segment-flush",
        "segment-close",
    }
)
_ForwardingOperation = Literal[
    "forward-configure",
    "forward-accept",
    "forward-protocol",
    "forward-receive",
    "forward-drain",
    "forward-sink",
    "forward-send",
    "forward-close",
]
_FORWARDING_OPERATIONS = frozenset(
    {
        "forward-configure",
        "forward-accept",
        "forward-protocol",
        "forward-receive",
        "forward-drain",
        "forward-sink",
        "forward-send",
        "forward-close",
    }
)


class LogLevel(StrEnum):
    """Supported process logging thresholds."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def number(self) -> int:
        """Return the corresponding :mod:`logging` numeric level."""
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


def _numeric_error_code(error: BaseException, attribute: str) -> int | None:
    try:
        value = getattr(error, attribute, None)
    except BaseException:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_SAFE_INTEGER
    ):
        return None
    return value


def _sanitized_error_type(error: BaseException) -> str:
    try:
        raw_name = type(error).__name__
    except BaseException:
        return "UnknownError"
    if (
        len(raw_name) > _MAX_ERROR_TYPE_LENGTH
        or _SAFE_ERROR_TYPE_PATTERN.fullmatch(raw_name) is None
    ):
        return "UnknownError"
    return raw_name


def _safe_process_id(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_SAFE_INTEGER
    ):
        return None
    return value


def _current_process_id() -> int | None:
    try:
        return _safe_process_id(os.getpid())
    except BaseException:
        return None


class LogPersistenceError(RuntimeError):
    """A configured application-log sink could not persist a record safely.

    The rendered exception deliberately excludes filesystem paths and operating
    system messages. The original exception remains available through
    ``__cause__`` for trusted in-process diagnostics.
    """

    def __init__(
        self,
        operation: _PersistenceOperation,
        error: BaseException,
        *,
        segment_index: int | None = None,
        writer_pid: int | None = None,
    ) -> None:
        if operation not in _PERSISTENCE_OPERATIONS:
            raise ValueError(f"Unsupported log persistence operation: {operation}")
        if segment_index is not None and (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or not 1 <= segment_index <= _MAX_SAFE_INTEGER
        ):
            raise ValueError("segment_index must be a positive 32-bit integer or None")
        if writer_pid is not None and (
            isinstance(writer_pid, bool)
            or not isinstance(writer_pid, int)
            or not 1 <= writer_pid <= _MAX_SAFE_INTEGER
        ):
            raise ValueError("writer_pid must be a positive 32-bit integer or None")
        self.operation = operation
        self.error_type = _sanitized_error_type(error)
        self.errno = _numeric_error_code(error, "errno")
        self.winerror = _numeric_error_code(error, "winerror")
        self.writer_pid = _current_process_id() if writer_pid is None else writer_pid
        self.segment_index = segment_index
        self.segment_name = (
            None if segment_index is None else f"events-{segment_index:06d}.jsonl"
        )
        self.__cause__ = error
        details = [
            f"operation={operation}",
            f"error_type={self.error_type}",
            f"writer_pid={self.writer_pid}",
        ]
        if self.segment_index is not None:
            details.append(f"segment_index={self.segment_index}")
            details.append(f"segment_name={self.segment_name}")
        if self.errno is not None:
            details.append(f"errno={self.errno}")
        if self.winerror is not None:
            details.append(f"winerror={self.winerror}")
        super().__init__("Process log persistence failed: " + " ".join(details))


@dataclass(frozen=True, slots=True)
class LoggingHealth:
    """Path-free snapshot of trace persistence and forwarding health."""

    trace_degraded: bool
    operation: str | None = None
    error_type: str | None = None
    errno: int | None = None
    winerror: int | None = None
    writer_pid: int | None = None
    segment_index: int | None = None
    segment_name: str | None = None
    forwarding_degraded: bool = False
    forwarding_operation: str | None = None
    forwarding_error_type: str | None = None
    forwarding_errno: int | None = None
    forwarding_winerror: int | None = None
    forwarding_writer_pid: int | None = None


class _LogForwardingFailure(RuntimeError):
    """First path-free transport failure retained for health diagnostics."""

    def __init__(
        self,
        operation: _ForwardingOperation,
        error: BaseException,
    ) -> None:
        if operation not in _FORWARDING_OPERATIONS:
            raise ValueError(f"Unsupported log forwarding operation: {operation}")
        self.operation = operation
        self.error_type = _sanitized_error_type(error)
        self.errno = _numeric_error_code(error, "errno")
        self.winerror = _numeric_error_code(error, "winerror")
        self.writer_pid = _current_process_id()
        self.__cause__ = error
        super().__init__(
            "Log forwarding degraded: "
            f"operation={operation} error_type={self.error_type}"
        )


@dataclass(frozen=True, slots=True)
class _LoggingConfiguration:
    console_level: LogLevel = LogLevel.INFO
    file_level: LogLevel = LogLevel.DEBUG
    segment_bytes: int = _DEFAULT_SEGMENT_BYTES
    require_file_sink: bool = True


_LOGGING_CONFIGURATION = _LoggingConfiguration()
_LOGGING_CONFIGURED = False
_LOGGING_CLOSED = False
_FILE_SINK_ATTEMPTED = False
_PROCESS_LOG_HANDLER: _AppendOnlyJsonlHandler | None = None
_LOG_PERSISTENCE_FAILURE: LogPersistenceError | None = None
_LOG_FORWARDING_FAILURE: _LogForwardingFailure | None = None
_LOG_FORWARDING_RECEIVER: LogForwardingReceiver | None = None
_LOG_FORWARDING_RECEIVER_ATTEMPTED = False
_FORWARDING_HANDLER: ForwardingHandler | None = None
_FORWARDED_LOGGING_CONFIGURED = False
_FORWARDED_LOGGING_CLOSED = False


@dataclass(frozen=True, slots=True)
class _LogContext:
    account: str | None = None
    realm: str | None = None
    tab_role: str | None = None
    activity: str | None = None
    scope: str | None = None


_EMPTY_LOG_CONTEXT = _LogContext()
_CURRENT_LOG_CONTEXT: ContextVar[_LogContext] = ContextVar(
    "hbrowser_log_context",
    default=_EMPTY_LOG_CONTEXT,
)


def _normalize_context_value(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


@contextmanager
def log_context(
    *,
    account: str | None = None,
    realm: str | None = None,
    tab_role: str | None = None,
    activity: str | None = None,
    scope: str | None = None,
) -> Iterator[None]:
    """Add semantic fields to log records within one synchronous scope."""
    current = _CURRENT_LOG_CONTEXT.get()
    supplied = {
        "account": _normalize_context_value("account", account),
        "realm": _normalize_context_value("realm", realm),
        "tab_role": _normalize_context_value("tab_role", tab_role),
        "activity": _normalize_context_value("activity", activity),
        "scope": _normalize_context_value("scope", scope),
    }
    merged = _LogContext(
        **{
            field: (
                supplied[field]
                if supplied[field] is not None
                else getattr(current, field)
            )
            for field in _LOG_CONTEXT_FIELDS
        }
    )
    token = _CURRENT_LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _CURRENT_LOG_CONTEXT.reset(token)


def _record_context_value(
    record: logging.LogRecord,
    field: str,
    inherited: str | None,
) -> str | None:
    value = record.__dict__.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return inherited


def _display_context_value(value: str) -> str:
    aliases = {
        "browser": "Browser",
        "isekai": "Isekai",
        "persistent": "Persistent",
        "system": "System",
    }
    normalized = value.strip()
    alias = aliases.get(normalized.casefold())
    if alias is not None:
        return alias
    if normalized.islower():
        return normalized.replace("_", " ").title()
    return normalized


def _semantic_label(context: _LogContext) -> str:
    if context.scope is not None:
        return _display_context_value(context.scope)

    target = context.realm or context.tab_role
    components = []
    if target is not None:
        components.append(_display_context_value(target))
    if context.activity is not None:
        activity = _display_context_value(context.activity)
        if not components or activity.casefold() != components[-1].casefold():
            components.append(activity)
    return " · ".join(components) if components else "System"


def _inject_log_context(record: logging.LogRecord) -> None:
    inherited = _CURRENT_LOG_CONTEXT.get()
    values = {
        field: _record_context_value(record, field, getattr(inherited, field))
        for field in _LOG_CONTEXT_FIELDS
    }
    for field, value in values.items():
        record.__dict__[field] = value
    record.__dict__["semantic_label"] = _semantic_label(_LogContext(**values))


class _LogContextFilter(logging.Filter):
    """Attach semantic context before a managed handler consumes a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        _inject_log_context(record)
        return True


_LOG_CONTEXT_FILTER = _LogContextFilter()


class _SemanticFormatter(logging.Formatter):
    """Render a concise user label while preserving logger diagnostics."""

    def format(self, record: logging.LogRecord) -> str:
        _inject_log_context(record)
        return super().format(record)


def _configure_handler_context(handler: logging.Handler) -> None:
    if _LOG_CONTEXT_FILTER not in handler.filters:
        handler.addFilter(_LOG_CONTEXT_FILTER)


def _formatter() -> logging.Formatter:
    return _SemanticFormatter(
        "%(asctime)s - %(levelname)s - [%(semantic_label)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _record_persistence_failure(failure: LogPersistenceError) -> None:
    global _LOG_PERSISTENCE_FAILURE

    first_failure = False
    with _LOG_PERSISTENCE_HEALTH_LOCK:
        if _LOG_PERSISTENCE_FAILURE is None:
            _LOG_PERSISTENCE_FAILURE = failure
            first_failure = True
    if first_failure:
        _write_health_record_in_background(
            lambda: _write_emergency_health_record(failure),
        )


def _record_forwarding_failure(stage: str, error: BaseException) -> None:
    """Latch one transport failure without disrupting an ordinary log call."""
    global _LOG_FORWARDING_FAILURE

    safe_stage = stage if stage in _FORWARDING_OPERATIONS else "forward-configure"
    try:
        underlying_error = error.__cause__
    except BaseException:
        underlying_error = None
    diagnostic_error = (
        underlying_error if isinstance(underlying_error, BaseException) else error
    )
    failure = _LogForwardingFailure(
        cast(_ForwardingOperation, safe_stage),
        diagnostic_error,
    )
    first_failure = False
    with _LOG_PERSISTENCE_HEALTH_LOCK:
        if _LOG_FORWARDING_FAILURE is None:
            _LOG_FORWARDING_FAILURE = failure
            first_failure = True
    if first_failure:
        _write_health_record_in_background(
            lambda: _write_forwarding_emergency_health_record(failure),
        )


def _write_health_record_in_background(
    writer: Callable[[], None],
) -> None:
    """Keep diagnostic stderr I/O outside business and close call stacks."""
    try:
        Thread(
            target=writer,
            name="hbrowser-log-health-diagnostic",
            daemon=True,
        ).start()
    except Exception:
        return


def _write_emergency_health_record(failure: LogPersistenceError) -> None:
    """Best-effort path-free stderr record that never uses :mod:`logging`."""
    payload = {
        "event": "hbrowser.trace_degraded",
        "stage": failure.operation,
        "error_type": failure.error_type,
        "errno": failure.errno,
        "winerror": failure.winerror,
        "writer_pid": failure.writer_pid,
        "segment_index": failure.segment_index,
    }
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        os.write(2, encoded)
    except Exception:
        return


def _write_forwarding_emergency_health_record(
    failure: _LogForwardingFailure,
) -> None:
    """Best-effort bounded forwarding health record written directly to fd 2."""
    payload = {
        "event": "hbrowser.forwarding_degraded",
        "stage": failure.operation,
        "error_type": failure.error_type,
        "errno": failure.errno,
        "winerror": failure.winerror,
        "writer_pid": failure.writer_pid,
    }
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        os.write(2, encoded)
    except Exception:
        return


def logging_health() -> LoggingHealth:
    """Return a non-raising, path-free snapshot of trace/forwarding health."""
    with _LOG_PERSISTENCE_HEALTH_LOCK:
        failure = _LOG_PERSISTENCE_FAILURE
        forwarding_failure = _LOG_FORWARDING_FAILURE
    return LoggingHealth(
        trace_degraded=failure is not None,
        operation=None if failure is None else failure.operation,
        error_type=None if failure is None else failure.error_type,
        errno=None if failure is None else failure.errno,
        winerror=None if failure is None else failure.winerror,
        writer_pid=None if failure is None else failure.writer_pid,
        segment_index=None if failure is None else failure.segment_index,
        segment_name=None if failure is None else failure.segment_name,
        forwarding_degraded=forwarding_failure is not None,
        forwarding_operation=(
            None if forwarding_failure is None else forwarding_failure.operation
        ),
        forwarding_error_type=(
            None if forwarding_failure is None else forwarding_failure.error_type
        ),
        forwarding_errno=(
            None if forwarding_failure is None else forwarding_failure.errno
        ),
        forwarding_winerror=(
            None if forwarding_failure is None else forwarding_failure.winerror
        ),
        forwarding_writer_pid=(
            None if forwarding_failure is None else forwarding_failure.writer_pid
        ),
    )


def raise_for_log_persistence_failure() -> None:
    """Raise the first latched file-sink failure at an application safe boundary."""
    with _LOG_PERSISTENCE_HEALTH_LOCK:
        failure = _LOG_PERSISTENCE_FAILURE
    if failure is not None:
        raise failure


def _reset_logging_locks_after_fork() -> None:
    global _LOGGING_CONFIGURATION_LOCK, _LOG_PERSISTENCE_HEALTH_LOCK
    global _FORWARDING_HANDLER, _LOG_FORWARDING_RECEIVER
    global _PROCESS_LOG_HANDLER

    _LOGGING_CONFIGURATION_LOCK = RLock()
    _LOG_PERSISTENCE_HEALTH_LOCK = Lock()
    receiver = _LOG_FORWARDING_RECEIVER
    _LOG_FORWARDING_RECEIVER = None
    if receiver is not None:
        receiver._discard_after_fork()
    forwarding_handler = _FORWARDING_HANDLER
    _FORWARDING_HANDLER = None
    if forwarding_handler is not None:
        forwarding_handler._discard_after_fork()
        _record_forwarding_failure(
            "forward-send",
            RuntimeError("Log forwarding capability does not survive fork"),
        )
    for logger_name in _handler_owner_logger_names():
        logger = logging.getLogger(logger_name)
        if forwarding_handler is not None:
            logger.removeHandler(forwarding_handler)
    process_handler = _PROCESS_LOG_HANDLER
    if process_handler is None:
        return
    failure = LogPersistenceError(
        "segment-write",
        RuntimeError("Process log ownership does not survive fork"),
        segment_index=process_handler._next_segment_index - 1,
        writer_pid=process_handler._writer_pid,
    )
    process_handler._disable_after_failure(failure)
    for logger_name in _handler_owner_logger_names():
        logging.getLogger(logger_name).removeHandler(process_handler)
    _PROCESS_LOG_HANDLER = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_logging_locks_after_fork)


class _AppendOnlyJsonlHandler(logging.Handler):
    """Write private JSONL segments without renaming or reopening old files."""

    def __init__(
        self,
        directory: Path,
        *,
        segment_bytes: int,
        level: int,
    ) -> None:
        super().__init__(level)
        self.directory = directory
        self.segment_bytes = segment_bytes
        self._stream: BinaryIO | None = None
        self._segment_path: Path | None = None
        self._written_bytes = 0
        self._writer_pid = os.getpid()
        self._directory_descriptor: int | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._owner_descriptor: int | None = None
        self._enabled = True
        self._abandoned_until_process_exit = False
        self._write_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=_MAX_QUEUED_PROCESS_RECORDS
        )
        self._stop_requested = Event()
        self._stopped = Event()
        self._idle = Event()
        self._idle.set()
        self._close_lock = Lock()
        self._enqueue_lock = Lock()
        self._failure_lock = Lock()
        self._close_failure: LogPersistenceError | None = None
        self._worker = Thread(
            target=self._run_writer,
            name="hbrowser-process-log-writer",
            daemon=True,
        )
        self._worker_started = False
        _configure_handler_context(self)
        try:
            self._pin_directory()
            self._acquire_owner_lock()
            self._next_segment_index = self._discover_next_segment_index()
            self._open_next_segment()
            try:
                self._worker.start()
                self._worker_started = True
            except Exception as error:
                raise LogPersistenceError(
                    "configure",
                    error,
                    writer_pid=self._writer_pid,
                ) from error
        except LogPersistenceError:
            self._stopped.set()
            self._close_stream_best_effort()
            self._close_owner_lock_best_effort()
            self._close_directory_best_effort()
            logging.Handler.close(self)
            raise
        except OSError as error:
            self._stopped.set()
            self._close_stream_best_effort()
            self._close_owner_lock_best_effort()
            self._close_directory_best_effort()
            logging.Handler.close(self)
            raise LogPersistenceError(
                "segment-open",
                error,
                writer_pid=self._writer_pid,
            ) from error
        except Exception:
            self._stopped.set()
            self._close_stream_best_effort()
            self._close_owner_lock_best_effort()
            self._close_directory_best_effort()
            logging.Handler.close(self)
            raise

    @property
    def healthy(self) -> bool:
        """Whether this handler still accepts file records."""
        return self._enabled

    def reconfigure(self, *, level: int, segment_bytes: int) -> None:
        """Update thresholds under the short record-enqueue lock."""
        self.acquire()
        try:
            self.setLevel(level)
            self.segment_bytes = segment_bytes
        finally:
            self.release()

    @staticmethod
    def _is_reparse_point(path_stat: os.stat_result) -> bool:
        attributes = getattr(path_stat, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return bool(attributes & reparse_flag)

    @classmethod
    def _validate_directory(cls, directory: Path) -> os.stat_result:
        directory_stat = directory.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or cls._is_reparse_point(directory_stat)
        ):
            raise OSError(
                f"Process log directory must be a non-reparse directory: {directory}"
            )
        return directory_stat

    def _pin_directory(self) -> None:
        path_stat = self._validate_directory(self.directory)
        identity = (path_stat.st_dev, path_stat.st_ino)
        if os.name == "posix":
            flags = os.O_RDONLY
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.directory, flags)
            try:
                descriptor_stat = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(descriptor_stat.st_mode)
                    or (descriptor_stat.st_dev, descriptor_stat.st_ino) != identity
                ):
                    raise OSError(
                        "Process log directory changed while ownership was acquired"
                    )
            except Exception:
                os.close(descriptor)
                raise
            self._directory_descriptor = descriptor
        self._directory_identity = identity

    def _validate_directory_identity(self) -> None:
        path_stat = self._validate_directory(self.directory)
        identity = (path_stat.st_dev, path_stat.st_ino)
        if identity != self._directory_identity:
            raise OSError("Process log directory identity changed")
        descriptor = self._directory_descriptor
        if descriptor is not None:
            descriptor_stat = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(descriptor_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino) != identity
            ):
                raise OSError("Process log directory descriptor identity changed")

    def _close_directory_best_effort(self) -> None:
        descriptor = self._directory_descriptor
        self._directory_descriptor = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _open_relative_to_owned_directory(
        self,
        path: Path,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        descriptor = self._directory_descriptor
        if descriptor is None:
            return os.open(path, flags, mode)
        return os.open(path.name, flags, mode, dir_fd=descriptor)

    def _acquire_owner_lock(self) -> None:
        path = self.directory / _OWNER_LOCK_NAME
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        try:
            self._validate_directory_identity()
            descriptor = self._open_relative_to_owned_directory(path, flags, 0o600)
            descriptor_stat = self._validate_open_descriptor(descriptor, path)
            path_stat = path.lstat()
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or stat.S_ISLNK(path_stat.st_mode)
                or self._is_reparse_point(path_stat)
                or path_stat.st_nlink != 1
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise OSError("Process log owner lock path is unsafe")
            self._validate_directory_identity()
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            if sys.platform == "win32":
                if descriptor_stat.st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._owner_descriptor = descriptor
            descriptor = None
        except OSError as error:
            raise LogPersistenceError(
                "configure",
                error,
                writer_pid=self._writer_pid,
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _close_owner_lock_best_effort(self) -> None:
        descriptor = self._owner_descriptor
        self._owner_descriptor = None
        if descriptor is None:
            return
        if os.getpid() != self._writer_pid:
            try:
                os.close(descriptor)
            except OSError:
                pass
            return
        try:
            if sys.platform == "win32":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _close_owner_lock(self) -> None:
        descriptor = self._owner_descriptor
        self._owner_descriptor = None
        if descriptor is None:
            return
        if os.getpid() != self._writer_pid:
            try:
                os.close(descriptor)
            except OSError as error:
                raise LogPersistenceError(
                    "segment-close",
                    error,
                    segment_index=self._next_segment_index - 1,
                    writer_pid=self._writer_pid,
                ) from error
            return
        failure: OSError | None = None
        try:
            if sys.platform == "win32":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as error:
            failure = error
        try:
            os.close(descriptor)
        except OSError as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise LogPersistenceError(
                "segment-close",
                failure,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from failure

    def _close_directory(self) -> None:
        descriptor = self._directory_descriptor
        self._directory_descriptor = None
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except OSError as error:
            raise LogPersistenceError(
                "segment-close",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error

    @staticmethod
    def _validate_open_descriptor(descriptor: int, path: Path) -> os.stat_result:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(f"Process log segment must be a regular file: {path}")
        if descriptor_stat.st_nlink != 1:
            raise OSError(f"Process log segment must have exactly one link: {path}")
        return descriptor_stat

    @classmethod
    def _validate_existing_segment(cls, path: Path) -> None:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        try:
            path_stat = path.lstat()
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or stat.S_ISLNK(path_stat.st_mode)
                or cls._is_reparse_point(path_stat)
                or path_stat.st_nlink != 1
            ):
                raise OSError(
                    "Process log segment must be a regular, single-link, "
                    f"non-reparse file: {path}"
                )
            descriptor = os.open(path, flags)
            descriptor_stat = cls._validate_open_descriptor(descriptor, path)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                path_stat.st_dev,
                path_stat.st_ino,
            ):
                raise OSError(f"Process log segment changed during validation: {path}")
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _discover_next_segment_index(self) -> int:
        highest = 0
        try:
            self._validate_directory_identity()
            entries = tuple(self.directory.iterdir())
            for entry in entries:
                match = _SEGMENT_NAME_PATTERN.fullmatch(entry.name)
                if match is None:
                    continue
                segment_index = int(match.group(1))
                if not 1 <= segment_index < _MAX_SAFE_INTEGER:
                    raise OSError("Process log segment sequence is outside its bound")
                if entry.name != f"events-{segment_index:06d}.jsonl":
                    raise OSError("Process log segment name is not canonical")
                self._validate_existing_segment(entry)
                highest = max(highest, segment_index)
            self._validate_directory_identity()
        except OSError as error:
            raise LogPersistenceError(
                "segment-open",
                error,
                writer_pid=self._writer_pid,
            ) from error
        return highest + 1

    def _open_next_segment(self) -> None:
        segment_index = self._next_segment_index
        if not 1 <= segment_index <= _MAX_SAFE_INTEGER:
            raise LogPersistenceError(
                "segment-open",
                OSError("Process log segment sequence is exhausted"),
                writer_pid=self._writer_pid,
            )
        path = self.directory / f"events-{segment_index:06d}.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        try:
            self._validate_directory_identity()
            descriptor = self._open_relative_to_owned_directory(path, flags, 0o600)
            descriptor_stat = self._validate_open_descriptor(descriptor, path)
            path_stat = path.lstat()
            if self._is_reparse_point(path_stat) or (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ) != (path_stat.st_dev, path_stat.st_ino):
                raise OSError(f"Process log segment changed during creation: {path}")
            self._validate_directory_identity()
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            self._stream = os.fdopen(descriptor, "wb", buffering=0)
            descriptor = None
            self._segment_path = path
            self._written_bytes = 0
            self._next_segment_index += 1
        except OSError as error:
            raise LogPersistenceError(
                "segment-open",
                error,
                segment_index=segment_index,
                writer_pid=self._writer_pid,
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        self._segment_path = None
        if stream is None:
            return
        try:
            stream.flush()
        except OSError as error:
            try:
                stream.close()
            except OSError:
                pass
            raise LogPersistenceError(
                "segment-flush",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error
        try:
            stream.close()
        except OSError as error:
            raise LogPersistenceError(
                "segment-close",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error

    def _close_stream_best_effort(self) -> None:
        stream = self._stream
        self._stream = None
        self._segment_path = None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _format_exception(record: logging.LogRecord) -> str | None:
        if record.exc_info is None:
            return None
        return logging.Formatter().formatException(record.exc_info)

    @classmethod
    def _encode_record(cls, record: logging.LogRecord) -> bytes:
        _inject_log_context(record)
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "semantic_label": record.__dict__["semantic_label"],
            "account": record.__dict__["account"],
            "realm": record.__dict__["realm"],
            "tab_role": record.__dict__["tab_role"],
            "activity": record.__dict__["activity"],
            "scope": record.__dict__["scope"],
            "process_id": record.process,
            "thread_id": record.thread,
        }
        forwarded_exception = record.__dict__.get("_hbrowser_forwarded_exception")
        exception = (
            forwarded_exception
            if isinstance(forwarded_exception, str)
            else cls._format_exception(record)
        )
        if exception is not None:
            payload["exception"] = exception
        forwarded_stack = record.__dict__.get("_hbrowser_forwarded_stack")
        stack = (
            forwarded_stack if isinstance(forwarded_stack, str) else record.stack_info
        )
        if stack is not None:
            payload["stack"] = stack
        return (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _write_bytes(self, data: bytes) -> None:
        stream = self._stream
        if stream is None:
            raise OSError("Process log segment is not open")
        remaining = memoryview(data)
        while remaining:
            written = stream.write(remaining)
            if written is None or written <= 0:
                raise OSError("Process log segment write made no progress")
            remaining = remaining[written:]

    def _validate_stream_path_identity(self) -> None:
        stream = self._stream
        path = self._segment_path
        if stream is None or path is None:
            raise OSError("Process log segment is not open")
        descriptor_stat = self._validate_open_descriptor(stream.fileno(), path)
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or self._is_reparse_point(path_stat)
            or path_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise OSError("Process log path no longer names the opened segment")

    def _flush_stream(self) -> None:
        stream = self._stream
        if stream is None:
            raise OSError("Process log segment is not open")
        stream.flush()

    def _request_stop_after_failure(self, failure: LogPersistenceError) -> None:
        """Latch degradation and let the writer own descriptor cleanup."""
        self._enabled = False
        self._stop_requested.set()
        with self._failure_lock:
            if self._close_failure is None:
                self._close_failure = failure
        _record_persistence_failure(failure)

    def _disable_after_failure(self, failure: LogPersistenceError) -> None:
        """Discard inherited resources in a fork child whose writer vanished."""
        # None of the parent's synchronization primitives are safe to acquire
        # after a multithreaded fork. Replace them before touching lifecycle
        # state; the copied queue is intentionally discarded with its records.
        self._write_queue = queue.Queue(maxsize=_MAX_QUEUED_PROCESS_RECORDS)
        self._stop_requested = Event()
        self._stopped = Event()
        self._idle = Event()
        self._close_lock = Lock()
        self._enqueue_lock = Lock()
        self._failure_lock = Lock()
        self._worker_started = False
        self._enabled = False
        self._stop_requested.set()
        self._close_failure = failure
        self._close_stream_best_effort()
        self._close_owner_lock_best_effort()
        self._close_directory_best_effort()
        self._stopped.set()
        self._idle.set()
        _record_persistence_failure(failure)

    def _roll_segment_if_needed(self, record_bytes: int) -> None:
        if self._written_bytes == 0:
            return
        if self._written_bytes < self.segment_bytes and (
            self._written_bytes + record_bytes <= self.segment_bytes
        ):
            return
        self._close_stream()
        self._open_next_segment()

    def _write_encoded_record(self, data: bytes) -> None:
        self._roll_segment_if_needed(len(data))
        try:
            self._validate_stream_path_identity()
            self._write_bytes(data)
        except OSError as error:
            raise LogPersistenceError(
                "segment-write",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error
        try:
            self._flush_stream()
        except OSError as error:
            raise LogPersistenceError(
                "segment-flush",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error
        try:
            self._validate_stream_path_identity()
        except OSError as error:
            raise LogPersistenceError(
                "segment-write",
                error,
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            ) from error
        self._written_bytes += len(data)

    def _mark_record_finished(self) -> None:
        with self._enqueue_lock:
            self._write_queue.task_done()
            with self._write_queue.mutex:
                unfinished = self._write_queue.unfinished_tasks
            if unfinished == 0:
                self._idle.set()

    def _discard_queued_records(self) -> None:
        while True:
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                return
            self._mark_record_finished()

    def _close_owned_resources(self) -> LogPersistenceError | None:
        failure: LogPersistenceError | None = None
        for close_resource in (
            self._close_stream,
            self._close_owner_lock,
            self._close_directory,
        ):
            try:
                close_resource()
            except LogPersistenceError as error:
                if failure is None:
                    failure = error
        return failure

    def _run_writer(self) -> None:
        try:
            while True:
                if self._stop_requested.is_set() and self._write_queue.empty():
                    break
                try:
                    data = self._write_queue.get(timeout=_PROCESS_SINK_POLL_SECONDS)
                except queue.Empty:
                    continue
                write_failure: LogPersistenceError | None = None
                try:
                    self._write_encoded_record(data)
                except LogPersistenceError as error:
                    write_failure = error
                except Exception as error:
                    write_failure = LogPersistenceError(
                        "segment-write",
                        error,
                        segment_index=self._next_segment_index - 1,
                        writer_pid=self._writer_pid,
                    )
                if write_failure is not None:
                    self._request_stop_after_failure(write_failure)
                self._mark_record_finished()
                if write_failure is not None:
                    self._discard_queued_records()
                    break

            close_failure = self._close_owned_resources()
            if close_failure is not None:
                self._request_stop_after_failure(close_failure)
        finally:
            self._enabled = False
            self._idle.set()
            self._stopped.set()

    def emit(self, record: logging.LogRecord) -> None:
        if not self._enabled:
            return
        if os.getpid() != self._writer_pid:
            failure = LogPersistenceError(
                "segment-write",
                RuntimeError("Process log handler cannot be used after fork"),
                segment_index=self._next_segment_index - 1,
                writer_pid=self._writer_pid,
            )
            self._request_stop_after_failure(failure)
            return
        try:
            data = self._encode_record(record)
            with self._enqueue_lock:
                if not self._enabled:
                    # Another thread may close the handler after the outer check.
                    return  # type: ignore[unreachable]
                self._idle.clear()
                self._write_queue.put_nowait(data)
        except queue.Full as error:
            self._request_stop_after_failure(
                LogPersistenceError(
                    "segment-write",
                    error,
                    segment_index=self._next_segment_index - 1,
                    writer_pid=self._writer_pid,
                )
            )
        except Exception as error:
            self._request_stop_after_failure(
                LogPersistenceError(
                    "segment-write",
                    error,
                    segment_index=self._next_segment_index - 1,
                    writer_pid=self._writer_pid,
                )
            )

    def wait_until_idle(self, *, timeout: float) -> bool:
        """Wait for queued records in tests and explicit lifecycle boundaries."""
        return self._idle.wait(timeout=timeout)

    def close_sink(self) -> None:
        """Boundedly drain and close, surfacing the first sink failure."""
        with self._close_lock:
            with self._enqueue_lock:
                self._enabled = False
                self._stop_requested.set()
            if self._worker_started:
                self._worker.join(timeout=_PROCESS_SINK_CLOSE_TIMEOUT_SECONDS)
            if not self._stopped.is_set():
                self._abandoned_until_process_exit = True
                timeout_failure = LogPersistenceError(
                    "segment-close",
                    TimeoutError("Process log writer exceeded close deadline"),
                    segment_index=self._next_segment_index - 1,
                    writer_pid=self._writer_pid,
                )
                with self._failure_lock:
                    if self._close_failure is None:
                        self._close_failure = timeout_failure
                _record_persistence_failure(timeout_failure)
            with self._failure_lock:
                failure = self._close_failure
            logging.Handler.close(self)
        if failure is not None:
            raise failure

    def abandon_until_process_exit(self) -> None:
        """Detach a sink whose forwarding writer cannot be proven stopped.

        No descriptor is closed by the caller. The writer may finish and close
        safely in its daemon thread; otherwise the OS reclaims every descriptor
        before the launcher can observe process exit and archive the run.
        """

        with self._enqueue_lock:
            self._enabled = False
            self._stop_requested.set()
        self._abandoned_until_process_exit = True
        logging.Handler.close(self)

    def close(self) -> None:
        """Best-effort close for :mod:`logging` shutdown and test cleanup."""
        if self._abandoned_until_process_exit:
            logging.Handler.close(self)
            return
        try:
            self.close_sink()
        except LogPersistenceError as failure:
            _record_persistence_failure(failure)


def _validate_managed_logger_handlers(logger: logging.Logger) -> None:
    unmanaged_handlers = [
        handler
        for handler in logger.handlers
        if not getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
        and not getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
        and not getattr(handler, _MANAGED_FORWARDING_HANDLER_ATTRIBUTE, False)
    ]
    if unmanaged_handlers:
        raise ValueError(
            f"Logger {logger.name!r} already has unmanaged handlers; "
            "setup_logger requires exclusive handler ownership"
        )


def _handler_owner_logger_names() -> tuple[str, ...]:
    return tuple(sorted(set(_NAMESPACE_LOGGER_NAMES) | _MANAGED_LOGGER_NAMES))


def _is_namespace_logger_name(name: str) -> bool:
    return any(
        name == namespace or name.startswith(f"{namespace}.")
        for namespace in _NAMESPACE_LOGGER_NAMES
    )


def _configure_managed_logger(
    logger: logging.Logger,
    configuration: _LoggingConfiguration,
    process_handler: _AppendOnlyJsonlHandler | None,
) -> None:
    managed_stdout_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
    ]
    if managed_stdout_handlers:
        stdout_handler = managed_stdout_handlers[0]
        for duplicate in managed_stdout_handlers[1:]:
            logger.removeHandler(duplicate)
            duplicate.close()
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
        setattr(stdout_handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, True)
        logger.addHandler(stdout_handler)

    stdout_handler.setLevel(configuration.console_level.number)
    _configure_handler_context(stdout_handler)
    stdout_handler.setFormatter(_formatter())

    for handler in tuple(logger.handlers):
        if (
            getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
            and handler is not process_handler
        ):
            logger.removeHandler(handler)
    if process_handler is not None and process_handler not in logger.handlers:
        logger.addHandler(process_handler)

    active_levels = [configuration.console_level.number]
    if process_handler is not None:
        active_levels.append(configuration.file_level.number)
    logger.setLevel(min(active_levels))
    logger.propagate = False


def _configure_forwarded_logger(
    logger: logging.Logger,
    forwarding_handler: ForwardingHandler | None,
) -> None:
    """Configure one handler-owning logger for child-only forwarding."""
    for handler in tuple(logger.handlers):
        if handler is not forwarding_handler:
            logger.removeHandler(handler)
    if forwarding_handler is not None and forwarding_handler not in logger.handlers:
        logger.addHandler(forwarding_handler)
    logger.setLevel(
        logging.DEBUG if forwarding_handler is not None else logging.CRITICAL + 1
    )
    logger.propagate = False


def _deliver_forwarded_record(record: ForwardedRecord) -> bool:
    """Deliver validated child data directly to the one parent file handler."""
    process_handler = _PROCESS_LOG_HANDLER
    if (
        _LOGGING_CLOSED
        or not _LOGGING_CONFIGURED
        or process_handler is None
        or not process_handler.healthy
    ):
        return False
    level = LogLevel(record.level)
    if level.number < process_handler.level:
        return True
    forwarded = logging.LogRecord(
        record.logger,
        level.number,
        "",
        0,
        record.message,
        (),
        None,
    )
    forwarded.created = record.created
    forwarded.process = record.process_id
    forwarded.thread = record.thread_id
    for field in _LOG_CONTEXT_FIELDS:
        forwarded.__dict__[field] = getattr(record, field)
    forwarded.__dict__["_hbrowser_forwarded_exception"] = record.exception
    forwarded.__dict__["_hbrowser_forwarded_stack"] = record.stack
    process_handler.handle(forwarded)
    return process_handler.healthy and not logging_health().trace_degraded


def start_log_forwarding_receiver() -> LogForwardingReceiver | None:
    """Best-effort start the parent's authenticated loopback trace receiver.

    Runtime sink and network availability failures are trace degradation, not
    business-startup failures. They latch safe health and return ``None``.
    """
    global _LOG_FORWARDING_RECEIVER, _LOG_FORWARDING_RECEIVER_ATTEMPTED

    with _LOGGING_CONFIGURATION_LOCK:
        if _LOGGING_CLOSED:
            raise RuntimeError("Logging lifecycle is already closed")
        if _FORWARDED_LOGGING_CONFIGURED or _FORWARDED_LOGGING_CLOSED:
            raise RuntimeError("This process owns a forwarded-logging lifecycle")
        if not _LOGGING_CONFIGURED:
            raise RuntimeError("configure_logging must run before log forwarding")
        if _LOG_FORWARDING_RECEIVER_ATTEMPTED:
            return _LOG_FORWARDING_RECEIVER
        _LOG_FORWARDING_RECEIVER_ATTEMPTED = True
        process_handler = _PROCESS_LOG_HANDLER
        if process_handler is None or not process_handler.healthy:
            _record_forwarding_failure(
                "forward-configure",
                RuntimeError("The parent trace sink is unavailable"),
            )
            return None
        receiver = _LOG_FORWARDING_RECEIVER
        if receiver is not None:
            return receiver
        try:
            receiver = LogForwardingReceiver(
                delivery_callback=_deliver_forwarded_record,
                failure_callback=_record_forwarding_failure,
            )
        except Exception as error:
            _record_forwarding_failure("forward-configure", error)
            return None
        _LOG_FORWARDING_RECEIVER = receiver
        return receiver


def _log_forwarding_environment_for_child() -> dict[str, str]:
    """Return a capability when available, otherwise a non-raising empty map."""
    with _LOGGING_CONFIGURATION_LOCK:
        receiver = _LOG_FORWARDING_RECEIVER
        if receiver is None:
            _record_forwarding_failure(
                "forward-configure",
                RuntimeError("Log forwarding receiver is not active"),
            )
            return {}
        try:
            return receiver.child_environment()
        except Exception as error:
            _record_forwarding_failure("forward-configure", error)
            return {}


def configure_forwarded_logging() -> None:
    """Configure a child to forward trace records without opening local sinks.

    The capability is consumed from the environment and removed immediately so
    later descendants cannot inherit it accidentally. Connection/configuration
    failures latch forwarding-degraded health and keep ordinary log calls quiet.
    """
    global _FORWARDED_LOGGING_CONFIGURED, _FORWARDING_HANDLER

    endpoint = os.environ.pop(FORWARD_ENDPOINT_ENVIRONMENT_VARIABLE, None)
    token = os.environ.pop(FORWARD_TOKEN_ENVIRONMENT_VARIABLE, None)
    with _LOGGING_CONFIGURATION_LOCK:
        if _FORWARDED_LOGGING_CLOSED:
            _record_forwarding_failure(
                "forward-configure",
                RuntimeError("Forwarded logging lifecycle is already closed"),
            )
            return
        if _FORWARDED_LOGGING_CONFIGURED:
            return
        _FORWARDED_LOGGING_CONFIGURED = True
        forwarding_handler: ForwardingHandler | None = None
        if _LOGGING_CONFIGURED or _LOGGING_CLOSED:
            _record_forwarding_failure(
                "forward-configure",
                RuntimeError("A parent logging lifecycle already exists"),
            )
        elif endpoint is None or token is None:
            _record_forwarding_failure(
                "forward-configure",
                ForwardingConfigurationError(
                    "The log forwarding capability is unavailable"
                ),
            )
        else:
            try:
                forwarding_handler = ForwardingHandler(
                    endpoint,
                    token,
                    failure_callback=_record_forwarding_failure,
                )
            except Exception as error:
                _record_forwarding_failure("forward-configure", error)
            else:
                setattr(
                    forwarding_handler,
                    _MANAGED_FORWARDING_HANDLER_ATTRIBUTE,
                    True,
                )
                _configure_handler_context(forwarding_handler)
                _FORWARDING_HANDLER = forwarding_handler

        for logger_name in _handler_owner_logger_names():
            _configure_forwarded_logger(
                logging.getLogger(logger_name),
                forwarding_handler,
            )


def close_forwarded_logging() -> None:
    """Gracefully close a child forwarding lifecycle without raising."""
    global _FORWARDED_LOGGING_CLOSED, _FORWARDED_LOGGING_CONFIGURED
    global _FORWARDING_HANDLER

    with _LOGGING_CONFIGURATION_LOCK:
        if _FORWARDED_LOGGING_CLOSED:
            return
        _FORWARDED_LOGGING_CLOSED = True
        _FORWARDED_LOGGING_CONFIGURED = False
        forwarding_handler = _FORWARDING_HANDLER
        for logger_name in _handler_owner_logger_names():
            logger = logging.getLogger(logger_name)
            if forwarding_handler is not None:
                logger.removeHandler(forwarding_handler)
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False
        _FORWARDING_HANDLER = None
        if forwarding_handler is not None:
            try:
                forwarding_handler.close_forwarding()
            except Exception as error:
                _record_forwarding_failure("forward-close", error)


def configure_logging(
    *,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    segment_bytes: int = _DEFAULT_SEGMENT_BYTES,
    require_file_sink: bool = True,
) -> None:
    """Explicitly configure console output and one append-only JSONL trace sink.

    ``HBROWSER_LOG_DIR`` selects the directory. Every segment is exclusively
    created as ``events-NNNNNN.jsonl``. A completed segment is never renamed,
    reopened, overwritten, or deleted by this module. When the file sink is not
    required, its one startup failure becomes degraded health and console
    logging remains available.
    """
    global _FILE_SINK_ATTEMPTED, _LOGGING_CONFIGURATION, _LOGGING_CONFIGURED
    global _PROCESS_LOG_HANDLER

    if not isinstance(console_level, LogLevel):
        raise TypeError("console_level must be a LogLevel")
    if not isinstance(file_level, LogLevel):
        raise TypeError("file_level must be a LogLevel")
    if isinstance(segment_bytes, bool) or not isinstance(segment_bytes, int):
        raise TypeError("segment_bytes must be an integer")
    if segment_bytes <= 0:
        raise ValueError("segment_bytes must be positive")
    if not isinstance(require_file_sink, bool):
        raise TypeError("require_file_sink must be a bool")

    configuration = _LoggingConfiguration(
        console_level=console_level,
        file_level=file_level,
        segment_bytes=segment_bytes,
        require_file_sink=require_file_sink,
    )
    with _LOGGING_CONFIGURATION_LOCK:
        if _LOGGING_CLOSED:
            raise RuntimeError("Logging lifecycle is already closed")
        if _FORWARDED_LOGGING_CONFIGURED or _FORWARDED_LOGGING_CLOSED:
            raise RuntimeError("This process owns a forwarded-logging lifecycle")
        health = logging_health()
        if health.trace_degraded and require_file_sink:
            raise_for_log_persistence_failure()
        for logger_name in _handler_owner_logger_names():
            _validate_managed_logger_handlers(logging.getLogger(logger_name))
        process_handler = _PROCESS_LOG_HANDLER
        if not _FILE_SINK_ATTEMPTED:
            _FILE_SINK_ATTEMPTED = True
            try:
                directory = get_log_dir()
            except OSError as error:
                directory_failure = LogPersistenceError("configure", error)
                _record_persistence_failure(directory_failure)
                if require_file_sink:
                    raise directory_failure from error
            else:
                try:
                    process_handler = _AppendOnlyJsonlHandler(
                        directory,
                        segment_bytes=segment_bytes,
                        level=file_level.number,
                    )
                except LogPersistenceError as persistence_failure:
                    _record_persistence_failure(persistence_failure)
                    if require_file_sink:
                        raise
                except Exception as error:
                    configuration_failure = LogPersistenceError("configure", error)
                    _record_persistence_failure(configuration_failure)
                    if require_file_sink:
                        raise configuration_failure from error
                else:
                    setattr(
                        process_handler,
                        _MANAGED_PROCESS_HANDLER_ATTRIBUTE,
                        True,
                    )
                    _PROCESS_LOG_HANDLER = process_handler
        elif process_handler is not None:
            requested_directory = _log_dir_path()
            if requested_directory != process_handler.directory:
                raise ValueError(
                    "HBROWSER_LOG_DIR cannot change after logging is configured"
                )
            if process_handler.healthy:
                process_handler.reconfigure(
                    level=file_level.number,
                    segment_bytes=segment_bytes,
                )

        active_process_handler = (
            process_handler
            if process_handler is not None and process_handler.healthy
            else None
        )

        for logger_name in _handler_owner_logger_names():
            _configure_managed_logger(
                logging.getLogger(logger_name),
                configuration,
                active_process_handler,
            )

        _LOGGING_CONFIGURATION = configuration
        _LOGGING_CONFIGURED = True


def log_to_process_file(
    logger: logging.Logger,
    level: LogLevel,
    message: str,
) -> None:
    """Write only to the process file and boundedly verify persistence."""
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    if not isinstance(level, LogLevel):
        raise TypeError("level must be a LogLevel")
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    with _LOGGING_CONFIGURATION_LOCK:
        if _LOGGING_CLOSED:
            raise RuntimeError("Logging lifecycle is already closed")
        if logging.getLogger(logger.name) is not logger or (
            logger.name not in _MANAGED_LOGGER_NAMES
            and not _is_namespace_logger_name(logger.name)
        ):
            raise ValueError(
                "logger must be a configured namespace logger or registered "
                "by setup_logger"
            )
        if not _LOGGING_CONFIGURED:
            return
        raise_for_log_persistence_failure()
        process_handler = _PROCESS_LOG_HANDLER
        if process_handler is None:
            return
        if level.number < process_handler.level:
            return
        record = logger.makeRecord(
            logger.name,
            level.number,
            "",
            0,
            message,
            (),
            None,
        )
        process_handler.handle(record)
        if not process_handler.wait_until_idle(
            timeout=_PROCESS_SINK_CLOSE_TIMEOUT_SECONDS
        ):
            timeout_failure = LogPersistenceError(
                "segment-flush",
                TimeoutError("Process log writer exceeded flush deadline"),
                segment_index=process_handler._next_segment_index - 1,
                writer_pid=process_handler._writer_pid,
            )
            process_handler._request_stop_after_failure(timeout_failure)
        raise_for_log_persistence_failure()


def setup_logger(name: str) -> logging.Logger:
    """Register a logger without configuring or opening any output sink."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must not be empty")

    with _LOGGING_CONFIGURATION_LOCK:
        if _LOGGING_CLOSED or _FORWARDED_LOGGING_CLOSED:
            raise RuntimeError("Logging lifecycle is already closed")
        logger = logging.getLogger(name)
        if name not in _MANAGED_LOGGER_NAMES:
            _validate_managed_logger_handlers(logger)
        _MANAGED_LOGGER_NAMES.add(name)
        if _FORWARDED_LOGGING_CONFIGURED:
            _configure_forwarded_logger(logger, _FORWARDING_HANDLER)
        elif _LOGGING_CONFIGURED:
            process_handler = _PROCESS_LOG_HANDLER
            _configure_managed_logger(
                logger,
                _LOGGING_CONFIGURATION,
                (
                    process_handler
                    if process_handler is not None and process_handler.healthy
                    else None
                ),
            )
        else:
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False
        return logger


def close_logging() -> None:
    """Permanently close this process's logging lifecycle at a safe boundary.

    The first call detaches and closes every managed sink. Repeated healthy
    calls are no-ops; when persistence health is latched, every call raises the
    same :class:`LogPersistenceError`.
    """
    global _LOG_FORWARDING_RECEIVER, _LOGGING_CLOSED, _LOGGING_CONFIGURED
    global _PROCESS_LOG_HANDLER

    with _LOGGING_CONFIGURATION_LOCK:
        if not _LOGGING_CLOSED:
            if _FORWARDED_LOGGING_CONFIGURED or _FORWARDED_LOGGING_CLOSED:
                raise RuntimeError("Use close_forwarded_logging in a forwarding child")
            receiver = _LOG_FORWARDING_RECEIVER
            _LOG_FORWARDING_RECEIVER = None
            forwarding_ownership_unresolved = False
            if receiver is not None:
                try:
                    receiver.close()
                except Exception as error:
                    forwarding_ownership_unresolved = True
                    _record_forwarding_failure("forward-drain", error)
                    receiver._discard_after_fork()
            _LOGGING_CLOSED = True
            _LOGGING_CONFIGURED = False
            process_handler = _PROCESS_LOG_HANDLER
            console_handlers: set[logging.Handler] = set()
            for logger_name in _handler_owner_logger_names():
                logger = logging.getLogger(logger_name)
                for handler in tuple(logger.handlers):
                    if getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False):
                        console_handlers.add(handler)
                        logger.removeHandler(handler)
                    elif getattr(
                        handler,
                        _MANAGED_PROCESS_HANDLER_ATTRIBUTE,
                        False,
                    ):
                        logger.removeHandler(handler)
                logger.setLevel(logging.CRITICAL + 1)
                logger.propagate = False
            for handler in console_handlers:
                handler.close()
            _PROCESS_LOG_HANDLER = None
            if process_handler is not None:
                if forwarding_ownership_unresolved:
                    process_handler.abandon_until_process_exit()
                else:
                    try:
                        process_handler.close_sink()
                    except LogPersistenceError as failure:
                        _record_persistence_failure(failure)
    raise_for_log_persistence_failure()


@contextmanager
def _isolated_logging_state_for_testing() -> Iterator[None]:
    """Isolate mutable process logging state for one unit test."""
    global _FILE_SINK_ATTEMPTED, _LOGGING_CONFIGURATION, _LOGGING_CONFIGURED
    global _FORWARDED_LOGGING_CLOSED, _FORWARDED_LOGGING_CONFIGURED
    global _FORWARDING_HANDLER, _LOG_FORWARDING_FAILURE
    global _LOG_FORWARDING_RECEIVER, _LOG_FORWARDING_RECEIVER_ATTEMPTED
    global _LOGGING_CLOSED
    global _LOG_PERSISTENCE_FAILURE, _PROCESS_LOG_HANDLER

    with _LOGGING_CONFIGURATION_LOCK:
        if _LOG_FORWARDING_RECEIVER is not None:
            raise RuntimeError("Cannot isolate an active forwarding receiver")
        previous_configuration = _LOGGING_CONFIGURATION
        previous_configured = _LOGGING_CONFIGURED
        previous_closed = _LOGGING_CLOSED
        previous_forwarded_configured = _FORWARDED_LOGGING_CONFIGURED
        previous_forwarded_closed = _FORWARDED_LOGGING_CLOSED
        previous_file_sink_attempted = _FILE_SINK_ATTEMPTED
        previous_logger_names = set(_MANAGED_LOGGER_NAMES)
        previous_handler = _PROCESS_LOG_HANDLER
        previous_forwarding_handler = _FORWARDING_HANDLER
        previous_receiver_attempted = _LOG_FORWARDING_RECEIVER_ATTEMPTED
        previous_failure = _LOG_PERSISTENCE_FAILURE
        previous_forwarding_failure = _LOG_FORWARDING_FAILURE
        snapshot_names = set(_NAMESPACE_LOGGER_NAMES) | previous_logger_names
        logger_snapshots = {
            name: (
                logging.getLogger(name).handlers[:],
                logging.getLogger(name).level,
                logging.getLogger(name).propagate,
            )
            for name in snapshot_names
        }
        for name in snapshot_names:
            logger = logging.getLogger(name)
            logger.handlers = [
                handler
                for handler in logger.handlers
                if not getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
                and not getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
                and not getattr(
                    handler,
                    _MANAGED_FORWARDING_HANDLER_ATTRIBUTE,
                    False,
                )
            ]
        _LOGGING_CONFIGURATION = _LoggingConfiguration()
        _LOGGING_CONFIGURED = False
        _LOGGING_CLOSED = False
        _FORWARDED_LOGGING_CONFIGURED = False
        _FORWARDED_LOGGING_CLOSED = False
        _FILE_SINK_ATTEMPTED = False
        _MANAGED_LOGGER_NAMES.clear()
        _PROCESS_LOG_HANDLER = None
        _FORWARDING_HANDLER = None
        _LOG_FORWARDING_RECEIVER = None
        _LOG_FORWARDING_RECEIVER_ATTEMPTED = False
        _LOG_PERSISTENCE_FAILURE = None
        _LOG_FORWARDING_FAILURE = None
    try:
        yield
    finally:
        with _LOGGING_CONFIGURATION_LOCK:
            current_handler = _PROCESS_LOG_HANDLER
            current_forwarding_handler = _FORWARDING_HANDLER
            current_receiver = _LOG_FORWARDING_RECEIVER
            current_names = set(_NAMESPACE_LOGGER_NAMES) | set(_MANAGED_LOGGER_NAMES)
            if current_receiver is not None:
                # The yielded test body may configure a receiver.
                try:  # type: ignore[unreachable]
                    current_receiver.close()
                except Exception:
                    current_receiver._discard_after_fork()
            for name in current_names:
                logger = logging.getLogger(name)
                logger.handlers = [
                    handler
                    for handler in logger.handlers
                    if not getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
                    and not getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
                    and not getattr(
                        handler,
                        _MANAGED_FORWARDING_HANDLER_ATTRIBUTE,
                        False,
                    )
                ]
            if current_forwarding_handler is not None:
                # The yielded test body may configure forwarding.
                current_forwarding_handler.close_forwarding()  # type: ignore[unreachable]
            if current_handler is not None:
                # The yielded test body may configure the process handler.
                current_handler.close()  # type: ignore[unreachable]
            _LOGGING_CONFIGURATION = previous_configuration
            _LOGGING_CONFIGURED = previous_configured
            _LOGGING_CLOSED = previous_closed
            _FORWARDED_LOGGING_CONFIGURED = previous_forwarded_configured
            _FORWARDED_LOGGING_CLOSED = previous_forwarded_closed
            _FILE_SINK_ATTEMPTED = previous_file_sink_attempted
            _MANAGED_LOGGER_NAMES.clear()
            _MANAGED_LOGGER_NAMES.update(previous_logger_names)
            _PROCESS_LOG_HANDLER = previous_handler
            _FORWARDING_HANDLER = previous_forwarding_handler
            _LOG_FORWARDING_RECEIVER = None
            _LOG_FORWARDING_RECEIVER_ATTEMPTED = previous_receiver_attempted
            _LOG_PERSISTENCE_FAILURE = previous_failure
            _LOG_FORWARDING_FAILURE = previous_forwarding_failure
            for name, (handlers, level, propagate) in logger_snapshots.items():
                logger = logging.getLogger(name)
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate


def _log_dir_path() -> Path:
    configured_directory = os.getenv(_LOG_DIR_ENVIRONMENT_VARIABLE)
    if configured_directory:
        log_dir = Path(configured_directory).expanduser()
        if not log_dir.is_absolute():
            log_dir = Path.cwd() / log_dir
        return Path(os.path.abspath(log_dir))

    script_name = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if script_name and not script_name.startswith("-"):
        script_dir = Path(script_name).expanduser().resolve().parent
    else:
        script_dir = Path.cwd().resolve()
    return script_dir / "log"


def get_log_dir() -> Path:
    """Return and create the directory for private diagnostics and log segments."""
    log_dir = _log_dir_path()

    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return log_dir
