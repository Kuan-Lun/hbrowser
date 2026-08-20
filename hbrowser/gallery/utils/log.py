"""日誌相關工具函數"""

from __future__ import annotations

import logging
import logging.handlers
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock

_LOG_DIR_ENVIRONMENT_VARIABLE = "HBROWSER_LOG_DIR"
_PROCESS_LOG_FILE_ENVIRONMENT_VARIABLE = "HBROWSER_PROCESS_LOG_FILE"
_DEFAULT_PROCESS_LOG_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_PROCESS_LOG_BACKUP_COUNT = 5
_LOGGING_CONFIGURATION_LOCK = RLock()
_PROCESS_LOG_HANDLERS: dict[Path, _PrivateRotatingFileHandler] = {}
_MANAGED_LOGGER_NAMES: set[str] = set()
_MANAGED_STDOUT_HANDLER_ATTRIBUTE = "_hbrowser_managed_stdout_handler"
_MANAGED_PROCESS_HANDLER_ATTRIBUTE = "_hbrowser_managed_process_handler"
_LOG_CONTEXT_FIELDS = ("account", "realm", "tab_role", "activity", "scope")


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


class LogPersistenceError(RuntimeError):
    """A configured application-log sink could not persist a record safely."""

    def __init__(self, operation: str, error: BaseException) -> None:
        self.operation = operation
        self.error_type = type(error).__name__
        super().__init__(
            "Process log persistence failed: "
            f"operation={operation} error_type={self.error_type}"
        )


@dataclass(frozen=True, slots=True)
class _LoggingConfiguration:
    console_level: LogLevel = LogLevel.INFO
    file_level: LogLevel = LogLevel.DEBUG
    max_bytes: int = _DEFAULT_PROCESS_LOG_MAX_BYTES
    backup_count: int = _DEFAULT_PROCESS_LOG_BACKUP_COUNT


_LOGGING_CONFIGURATION = _LoggingConfiguration()


@dataclass(frozen=True, slots=True)
class _LogContext:
    account: str | None = None
    realm: str | None = None
    tab_role: str | None = None
    activity: str | None = None
    scope: str | None = None


_CURRENT_LOG_CONTEXT: ContextVar[_LogContext] = ContextVar(
    "hbrowser_log_context",
    default=_LogContext(),
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
    """Add semantic fields to log records within one synchronous scope.

    Unspecified fields inherit their current values. Context variables make the
    scope safe to nest and keep concurrent asyncio tasks isolated from one
    another, even when the context remains active across ``await`` expressions.
    """
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


def _configure_user_facing_handler(
    handler: logging.Handler,
    formatter: logging.Formatter,
) -> None:
    if _LOG_CONTEXT_FILTER not in handler.filters:
        handler.addFilter(_LOG_CONTEXT_FILTER)
    handler.setFormatter(formatter)


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Keep the process log private and surface sink failures to the caller."""

    @staticmethod
    def _validate_open_descriptor(descriptor: int, path: Path) -> None:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(f"Process log target must be a regular file: {path}")
        if descriptor_stat.st_nlink != 1:
            raise OSError(f"Process log target must have exactly one link: {path}")

        try:
            path_stat = path.stat(follow_symlinks=False)
        except OSError as error:
            raise OSError(
                f"Process log target is no longer addressable: {path}"
            ) from error
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError(f"Process log target must be a regular file: {path}")
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise OSError(f"Process log path no longer names the opened file: {path}")

    def _open(self):  # type: ignore[no-untyped-def]
        path = Path(self.baseFilename)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            self._validate_open_descriptor(descriptor, path)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            stream = os.fdopen(
                descriptor,
                self.mode,
                encoding=self.encoding,
                errors=self.errors,
            )
            descriptor = None
            return stream
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _validate_stream_path_identity(self) -> None:
        if self.stream is None:
            return
        self._validate_open_descriptor(
            self.stream.fileno(),
            Path(self.baseFilename),
        )

    @classmethod
    def _secure_existing_rollover_path(
        cls,
        path: Path,
        *,
        required: bool,
    ) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            if required:
                raise
            return
        if (
            stat.S_ISLNK(path_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
        ):
            raise OSError(
                "Process log rollover path must be a regular, "
                f"single-link file: {path}"
            )

        flags = os.O_WRONLY | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            if required:
                raise
            return
        try:
            cls._validate_open_descriptor(descriptor, path)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            cls._validate_open_descriptor(descriptor, path)
        finally:
            os.close(descriptor)

    def _secure_rollover_paths(self) -> None:
        base_path = Path(self.baseFilename)
        self._secure_existing_rollover_path(base_path, required=True)
        self._validate_stream_path_identity()
        for backup_index in range(1, self.backupCount + 1):
            self._secure_existing_rollover_path(
                Path(f"{base_path}.{backup_index}"),
                required=False,
            )

    def doRollover(self) -> None:  # noqa: N802
        """Rotate only private, single-link regular files."""
        try:
            self._secure_rollover_paths()
            super().doRollover()
            self._secure_rollover_paths()
        except LogPersistenceError:
            raise
        except Exception as error:
            raise LogPersistenceError("rollover", error) from error

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._validate_stream_path_identity()
            super().emit(record)
            self._validate_stream_path_identity()
        except LogPersistenceError:
            raise
        except Exception as error:
            raise LogPersistenceError("emit", error) from error

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        """Never let a configured process-log failure remain best effort."""
        del record
        error = sys.exception()
        if error is None:
            error = RuntimeError("Process log handler failed without an exception")
        if isinstance(error, LogPersistenceError):
            raise error
        raise LogPersistenceError("emit", error) from error


def _configured_process_log_path() -> Path | None:
    configured = os.getenv(_PROCESS_LOG_FILE_ENVIRONMENT_VARIABLE, "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _validate_process_log_target(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
        raise OSError(f"Process log target must be a regular file: {path}")


def _process_log_handler(
    path: Path | None,
    configuration: _LoggingConfiguration,
) -> _PrivateRotatingFileHandler | None:
    if path is None:
        return None

    handler = _PROCESS_LOG_HANDLERS.get(path)
    if handler is None:
        try:
            _validate_process_log_target(path)
            handler = _PrivateRotatingFileHandler(
                path,
                maxBytes=configuration.max_bytes,
                backupCount=configuration.backup_count,
                encoding="utf-8",
            )
        except LogPersistenceError:
            raise
        except Exception as error:
            raise LogPersistenceError("configure", error) from error
        setattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, True)
        _PROCESS_LOG_HANDLERS[path] = handler

    handler.acquire()
    try:
        handler.maxBytes = configuration.max_bytes
        handler.backupCount = configuration.backup_count
        handler.setLevel(configuration.file_level.number)
    finally:
        handler.release()
    return handler


def _close_inactive_process_log_handlers(active_path: Path | None) -> None:
    inactive = tuple(
        (path, handler)
        for path, handler in _PROCESS_LOG_HANDLERS.items()
        if path != active_path
    )
    if not inactive:
        return

    for logger_name in _MANAGED_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        for _, handler in inactive:
            logger.removeHandler(handler)
    for path, handler in inactive:
        try:
            handler.close()
        except Exception as error:
            raise LogPersistenceError("close", error) from error
        _PROCESS_LOG_HANDLERS.pop(path, None)


@contextmanager
def _isolated_process_log_handlers_for_testing() -> Iterator[None]:
    """Close shared process-log handlers after one isolated unit test."""
    try:
        yield
    finally:
        with _LOGGING_CONFIGURATION_LOCK:
            handlers = tuple(_PROCESS_LOG_HANDLERS.values())
            _PROCESS_LOG_HANDLERS.clear()
        for handler in handlers:
            for logger_object in logging.Logger.manager.loggerDict.values():
                if isinstance(logger_object, logging.Logger):
                    logger_object.removeHandler(handler)
            handler.close()


@contextmanager
def _isolated_logging_state_for_testing() -> Iterator[None]:
    """Isolate mutable process logging state for one unit test."""
    global _LOGGING_CONFIGURATION

    with _LOGGING_CONFIGURATION_LOCK:
        previous_configuration = _LOGGING_CONFIGURATION
        previous_logger_names = set(_MANAGED_LOGGER_NAMES)
        previous_handlers = dict(_PROCESS_LOG_HANDLERS)
        _LOGGING_CONFIGURATION = _LoggingConfiguration()
        _MANAGED_LOGGER_NAMES.clear()
        _PROCESS_LOG_HANDLERS.clear()
    try:
        yield
    finally:
        with _LOGGING_CONFIGURATION_LOCK:
            current_handlers = tuple(_PROCESS_LOG_HANDLERS.values())
            current_logger_names = tuple(_MANAGED_LOGGER_NAMES)
            for logger_name in current_logger_names:
                logger = logging.getLogger(logger_name)
                for handler in current_handlers:
                    logger.removeHandler(handler)
            _PROCESS_LOG_HANDLERS.clear()
            _MANAGED_LOGGER_NAMES.clear()
            _LOGGING_CONFIGURATION = previous_configuration
            _PROCESS_LOG_HANDLERS.update(previous_handlers)
            _MANAGED_LOGGER_NAMES.update(previous_logger_names)
        for handler in current_handlers:
            handler.close()


def _formatter() -> logging.Formatter:
    return _SemanticFormatter(
        "%(asctime)s - %(levelname)s - [%(semantic_label)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _configure_managed_logger(
    logger: logging.Logger,
    configuration: _LoggingConfiguration,
    process_handler: _PrivateRotatingFileHandler | None,
    formatter: logging.Formatter,
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
    _configure_user_facing_handler(stdout_handler, formatter)

    for handler in tuple(logger.handlers):
        if (
            getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
            and handler is not process_handler
        ):
            logger.removeHandler(handler)
    if process_handler is not None:
        process_handler.setLevel(configuration.file_level.number)
        _configure_user_facing_handler(process_handler, formatter)
        if process_handler not in logger.handlers:
            logger.addHandler(process_handler)

    active_levels = [configuration.console_level.number]
    if process_handler is not None:
        active_levels.append(configuration.file_level.number)
    logger.setLevel(min(active_levels))
    logger.propagate = False


def _validate_managed_logger_handlers(logger: logging.Logger) -> None:
    unmanaged_handlers = [
        handler
        for handler in logger.handlers
        if not getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
        and not getattr(handler, _MANAGED_PROCESS_HANDLER_ATTRIBUTE, False)
    ]
    if unmanaged_handlers:
        raise ValueError(
            f"Logger {logger.name!r} already has unmanaged handlers; "
            "setup_logger requires exclusive handler ownership"
        )


def _apply_logging_configuration(configuration: _LoggingConfiguration) -> None:
    managed_loggers = tuple(
        logging.getLogger(logger_name) for logger_name in _MANAGED_LOGGER_NAMES
    )
    process_path = _configured_process_log_path()
    process_handler = _process_log_handler(process_path, configuration)
    formatter = _formatter()
    for logger in managed_loggers:
        _configure_managed_logger(
            logger,
            configuration,
            process_handler,
            formatter,
        )
    _close_inactive_process_log_handlers(process_path)


def configure_logging(
    *,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    max_bytes: int = _DEFAULT_PROCESS_LOG_MAX_BYTES,
    backup_count: int = _DEFAULT_PROCESS_LOG_BACKUP_COUNT,
) -> None:
    """Configure every managed logger in the current process.

    The console and optional rotating file sink have independent thresholds.
    ``HBROWSER_PROCESS_LOG_FILE`` selects the file path; when it is absent,
    only the console threshold participates in each logger's effective level.
    Existing managed loggers are updated immediately.
    """
    global _LOGGING_CONFIGURATION

    if not isinstance(console_level, LogLevel):
        raise TypeError("console_level must be a LogLevel")
    if not isinstance(file_level, LogLevel):
        raise TypeError("file_level must be a LogLevel")
    for field, value in (("max_bytes", max_bytes), ("backup_count", backup_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        if value <= 0:
            raise ValueError(f"{field} must be positive")

    configuration = _LoggingConfiguration(
        console_level=console_level,
        file_level=file_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    with _LOGGING_CONFIGURATION_LOCK:
        previous_configuration = _LOGGING_CONFIGURATION
        try:
            _apply_logging_configuration(configuration)
        except LogPersistenceError as error:
            _LOGGING_CONFIGURATION = (
                configuration if error.operation == "close" else previous_configuration
            )
            raise
        else:
            _LOGGING_CONFIGURATION = configuration


def log_to_process_file(
    logger: logging.Logger,
    level: LogLevel,
    message: str,
) -> None:
    """Write one record only to the configured process-file sink.

    This is intended for a terminal record that another boundary renders to
    the console in a different representation. It avoids duplicate console
    output without adding routing fields to ordinary log records.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    if not isinstance(level, LogLevel):
        raise TypeError("level must be a LogLevel")
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    with _LOGGING_CONFIGURATION_LOCK:
        if (
            logger.name not in _MANAGED_LOGGER_NAMES
            or logging.getLogger(logger.name) is not logger
        ):
            raise ValueError("logger must be registered by setup_logger")
        process_path = _configured_process_log_path()
        _apply_logging_configuration(_LOGGING_CONFIGURATION)
        if process_path is None:
            return
        process_handler = _PROCESS_LOG_HANDLERS[process_path]
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


def setup_logger(name: str) -> logging.Logger:
    """Create or retrieve one logger managed by the process configuration."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must not be empty")

    with _LOGGING_CONFIGURATION_LOCK:
        logger = logging.getLogger(name)
        if name not in _MANAGED_LOGGER_NAMES:
            _validate_managed_logger_handlers(logger)
        _MANAGED_LOGGER_NAMES.add(name)
        _apply_logging_configuration(_LOGGING_CONFIGURATION)
        return logger


def get_log_dir() -> Path:
    """
    獲取並建立診斷資料夾。

    ``HBROWSER_LOG_DIR`` 可指定整合應用的共同日誌目錄。未指定時，維持
    原有行為，使用主腳本所在目錄下的 ``log`` 資料夾；若沒有可用的主腳本
    路徑則使用目前工作目錄。

    Returns:
        log 資料夾的絕對路徑
    """
    configured_directory = os.getenv(_LOG_DIR_ENVIRONMENT_VARIABLE)
    if configured_directory:
        log_dir = Path(configured_directory).expanduser()
        if not log_dir.is_absolute():
            log_dir = Path.cwd() / log_dir
        log_dir = log_dir.resolve()
    else:
        script_name = sys.argv[0] if sys.argv and sys.argv[0] else ""
        if script_name and not script_name.startswith("-"):
            script_dir = Path(script_name).expanduser().resolve().parent
        else:
            script_dir = Path.cwd().resolve()
        log_dir = script_dir / "log"

    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return log_dir
