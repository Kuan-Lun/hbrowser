"""日誌相關工具函數"""

import logging
import logging.handlers
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

_LOG_DIR_ENVIRONMENT_VARIABLE = "HBROWSER_LOG_DIR"
_PROCESS_LOG_FILE_ENVIRONMENT_VARIABLE = "HBROWSER_PROCESS_LOG_FILE"
_PROCESS_LOG_MAX_BYTES = 10 * 1024 * 1024
_PROCESS_LOG_BACKUP_COUNT = 5
_PROCESS_LOG_HANDLER_LOCK = Lock()
_PROCESS_LOG_HANDLERS: dict[Path, logging.Handler] = {}
_MANAGED_STDOUT_HANDLER_ATTRIBUTE = "_hbrowser_managed_stdout_handler"
_LOG_CONTEXT_FIELDS = ("account", "realm", "tab_role", "activity", "scope")


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

    def emit(self, record: logging.LogRecord) -> None:
        self._validate_stream_path_identity()
        super().emit(record)
        self._validate_stream_path_identity()

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        """Never let a configured process-log failure remain best effort."""
        del record
        error = sys.exception()
        if error is None:
            raise RuntimeError("Process log handler failed without an exception")
        raise error


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


def _process_log_handler() -> logging.Handler | None:
    path = _configured_process_log_path()
    if path is None:
        return None

    with _PROCESS_LOG_HANDLER_LOCK:
        handler = _PROCESS_LOG_HANDLERS.get(path)
        if handler is not None:
            return handler
        _validate_process_log_target(path)
        handler = _PrivateRotatingFileHandler(
            path,
            maxBytes=_PROCESS_LOG_MAX_BYTES,
            backupCount=_PROCESS_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _PROCESS_LOG_HANDLERS[path] = handler
        return handler


@contextmanager
def _isolated_process_log_handlers_for_testing() -> Iterator[None]:
    """Close shared process-log handlers after one isolated unit test."""
    try:
        yield
    finally:
        with _PROCESS_LOG_HANDLER_LOCK:
            handlers = tuple(_PROCESS_LOG_HANDLERS.values())
            _PROCESS_LOG_HANDLERS.clear()
        for handler in handlers:
            for logger_object in logging.Logger.manager.loggerDict.values():
                if isinstance(logger_object, logging.Logger):
                    logger_object.removeHandler(handler)
            handler.close()


def _formatter() -> logging.Formatter:
    return _SemanticFormatter(
        "%(asctime)s - %(levelname)s - [%(semantic_label)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logger(name: str) -> logging.Logger:
    """
    創建或獲取 logger

    日誌級別可通過環境變量 HBROWSER_LOG_LEVEL 控制：
    - DEBUG: 詳細的調試信息
    - INFO: 一般信息（默認）
    - WARNING: 警告信息
    - ERROR: 錯誤信息

    Args:
        name: logger 名稱（通常使用 __name__）

    Returns:
        配置好的 Logger 實例

    Example:
        >>> import os
        >>> os.environ["HBROWSER_LOG_LEVEL"] = "DEBUG"
        >>> logger = setup_logger(__name__)
        >>> logger.debug("這是調試信息")
    """
    logger = logging.getLogger(name)

    # 從環境變量獲取日誌級別
    level_str = os.getenv("HBROWSER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    managed_stdout_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, False)
    ]
    if not logger.handlers:
        stdout_handler = logging.StreamHandler(sys.stdout)
        setattr(stdout_handler, _MANAGED_STDOUT_HANDLER_ATTRIBUTE, True)
        stdout_handler.setLevel(level)
        _configure_user_facing_handler(stdout_handler, _formatter())
        logger.addHandler(stdout_handler)
    else:
        for managed_handler in managed_stdout_handlers:
            managed_handler.setLevel(level)
            _configure_user_facing_handler(managed_handler, _formatter())

    process_handler = _process_log_handler()
    if process_handler is not None:
        process_handler.setLevel(logging.NOTSET)
        _configure_user_facing_handler(process_handler, _formatter())
        if process_handler not in logger.handlers:
            logger.addHandler(process_handler)

    # 防止日誌向上傳播到 root logger（避免重複輸出）
    logger.propagate = False

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
