"""Authenticated bounded JSON transport for child-process trace records."""

from __future__ import annotations

import hmac
import json
import logging
import math
import queue
import re
import secrets
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

FORWARD_ENDPOINT_ENVIRONMENT_VARIABLE: Final = "HBROWSER_LOG_FORWARD_ENDPOINT"
FORWARD_TOKEN_ENVIRONMENT_VARIABLE: Final = "HBROWSER_LOG_FORWARD_TOKEN"

_LOOPBACK_HOST: Final = "127.0.0.1"
_TOKEN_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_LOGGER_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,255}\Z")
_MAX_FRAME_BYTES: Final = 32 * 1024
_MAX_MESSAGE_BYTES: Final = 8 * 1024
_MAX_DETAIL_BYTES: Final = 4 * 1024
_MAX_CONTEXT_BYTES: Final = 256
_MAX_UINT32: Final = 0xFFFFFFFF
_MAX_UINT64: Final = 0xFFFFFFFFFFFFFFFF
_MAX_CREATED_TIMESTAMP: Final = 4_102_444_800.0
_SOCKET_TIMEOUT_SECONDS: Final = 0.2
_CONNECT_TIMEOUT_SECONDS: Final = 2.0
_AUTHENTICATION_TIMEOUT_SECONDS: Final = 2.0
_DEFAULT_DRAIN_TIMEOUT_SECONDS: Final = 2.0
_SENDER_CLOSE_TIMEOUT_SECONDS: Final = 0.5
_MAX_CLIENTS: Final = 64
_MAX_QUEUED_RECORDS: Final = 256
_RECORD_FIELDS: Final = frozenset(
    {
        "logger",
        "level",
        "message",
        "created",
        "account",
        "realm",
        "tab_role",
        "activity",
        "scope",
        "process_id",
        "thread_id",
        "exception",
        "stack",
    }
)
_LEVEL_NUMBERS: Final = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

type FailureCallback = Callable[[str, BaseException], None]
type DeliveryCallback = Callable[[ForwardedRecord], bool]


class ForwardingConfigurationError(RuntimeError):
    """The forwarding capability could not be configured safely."""


class ForwardingProtocolError(RuntimeError):
    """An authenticated peer violated the bounded JSON protocol."""


class ForwardingReceiveError(RuntimeError):
    """An authenticated peer disconnected without a graceful close frame."""


class ForwardingDrainError(RuntimeError):
    """An authenticated peer exceeded the receiver drain deadline."""


class ForwardingSinkError(RuntimeError):
    """A validated record could not be delivered to the parent trace sink."""


@dataclass(frozen=True, slots=True)
class ForwardedRecord:
    """Strict data-only record accepted from an authenticated child."""

    logger: str
    level: str
    message: str
    created: float
    account: str | None
    realm: str | None
    tab_role: str | None
    activity: str | None
    scope: str | None
    process_id: int
    thread_id: int
    exception: str | None
    stack: str | None


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _encode_frame(value: Mapping[str, object]) -> bytes:
    payload = _canonical_json(value)
    if not payload or len(payload) > _MAX_FRAME_BYTES:
        raise ForwardingProtocolError("forwarding frame exceeded its byte bound")
    return struct.pack(">I", len(payload)) + payload


def _receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    received = bytearray()
    while len(received) < byte_count:
        chunk = connection.recv(byte_count - len(received))
        if not chunk:
            raise ForwardingReceiveError("forwarding peer closed during handshake")
        received.extend(chunk)
    return bytes(received)


def _bounded_text(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _optional_bounded_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ForwardingProtocolError("forwarding text field has invalid type")
    if len(value.encode("utf-8")) > limit:
        raise ForwardingProtocolError("forwarding text field exceeded its bound")
    return value


def _bounded_unsigned_integer(
    value: object,
    *,
    maximum: int,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForwardingProtocolError("forwarding integer field has invalid type")
    if not minimum <= value <= maximum:
        raise ForwardingProtocolError("forwarding integer field exceeded its bound")
    return value


def _decode_record(value: object) -> ForwardedRecord:
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise ForwardingProtocolError("forwarding record schema is invalid")

    logger_name = value["logger"]
    if (
        not isinstance(logger_name, str)
        or _LOGGER_PATTERN.fullmatch(logger_name) is None
    ):
        raise ForwardingProtocolError("forwarding logger name is invalid")
    level = value["level"]
    if not isinstance(level, str) or level not in _LEVEL_NUMBERS:
        raise ForwardingProtocolError("forwarding level is invalid")
    message = value["message"]
    if not isinstance(message, str):
        raise ForwardingProtocolError("forwarding message has invalid type")
    if len(message.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise ForwardingProtocolError("forwarding message exceeded its bound")
    created = value["created"]
    if (
        isinstance(created, bool)
        or not isinstance(created, int | float)
        or not math.isfinite(float(created))
        or not 0 <= float(created) <= _MAX_CREATED_TIMESTAMP
    ):
        raise ForwardingProtocolError("forwarding timestamp is invalid")

    return ForwardedRecord(
        logger=logger_name,
        level=level,
        message=message,
        created=float(created),
        account=_optional_bounded_text(value["account"], limit=_MAX_CONTEXT_BYTES),
        realm=_optional_bounded_text(value["realm"], limit=_MAX_CONTEXT_BYTES),
        tab_role=_optional_bounded_text(
            value["tab_role"],
            limit=_MAX_CONTEXT_BYTES,
        ),
        activity=_optional_bounded_text(
            value["activity"],
            limit=_MAX_CONTEXT_BYTES,
        ),
        scope=_optional_bounded_text(value["scope"], limit=_MAX_CONTEXT_BYTES),
        process_id=_bounded_unsigned_integer(
            value["process_id"],
            maximum=_MAX_UINT32,
            minimum=1,
        ),
        thread_id=_bounded_unsigned_integer(
            value["thread_id"],
            maximum=_MAX_UINT64,
            minimum=1,
        ),
        exception=_optional_bounded_text(
            value["exception"],
            limit=_MAX_DETAIL_BYTES,
        ),
        stack=_optional_bounded_text(value["stack"], limit=_MAX_DETAIL_BYTES),
    )


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, raw_port = endpoint.partition(":")
    if separator != ":" or host != _LOOPBACK_HOST or not raw_port.isascii():
        raise ForwardingConfigurationError("forwarding endpoint is invalid")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ForwardingConfigurationError("forwarding endpoint is invalid") from error
    if not 1 <= port <= 65535:
        raise ForwardingConfigurationError("forwarding endpoint is invalid")
    return host, port


def _validate_token(token: str) -> str:
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise ForwardingConfigurationError("forwarding token is invalid")
    return token


class ForwardingHandler(logging.Handler):
    """Queue trace records without putting socket latency on business calls."""

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        failure_callback: FailureCallback,
    ) -> None:
        super().__init__(logging.DEBUG)
        self._token = _validate_token(token)
        self._failure_callback = failure_callback
        self._endpoint = _parse_endpoint(endpoint)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=_MAX_QUEUED_RECORDS)
        self._stop_requested = threading.Event()
        self._stopped = threading.Event()
        self._failure_lock = threading.Lock()
        self._enqueue_lock = threading.Lock()
        self._failure_reported = False
        self._socket: socket.socket | None = None
        self._enabled = True
        self._worker = threading.Thread(
            target=self._run,
            name="hbrowser-log-forwarding-send",
            daemon=True,
        )
        self._worker_survived_fork = True
        self._worker.start()

    @staticmethod
    def _receive_ready(connection: socket.socket) -> None:
        header = _receive_exact(connection, 4)
        frame_bytes = struct.unpack(">I", header)[0]
        if not 0 < frame_bytes <= _MAX_FRAME_BYTES:
            raise ForwardingProtocolError("forwarding ready frame length is invalid")
        payload = _receive_exact(connection, frame_bytes)
        try:
            decoded = json.loads(payload.decode("utf-8"))
            if decoded != {"type": "ready"} or _canonical_json(decoded) != payload:
                raise ForwardingProtocolError("forwarding ready frame is invalid")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RecursionError,
            OverflowError,
        ) as error:
            raise ForwardingProtocolError(
                "forwarding ready frame is invalid"
            ) from error

    @staticmethod
    def _formatted_exception(record: logging.LogRecord) -> str | None:
        if record.exc_info is None:
            return None
        return _bounded_text(
            logging.Formatter().formatException(record.exc_info),
            _MAX_DETAIL_BYTES,
        )

    @staticmethod
    def _context_value(record: logging.LogRecord, field: str) -> str | None:
        value = record.__dict__.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        return _bounded_text(value, _MAX_CONTEXT_BYTES)

    @classmethod
    def _record_payload(cls, record: logging.LogRecord) -> dict[str, object]:
        if _LOGGER_PATTERN.fullmatch(record.name) is None:
            raise ForwardingProtocolError("forwarding logger name is invalid")
        level = record.levelname
        if level not in _LEVEL_NUMBERS:
            raise ForwardingProtocolError("forwarding level is invalid")
        process_id = record.process
        thread_id = record.thread
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or not 1 <= process_id <= _MAX_UINT32
        ):
            raise ForwardingProtocolError("forwarding process id is invalid")
        if (
            isinstance(thread_id, bool)
            or not isinstance(thread_id, int)
            or not 1 <= thread_id <= _MAX_UINT64
        ):
            raise ForwardingProtocolError("forwarding thread id is invalid")
        stack = record.stack_info
        return {
            "logger": record.name,
            "level": level,
            "message": _bounded_text(record.getMessage(), _MAX_MESSAGE_BYTES),
            "created": min(
                max(float(record.created), 0.0),
                _MAX_CREATED_TIMESTAMP,
            ),
            "account": cls._context_value(record, "account"),
            "realm": cls._context_value(record, "realm"),
            "tab_role": cls._context_value(record, "tab_role"),
            "activity": cls._context_value(record, "activity"),
            "scope": cls._context_value(record, "scope"),
            "process_id": process_id,
            "thread_id": thread_id,
            "exception": cls._formatted_exception(record),
            "stack": (
                None if stack is None else _bounded_text(stack, _MAX_DETAIL_BYTES)
            ),
        }

    def _report_failure_once(self, stage: str, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure_reported:
                return
            self._failure_reported = True

        def report() -> None:
            try:
                self._failure_callback(stage, error)
            except Exception:
                pass

        try:
            threading.Thread(
                target=report,
                name="hbrowser-log-forwarding-failure",
                daemon=True,
            ).start()
        except Exception:
            pass

    def _fail(self, stage: str, error: BaseException) -> None:
        self._enabled = False
        self._stop_requested.set()
        self._report_failure_once(stage, error)

    def _run(self) -> None:
        connection: socket.socket | None = None
        failure_stage = "forward-configure"
        try:
            if self._stop_requested.is_set() and self._queue.empty():
                return
            host, port = self._endpoint
            connection = socket.create_connection(
                (host, port),
                timeout=_CONNECT_TIMEOUT_SECONDS,
            )
            connection.settimeout(_CONNECT_TIMEOUT_SECONDS)
            self._socket = connection
            connection.sendall(_encode_frame({"token": self._token, "type": "hello"}))
            self._receive_ready(connection)
            failure_stage = "forward-send"
            while True:
                if self._stop_requested.is_set() and self._queue.empty():
                    failure_stage = "forward-close"
                    connection.sendall(
                        _encode_frame({"token": self._token, "type": "close"})
                    )
                    return
                try:
                    encoded = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                failure_stage = "forward-send"
                connection.sendall(encoded)
        except Exception as error:
            self._fail(failure_stage, error)
        finally:
            self._socket = None
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    connection.close()
                except OSError:
                    pass
            self._stopped.set()

    def emit(self, record: logging.LogRecord) -> None:
        if not self._enabled:
            return
        try:
            encoded = _encode_frame(
                {
                    "record": self._record_payload(record),
                    "token": self._token,
                    "type": "record",
                }
            )
            with self._enqueue_lock:
                if not self._enabled:
                    return
                self._queue.put_nowait(encoded)
        except Exception as error:
            self._fail("forward-send", error)

    def close_forwarding(self) -> None:
        """Request a bounded drain without exposing socket calls to the caller."""
        with self._enqueue_lock:
            self._enabled = False
            self._stop_requested.set()
        if self._worker_survived_fork:
            self._worker.join(timeout=_SENDER_CLOSE_TIMEOUT_SECONDS)
        if not self._stopped.is_set():
            self._report_failure_once(
                "forward-close",
                ForwardingDrainError("forwarding sender exceeded close deadline"),
            )
        logging.Handler.close(self)

    def close(self) -> None:
        self.close_forwarding()

    def _discard_after_fork(self) -> None:
        """Drop an inherited capability without sending on the parent's socket."""
        self._queue = queue.Queue(maxsize=_MAX_QUEUED_RECORDS)
        self._stop_requested = threading.Event()
        self._stopped = threading.Event()
        self._failure_lock = threading.Lock()
        self._enqueue_lock = threading.Lock()
        self._worker_survived_fork = False
        self._enabled = False
        self._stop_requested.set()
        self._stopped.set()
        connection = self._socket
        self._socket = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


@dataclass(eq=False, slots=True)
class _ClientConnection:
    connection: socket.socket
    accepted_at: float
    authenticated: bool = False
    graceful_close: bool = False
    thread: threading.Thread | None = None


class LogForwardingReceiver:
    """Parent-owned authenticated loopback receiver with bounded draining."""

    def __init__(
        self,
        *,
        delivery_callback: DeliveryCallback,
        failure_callback: FailureCallback,
    ) -> None:
        self._delivery_callback = delivery_callback
        self._failure_callback = failure_callback
        self._token = secrets.token_hex(32)
        self._stopping = threading.Event()
        self._stopped = False
        self._discarded = False
        self._drain_deadline: float | None = None
        self._clients: set[_ClientConnection] = set()
        self._clients_lock = threading.Lock()
        self._delivery_lock = threading.Lock()
        self._delivery_enabled = True

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((_LOOPBACK_HOST, 0))
            listener.listen(_MAX_CLIENTS)
            listener.settimeout(_SOCKET_TIMEOUT_SECONDS)
            address = listener.getsockname()
            if address[0] != _LOOPBACK_HOST:
                raise ForwardingConfigurationError(
                    "forwarding receiver did not bind loopback"
                )
        except Exception:
            listener.close()
            raise
        self._listener = listener
        self._endpoint = f"{_LOOPBACK_HOST}:{address[1]}"
        self._accept_thread = threading.Thread(
            target=self._accept_connections,
            name="hbrowser-log-forwarding-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def _report_failure(self, stage: str, error: BaseException) -> None:
        if self._discarded:
            return

        def report() -> None:
            try:
                self._failure_callback(stage, error)
            except Exception:
                pass

        try:
            threading.Thread(
                target=report,
                name="hbrowser-log-forwarding-receiver-failure",
                daemon=True,
            ).start()
        except Exception:
            pass

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def child_environment(self) -> dict[str, str]:
        """Return the capability environment for one explicitly opted-in child."""
        if self._stopped or self._stopping.is_set():
            raise RuntimeError("log forwarding receiver is stopping")
        return {
            FORWARD_ENDPOINT_ENVIRONMENT_VARIABLE: self._endpoint,
            FORWARD_TOKEN_ENVIRONMENT_VARIABLE: self._token,
        }

    def _accept_connections(self) -> None:
        while not self._stopping.is_set():
            try:
                connection, address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError as error:
                if not self._stopping.is_set():
                    self._report_failure("forward-accept", error)
                return
            if address[0] != _LOOPBACK_HOST:
                connection.close()
                continue
            connection.settimeout(_SOCKET_TIMEOUT_SECONDS)
            client = _ClientConnection(connection, time.monotonic())
            client.thread = threading.Thread(
                target=self._serve_client,
                args=(client,),
                name="hbrowser-log-forwarding-client",
                daemon=True,
            )
            with self._clients_lock:
                if len(self._clients) >= _MAX_CLIENTS:
                    connection.close()
                    continue
                self._clients.add(client)
            client.thread.start()

    def _read_exact(self, client: _ClientConnection, byte_count: int) -> bytes:
        received = bytearray()
        while len(received) < byte_count:
            if (
                not client.authenticated
                and time.monotonic() - client.accepted_at
                >= _AUTHENTICATION_TIMEOUT_SECONDS
            ):
                raise ForwardingReceiveError(
                    "forwarding peer did not authenticate before its deadline"
                )
            try:
                chunk = client.connection.recv(byte_count - len(received))
            except TimeoutError:
                if (
                    not client.authenticated
                    and time.monotonic() - client.accepted_at
                    >= _AUTHENTICATION_TIMEOUT_SECONDS
                ):
                    raise ForwardingReceiveError(
                        "forwarding peer did not authenticate before its deadline"
                    ) from None
                deadline = self._drain_deadline
                if (
                    self._stopping.is_set()
                    and deadline is not None
                    and time.monotonic() >= deadline
                ):
                    raise ForwardingDrainError(
                        "forwarding client exceeded drain deadline"
                    ) from None
                continue
            if not chunk:
                raise ForwardingReceiveError(
                    "forwarding client disconnected without close frame"
                )
            received.extend(chunk)
        return bytes(received)

    def _decode_frame(
        self,
        payload: bytes,
        client: _ClientConnection,
    ) -> tuple[bool, str, ForwardedRecord | None]:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            return False, "noise", None
        if not isinstance(decoded, dict):
            return False, "noise", None
        supplied_token = decoded.get("token")
        authenticated = isinstance(supplied_token, str) and hmac.compare_digest(
            supplied_token,
            self._token,
        )
        if not authenticated:
            return False, "noise", None
        # Authentication is established before canonical/schema validation so
        # a peer possessing the capability cannot disguise protocol corruption
        # as unauthenticated network noise.
        client.authenticated = True
        try:
            if _canonical_json(decoded) != payload:
                raise ForwardingProtocolError("forwarding JSON is not canonical")
            frame_type = decoded.get("type")
            if frame_type == "hello":
                if set(decoded) != {"token", "type"}:
                    raise ForwardingProtocolError("forwarding hello schema is invalid")
                return True, "hello", None
            if frame_type == "close":
                if set(decoded) != {"token", "type"}:
                    raise ForwardingProtocolError("forwarding close schema is invalid")
                return True, "close", None
            if frame_type == "record":
                if set(decoded) != {"record", "token", "type"}:
                    raise ForwardingProtocolError("forwarding frame schema is invalid")
                return True, "record", _decode_record(decoded["record"])
            raise ForwardingProtocolError("forwarding frame type is invalid")
        except (TypeError, ValueError, RecursionError, OverflowError) as error:
            if isinstance(error, ForwardingProtocolError):
                raise
            raise ForwardingProtocolError(
                "forwarding canonical JSON is invalid"
            ) from error

    def _serve_client(self, client: _ClientConnection) -> None:
        try:
            while True:
                try:
                    header = self._read_exact(client, 4)
                    frame_bytes = struct.unpack(">I", header)[0]
                    if not 0 < frame_bytes <= _MAX_FRAME_BYTES:
                        if client.authenticated:
                            raise ForwardingProtocolError(
                                "forwarding frame length is invalid"
                            )
                        return
                    payload = self._read_exact(client, frame_bytes)
                    authenticated, frame_type, record = self._decode_frame(
                        payload,
                        client,
                    )
                    if not authenticated:
                        return
                    if frame_type == "close":
                        client.graceful_close = True
                        return
                    if frame_type == "hello":
                        try:
                            client.connection.sendall(_encode_frame({"type": "ready"}))
                        except Exception as error:
                            raise ForwardingReceiveError(
                                "forwarding ready receipt could not be sent"
                            ) from error
                        continue
                    if record is None:
                        raise ForwardingProtocolError(
                            "forwarding record frame is empty"
                        )
                    with self._delivery_lock:
                        if not self._delivery_enabled:
                            raise ForwardingDrainError(
                                "forwarding delivery is already closed"
                            )
                        try:
                            delivered = self._delivery_callback(record)
                        except Exception as error:
                            raise ForwardingSinkError(
                                "parent trace delivery raised"
                            ) from error
                    if not delivered:
                        raise ForwardingSinkError(
                            "parent trace sink rejected forwarded record"
                        )
                except ForwardingProtocolError as error:
                    if client.authenticated:
                        self._report_failure("forward-protocol", error)
                    return
                except ForwardingDrainError as error:
                    if client.authenticated:
                        self._report_failure("forward-drain", error)
                    return
                except ForwardingSinkError as error:
                    self._report_failure("forward-sink", error)
                    return
                except (OSError, ForwardingReceiveError) as error:
                    if client.authenticated:
                        stage = (
                            "forward-drain"
                            if self._stopping.is_set()
                            else "forward-receive"
                        )
                        self._report_failure(stage, error)
                    return
        finally:
            try:
                client.connection.close()
            except OSError:
                pass
            with self._clients_lock:
                self._clients.discard(client)

    def close(self, *, drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT_SECONDS) -> None:
        """Stop accepting, boundedly drain clients, then disable delivery."""
        if self._stopped:
            return
        if (
            isinstance(drain_timeout, bool)
            or not isinstance(drain_timeout, int | float)
            or not math.isfinite(float(drain_timeout))
            or drain_timeout <= 0
        ):
            raise ValueError("forwarding drain timeout must be finite and positive")
        self._stopping.set()
        self._drain_deadline = time.monotonic() + float(drain_timeout)

        def remaining_budget() -> float:
            deadline = self._drain_deadline
            if deadline is None:
                return 0.0
            return max(0.0, deadline - time.monotonic())

        try:
            self._listener.close()
        except OSError:
            pass
        self._accept_thread.join(timeout=min(remaining_budget(), 0.5))

        while remaining_budget() > 0:
            with self._clients_lock:
                active = tuple(self._clients)
            if not active:
                break
            time.sleep(min(0.01, remaining_budget()))

        # This flag is intentionally set without waiting for _delivery_lock.
        # A delivery callback is arbitrary application code and may never
        # return. Existing callbacks remain visible as live client threads;
        # waiting callbacks observe the disabled flag after acquiring the lock.
        self._delivery_enabled = False
        with self._clients_lock:
            remaining = tuple(self._clients)
        for client in remaining:
            try:
                client.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for client in remaining:
            thread = client.thread
            if thread is not None:
                budget = remaining_budget()
                if budget <= 0:
                    break
                thread.join(timeout=budget)

        with self._clients_lock:
            still_running = tuple(
                client
                for client in self._clients
                if client.thread is not None and client.thread.is_alive()
            )
        if still_running:
            drain_error = ForwardingDrainError("forwarding client thread did not stop")
            self._report_failure("forward-drain", drain_error)
            self._discarded = True
            self._stopped = True
            raise drain_error
        self._discarded = True
        self._stopped = True

    def _discard_after_fork(self) -> None:
        """Close inherited descriptors in a fork child whose threads vanished."""
        self._discarded = True
        self._delivery_enabled = False
        try:
            self._listener.close()
        except OSError:
            pass
        for client in tuple(self._clients):
            try:
                client.connection.close()
            except OSError:
                pass
        self._stopped = True


__all__ = [
    "FORWARD_ENDPOINT_ENVIRONMENT_VARIABLE",
    "FORWARD_TOKEN_ENVIRONMENT_VARIABLE",
    "ForwardedRecord",
    "ForwardingConfigurationError",
    "ForwardingHandler",
    "LogForwardingReceiver",
]
