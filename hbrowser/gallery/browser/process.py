"""Own browser-related process trees without sharing terminal signals."""

from __future__ import annotations

import atexit
import ctypes
import math
import ntpath
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

_STARTUP_TIMEOUT_SECONDS: Final = 10.0
_PROCESS_WAIT_TIMEOUT_SECONDS: Final = 15.0
_SHUTDOWN_PHASE_TIMEOUT_SECONDS: Final = 5.0
_SHUTDOWN_TOTAL_TIMEOUT_SECONDS: Final = 15.0
_STATUS_POLL_SECONDS: Final = 0.01
_PRIVATE_CLEANUP_TIMEOUT_SECONDS: Final = 5.0
_PRIVATE_CLEANUP_KILL_RESERVE_SECONDS: Final = 1.0
_PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS: Final = 0.02
_PRIVATE_CLEANUP_RETRY_MAX_SECONDS: Final = 0.25
_WINDOWS_RETRYABLE_PRIVATE_CLEANUP_ERRORS: Final = frozenset({32, 33})
_OUTPUT_RING_BYTES: Final = 256 * 1024
_SUPERVISOR_READY_PREFIX: Final = "ready "
_SUPERVISOR_ERROR_PREFIX: Final = "error "


class ProcessOwnershipError(RuntimeError):
    """A process tree or its private material remains under unresolved ownership."""


@dataclass(slots=True)
class _ProvisionalProcessOwner:
    """Durable ownership from supervisor Popen until OwnedProcess takes over."""

    supervisor: subprocess.Popen[bytes]
    status_guard: _PrivateDirectory
    cleanup_guards: tuple[_PrivateDirectory, ...]
    windows_job: _WindowsJob | None = None


_PROVISIONAL_OWNERS: dict[int, _ProvisionalProcessOwner] = {}
_PROVISIONAL_OWNERS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _SupervisorReady:
    target_pid: int


@dataclass(frozen=True, slots=True)
class _SupervisorTargetNotStarted:
    error_type: str


type _SupervisorStatus = _SupervisorReady | _SupervisorTargetNotStarted


@dataclass(frozen=True, slots=True)
class _PrivateDirectory:
    path: Path
    device: int
    inode: int
    platform_name: str

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        platform_name: str | None = None,
    ) -> _PrivateDirectory:
        path = path.absolute()
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProcessOwnershipError(
                "Private generation path is not an owned directory"
            )
        return cls(
            path,
            metadata.st_dev,
            metadata.st_ino,
            _ownership_platform() if platform_name is None else platform_name,
        )

    def _assert_identity_or_absence(self) -> bool:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_dev != self.device
            or metadata.st_ino != self.inode
        ):
            raise ProcessOwnershipError(
                "Private generation directory identity changed before cleanup"
            )
        return True

    def _is_retryable_error(self, error: OSError) -> bool:
        return (
            self.platform_name == "nt"
            and getattr(error, "winerror", None)
            in _WINDOWS_RETRYABLE_PRIVATE_CLEANUP_ERRORS
        )

    def _deadline_error(
        self,
        error: OSError,
        *,
        attempts: int,
    ) -> ProcessOwnershipError:
        timeout_error = ProcessOwnershipError(
            f"Private generation directory cleanup timed out: {self.path}"
        )
        timeout_error.add_note(
            "Last Windows cleanup error: "
            f"winerror={getattr(error, 'winerror', None)} attempts={attempts}"
        )
        return timeout_error

    def remove(self, *, deadline: float) -> None:
        """Remove this directory through a killable, durably owned worker."""

        _remove_private_directory_owned(self, deadline=deadline)

    def _remove_inline(self, *, deadline: float) -> None:
        """Worker-only removal with per-attempt identity validation.

        Windows can keep a just-exited Chromium database handle alive briefly.
        Retry only the two native sharing/lock violations, revalidating the root
        identity before every attempt and retaining ownership when the deadline
        expires.
        """

        retry_delay = _PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
        last_retryable_error: OSError | None = None
        attempts = 0
        while True:
            if time.monotonic() >= deadline:
                if last_retryable_error is not None:
                    raise self._deadline_error(
                        last_retryable_error,
                        attempts=attempts,
                    ) from last_retryable_error
                raise ProcessOwnershipError(
                    "Private generation directory cleanup deadline expired before "
                    "filesystem mutation"
                )
            if not self._assert_identity_or_absence():
                return
            if time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    "Private generation directory identity proof completed after its "
                    "cleanup deadline"
                )
            try:
                attempts += 1
                shutil.rmtree(self.path)
            except FileNotFoundError:
                return
            except OSError as error:
                if not self._is_retryable_error(error):
                    raise
                last_retryable_error = error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._deadline_error(
                        error,
                        attempts=attempts,
                    ) from error
                time.sleep(min(retry_delay, remaining))
                retry_delay = min(
                    retry_delay * 2,
                    _PRIVATE_CLEANUP_RETRY_MAX_SECONDS,
                )
                continue

            if not self._assert_identity_or_absence():
                if time.monotonic() >= deadline:
                    raise ProcessOwnershipError(
                        "Private generation directory was removed after its deadline"
                    )
                return
            raise ProcessOwnershipError(
                "Private generation directory remained after cleanup"
            )


class _BoundedPipeDrain:
    """Continuously drain one pipe while retaining a bounded diagnostic tail."""

    def __init__(self, pipe: BinaryIO, *, limit: int = _OUTPUT_RING_BYTES) -> None:
        if limit <= 0:
            raise ValueError("pipe drain limit must be positive")
        self._pipe = pipe
        self._limit = limit
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="hbrowser-process-output-drain",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while chunk := self._pipe.read(64 * 1024):
                with self._lock:
                    self._chunks.append(chunk)
                    self._size += len(chunk)
                    while self._size > self._limit and self._chunks:
                        removed = self._chunks.popleft()
                        self._size -= len(removed)
        finally:
            try:
                self._pipe.close()
            except OSError:
                pass

    def tail(self) -> bytes:
        with self._lock:
            joined = b"".join(self._chunks)
        return joined[-self._limit :]


class _WindowsJob:
    """A non-inheritable Windows Job with kill-on-owner-close semantics."""

    _KILL_ON_JOB_CLOSE: Final = 0x00002000
    _EXTENDED_LIMIT_INFORMATION: Final = 9
    _BASIC_ACCOUNTING_INFORMATION: Final = 1

    def __init__(
        self,
        kernel32: Any,
        handle: int,
        accounting_information_type: type[ctypes.Structure],
    ) -> None:
        self._kernel32 = kernel32
        self._handle: int | None = handle
        self._accounting_information_type = accounting_information_type
        self._lock = threading.Lock()

    @classmethod
    def create(cls) -> _WindowsJob:
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects are unavailable on this platform")

        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        win_dll = getattr(ctypes, "WinDLL", None)
        if not callable(win_dll):
            raise RuntimeError("Windows Job Object APIs are unavailable")
        kernel32 = win_dll("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        raw_handle = kernel32.CreateJobObjectW(None, None)
        if not raw_handle:
            raise cls._last_error("CreateJobObjectW")
        handle = int(raw_handle)
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = cls._KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            cls._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = cls._last_error("SetInformationJobObject")
            kernel32.CloseHandle(handle)
            raise error
        return cls(kernel32, handle, _BasicAccountingInformation)

    @staticmethod
    def _last_error(operation: str) -> OSError:
        get_last_error = cast(Any, getattr(ctypes, "get_last_error", lambda: 0))
        code = int(get_last_error())
        return OSError(code, f"{operation} failed with Windows error {code}")

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise RuntimeError("Python did not expose the child process handle")
        with self._lock:
            if self._handle is None:
                raise RuntimeError("Windows Job Object is already closed")
            if not self._kernel32.AssignProcessToJobObject(
                self._handle,
                int(process_handle),
            ):
                raise self._last_error("AssignProcessToJobObject")

    def terminate(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            if not self._kernel32.TerminateJobObject(self._handle, 1):
                get_last_error = cast(
                    Any,
                    getattr(ctypes, "get_last_error", lambda: 0),
                )
                code = int(get_last_error())
                raise OSError(
                    code,
                    f"TerminateJobObject failed with Windows error {code}",
                )

    def active_processes(self) -> int:
        with self._lock:
            if self._handle is None:
                return 0
            information = self._accounting_information_type()
            if not self._kernel32.QueryInformationJobObject(
                self._handle,
                self._BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
                None,
            ):
                raise self._last_error("QueryInformationJobObject")
            return int(information.ActiveProcesses)

    def wait_empty(self, *, timeout: float | None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.active_processes() != 0:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Windows Job process tree did not terminate")
            time.sleep(_STATUS_POLL_SECONDS)

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            if not self._kernel32.CloseHandle(handle):
                raise self._last_error("CloseHandle")
            self._handle = None


def _register_provisional_owner(owner: _ProvisionalProcessOwner) -> None:
    with _PROVISIONAL_OWNERS_LOCK:
        _PROVISIONAL_OWNERS[id(owner)] = owner


def _release_provisional_owner(owner: _ProvisionalProcessOwner) -> None:
    with _PROVISIONAL_OWNERS_LOCK:
        _PROVISIONAL_OWNERS.pop(id(owner), None)


def _cleanup_provisional_owner(
    owner: _ProvisionalProcessOwner,
    *,
    deadline: float,
) -> None:
    """Reap a start-gated supervisor that never transferred ownership."""

    cleanup_deadline = min(
        deadline,
        time.monotonic() + _PRIVATE_CLEANUP_TIMEOUT_SECONDS,
    )
    supervisor = owner.supervisor
    windows_job = owner.windows_job
    if windows_job is not None:
        windows_job.terminate()
        windows_job.wait_empty(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    if supervisor.poll() is None:
        try:
            supervisor.terminate()
        except ProcessLookupError:
            pass
        try:
            supervisor.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                supervisor.kill()
            except ProcessLookupError:
                pass
            supervisor.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    if supervisor.poll() is None:
        raise ProcessOwnershipError(
            "Start-gated process supervisor could not be reaped"
        )
    if windows_job is not None:
        windows_job.close()
        owner.windows_job = None
    try:
        if supervisor.stdin is not None:
            supervisor.stdin.close()
    except OSError:
        pass
    for guard in (owner.status_guard, *owner.cleanup_guards):
        guard.remove(deadline=cleanup_deadline)
    _release_provisional_owner(owner)


def _cleanup_provisional_owners_at_exit() -> None:
    deadline = time.monotonic() + 15.0
    with _PROVISIONAL_OWNERS_LOCK:
        owners = tuple(_PROVISIONAL_OWNERS.values())
    for owner in owners:
        try:
            _cleanup_provisional_owner(owner, deadline=deadline)
        except BaseException:
            pass


atexit.register(_cleanup_provisional_owners_at_exit)


class OwnedProcess:
    """A minimal Popen-compatible facade that owns the complete target tree."""

    def __init__(
        self,
        supervisor: subprocess.Popen[bytes],
        *,
        target_process_group: int | None,
        windows_job: _WindowsJob | None,
        stdout_drain: _BoundedPipeDrain | None,
        stderr_drain: _BoundedPipeDrain | None,
        status_directory: Path | _PrivateDirectory,
        cleanup_paths: Sequence[Path | _PrivateDirectory],
    ) -> None:
        self._supervisor = supervisor
        self._target_process_group = target_process_group
        self._windows_job = windows_job
        self._stdout_drain = stdout_drain
        self._stderr_drain = stderr_drain
        self._status_directory: _PrivateDirectory | None = (
            status_directory
            if isinstance(status_directory, _PrivateDirectory)
            else _PrivateDirectory.capture(status_directory)
        )
        self._cleanup_paths = [
            (
                path
                if isinstance(path, _PrivateDirectory)
                else _PrivateDirectory.capture(path)
            )
            for path in cleanup_paths
        ]
        self._state_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_attempt_lock = threading.Lock()
        self._shutdown_attempt_deadline: float | None = None
        self._shutdown_attempt_users = 0
        self._supervisor_reaped = False
        self._supervisor_returncode: int | None = None
        self._target_not_started = False
        self._tree_reaped = False
        self._closed = False
        atexit.register(self._atexit_cleanup)

    @property
    def pid(self) -> int:
        return self._supervisor.pid

    @property
    def target_pid(self) -> int | None:
        return self._target_process_group

    def bind_target_process_group(self, target_pid: int) -> None:
        if target_pid <= 0:
            raise ValueError("target_pid must be positive")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot bind a closed process owner")
            if self._target_process_group is not None:
                raise RuntimeError("Target process identity is already bound")
            if self._target_not_started:
                raise RuntimeError("Target absence is already established")
            self._target_process_group = target_pid

    def bind_target_not_started(self) -> None:
        """Record the supervisor's atomic proof that no target was created."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot bind a closed process owner")
            if self._target_process_group is not None:
                raise RuntimeError("Target process identity is already bound")
            self._target_not_started = True

    @property
    def stdin(self) -> None:
        return None

    @property
    def stdout(self) -> BinaryIO | None:
        if self._stdout_drain is not None:
            return None
        return cast(BinaryIO | None, self._supervisor.stdout)

    @property
    def stderr(self) -> BinaryIO | None:
        if self._stderr_drain is not None:
            return None
        return cast(BinaryIO | None, self._supervisor.stderr)

    @property
    def returncode(self) -> int | None:
        return self.poll()

    def poll(self) -> int | None:
        with self._state_lock:
            returncode = self._supervisor.poll()
            if returncode is not None:
                self._supervisor_reaped = True
                self._supervisor_returncode = returncode
            return returncode

    def diagnostic_tail(self) -> bytes:
        parts = []
        if self._stdout_drain is not None:
            parts.append(self._stdout_drain.tail())
        if self._stderr_drain is not None:
            parts.append(self._stderr_drain.tail())
        return b"\n".join(part for part in parts if part)

    def begin_output_draining(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cannot drain output from a closed process owner")
            if self._stdout_drain is not None or self._stderr_drain is not None:
                raise RuntimeError("Process output draining is already active")
            if self._supervisor.stdout is not None:
                self._stdout_drain = _BoundedPipeDrain(
                    cast(BinaryIO, self._supervisor.stdout)
                )
            if self._supervisor.stderr is not None:
                self._stderr_drain = _BoundedPipeDrain(
                    cast(BinaryIO, self._supervisor.stderr)
                )

    @contextmanager
    def _state_before(
        self,
        *,
        deadline: float,
        phase: str,
        allow_expired_immediate_check: bool = False,
    ) -> Iterator[None]:
        remaining = max(0.0, deadline - time.monotonic())
        acquired = self._state_lock.acquire(timeout=remaining)
        if not acquired:
            raise ProcessOwnershipError(
                f"Process ownership deadline expired waiting for {phase} state"
            )
        if time.monotonic() >= deadline and not allow_expired_immediate_check:
            self._state_lock.release()
            raise ProcessOwnershipError(
                f"Process ownership deadline expired while acquiring {phase} state"
            )
        try:
            yield
        finally:
            self._state_lock.release()

    def _request_supervisor_shutdown(
        self,
        command: bytes,
        *,
        deadline: float,
    ) -> None:
        if time.monotonic() >= deadline:
            raise ProcessOwnershipError(
                "Process shutdown deadline expired before supervisor control write"
            )
        returncode = self._supervisor.returncode
        if returncode is not None:
            with self._state_before(
                deadline=deadline,
                phase="supervisor exit receipt",
            ):
                self._supervisor_reaped = True
                self._supervisor_returncode = returncode
            return
        control_pipe = self._supervisor.stdin
        if control_pipe is None:
            return
        try:
            descriptor = control_pipe.fileno()
            if isinstance(descriptor, int):
                os.set_blocking(descriptor, False)
                if os.write(descriptor, command) != len(command):
                    raise BlockingIOError("partial supervisor control write")
            else:
                # OwnedProcess is only constructed with Popen pipes in
                # production. This branch keeps structural test doubles useful.
                control_pipe.write(command)
                control_pipe.flush()
        except OSError:
            # A failed control channel provides no safe basis for signalling a
            # cached numeric PID. wait() will either obtain the supervisor's
            # proof or fail closed while retaining private material.
            return
        if time.monotonic() >= deadline:
            raise ProcessOwnershipError(
                "Supervisor control receipt arrived after the shutdown deadline"
            )

    def terminate(self, *, deadline: float | None = None) -> None:
        control_deadline = (
            time.monotonic() + _SHUTDOWN_PHASE_TIMEOUT_SECONDS
            if deadline is None
            else deadline
        )
        with self._state_before(
            deadline=control_deadline,
            phase="terminate",
        ):
            if self._closed or self._tree_reaped:
                return
        # Never hold the owner state lock across a potentially backpressured
        # control channel. The real Popen descriptor is always nonblocking.
        self._request_supervisor_shutdown(
            b"terminate\n",
            deadline=control_deadline,
        )

    def kill(self, *, deadline: float | None = None) -> None:
        control_deadline = (
            time.monotonic() + _SHUTDOWN_PHASE_TIMEOUT_SECONDS
            if deadline is None
            else deadline
        )
        with self._state_before(
            deadline=control_deadline,
            phase="kill",
        ):
            if self._closed or self._tree_reaped:
                return
            windows_job = self._windows_job
        if windows_job is not None:
            windows_job.terminate()
            if time.monotonic() >= control_deadline:
                raise ProcessOwnershipError(
                    "Windows Job termination completed after the shutdown deadline"
                )
            return
        self._request_supervisor_shutdown(
            b"kill\n",
            deadline=control_deadline,
        )

    def _join_shutdown_attempt(
        self,
        caller_deadline: float,
        *,
        allow_expired_immediate_check: bool = False,
    ) -> float:
        """Join concurrent ownership calls to one non-resetting deadline."""

        remaining = max(0.0, caller_deadline - time.monotonic())
        acquired = self._shutdown_attempt_lock.acquire(timeout=remaining)
        if not acquired:
            raise ProcessOwnershipError(
                "Process ownership deadline expired joining shutdown attempt"
            )
        try:
            if (
                time.monotonic() >= caller_deadline
                and not allow_expired_immediate_check
            ):
                raise ProcessOwnershipError(
                    "Process ownership deadline expired while joining shutdown "
                    "attempt"
                )
            attempt_deadline = self._shutdown_attempt_deadline
            if attempt_deadline is None:
                attempt_deadline = caller_deadline
                self._shutdown_attempt_deadline = attempt_deadline
            self._shutdown_attempt_users += 1
        finally:
            self._shutdown_attempt_lock.release()
        return min(caller_deadline, attempt_deadline)

    def _leave_shutdown_attempt(self) -> None:
        with self._shutdown_attempt_lock:
            self._shutdown_attempt_users -= 1
            if self._shutdown_attempt_users < 0:
                raise ProcessOwnershipError(
                    "Process shutdown attempt accounting became inconsistent"
                )
            if self._shutdown_attempt_users == 0:
                self._shutdown_attempt_deadline = None

    def _acquire_shutdown_lock(
        self,
        *,
        deadline: float,
        allow_expired_immediate_check: bool = False,
    ) -> None:
        remaining = max(0.0, deadline - time.monotonic())
        acquired = self._shutdown_lock.acquire(timeout=remaining)
        if not acquired:
            raise ProcessOwnershipError(
                "Process ownership deadline expired waiting for concurrent cleanup"
            )
        if time.monotonic() >= deadline and not allow_expired_immediate_check:
            self._shutdown_lock.release()
            raise ProcessOwnershipError(
                "Process ownership deadline expired while waiting for concurrent "
                "cleanup"
            )

    def wait(self, timeout: float | None = None) -> int:
        """Prove tree exit and ownership release within one bounded deadline."""

        if timeout is None:
            wait_timeout = _PROCESS_WAIT_TIMEOUT_SECONDS
        else:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, int | float)
                or not math.isfinite(float(timeout))
                or not 0 <= timeout <= _PROCESS_WAIT_TIMEOUT_SECONDS
            ):
                raise ValueError(
                    "process wait timeout must be finite and in [0, "
                    f"{_PROCESS_WAIT_TIMEOUT_SECONDS:g}]"
                )
            wait_timeout = float(timeout)

        deadline = self._join_shutdown_attempt(
            time.monotonic() + wait_timeout,
            allow_expired_immediate_check=wait_timeout == 0,
        )
        acquired = False
        try:
            self._acquire_shutdown_lock(
                deadline=deadline,
                allow_expired_immediate_check=wait_timeout == 0,
            )
            acquired = True
            with self._state_before(
                deadline=deadline,
                phase="cached wait result",
                allow_expired_immediate_check=wait_timeout == 0,
            ):
                if self._closed:
                    return self._supervisor_returncode or 0
            if time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    "Process wait deadline expired before process-tree proof"
                )
            returncode = self._wait_for_process_tree(
                deadline=deadline,
                allow_expired_immediate_check=wait_timeout == 0,
            )
            if time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    "Process wait deadline expired before private cleanup"
                )
            self._release_ownership(deadline=deadline)
            if wait_timeout > 0 and time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    "Process ownership was released after the caller's wait deadline"
                )
            return returncode
        finally:
            if acquired:
                self._shutdown_lock.release()
            self._leave_shutdown_attempt()

    def _wait_for_process_tree(
        self,
        *,
        deadline: float,
        state_deadline: float | None = None,
        allow_expired_immediate_check: bool = False,
    ) -> int:
        """Advance only the process-proof phase of the ownership state machine."""

        coordination_deadline = deadline if state_deadline is None else state_deadline
        with self._state_before(
            deadline=coordination_deadline,
            phase="process-tree snapshot",
            allow_expired_immediate_check=allow_expired_immediate_check,
        ):
            if self._closed:
                return self._supervisor_returncode or 0
            supervisor_reaped = self._supervisor_reaped
            returncode = self._supervisor_returncode

        if supervisor_reaped:
            if returncode is None:
                raise ProcessOwnershipError(
                    "Process supervisor reaping state is inconsistent"
                )
        else:
            remaining = max(0.0, deadline - time.monotonic())
            returncode = self._supervisor.wait(timeout=remaining)
            if time.monotonic() >= deadline and not allow_expired_immediate_check:
                raise TimeoutError(
                    "Process supervisor exit receipt arrived after its phase deadline"
                )
            with self._state_before(
                deadline=coordination_deadline,
                phase="supervisor reap receipt",
                allow_expired_immediate_check=allow_expired_immediate_check,
            ):
                self._supervisor_reaped = True
                self._supervisor_returncode = returncode

        with self._state_before(
            deadline=coordination_deadline,
            phase="target-tree snapshot",
            allow_expired_immediate_check=allow_expired_immediate_check,
        ):
            if self._tree_reaped:
                return returncode
            windows_job = self._windows_job
            target_process_group = self._target_process_group
            target_not_started = self._target_not_started

        if windows_job is not None:
            remaining = max(0.0, deadline - time.monotonic())
            windows_job.wait_empty(timeout=remaining)
            if time.monotonic() >= deadline and not allow_expired_immediate_check:
                raise TimeoutError(
                    "Windows Job exit receipt arrived after its phase deadline"
                )
        elif target_process_group is not None:
            if returncode != 0:
                raise ProcessOwnershipError(
                    "Process supervisor did not prove target-tree cleanup"
                )
        elif not target_not_started:
            raise ProcessOwnershipError(
                "Process supervisor exited before target cleanup was proven"
            )

        with self._state_before(
            deadline=coordination_deadline,
            phase="target-tree receipt",
            allow_expired_immediate_check=allow_expired_immediate_check,
        ):
            self._tree_reaped = True
        return returncode

    def shutdown(
        self,
        *,
        graceful_timeout: float,
        terminate_timeout: float,
        kill_timeout: float,
        cleanup_timeout: float = _PRIVATE_CLEANUP_TIMEOUT_SECONDS,
        deadline: float | None = None,
    ) -> int:
        """Close one owned generation through monotonic, bounded phases.

        Natural exit gets the first opportunity so Chromium can flush its
        profile. Only a process-tree timeout advances to supervisor terminate,
        then to the Windows Job / POSIX force-kill path. Private material is
        released last and never causes an already-proven tree to be signalled
        again.
        """

        timeouts = (
            graceful_timeout,
            terminate_timeout,
            kill_timeout,
            cleanup_timeout,
        )
        if any(
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or not 0 <= timeout <= _SHUTDOWN_PHASE_TIMEOUT_SECONDS
            for timeout in timeouts
        ):
            raise ValueError(
                "Process shutdown phase timeouts must be finite and in [0, "
                f"{_SHUTDOWN_PHASE_TIMEOUT_SECONDS:g}]"
            )
        started_at = time.monotonic()
        policy_deadline = started_at + min(
            _SHUTDOWN_TOTAL_TIMEOUT_SECONDS,
            sum(float(timeout) for timeout in timeouts),
        )
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, int | float):
                raise TypeError("Process shutdown deadline must be a real number")
            if not math.isfinite(float(deadline)):
                raise ValueError("Process shutdown deadline must be finite")
            policy_deadline = min(policy_deadline, float(deadline))
        immediate_check = not any(timeout > 0 for timeout in timeouts)
        deadline = self._join_shutdown_attempt(
            policy_deadline,
            allow_expired_immediate_check=immediate_check,
        )

        def phase_deadline(timeout: float) -> float:
            candidate = time.monotonic() + timeout
            return min(candidate, deadline)

        def require_overall_budget(phase: str) -> None:
            if time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    f"Process shutdown deadline expired before {phase}"
                )

        acquired = False
        try:
            self._acquire_shutdown_lock(
                deadline=deadline,
                allow_expired_immediate_check=immediate_check,
            )
            acquired = True
            with self._state_before(
                deadline=deadline,
                phase="cached shutdown result",
                allow_expired_immediate_check=immediate_check,
            ):
                if self._closed:
                    return self._supervisor_returncode or 0
                tree_reaped = self._tree_reaped

            if time.monotonic() >= deadline and any(
                timeout > 0 for timeout in timeouts
            ):
                raise ProcessOwnershipError(
                    "Process shutdown deadline expired before process-tree proof"
                )

            if not tree_reaped:
                try:
                    returncode = self._wait_for_process_tree(
                        deadline=phase_deadline(graceful_timeout),
                        state_deadline=deadline,
                        allow_expired_immediate_check=graceful_timeout == 0,
                    )
                except subprocess.TimeoutExpired, TimeoutError:
                    require_overall_budget("terminate")
                    self.terminate(deadline=deadline)
                    try:
                        returncode = self._wait_for_process_tree(
                            deadline=phase_deadline(terminate_timeout),
                            state_deadline=deadline,
                            allow_expired_immediate_check=terminate_timeout == 0,
                        )
                    except subprocess.TimeoutExpired, TimeoutError:
                        require_overall_budget("kill")
                        self.kill(deadline=deadline)
                        returncode = self._wait_for_process_tree(
                            deadline=phase_deadline(kill_timeout),
                            state_deadline=deadline,
                            allow_expired_immediate_check=kill_timeout == 0,
                        )
            else:
                with self._state_before(
                    deadline=deadline,
                    phase="reaped return code",
                ):
                    known_returncode = self._supervisor_returncode
                if known_returncode is None:
                    raise ProcessOwnershipError(
                        "Process-tree proof has no supervisor return code"
                    )
                returncode = known_returncode

            # A zero-second cleanup phase still permits one immediate,
            # non-waiting ownership reconciliation.  Positive cleanup work
            # must not begin after the shared overall deadline.
            if cleanup_timeout > 0:
                require_overall_budget("private cleanup")
            self._release_ownership(deadline=phase_deadline(cleanup_timeout))
            if not immediate_check and time.monotonic() >= deadline:
                raise ProcessOwnershipError(
                    "Process ownership was released after the shutdown deadline"
                )
            return returncode
        finally:
            if acquired:
                self._shutdown_lock.release()
            self._leave_shutdown_attempt()

    def _release_ownership(self, *, deadline: float) -> None:
        """Release private paths and the Job only after tree exit is proven."""

        with self._state_before(
            deadline=deadline,
            phase="private-release precondition",
        ):
            if self._closed:
                return
            if not self._tree_reaped:
                raise ProcessOwnershipError(
                    "Private material cannot be released before process-tree cleanup"
                )
        try:
            if self._supervisor.stdin is not None:
                self._supervisor.stdin.close()
        except OSError:
            pass
        with self._state_before(
            deadline=deadline,
            phase="status-directory snapshot",
        ):
            status_directory = self._status_directory
        if status_directory is not None:
            status_directory.remove(deadline=deadline)
            with self._state_before(
                deadline=deadline,
                phase="status-directory receipt",
            ):
                self._status_directory = None
        while True:
            with self._state_before(
                deadline=deadline,
                phase="private-directory snapshot",
            ):
                cleanup_path = self._cleanup_paths[0] if self._cleanup_paths else None
            if cleanup_path is None:
                break
            cleanup_path.remove(deadline=deadline)
            with self._state_before(
                deadline=deadline,
                phase="private-directory receipt",
            ):
                if (
                    not self._cleanup_paths
                    or self._cleanup_paths[0] is not cleanup_path
                ):
                    raise ProcessOwnershipError(
                        "Private cleanup ownership state changed unexpectedly"
                    )
                self._cleanup_paths.pop(0)
        with self._state_before(
            deadline=deadline,
            phase="Windows Job snapshot",
        ):
            windows_job = self._windows_job
        if windows_job is not None:
            windows_job.close()
            with self._state_before(
                deadline=deadline,
                phase="Windows Job release receipt",
            ):
                if self._windows_job is not windows_job:
                    raise ProcessOwnershipError(
                        "Windows Job ownership state changed unexpectedly"
                    )
                self._windows_job = None
        with self._state_before(
            deadline=deadline,
            phase="process ownership receipt",
        ):
            self._closed = True
        try:
            atexit.unregister(self._atexit_cleanup)
        except Exception:
            pass

    def _atexit_cleanup(self) -> None:
        with self._state_lock:
            if self._closed:
                return
        try:
            self.shutdown(
                graceful_timeout=0,
                terminate_timeout=5,
                kill_timeout=5,
                cleanup_timeout=_PRIVATE_CLEANUP_TIMEOUT_SECONDS,
            )
        except BaseException:
            pass


def _ownership_platform() -> str:
    if os.name in {"nt", "posix"}:
        return os.name
    raise RuntimeError(f"Unsupported process ownership platform: {os.name}")


def _supervisor_creation_options() -> dict[str, Any]:
    platform_name = _ownership_platform()
    if platform_name == "posix":
        return {"start_new_session": True}
    if platform_name == "nt":
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", None)
        if not isinstance(creation_flag, int) or creation_flag <= 0:
            raise RuntimeError("Windows process-group isolation is unavailable")
        return {"creationflags": creation_flag}
    raise AssertionError("unreachable process ownership platform")


def _supervisor_launch_context(
    platform_name: str,
) -> tuple[str, dict[str, str] | None]:
    """Select an interpreter whose process identity can be owned directly."""

    executable = sys.executable
    if not isinstance(executable, str) or not executable:
        raise ProcessOwnershipError("Python did not expose its executable path")
    if platform_name != "nt":
        return executable, None

    base_executable = getattr(sys, "_base_executable", None)
    if not isinstance(base_executable, str) or not base_executable:
        raise ProcessOwnershipError(
            "Windows Python did not expose its base executable path"
        )
    executable_identity = ntpath.normcase(ntpath.normpath(executable))
    base_identity = ntpath.normcase(ntpath.normpath(base_executable))
    if executable_identity == base_identity:
        return executable, None

    # A Windows venv's python.exe is a redirector process. Launching it would
    # make Popen and the Job own that short-lived redirector rather than the
    # real supervisor. This is the same launch contract used by multiprocessing.
    if not Path(base_executable).is_file():
        raise ProcessOwnershipError(
            "Windows Python base executable is unavailable for process ownership"
        )

    environment = os.environ.copy()
    environment["__PYVENV_LAUNCHER__"] = executable
    return base_executable, environment


@dataclass(slots=True)
class _ActivePrivateDirectoryCleanup:
    """A start-gated cleanup child retained until process reaping is proven."""

    guard: _PrivateDirectory
    process: subprocess.Popen[bytes] | None = None
    windows_job: _WindowsJob | None = None
    windows_job_assigned: bool = False


_PRIVATE_DIRECTORY_CLEANUP_LOCK = threading.Lock()
_PENDING_PRIVATE_DIRECTORIES: dict[tuple[str, int, int], _PrivateDirectory] = {}
_ACTIVE_PRIVATE_DIRECTORY_CLEANUPS: dict[
    tuple[str, int, int], _ActivePrivateDirectoryCleanup
] = {}
_PRIVATE_DIRECTORY_OPERATION_LOCKS: dict[tuple[str, int, int], threading.Lock] = {}


def _private_directory_key(guard: _PrivateDirectory) -> tuple[str, int, int]:
    return str(guard.path), guard.device, guard.inode


def _private_directory_operation_lock(guard: _PrivateDirectory) -> threading.Lock:
    key = _private_directory_key(guard)
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        operation_lock = _PRIVATE_DIRECTORY_OPERATION_LOCKS.get(key)
        if operation_lock is None:
            operation_lock = threading.Lock()
            _PRIVATE_DIRECTORY_OPERATION_LOCKS[key] = operation_lock
        return operation_lock


def _register_pending_private_directory(guard: _PrivateDirectory) -> None:
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        _PENDING_PRIVATE_DIRECTORIES[_private_directory_key(guard)] = guard


def _release_pending_private_directory(guard: _PrivateDirectory) -> None:
    key = _private_directory_key(guard)
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        _PENDING_PRIVATE_DIRECTORIES.pop(key, None)
        if key not in _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS:
            _PRIVATE_DIRECTORY_OPERATION_LOCKS.pop(key, None)


def _register_active_private_directory_cleanup(
    active: _ActivePrivateDirectoryCleanup,
) -> None:
    key = _private_directory_key(active.guard)
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        if key in _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS:
            raise ProcessOwnershipError(
                "Private directory already has an active cleanup owner"
            )
        _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS[key] = active


def _active_private_directory_cleanup(
    guard: _PrivateDirectory,
) -> _ActivePrivateDirectoryCleanup | None:
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        return _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS.get(_private_directory_key(guard))


def _release_active_private_directory_cleanup(
    active: _ActivePrivateDirectoryCleanup,
) -> None:
    key = _private_directory_key(active.guard)
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        if _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS.get(key) is active:
            _ACTIVE_PRIVATE_DIRECTORY_CLEANUPS.pop(key)


def _close_cleanup_control_pipe(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()
    except OSError:
        pass


def _release_reaped_private_directory_cleanup(
    active: _ActivePrivateDirectoryCleanup,
    *,
    deadline: float,
) -> None:
    process = active.process
    if process is not None:
        _close_cleanup_control_pipe(process)
    windows_job = active.windows_job
    if windows_job is not None:
        remaining = max(0.0, deadline - time.monotonic())
        windows_job.wait_empty(timeout=remaining)
        windows_job.close()
        active.windows_job = None
    _release_active_private_directory_cleanup(active)


def _force_reap_private_directory_cleanup(
    active: _ActivePrivateDirectoryCleanup,
    *,
    deadline: float,
) -> None:
    process = active.process
    if process is None:
        windows_job = active.windows_job
        if windows_job is not None:
            windows_job.close()
            active.windows_job = None
        _release_active_private_directory_cleanup(active)
        return
    if process.poll() is None:
        windows_job = active.windows_job
        if windows_job is not None and active.windows_job_assigned:
            try:
                windows_job.terminate()
            except OSError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    remaining = max(0.0, deadline - time.monotonic())
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        raise ProcessOwnershipError(
            "Private directory cleanup worker could not be reaped before its "
            "ownership deadline"
        ) from error
    _release_reaped_private_directory_cleanup(active, deadline=deadline)
    if time.monotonic() >= deadline:
        raise ProcessOwnershipError(
            "Private directory cleanup worker was reaped after its ownership deadline"
        )


def _spawn_private_directory_cleanup(
    guard: _PrivateDirectory,
    *,
    work_deadline: float,
    ownership_deadline: float,
) -> _ActivePrivateDirectoryCleanup:
    active = _ActivePrivateDirectoryCleanup(guard=guard)
    _register_active_private_directory_cleanup(active)
    try:
        platform_name = _ownership_platform()
        executable, environment = _supervisor_launch_context(platform_name)
        if platform_name == "nt":
            active.windows_job = _WindowsJob.create()
        process = subprocess.Popen(
            [
                executable,
                "-m",
                "hbrowser.gallery.browser._directory_cleanup_worker",
                str(guard.path),
                str(guard.device),
                str(guard.inode),
                guard.platform_name,
                repr(work_deadline),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=environment,
            **_supervisor_creation_options(),
        )
        active.process = process
        if active.windows_job is not None:
            active.windows_job.assign(process)
            active.windows_job_assigned = True
        return active
    except BaseException as startup_error:
        try:
            _force_reap_private_directory_cleanup(
                active,
                deadline=ownership_deadline,
            )
        except BaseException as cleanup_error:
            ownership_error = ProcessOwnershipError(
                "Private directory cleanup worker startup failed and its process "
                "ownership remains unresolved"
            )
            ownership_error.add_note(
                f"Startup failure type: {type(startup_error).__name__}"
            )
            raise ownership_error from cleanup_error
        raise


def _start_private_directory_cleanup(
    active: _ActivePrivateDirectoryCleanup,
) -> None:
    process = active.process
    if process is None:
        raise ProcessOwnershipError("Private cleanup worker was not launched")
    try:
        if process.stdin is None:
            raise BrokenPipeError("Private cleanup start gate is unavailable")
        process.stdin.write(b"start\n")
        process.stdin.flush()
    except OSError as error:
        raise ProcessOwnershipError(
            "Private directory cleanup start gate failed"
        ) from error


def _remove_private_directory_owned(
    guard: _PrivateDirectory,
    *,
    deadline: float,
) -> None:
    """Delete one exact directory without abandoning an executor mutation."""

    if isinstance(deadline, bool) or not isinstance(deadline, int | float):
        raise TypeError("Private cleanup deadline must be a real number")
    absolute_deadline = float(deadline)
    if not math.isfinite(absolute_deadline):
        raise ValueError("Private cleanup deadline must be finite")

    _register_pending_private_directory(guard)
    operation_lock = _private_directory_operation_lock(guard)
    remaining = max(0.0, absolute_deadline - time.monotonic())
    if not operation_lock.acquire(timeout=remaining):
        raise ProcessOwnershipError(
            "Private directory cleanup deadline expired waiting for its owner lock"
        )
    try:
        if time.monotonic() >= absolute_deadline:
            raise ProcessOwnershipError(
                "Private directory cleanup deadline expired before identity proof"
            )
        if not guard._assert_identity_or_absence():
            _release_pending_private_directory(guard)
            return

        prior_active = _active_private_directory_cleanup(guard)
        if prior_active is not None:
            _force_reap_private_directory_cleanup(
                prior_active,
                deadline=absolute_deadline,
            )
            if not guard._assert_identity_or_absence():
                _release_pending_private_directory(guard)
                return

        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            raise ProcessOwnershipError(
                "Private directory cleanup deadline expired before worker startup"
            )
        kill_reserve = min(
            _PRIVATE_CLEANUP_KILL_RESERVE_SECONDS,
            remaining / 2.0,
        )
        work_deadline = absolute_deadline - kill_reserve
        active = _spawn_private_directory_cleanup(
            guard,
            work_deadline=work_deadline,
            ownership_deadline=absolute_deadline,
        )
        process = active.process
        if process is None:
            raise ProcessOwnershipError(
                "Private directory cleanup worker ownership was not published"
            )
        returncode: int | None = None
        try:
            if time.monotonic() >= work_deadline:
                raise subprocess.TimeoutExpired(
                    process.args,
                    0,
                )
            _start_private_directory_cleanup(active)
            process.wait(timeout=max(0.0, work_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _force_reap_private_directory_cleanup(
                active,
                deadline=absolute_deadline,
            )
            returncode = process.returncode
        except BaseException:
            _force_reap_private_directory_cleanup(
                active,
                deadline=absolute_deadline,
            )
            raise
        else:
            returncode = process.returncode
            _release_reaped_private_directory_cleanup(
                active,
                deadline=absolute_deadline,
            )

        removed = not guard._assert_identity_or_absence()
        if removed:
            _release_pending_private_directory(guard)
        if time.monotonic() >= absolute_deadline:
            raise ProcessOwnershipError(
                "Private directory cleanup completed after its ownership deadline"
            )
        if not removed:
            raise ProcessOwnershipError(
                "Private directory cleanup worker exited without removing its exact "
                f"owned directory (exit_code={returncode})"
            )
    finally:
        operation_lock.release()


def _cleanup_pending_private_directories_at_exit() -> None:
    deadline = time.monotonic() + _PROCESS_WAIT_TIMEOUT_SECONDS
    with _PRIVATE_DIRECTORY_CLEANUP_LOCK:
        pending = tuple(_PENDING_PRIVATE_DIRECTORIES.values())
    for guard in pending:
        if time.monotonic() >= deadline:
            break
        try:
            guard.remove(deadline=deadline)
        except BaseException:
            pass


atexit.register(_cleanup_pending_private_directories_at_exit)


def _read_supervisor_status(
    owner: OwnedProcess,
    status_path: Path,
    *,
    deadline: float,
) -> _SupervisorStatus:
    if isinstance(deadline, bool) or not isinstance(deadline, int | float):
        raise TypeError("supervisor READY deadline must be a real number")
    supplied_deadline = float(deadline)
    if not math.isfinite(supplied_deadline):
        raise ValueError("supervisor READY deadline must be finite")
    started_at = time.monotonic()
    ready_deadline = min(
        supplied_deadline,
        started_at + _STARTUP_TIMEOUT_SECONDS,
    )
    if ready_deadline <= started_at:
        raise TimeoutError("Process supervisor READY deadline already expired")

    def require_ready_budget() -> None:
        if time.monotonic() >= ready_deadline:
            raise TimeoutError(
                "Process supervisor did not launch target before its READY deadline"
            )

    while time.monotonic() < ready_deadline:
        if status_path.is_file():
            status = status_path.read_text(encoding="utf-8").strip()
            require_ready_budget()
            if status.startswith(_SUPERVISOR_READY_PREFIX):
                raw_pid = status.removeprefix(_SUPERVISOR_READY_PREFIX)
                try:
                    target_pid = int(raw_pid)
                except ValueError as error:
                    raise RuntimeError(
                        "Process supervisor returned an invalid PID"
                    ) from error
                if target_pid <= 0:
                    raise RuntimeError("Process supervisor returned an invalid PID")
                require_ready_budget()
                return _SupervisorReady(target_pid)
            if status.startswith(_SUPERVISOR_ERROR_PREFIX):
                error_type = status.removeprefix(_SUPERVISOR_ERROR_PREFIX)
                if not error_type.isascii() or not error_type.isidentifier():
                    raise RuntimeError("Process supervisor returned an invalid status")
                require_ready_budget()
                return _SupervisorTargetNotStarted(error_type)
            raise RuntimeError("Process supervisor returned an invalid status")
        returncode = owner.poll()
        if returncode is not None:
            raise RuntimeError(
                "Process supervisor exited before target startup "
                f"(exit_code={returncode})"
            )
        time.sleep(_STATUS_POLL_SECONDS)
    raise TimeoutError(
        "Process supervisor did not launch target before its READY deadline"
    )


def start_owned_process(
    executable: str | Path,
    parameters: Sequence[str],
    *,
    stdout: int | None = subprocess.PIPE,
    stderr: int | None = subprocess.PIPE,
    drain_output: bool = False,
    cleanup_paths: Sequence[str | Path] = (),
    startup_timeout: float = _STARTUP_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> OwnedProcess:
    """Start one target under one absolute startup-and-cleanup deadline.

    ``startup_timeout`` caps the supervisor READY phase. ``deadline`` is a
    monotonic caller deadline shared with every failure-cleanup phase. When no
    caller deadline is supplied, ``startup_timeout`` is the complete operation
    budget and half (at most five seconds) is reserved for ownership proof.
    """

    if (
        isinstance(startup_timeout, bool)
        or not isinstance(startup_timeout, int | float)
        or not math.isfinite(float(startup_timeout))
        or startup_timeout <= 0
    ):
        raise ValueError("process startup timeout must be finite and positive")
    effective_startup_timeout = min(
        _STARTUP_TIMEOUT_SECONDS,
        float(startup_timeout),
    )
    started_at = time.monotonic()
    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, int | float):
            raise TypeError("process startup deadline must be a real number")
        overall_deadline = float(deadline)
        if not math.isfinite(overall_deadline):
            raise ValueError("process startup deadline must be finite")
    else:
        overall_deadline = started_at + effective_startup_timeout
    total_budget = overall_deadline - started_at
    if total_budget <= 0:
        raise TimeoutError("process startup deadline already expired")
    cleanup_reserve = min(_PRIVATE_CLEANUP_TIMEOUT_SECONDS, total_budget / 2.0)

    def remaining() -> float:
        return max(0.0, overall_deadline - time.monotonic())

    def startup_phase_timeout() -> float:
        available = remaining() - cleanup_reserve
        if available <= 0:
            raise TimeoutError(
                "process startup deadline has no budget before ownership cleanup"
            )
        return min(effective_startup_timeout, available)

    status_directory = Path(tempfile.mkdtemp(prefix="hbrowser-process-owner-"))
    status_path = status_directory / "status"
    windows_job: _WindowsJob | None = None
    supervisor: subprocess.Popen[bytes] | None = None
    target_pid: int | None = None
    owner: OwnedProcess | None = None
    provisional_owner: _ProvisionalProcessOwner | None = None
    platform_name = _ownership_platform()
    status_guard: _PrivateDirectory | None = None
    cleanup_guards: tuple[_PrivateDirectory, ...] = ()
    cleanup_guard_list: list[_PrivateDirectory] = []
    try:
        status_guard = _PrivateDirectory.capture(status_directory)
        for path in cleanup_paths:
            cleanup_guard_list.append(_PrivateDirectory.capture(Path(path)))
        cleanup_guards = tuple(cleanup_guard_list)
        supervisor_executable, supervisor_environment = _supervisor_launch_context(
            platform_name
        )
        command = [
            supervisor_executable,
            "-m",
            "hbrowser.gallery.browser._process_supervisor",
            str(status_path),
            "--",
            str(executable),
            *parameters,
        ]
        supervisor = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            env=supervisor_environment,
            **_supervisor_creation_options(),
        )
        assert status_guard is not None
        provisional_owner = _ProvisionalProcessOwner(
            supervisor=supervisor,
            status_guard=status_guard,
            cleanup_guards=cleanup_guards,
        )
        _register_provisional_owner(provisional_owner)
        if platform_name == "nt":
            windows_job = _WindowsJob.create()
            provisional_owner.windows_job = windows_job
            windows_job.assign(supervisor)
        if supervisor.stdin is None:
            raise RuntimeError("Process supervisor control pipe is unavailable")
        owner = OwnedProcess(
            supervisor,
            target_process_group=None,
            windows_job=windows_job,
            stdout_drain=None,
            stderr_drain=None,
            status_directory=status_guard,
            cleanup_paths=cleanup_guards,
        )
        _release_provisional_owner(provisional_owner)
        provisional_owner = None
        windows_job = None
        supervisor.stdin.write(b"start\n")
        supervisor.stdin.flush()
        ready_deadline = min(
            overall_deadline,
            time.monotonic() + startup_phase_timeout(),
        )
        startup_status = _read_supervisor_status(
            owner,
            status_path,
            deadline=ready_deadline,
        )
        if isinstance(startup_status, _SupervisorTargetNotStarted):
            owner.bind_target_not_started()
            raise RuntimeError(
                "Process supervisor could not launch target: "
                f"{startup_status.error_type}"
            ) from None
        target_pid = startup_status.target_pid
        owner.bind_target_process_group(target_pid)
        if platform_name == "posix":
            if os.getpgid(target_pid) != target_pid:
                raise RuntimeError("Target process group ownership was not established")
            if os.getsid(target_pid) != supervisor.pid:
                raise RuntimeError("Target process escaped its owned session")

        if drain_output:
            owner.begin_output_draining()
        if remaining() <= 0:
            raise TimeoutError("process startup receipt arrived after its deadline")
        return owner
    except BaseException as startup_error:
        if owner is not None:
            try:
                owner.shutdown(
                    graceful_timeout=0,
                    terminate_timeout=min(2.0, remaining()),
                    kill_timeout=min(_SHUTDOWN_PHASE_TIMEOUT_SECONDS, remaining()),
                    cleanup_timeout=min(
                        _SHUTDOWN_PHASE_TIMEOUT_SECONDS,
                        remaining(),
                    ),
                    deadline=overall_deadline,
                )
            except BaseException as owner_cleanup_error:
                ownership_error = ProcessOwnershipError(
                    "Process startup failed and target ownership remains unresolved"
                )
                ownership_error.add_note(
                    "Startup failure type: " f"{type(startup_error).__name__}"
                )
                raise ownership_error from owner_cleanup_error
            raise
        if provisional_owner is not None:
            try:
                _cleanup_provisional_owner(
                    provisional_owner,
                    deadline=overall_deadline,
                )
            except BaseException as provisional_cleanup_error:
                ownership_error = ProcessOwnershipError(
                    "Process startup failed before ownership transfer and the "
                    "start-gated supervisor remains unresolved"
                )
                ownership_error.add_note(
                    "Startup failure type: " f"{type(startup_error).__name__}"
                )
                raise ownership_error from provisional_cleanup_error
            raise
        if windows_job is not None:
            try:
                windows_job.terminate()
            except BaseException:
                pass
        if supervisor is not None:
            if platform_name == "posix" and target_pid is not None:
                try:
                    os.killpg(target_pid, signal.SIGTERM)
                except PermissionError, ProcessLookupError:
                    pass
            try:
                supervisor.terminate()
            except ProcessLookupError:
                pass
            try:
                supervisor.wait(timeout=remaining())
            except subprocess.TimeoutExpired, ProcessLookupError:
                if platform_name == "posix" and target_pid is not None:
                    try:
                        os.killpg(target_pid, signal.SIGKILL)
                    except PermissionError, ProcessLookupError:
                        pass
                try:
                    supervisor.kill()
                except ProcessLookupError:
                    pass
                try:
                    supervisor.wait(timeout=remaining())
                except subprocess.TimeoutExpired, ProcessLookupError:
                    pass
        if windows_job is not None:
            try:
                windows_job.close()
            except BaseException:
                pass
        private_cleanup_error: BaseException | None = None
        guarded_paths: tuple[_PrivateDirectory, ...] = (
            *((status_guard,) if status_guard is not None else ()),
            *cleanup_guard_list,
        )
        for cleanup_path in guarded_paths:
            try:
                cleanup_path.remove(deadline=overall_deadline)
            except BaseException as error:
                if private_cleanup_error is None:
                    private_cleanup_error = error
                else:
                    private_cleanup_error.add_note(
                        "Additional private cleanup failure: " f"{type(error).__name__}"
                    )
        if private_cleanup_error is not None:
            ownership_error = ProcessOwnershipError(
                "Process startup failed and private generation cleanup "
                "could not be completed"
            )
            ownership_error.add_note(
                "Startup failure type: " f"{type(startup_error).__name__}"
            )
            raise ownership_error from private_cleanup_error
        raise


def start_owned_browser_process(
    executable: str | Path,
    parameters: Sequence[str],
    *,
    cleanup_paths: Sequence[str | Path] = (),
    startup_timeout: float = _STARTUP_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> OwnedProcess:
    """Start Chromium with bounded output draining and full tree ownership."""

    return start_owned_process(
        executable,
        parameters,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        drain_output=True,
        cleanup_paths=cleanup_paths,
        startup_timeout=startup_timeout,
        deadline=deadline,
    )


__all__ = [
    "OwnedProcess",
    "ProcessOwnershipError",
    "start_owned_browser_process",
    "start_owned_process",
]
