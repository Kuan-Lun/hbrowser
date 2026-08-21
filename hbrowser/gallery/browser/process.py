"""Own browser-related process trees without sharing terminal signals."""

from __future__ import annotations

import atexit
import ctypes
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

_STARTUP_TIMEOUT_SECONDS: Final = 10.0
_STATUS_POLL_SECONDS: Final = 0.01
_PRIVATE_CLEANUP_TIMEOUT_SECONDS: Final = 5.0
_PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS: Final = 0.02
_PRIVATE_CLEANUP_RETRY_MAX_SECONDS: Final = 0.25
_WINDOWS_RETRYABLE_PRIVATE_CLEANUP_ERRORS: Final = frozenset({32, 33})
_OUTPUT_RING_BYTES: Final = 256 * 1024
_SUPERVISOR_READY_PREFIX: Final = "ready "
_SUPERVISOR_ERROR_PREFIX: Final = "error "


class ProcessOwnershipError(RuntimeError):
    """A process tree or its private material remains under unresolved ownership."""


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
        """Remove the captured directory with per-attempt identity validation.

        Windows can keep a just-exited Chromium database handle alive briefly.
        Retry only the two native sharing/lock violations, revalidating the root
        identity before every attempt and retaining ownership when the deadline
        expires.
        """

        retry_delay = _PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
        last_retryable_error: OSError | None = None
        attempts = 0
        while True:
            if last_retryable_error is not None and time.monotonic() >= deadline:
                raise self._deadline_error(
                    last_retryable_error,
                    attempts=attempts,
                ) from last_retryable_error
            if not self._assert_identity_or_absence():
                return
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

    def _request_supervisor_shutdown(self, command: bytes) -> None:
        returncode = self._supervisor.returncode
        if self._supervisor_reaped or returncode is not None:
            self._supervisor_reaped = True
            if returncode is not None:
                self._supervisor_returncode = returncode
            return
        try:
            if self._supervisor.stdin is None:
                raise BrokenPipeError("Process supervisor control pipe is unavailable")
            self._supervisor.stdin.write(command)
            self._supervisor.stdin.flush()
        except OSError:
            # A failed control channel provides no safe basis for signalling a
            # cached numeric PID. wait() will either obtain the supervisor's
            # proof or fail closed while retaining private material.
            return

    def terminate(self) -> None:
        with self._state_lock:
            if self._closed or self._tree_reaped:
                return
            if self._windows_job is not None:
                # Let the supervisor reap the Chrome root first. Terminating the
                # entire Job here interrupts SQLite/profile flushes and creates
                # the very sharing-violation race private cleanup must avoid.
                self._request_supervisor_shutdown(b"terminate\n")
                return
            self._request_supervisor_shutdown(b"terminate\n")

    def kill(self) -> None:
        with self._state_lock:
            if self._closed or self._tree_reaped:
                return
            if self._windows_job is not None:
                self._windows_job.terminate()
                return
            self._request_supervisor_shutdown(b"kill\n")

    def wait(self, timeout: float | None = None) -> int:
        """Prove tree exit, then release private material within its own budget."""

        with self._shutdown_lock:
            with self._state_lock:
                if self._closed:
                    return self._supervisor_returncode or 0
            deadline = None if timeout is None else time.monotonic() + timeout
            returncode = self._wait_for_process_tree(deadline=deadline)
            self._release_ownership(
                deadline=time.monotonic() + _PRIVATE_CLEANUP_TIMEOUT_SECONDS
            )
            return returncode

    def _wait_for_process_tree(self, *, deadline: float | None) -> int:
        """Advance only the process-proof phase of the ownership state machine."""

        with self._state_lock:
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
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            returncode = self._supervisor.wait(timeout=remaining)
            with self._state_lock:
                self._supervisor_reaped = True
                self._supervisor_returncode = returncode

        with self._state_lock:
            if self._tree_reaped:
                return returncode
            windows_job = self._windows_job
            target_process_group = self._target_process_group
            target_not_started = self._target_not_started

        if windows_job is not None:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            windows_job.wait_empty(timeout=remaining)
        elif target_process_group is not None:
            if returncode != 0:
                raise ProcessOwnershipError(
                    "Process supervisor did not prove target-tree cleanup"
                )
        elif not target_not_started:
            raise ProcessOwnershipError(
                "Process supervisor exited before target cleanup was proven"
            )

        with self._state_lock:
            self._tree_reaped = True
        return returncode

    def shutdown(
        self,
        *,
        graceful_timeout: float,
        terminate_timeout: float,
        kill_timeout: float,
        cleanup_timeout: float = _PRIVATE_CLEANUP_TIMEOUT_SECONDS,
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
        if any(timeout < 0 for timeout in timeouts):
            raise ValueError("Process shutdown timeouts must be non-negative")

        with self._shutdown_lock:
            with self._state_lock:
                if self._closed:
                    return self._supervisor_returncode or 0
                tree_reaped = self._tree_reaped

            if not tree_reaped:
                try:
                    returncode = self._wait_for_process_tree(
                        deadline=time.monotonic() + graceful_timeout
                    )
                except subprocess.TimeoutExpired, TimeoutError:
                    self.terminate()
                    try:
                        returncode = self._wait_for_process_tree(
                            deadline=time.monotonic() + terminate_timeout
                        )
                    except subprocess.TimeoutExpired, TimeoutError:
                        self.kill()
                        returncode = self._wait_for_process_tree(
                            deadline=time.monotonic() + kill_timeout
                        )
            else:
                with self._state_lock:
                    known_returncode = self._supervisor_returncode
                if known_returncode is None:
                    raise ProcessOwnershipError(
                        "Process-tree proof has no supervisor return code"
                    )
                returncode = known_returncode

            self._release_ownership(deadline=time.monotonic() + cleanup_timeout)
            return returncode

    def _release_ownership(self, *, deadline: float) -> None:
        """Release private paths and the Job only after tree exit is proven."""

        with self._state_lock:
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
        with self._state_lock:
            status_directory = self._status_directory
        if status_directory is not None:
            status_directory.remove(deadline=deadline)
            with self._state_lock:
                self._status_directory = None
        while True:
            with self._state_lock:
                cleanup_path = self._cleanup_paths[0] if self._cleanup_paths else None
            if cleanup_path is None:
                break
            cleanup_path.remove(deadline=deadline)
            with self._state_lock:
                if (
                    not self._cleanup_paths
                    or self._cleanup_paths[0] is not cleanup_path
                ):
                    raise ProcessOwnershipError(
                        "Private cleanup ownership state changed unexpectedly"
                    )
                self._cleanup_paths.pop(0)
        with self._state_lock:
            windows_job = self._windows_job
        if windows_job is not None:
            windows_job.close()
            with self._state_lock:
                if self._windows_job is not windows_job:
                    raise ProcessOwnershipError(
                        "Windows Job ownership state changed unexpectedly"
                    )
                self._windows_job = None
        with self._state_lock:
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


def _read_supervisor_status(
    owner: OwnedProcess,
    status_path: Path,
    *,
    timeout: float,
) -> _SupervisorStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status_path.is_file():
            status = status_path.read_text(encoding="utf-8").strip()
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
                return _SupervisorReady(target_pid)
            if status.startswith(_SUPERVISOR_ERROR_PREFIX):
                error_type = status.removeprefix(_SUPERVISOR_ERROR_PREFIX)
                if not error_type.isascii() or not error_type.isidentifier():
                    raise RuntimeError("Process supervisor returned an invalid status")
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
        f"Process supervisor did not launch target within {timeout:g} seconds"
    )


def start_owned_process(
    executable: str | Path,
    parameters: Sequence[str],
    *,
    stdout: int | None = subprocess.PIPE,
    stderr: int | None = subprocess.PIPE,
    drain_output: bool = False,
    cleanup_paths: Sequence[str | Path] = (),
) -> OwnedProcess:
    """Start one target behind an ownership gate and return its tree owner."""

    status_directory = Path(tempfile.mkdtemp(prefix="hbrowser-process-owner-"))
    status_path = status_directory / "status"
    windows_job: _WindowsJob | None = None
    supervisor: subprocess.Popen[bytes] | None = None
    target_pid: int | None = None
    owner: OwnedProcess | None = None
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
        if platform_name == "nt":
            windows_job = _WindowsJob.create()
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
        windows_job = None
        supervisor.stdin.write(b"start\n")
        supervisor.stdin.flush()
        startup_status = _read_supervisor_status(
            owner,
            status_path,
            timeout=_STARTUP_TIMEOUT_SECONDS,
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
        return owner
    except BaseException as startup_error:
        if owner is not None:
            try:
                owner.shutdown(
                    graceful_timeout=0,
                    terminate_timeout=5,
                    kill_timeout=5,
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
                supervisor.wait(timeout=5)
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
                    supervisor.wait(timeout=5)
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
                cleanup_path.remove(
                    deadline=time.monotonic() + _PRIVATE_CLEANUP_TIMEOUT_SECONDS
                )
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
) -> OwnedProcess:
    """Start Chromium with bounded output draining and full tree ownership."""

    return start_owned_process(
        executable,
        parameters,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        drain_output=True,
        cleanup_paths=cleanup_paths,
    )


__all__ = [
    "OwnedProcess",
    "ProcessOwnershipError",
    "start_owned_browser_process",
    "start_owned_process",
]
