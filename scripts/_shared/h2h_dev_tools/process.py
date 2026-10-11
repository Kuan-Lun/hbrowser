"""Own commands through POSIX sessions or start-gated Windows Job Objects.

A successful command requires an empty owned tree, not merely an exited leader.
Callers own their command selection, absolute deadline and cleanup reserve.
Detached POSIX sessions and daemon-owned resources are outside this contract.
"""

from __future__ import annotations

import ctypes
import math
import ntpath
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, BinaryIO, Final, cast

TERMINATION_GRACE_SECONDS = 2.0
TERMINATION_POLL_SECONDS = 0.05
TIMEOUT_EXIT_CODE = 124
TERMINATION_FAILED_EXIT_CODE = 125
INTERRUPTED_EXIT_CODE = 130
_WINDOWS_SUPERVISOR_MODE: Final = "--internal-windows-supervisor"
_WINDOWS_START_TOKEN: Final = b"\x01"


@dataclass(frozen=True)
class ProcessCommand:
    """Explicit command and execution context; no repository discovery."""

    label: str
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] | None = None
    stdout: BinaryIO | int | None = None
    stderr: BinaryIO | int | None = None


class RunnerSignalInterrupt(BaseException):
    """A POSIX termination signal converted into controlled runner shutdown."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


class ProcessOwnershipError(RuntimeError):
    """A command violated ownership; cleanup may still finish before propagation."""

    cleanup_proven: bool = False


@dataclass
class SignalController:
    """Defer asynchronous interruption only across ownership-critical regions."""

    pending_signal: int | None = None
    defer_depth: int = 0
    interruption_started: bool = False

    def receive(self, signal_number: int) -> None:
        if self.interruption_started:
            return
        if self.defer_depth:
            if self.pending_signal is None:
                self.pending_signal = signal_number
            return
        self._raise(signal_number)

    @contextmanager
    def defer(self) -> Iterator[None]:
        self.defer_depth += 1
        body_failed = True
        try:
            yield
            body_failed = False
        finally:
            self.defer_depth -= 1
            if self.defer_depth == 0 and not body_failed:
                self.raise_pending()

    def raise_pending(self) -> None:
        if self.pending_signal is None:
            return
        pending = self.pending_signal
        self.pending_signal = None
        self._raise(pending)

    def _raise(self, signal_number: int) -> None:
        self.interruption_started = True
        if signal_number == signal.SIGINT:
            raise KeyboardInterrupt
        raise RunnerSignalInterrupt(signal_number)


@contextmanager
def controlled_termination_signals(
    *, posix_signals: tuple[int, ...] | None = None
) -> Iterator[SignalController]:
    """Control platform termination signals and restore prior handlers."""

    signal_numbers: tuple[int, ...]
    if os.name == "nt":
        break_signal = cast(int | None, getattr(signal, "SIGBREAK", None))
        if break_signal is None:
            raise RuntimeError("Windows requires SIGBREAK support")
        signal_numbers = (signal.SIGINT, break_signal)
    else:
        signal_numbers = (
            (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            if posix_signals is None
            else posix_signals
        )
    previous_handlers = [
        (signal_number, signal.getsignal(signal_number))
        for signal_number in signal_numbers
    ]
    controller = SignalController()

    def interrupt(signal_number: int, frame: FrameType | None) -> None:
        del frame
        controller.receive(signal_number)

    installed: list[int] = []
    try:
        for signal_number, _previous_handler in previous_handlers:
            signal.signal(signal_number, interrupt)
            installed.append(signal_number)
        yield controller
    finally:
        previous_by_signal = dict(previous_handlers)
        for signal_number in reversed(installed):
            signal.signal(signal_number, previous_by_signal[signal_number])


@contextmanager
def defer_signal_interrupts(
    controller: SignalController | None,
) -> Iterator[None]:
    if controller is None:
        yield
        return
    with controller.defer():
        yield


class WindowsJob:
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

    @classmethod
    def create(cls) -> WindowsJob:
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects are unavailable")
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
        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        if not self._kernel32.AssignProcessToJobObject(
            self._handle,
            int(process_handle),
        ):
            raise self._last_error("AssignProcessToJobObject")

    def terminate(self) -> None:
        if self._handle is None:
            return
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            raise self._last_error("TerminateJobObject")

    def active_processes(self) -> int:
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

    def wait_empty(self, *, deadline: float) -> bool:
        while self.active_processes() != 0:
            remaining_seconds = seconds_remaining(deadline)
            if remaining_seconds == 0:
                return False
            time.sleep(min(TERMINATION_POLL_SECONDS, remaining_seconds))
        return True

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        if not self._kernel32.CloseHandle(handle):
            raise self._last_error("CloseHandle")
        self._handle = None


@dataclass(frozen=True)
class OwnedProcess:
    process: subprocess.Popen[bytes]
    windows_job: WindowsJob | None = None


def _start_posix_process(phase: ProcessCommand) -> subprocess.Popen[bytes]:
    command = phase.argv
    return subprocess.Popen(
        command,
        cwd=phase.cwd,
        env=phase.environment,
        stdout=phase.stdout,
        stderr=phase.stderr,
        start_new_session=True,
    )


def _windows_supervisor_launch(
    phase: ProcessCommand,
) -> tuple[tuple[str, ...], dict[str, str]]:
    executable = sys.executable
    base_executable = getattr(sys, "_base_executable", None)
    if not isinstance(base_executable, str) or not base_executable:
        raise RuntimeError("Windows Python did not expose its base executable")
    environment = dict(os.environ if phase.environment is None else phase.environment)
    environment.pop("__PYVENV_LAUNCHER__", None)
    if ntpath.normcase(ntpath.normpath(base_executable)) != ntpath.normcase(
        ntpath.normpath(executable)
    ):
        if not Path(base_executable).is_file():
            raise RuntimeError("Windows Python base executable is unavailable")
        executable = base_executable
    return (
        executable,
        # Python startup must not run environment/site hooks before Job.assign.
        # Flags isolate only this supervisor; the admitted child keeps its env.
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        _WINDOWS_SUPERVISOR_MODE,
        *phase.argv,
    ), environment


def _cleanup_windows_supervisor(
    process: subprocess.Popen[bytes],
    job: WindowsJob,
    *,
    assigned: bool,
    deadline: float,
) -> bool:
    cleanup_succeeded = True
    try:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                cleanup_succeeded = False
        if assigned:
            try:
                job.terminate()
            except OSError:
                cleanup_succeeded = False
            try:
                cleanup_succeeded = (
                    job.wait_empty(deadline=deadline) and cleanup_succeeded
                )
            except OSError:
                cleanup_succeeded = False
        elif process.poll() is None:
            try:
                process.kill()
            except OSError:
                cleanup_succeeded = False
        try:
            process.wait(timeout=seconds_remaining(deadline))
        except OSError, subprocess.TimeoutExpired:
            cleanup_succeeded = False
    finally:
        # Startup can fail before an OwnedProcess is returned. Its local Job
        # still needs the kill-on-close backstop if waiting or probing raises.
        try:
            job.close()
        except OSError:
            cleanup_succeeded = False
    return cleanup_succeeded and process.poll() is not None


def _start_windows_process(
    phase: ProcessCommand,
    *,
    deadline: float,
    signal_controller: SignalController | None,
) -> OwnedProcess:
    creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if not isinstance(creation_flag, int) or creation_flag == 0:
        raise ProcessOwnershipError(phase.label) from None
    job: WindowsJob | None = None
    supervisor: subprocess.Popen[bytes] | None = None
    assigned = False
    cleanup_performed = False
    try:
        with defer_signal_interrupts(signal_controller):
            try:
                job = WindowsJob.create()
                command, environment = _windows_supervisor_launch(phase)
                supervisor = subprocess.Popen(
                    command,
                    cwd=phase.cwd,
                    env=environment,
                    stdout=phase.stdout,
                    stderr=phase.stderr,
                    stdin=subprocess.PIPE,
                    creationflags=creation_flag,
                )
                job.assign(supervisor)
                assigned = True
                if signal_controller is not None:
                    signal_controller.raise_pending()
                if supervisor.stdin is None:
                    raise RuntimeError(
                        "Windows command supervisor start gate is unavailable"
                    )
                written = supervisor.stdin.write(_WINDOWS_START_TOKEN)
                if written != len(_WINDOWS_START_TOKEN):
                    raise BlockingIOError(
                        "Windows command supervisor start gate was partial"
                    )
                supervisor.stdin.close()
            except BaseException as startup_error:
                cleaned = True
                if job is None:
                    cleaned = supervisor is None
                elif supervisor is None:
                    try:
                        job.close()
                    except OSError:
                        cleaned = False
                else:
                    cleanup_performed = True
                    cleaned = _cleanup_windows_supervisor(
                        supervisor,
                        job,
                        assigned=assigned,
                        deadline=deadline,
                    )
                if not cleaned:
                    raise ProcessOwnershipError(phase.label) from startup_error
                if signal_controller is not None:
                    signal_controller.raise_pending()
                if isinstance(
                    startup_error,
                    (KeyboardInterrupt, RunnerSignalInterrupt),
                ):
                    raise
                raise ProcessOwnershipError(phase.label) from startup_error
        return OwnedProcess(supervisor, job)
    except (KeyboardInterrupt, RunnerSignalInterrupt) as startup_error:
        if job is not None and supervisor is not None and not cleanup_performed:
            cleaned = _cleanup_windows_supervisor(
                supervisor,
                job,
                assigned=assigned,
                deadline=deadline,
            )
            if not cleaned:
                raise ProcessOwnershipError(phase.label) from startup_error
        raise


def start_owned_process(
    phase: ProcessCommand,
    *,
    deadline: float,
    signal_controller: SignalController | None,
) -> OwnedProcess:
    if os.name == "nt":
        return _start_windows_process(
            phase,
            deadline=deadline,
            signal_controller=signal_controller,
        )
    try:
        return OwnedProcess(_start_posix_process(phase))
    except OSError as error:
        raise ProcessOwnershipError(phase.label) from error


def seconds_remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _wait_for_process_until(
    process: subprocess.Popen[bytes], *, deadline: float
) -> bool:
    remaining_seconds = seconds_remaining(deadline)
    if remaining_seconds == 0:
        return process.poll() is not None
    try:
        process.wait(timeout=remaining_seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def _wait_for_posix_process_group_exit(
    process: subprocess.Popen[bytes],
    *,
    kill_process_group: Callable[[int, int], None],
    deadline: float,
) -> bool:
    while True:
        # poll() reaps an exited leader; group membership, rather than leader
        # state, remains the authority for whether descendants survived.
        process.poll()
        try:
            kill_process_group(process.pid, 0)
        except ProcessLookupError:
            return _wait_for_process_until(process, deadline=deadline)
        except PermissionError:
            pass
        remaining_seconds = seconds_remaining(deadline)
        if remaining_seconds <= 0:
            return False
        time.sleep(min(TERMINATION_POLL_SECONDS, remaining_seconds))


def _terminate_posix_process_group(
    process: subprocess.Popen[bytes], *, deadline: float
) -> bool:
    kill_process_group = cast(
        Callable[[int, int], None] | None,
        getattr(os, "killpg", None),
    )
    if kill_process_group is None:
        return False

    try:
        kill_process_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return _wait_for_process_until(process, deadline=deadline)
    except OSError:
        return False
    now = time.monotonic()
    graceful_deadline = min(
        deadline,
        now + min(TERMINATION_GRACE_SECONDS, max(0.0, deadline - now) / 2),
    )
    if _wait_for_posix_process_group_exit(
        process,
        kill_process_group=kill_process_group,
        deadline=graceful_deadline,
    ):
        return True
    try:
        kill_process_group(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return _wait_for_process_until(process, deadline=deadline)
    except OSError:
        return False
    return _wait_for_posix_process_group_exit(
        process,
        kill_process_group=kill_process_group,
        deadline=deadline,
    )


def wait_for_owned_tree_exit(
    owner: OwnedProcess,
    *,
    deadline: float,
) -> bool:
    if owner.windows_job is not None:
        try:
            return owner.windows_job.wait_empty(
                deadline=deadline
            ) and _wait_for_process_until(
                owner.process,
                deadline=deadline,
            )
        except OSError:
            return False
    kill_process_group = cast(
        Callable[[int, int], None] | None,
        getattr(os, "killpg", None),
    )
    if kill_process_group is None:
        return False
    return _wait_for_posix_process_group_exit(
        owner.process,
        kill_process_group=kill_process_group,
        deadline=deadline,
    )


def _report_diagnostic(message: str) -> None:
    """Keep a failed diagnostic stream from interrupting ownership cleanup."""

    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        # Diagnostics are secondary to process ownership. Even an unusual stream
        # implementation must not prevent termination, fallback or Job closing.
        pass


def _taskkill_windows_tree(
    process: subprocess.Popen[bytes], *, deadline: float
) -> bool:
    remaining_seconds = seconds_remaining(deadline)
    if remaining_seconds == 0:
        return False
    try:
        completed = subprocess.run(
            ("taskkill", "/PID", str(process.pid), "/T", "/F"),
            check=False,
            capture_output=True,
            text=True,
            timeout=min(TERMINATION_GRACE_SECONDS, remaining_seconds),
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        _report_diagnostic(
            f"taskkill failed with exit {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
        return False
    return True


def _terminate_windows_tree(
    owner: OwnedProcess,
    *,
    deadline: float,
) -> bool:
    job = owner.windows_job
    if job is None or seconds_remaining(deadline) == 0:
        return False
    try:
        job.terminate()
    except OSError as error:
        _report_diagnostic(str(error))
        if not _taskkill_windows_tree(owner.process, deadline=deadline):
            return False
    return wait_for_owned_tree_exit(owner, deadline=deadline)


def terminate_owned_tree(
    owner: OwnedProcess,
    *,
    deadline: float,
) -> bool:
    """Terminate and verify the managed command process tree before deadline."""

    if owner.windows_job is not None:
        return _terminate_windows_tree(owner, deadline=deadline)
    return _terminate_posix_process_group(owner.process, deadline=deadline)


def close_owner(owner: OwnedProcess) -> bool:
    job = owner.windows_job
    if job is None:
        return True
    try:
        job.close()
    except OSError as error:
        _report_diagnostic(str(error))
        return False
    return True


def run_command(
    phase: ProcessCommand,
    *,
    hard_deadline: float | None,
    termination_reserve_seconds: float = 6.0,
    signal_controller: SignalController | None = None,
) -> int:
    """Run and close an owned command, controlling signals by default.

    Automatic signal handling requires the main thread. Callers that already
    manage process-wide handlers may pass their existing signal controller.
    """

    controller_context = (
        controlled_termination_signals()
        if signal_controller is None
        else nullcontext(signal_controller)
    )
    with controller_context as controller:
        return _run_command(
            phase,
            hard_deadline=hard_deadline,
            termination_reserve_seconds=termination_reserve_seconds,
            signal_controller=controller,
        )


def _run_command(
    phase: ProcessCommand,
    *,
    hard_deadline: float | None,
    termination_reserve_seconds: float,
    signal_controller: SignalController,
) -> int:
    if (
        not math.isfinite(termination_reserve_seconds)
        or termination_reserve_seconds <= 0
        or (hard_deadline is not None and not math.isfinite(hard_deadline))
    ):
        raise ValueError(
            "Deadline must be finite and cleanup reserve finite and positive"
        )
    started_at = time.monotonic()
    budget_detail = (
        "unbounded"
        if hard_deadline is None
        else f"{max(0.0, hard_deadline - started_at):.1f}s total remaining"
    )
    print(f"\n==> {phase.label} ({budget_detail})", flush=True)
    print("+", shlex.join(phase.argv), flush=True)
    execution_deadline = (
        None if hard_deadline is None else hard_deadline - termination_reserve_seconds
    )
    if execution_deadline is not None and execution_deadline <= started_at:
        _report_diagnostic(
            f"command budget lacks the {termination_reserve_seconds:.1f}s "
            f"termination reserve before {phase.label}"
        )
        return TIMEOUT_EXIT_CODE
    ownership_deadline = (
        hard_deadline
        if hard_deadline is not None
        else started_at + termination_reserve_seconds
    )
    owner: OwnedProcess | None = None
    tree_empty_proven = False
    ownership_error: ProcessOwnershipError | None = None
    termination_deadline: float | None = None
    return_code = TERMINATION_FAILED_EXIT_CODE
    try:
        # Do not deliver a controlled platform signal after Popen creates the
        # child but before this frame owns its process tree. The handler records
        # a pending signal and delivers it after ownership is established.
        with signal_controller.defer():
            owner = start_owned_process(
                phase,
                deadline=ownership_deadline,
                signal_controller=signal_controller,
            )
        timeout_seconds = (
            None
            if execution_deadline is None
            else max(0.0, execution_deadline - time.monotonic())
        )
        try:
            return_code = owner.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            termination_deadline = hard_deadline
            return_code = TIMEOUT_EXIT_CODE
            _report_diagnostic(
                f"command budget expired during {phase.label}; "
                "terminating its process tree"
            )
        else:
            cleanup_started = time.monotonic()
            termination_deadline = (
                cleanup_started + termination_reserve_seconds
                if hard_deadline is None
                else hard_deadline
            )
            clean_exit_deadline = min(
                termination_deadline,
                cleanup_started + TERMINATION_POLL_SECONDS * 5,
            )
            with signal_controller.defer():
                tree_empty_proven = wait_for_owned_tree_exit(
                    owner, deadline=clean_exit_deadline
                )
            if not tree_empty_proven:
                _report_diagnostic(
                    "command leader exited with surviving process-tree members "
                    f"during {phase.label}"
                )
                ownership_error = ProcessOwnershipError(phase.label)
                raise ownership_error
    except KeyboardInterrupt, RunnerSignalInterrupt:
        _report_diagnostic(
            f"command interrupted during {phase.label}; terminating its process tree"
        )
        raise
    finally:
        if owner is not None:
            # Every exit, including wait or diagnostic failure, passes through
            # bounded teardown. Defer signals through Job closure as well, and
            # never let failed termination skip the kill-on-close backstop.
            with signal_controller.defer():
                try:
                    if not tree_empty_proven:
                        if termination_deadline is None:
                            termination_deadline = (
                                time.monotonic() + termination_reserve_seconds
                            )
                            if hard_deadline is not None:
                                termination_deadline = min(
                                    hard_deadline, termination_deadline
                                )
                        try:
                            tree_empty_proven = terminate_owned_tree(
                                owner, deadline=termination_deadline
                            )
                        except Exception as error:
                            raise ProcessOwnershipError(phase.label) from error
                        if not tree_empty_proven:
                            _report_diagnostic(
                                "could not verify process-tree termination for "
                                f"{phase.label}"
                            )
                            raise ProcessOwnershipError(phase.label)
                finally:
                    if not close_owner(owner):
                        raise ProcessOwnershipError(phase.label)
                if ownership_error is not None:
                    ownership_error.cleanup_proven = tree_empty_proven
                print(f"owned_tree_empty={tree_empty_proven}", flush=True)
    finished_at = time.monotonic()
    elapsed = finished_at - started_at
    if hard_deadline is not None and finished_at > hard_deadline and return_code == 0:
        return_code = TIMEOUT_EXIT_CODE
    print(f"<== {phase.label}: {elapsed:.1f}s (exit {return_code})", flush=True)
    return return_code


def _windows_supervisor_main(command: Sequence[str]) -> int:
    """Wait for Job ownership before starting the requested command phase."""

    if os.name != "nt" or not command:
        return TERMINATION_FAILED_EXIT_CODE
    try:
        token = sys.stdin.buffer.read(len(_WINDOWS_START_TOKEN))
    except OSError:
        return TERMINATION_FAILED_EXIT_CODE
    if token != _WINDOWS_START_TOKEN:
        return TERMINATION_FAILED_EXIT_CODE
    try:
        process = subprocess.Popen(tuple(command))
    except OSError as error:
        print(f"could not start command: {error}", file=sys.stderr, flush=True)
        return TERMINATION_FAILED_EXIT_CODE
    return process.wait()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != _WINDOWS_SUPERVISOR_MODE:
        raise SystemExit(TERMINATION_FAILED_EXIT_CODE)
    raise SystemExit(_windows_supervisor_main(sys.argv[2:]))
