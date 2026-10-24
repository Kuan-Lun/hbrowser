"""Finite same-group diagnostic writers; never alter inherited descriptor flags."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from ._process_scope import (
    ENVIRONMENT_KEY,
    PosixScope,
    ScopeAdmissionClosed,
)

if TYPE_CHECKING:
    from .process import SignalController, WindowsJob

MAX_PAYLOAD_BYTES = 20 * 1024
DIAGNOSTIC_SECONDS = 0.25
MINIMUM_ADMISSION_SECONDS = 0.025
_TRUNCATED = "\n[diagnostic truncated]"
# A fixed isolated interpreter has no site hooks, imports only stdlib and never
# launches descendants. Windows reads its single start byte only after Job.assign.
_WRITER = """import base64, os, sys
if sys.argv[1] == 'gate' and sys.stdin.buffer.read(1) != b'\\x01':
    raise SystemExit(125)
payload = base64.b64decode(sys.argv[2], validate=True)
while payload:
    written = os.write(1, payload)
    if not written:
        break
    payload = payload[written:]
"""


def encode_diagnostic(
    message: str, *, encoding: str, errors: str, newline: bool = True
) -> bytes:
    """Bound argv bytes while preserving whole encoded Unicode characters."""
    terminator = "\n" if newline else ""
    upper = min(len(message), MAX_PAYLOAD_BYTES)
    prefix = message[:upper]
    suffix = _TRUNCATED if upper < len(message) else ""
    encoded = (prefix + suffix + terminator).encode(encoding, errors=errors)
    if len(encoded) <= MAX_PAYLOAD_BYTES:
        return encoded
    low, high = 0, upper
    while low < high:
        middle = (low + high + 1) // 2
        candidate = (prefix[:middle] + _TRUNCATED + terminator).encode(
            encoding, errors=errors
        )
        if len(candidate) <= MAX_PAYLOAD_BYTES:
            low = middle
        else:
            high = middle - 1
    return (prefix[:low] + _TRUNCATED + terminator).encode(encoding, errors=errors)


def write_diagnostic(
    descriptor: int,
    payload: bytes,
    *,
    deadline: float,
    signal_controller: SignalController,
) -> None:
    """Write or discard human output, proving helper cleanup before returning.

    Spawn, write, kill and reap share deadline. The first half is for execution;
    the rest is retained for cleanup, never a new grace period. Expired or tiny
    budgets do not admit a helper. An inherited POSIX owner must admit a leaf
    obligation before spawn; a closed parent discards this optional output. No
    process group, session or recursive command runner is created.
    """
    from .process import ProcessOwnershipError, WindowsJob, seconds_remaining

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Diagnostic payload exceeds its explicit argv bound")
    started = time.monotonic()
    if deadline - started < MINIMUM_ADMISSION_SECONDS:
        return
    execution_deadline = started + (deadline - started) / 2
    child: subprocess.Popen[bytes] | None = None
    job: WindowsJob | None = None
    assigned = False
    scope: PosixScope | None = None
    try:
        with signal_controller.defer():
            if seconds_remaining(execution_deadline) == 0:
                return
            executable = sys.executable
            environment = None
            if os.name == "nt":
                executable = getattr(sys, "_base_executable", "")
                if not isinstance(executable, str) or not executable:
                    raise OSError("Windows diagnostic base interpreter unavailable")
                environment = dict(os.environ)
                environment.pop("__PYVENV_LAUNCHER__", None)
                job = WindowsJob.create()
            command = (
                executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _WRITER,
                "gate" if job is not None else "write",
                base64.b64encode(payload).decode("ascii"),
            )
            if os.name == "nt" and (
                len(subprocess.list2cmdline(command).encode("utf-16-le")) // 2 >= 32767
            ):
                return
            if os.name == "posix" and ENVIRONMENT_KEY in os.environ:
                try:
                    scope = PosixScope.create(deadline=deadline)
                except ScopeAdmissionClosed:
                    return
                except Exception as error:
                    raise ProcessOwnershipError(
                        "Diagnostic writer admission is unproven"
                    ) from error
                # Admission may have consumed the execution slice. Even without
                # a child, the registered obligation must be closed below.
                signal_controller.raise_pending()
                if seconds_remaining(execution_deadline) == 0:
                    return
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if job is not None else subprocess.DEVNULL,
                stdout=descriptor,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            if job is not None:
                job.assign(child)
                assigned = True
                signal_controller.raise_pending()
                if seconds_remaining(execution_deadline) == 0:
                    return
                if child.stdin is None:
                    raise OSError("Diagnostic start gate unavailable")
                if child.stdin.write(b"\x01") != 1:
                    raise OSError("Diagnostic start gate was partial")
                child.stdin.close()
        child.wait(timeout=seconds_remaining(execution_deadline))
    except OSError, subprocess.TimeoutExpired:
        # The diagnostic itself is optional. Ownership below is not optional.
        pass
    finally:
        if child is not None or job is not None or scope is not None:
            with signal_controller.defer():
                clean = True
                cleanup_error: Exception | None = None
                if child is not None:
                    if child.stdin is not None:
                        try:
                            child.stdin.close()
                        except Exception as error:
                            clean = False
                            cleanup_error = cleanup_error or error
                    try:
                        if job is not None and assigned:
                            job.terminate()
                        elif child.poll() is None:
                            child.kill()
                    except Exception as error:
                        clean = False
                        cleanup_error = cleanup_error or error
                    if job is not None and assigned:
                        try:
                            clean = job.wait_empty(deadline=deadline) and clean
                        except Exception as error:
                            clean = False
                            cleanup_error = cleanup_error or error
                    try:
                        child.wait(timeout=seconds_remaining(deadline))
                    except Exception as error:
                        clean = False
                        cleanup_error = cleanup_error or error
                if job is not None:
                    try:
                        job.close()
                    except Exception as error:
                        clean = False
                        cleanup_error = cleanup_error or error
                if clean and scope is not None:
                    # The fixed isolated leaf cannot spawn descendants. A
                    # successful wait (or no Popen) is its own proof, independent
                    # of any main-command owner's previous empty-tree result.
                    scope.proven_empty = True
                    try:
                        scope.close(deadline=deadline)
                    except Exception as error:
                        clean = False
                        cleanup_error = cleanup_error or error
                if not clean:
                    # Never release an obligation without reap and local scope
                    # removal proof. A close failure retains that responsibility
                    # and must not be turned into optional discarded output.
                    raise ProcessOwnershipError(
                        "Diagnostic writer cleanup could not be proven"
                    ) from cleanup_error
