"""Start-gated subprocess supervisor used by hbrowser process owners."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from enum import Enum, auto
from pathlib import Path

_POLL_SECONDS = 0.05
_TERM_GRACE_SECONDS = 2.0
_KILL_PROOF_SECONDS = 2.0
_EXIT_PROOF_RECONCILIATION_SECONDS = 1.0
_TARGET_ONLY_ENVIRONMENT_KEYS = (
    "HBROWSER_LOG_FORWARD_ENDPOINT",
    "HBROWSER_LOG_FORWARD_TOKEN",
)


class _TargetGroupIdentity(Enum):
    LIVE_OWNED = auto()
    EXITED_PINNED = auto()


def _write_status(status_path: Path, value: str) -> None:
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(f"{value}\n", encoding="utf-8")
    os.replace(temporary, status_path)


def _target_exited_without_reaping(target: subprocess.Popen[bytes]) -> bool:
    if os.name != "posix" or not hasattr(os, "waitid"):
        return target.poll() is not None
    try:
        result = os.waitid(
            os.P_PID,
            target.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:
        raise RuntimeError("Owned target identity was reaped unexpectedly") from None
    return result is not None


def _wait_for_target_exit_proof(target: subprocess.Popen[bytes]) -> bool:
    """Boundedly reconcile an ESRCH identity lookup with a WNOWAIT receipt."""

    deadline = time.monotonic() + _EXIT_PROOF_RECONCILIATION_SECONDS
    while True:
        if _target_exited_without_reaping(target):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_POLL_SECONDS, remaining))


def _process_group_members(process_group: int) -> tuple[int, ...]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid="],
        check=True,
        capture_output=True,
        text=True,
        timeout=1,
    )
    members: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, pgid = (int(field) for field in fields)
        except ValueError:
            continue
        if pgid == process_group:
            members.append(pid)
    return tuple(members)


def _wait_for_killed_target_group(target: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + _KILL_PROOF_SECONDS
    while time.monotonic() < deadline:
        members = _process_group_members(target.pid)
        if _target_exited_without_reaping(target) and all(
            pid == target.pid for pid in members
        ):
            return
        time.sleep(_POLL_SECONDS)
    members = tuple(
        pid for pid in _process_group_members(target.pid) if pid != target.pid
    )
    raise RuntimeError(
        "Owned target process group remained after SIGKILL "
        f"(descendant_count={len(members)})"
    )


def _target_group_identity(
    target: subprocess.Popen[bytes],
) -> _TargetGroupIdentity:
    """Validate a live target or return its pinned post-exit identity state."""

    try:
        process_group = os.getpgid(target.pid)
        session = os.getsid(target.pid)
    except ProcessLookupError as error:
        # The target can exit after the caller's waitid(WNOWAIT) probe but
        # before either identity lookup.  Because it remains our unreaped
        # child, a positive second probe pins the PID and makes the cached
        # process-group identity safe to inspect during cleanup.
        if _wait_for_target_exit_proof(target):
            return _TargetGroupIdentity.EXITED_PINNED
        raise RuntimeError(
            "Owned target identity disappeared before cleanup"
        ) from error
    if process_group != target.pid or session != os.getsid(0):
        raise RuntimeError("Owned target escaped its assigned process group")
    return _TargetGroupIdentity.LIVE_OWNED


def _terminate_posix_target(
    target: subprocess.Popen[bytes],
    *,
    force_requested: threading.Event,
) -> None:
    target_exited = _target_exited_without_reaping(target)
    if not target_exited:
        target_exited = (
            _target_group_identity(target) is _TargetGroupIdentity.EXITED_PINNED
        )
    if target_exited and not any(
        pid != target.pid for pid in _process_group_members(target.pid)
    ):
        # A normally exiting target is still our unreaped child. Verify that
        # its process group is empty, then reap it without requiring a live
        # getpgid identity.
        target.wait()
        return
    if not force_requested.is_set():
        try:
            os.killpg(target.pid, signal.SIGTERM)
        except PermissionError, ProcessLookupError:
            pass
        deadline = time.monotonic() + _TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            if force_requested.is_set():
                break
            target_exited = _target_exited_without_reaping(target)
            descendants = tuple(
                pid for pid in _process_group_members(target.pid) if pid != target.pid
            )
            if target_exited and not descendants:
                target.wait()
                return
            time.sleep(_POLL_SECONDS)
    try:
        os.killpg(target.pid, signal.SIGKILL)
    except PermissionError, ProcessLookupError:
        pass
    _wait_for_killed_target_group(target)
    target.wait()


def _terminate_windows_target(
    target: subprocess.Popen[bytes],
    *,
    force_requested: threading.Event,
) -> None:
    if not force_requested.is_set():
        try:
            target.terminate()
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + _TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            if force_requested.is_set() or target.poll() is not None:
                return
            time.sleep(_POLL_SECONDS)
    try:
        target.kill()
    except ProcessLookupError:
        pass
    target.wait(timeout=_TERM_GRACE_SECONDS)


def _parse_arguments(arguments: Sequence[str]) -> tuple[Path, tuple[str, ...]]:
    if len(arguments) < 3 or arguments[1] != "--":
        raise ValueError("invalid supervisor arguments")
    status_path = Path(arguments[0])
    command = tuple(arguments[2:])
    if not command:
        raise ValueError("invalid supervisor arguments")
    return status_path, command


def _read_start_gate(file_descriptor: int) -> bytes:
    command = bytearray()
    while len(command) <= 64:
        chunk = os.read(file_descriptor, 1)
        if not chunk:
            break
        command.extend(chunk)
        if chunk == b"\n":
            break
    return bytes(command)


def _take_target_environment() -> dict[str, str]:
    """Consume target-only capabilities before the supervisor spawns helpers."""

    target_only: dict[str, str] = {}
    for key in _TARGET_ONLY_ENVIRONMENT_KEYS:
        value = os.environ.pop(key, None)
        if value is not None:
            target_only[key] = value
    target_environment = os.environ.copy()
    target_environment.update(target_only)
    return target_environment


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        status_path, command = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
    except OSError, ValueError:
        return 4

    target_environment = _take_target_environment()

    control_file_descriptor = sys.stdin.fileno()
    start_command = _read_start_gate(control_file_descriptor)
    if start_command != b"start\n":
        _write_status(status_path, "error InvalidStartGate")
        return 4

    shutdown_requested = threading.Event()
    force_shutdown_requested = threading.Event()
    if os.name == "posix":
        signal.signal(signal.SIGINT, lambda *_: shutdown_requested.set())
        signal.signal(signal.SIGTERM, lambda *_: shutdown_requested.set())
    elif os.name == "nt":
        pass
    else:
        _write_status(status_path, "error UnsupportedPlatform")
        return 4

    try:
        try:
            if os.name == "posix":
                target = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    env=target_environment,
                    process_group=0,
                )
            else:
                target = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    env=target_environment,
                )
        finally:
            target_environment.clear()
    except OSError as error:
        _write_status(status_path, f"error {type(error).__name__}")
        return 4
    try:
        _write_status(status_path, f"ready {target.pid}")

        def watch_control_pipe() -> None:
            # EOF is the parent-death signal. Unlike a cached parent PID, the
            # pipe cannot be reused by an unrelated process and is safe on Windows.
            pending = bytearray()
            try:
                while chunk := os.read(control_file_descriptor, 64 * 1024):
                    pending.extend(chunk)
                    while b"\n" in pending:
                        raw_command, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        if raw_command == b"kill":
                            force_shutdown_requested.set()
                            shutdown_requested.set()
                        elif raw_command == b"terminate":
                            shutdown_requested.set()
            except OSError:
                pass
            finally:
                shutdown_requested.set()

        threading.Thread(
            target=watch_control_pipe,
            name="hbrowser-owner-control",
            daemon=True,
        ).start()

        while not shutdown_requested.wait(_POLL_SECONDS):
            if _target_exited_without_reaping(target):
                break
    finally:
        if os.name == "posix":
            _terminate_posix_target(
                target,
                force_requested=force_shutdown_requested,
            )
        else:
            _terminate_windows_target(
                target,
                force_requested=force_shutdown_requested,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
