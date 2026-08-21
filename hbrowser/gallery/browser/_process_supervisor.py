"""Start-gated subprocess supervisor used by hbrowser process owners."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

_POLL_SECONDS = 0.05
_TERM_GRACE_SECONDS = 2.0
_KILL_PROOF_SECONDS = 2.0


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


def _assert_target_group_identity(target: subprocess.Popen[bytes]) -> None:
    try:
        process_group = os.getpgid(target.pid)
        session = os.getsid(target.pid)
    except ProcessLookupError as error:
        raise RuntimeError(
            "Owned target identity disappeared before cleanup"
        ) from error
    if process_group != target.pid or session != os.getsid(0):
        raise RuntimeError("Owned target escaped its assigned process group")


def _terminate_posix_target(
    target: subprocess.Popen[bytes],
    *,
    force_requested: threading.Event,
) -> None:
    _assert_target_group_identity(target)
    if not force_requested.is_set():
        try:
            os.killpg(target.pid, signal.SIGTERM)
        except PermissionError, ProcessLookupError:
            pass
        deadline = time.monotonic() + _TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            if force_requested.is_set() or _target_exited_without_reaping(target):
                break
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


def _parent_is_alive(expected_parent_pid: int) -> bool:
    if os.getppid() != expected_parent_pid:
        return False
    try:
        os.kill(expected_parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_arguments(arguments: Sequence[str]) -> tuple[Path, int, tuple[str, ...]]:
    if len(arguments) < 4 or arguments[2] != "--":
        raise ValueError("invalid supervisor arguments")
    status_path = Path(arguments[0])
    parent_pid = int(arguments[1])
    command = tuple(arguments[3:])
    if parent_pid <= 0 or not command:
        raise ValueError("invalid supervisor arguments")
    return status_path, parent_pid, command


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


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        status_path, parent_pid, command = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
    except OSError, ValueError:
        return 4

    control_file_descriptor = sys.stdin.fileno()
    start_command = _read_start_gate(control_file_descriptor)
    if start_command != b"start\n":
        _write_status(status_path, "error InvalidStartGate")
        return 4
    if not _parent_is_alive(parent_pid):
        _write_status(status_path, "error ParentUnavailable")
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
        if os.name == "posix":
            target = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                process_group=0,
            )
        else:
            target = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
    except OSError as error:
        _write_status(status_path, f"error {type(error).__name__}")
        return 4
    try:
        _write_status(status_path, f"ready {target.pid}")

        def watch_control_pipe() -> None:
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
            if not _parent_is_alive(parent_pid):
                shutdown_requested.set()
                break
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
