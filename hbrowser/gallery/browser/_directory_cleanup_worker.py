"""Start-gated worker for killable private-directory removal."""

from __future__ import annotations

import math
import os
import socket
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# Production executes this file directly to keep startup dependency-light;
# tests import it through the package so both module contexts are intentional.
if TYPE_CHECKING or __package__:
    from ._directory_identity import DirectoryPin, _WindowsDirectoryApi, send_pin
    from .process import ProcessOwnershipError, _PrivateDirectory
else:
    from _directory_identity import DirectoryPin, _WindowsDirectoryApi, send_pin
    from process import ProcessOwnershipError, _PrivateDirectory


_DIAGNOSTIC_LIMIT = 768
# A ready frame may still be in flight when this worker fails. Keep its native
# handles until atomic OS process exit; a live Windows process must never recycle
# a published HANDLE before the parent can DuplicateHandle it. Explicit Windows
# delete-pending acknowledgements below are the only earlier retirement points.
_PROCESS_LIFETIME_PINS: list[DirectoryPin] = []


def _single_line(value: object, *, limit: int) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")[:limit]


def _report_failure(stage: str, error: BaseException) -> None:
    notes = " | ".join(
        _single_line(note, limit=160) for note in getattr(error, "__notes__", ())
    )
    diagnostic = (
        f"stage={stage} error={type(error).__name__} "
        f"errno={getattr(error, 'errno', None)} "
        f"winerror={getattr(error, 'winerror', None)} "
        f"message={_single_line(error, limit=320)}"
    )
    if notes:
        diagnostic = f"{diagnostic} notes={notes}"
    payload = (diagnostic + "\n").encode("ascii", errors="backslashreplace")
    if len(payload) > _DIAGNOSTIC_LIMIT:
        payload = payload[: _DIAGNOSTIC_LIMIT - 1] + b"\n"
    try:
        os.write(sys.stderr.fileno(), payload)
    except AttributeError, OSError, ValueError:
        pass


def _parse_arguments(
    arguments: Sequence[str],
) -> tuple[_PrivateDirectory, float, Path, _PrivateDirectory | None, bool, int]:
    if len(arguments) != 12:
        raise ValueError("invalid private cleanup worker arguments")
    (
        path,
        raw_device,
        raw_inode,
        platform_name,
        raw_deadline,
        claim_path,
        raw_claim_device,
        raw_claim_inode,
        raw_confirmed,
        raw_root_handle,
        raw_namespace_handle,
        raw_channel,
    ) = arguments
    parsed_path = Path(path)
    device = int(raw_device)
    inode = int(raw_inode)
    deadline = float(raw_deadline)
    parsed_claim_path = Path(claim_path)
    claim_device = int(raw_claim_device)
    claim_inode = int(raw_claim_inode)
    root_handle = int(raw_root_handle)
    namespace_handle = int(raw_namespace_handle)
    channel = int(raw_channel)
    inherited = [value for value in (root_handle, namespace_handle) if value >= 0]
    if (
        not parsed_path.is_absolute()
        or device < 0
        or inode < 0
        or platform_name not in {"nt", "posix"}
        or not math.isfinite(deadline)
        or not parsed_claim_path.is_absolute()
        or parsed_claim_path.parent != parsed_path.parent
        or not parsed_claim_path.name.startswith(".hbrowser-cleanup-")
        or not (
            (claim_device == -1 and claim_inode == -1)
            or (claim_device >= 0 and claim_inode >= 0)
        )
        or raw_confirmed not in {"0", "1"}
        or (raw_confirmed == "1" and claim_device == -1)
        or root_handle < -1
        or (root_handle == -1 and raw_confirmed != "1")
        or (namespace_handle == -1) != (claim_device == -1)
        or namespace_handle < -1
        or (
            os.name == "posix"
            and (
                channel < 3
                or any(value in {0, 1, 2} for value in (root_handle, namespace_handle))
            )
        )
        or (os.name == "nt" and channel != -1)
        or len(set(inherited)) != len(inherited)
        or channel in inherited
    ):
        raise ValueError("invalid private cleanup worker identity")
    # Prepare before consuming any inherited handle. If preparation itself
    # fails, main exits the worker; no ownership handoff or ready frame occurred.
    windows_api = _WindowsDirectoryApi() if os.name == "nt" else None
    adopted: list[DirectoryPin] = []
    try:
        if root_handle >= 0:
            inherited.remove(root_handle)
        root_pin = (
            DirectoryPin.completed(device, inode)
            if root_handle == -1
            else DirectoryPin.inherit(root_handle, windows_api=windows_api)
        )
        adopted.append(root_pin)
        if (root_pin.device, root_pin.inode) != (device, inode):
            raise ProcessOwnershipError("Inherited private root identity changed")
        guard = _PrivateDirectory.from_pin(
            parsed_path, root_pin, platform_name=platform_name
        )
        namespace = None
        if namespace_handle != -1:
            inherited.remove(namespace_handle)
            namespace_pin = DirectoryPin.inherit(
                namespace_handle, windows_api=windows_api
            )
            adopted.append(namespace_pin)
            if (namespace_pin.device, namespace_pin.inode) != (
                claim_device,
                claim_inode,
            ):
                raise ProcessOwnershipError(
                    "Inherited private namespace identity changed"
                )
            namespace = _PrivateDirectory.from_pin(
                parsed_claim_path, namespace_pin, platform_name=platform_name
            )
        return (
            guard,
            deadline,
            parsed_claim_path,
            namespace,
            raw_confirmed == "1",
            channel,
        )
    except BaseException:
        for pin in adopted:
            pin.retire()
        for native in inherited:
            DirectoryPin.close_inherited(native, windows_api=windows_api)
        if channel >= 0:
            os.close(channel)
        raise


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


def _watch_parent_channel(
    file_descriptor: int,
    claim_allowed: threading.Event,
    removal_allowed: threading.Event,
    root_released: threading.Event | None = None,
    namespace_released: threading.Event | None = None,
) -> None:
    """Acknowledge phases in order; parent death always ends filesystem work."""

    try:
        if _read_start_gate(file_descriptor) != b"claim\n":
            os._exit(5)
        claim_allowed.set()
        if _read_start_gate(file_descriptor) != b"remove\n":
            os._exit(5)
        removal_allowed.set()
        if root_released is not None and namespace_released is not None:
            if _read_start_gate(file_descriptor) != b"release-root\n":
                os._exit(5)
            root_released.set()
            if _read_start_gate(file_descriptor) != b"release-namespace\n":
                os._exit(5)
            namespace_released.set()
        # No further command is valid; EOF and unexpected input both revoke work.
        os.read(file_descriptor, 1)
    except OSError:
        pass
    os._exit(5)


def _wait_permission(permission: threading.Event, *, deadline: float) -> None:
    if not permission.wait(max(0.0, deadline - time.monotonic())):
        raise ProcessOwnershipError("Private cleanup acknowledgement deadline expired")
    if time.monotonic() >= deadline:
        raise ProcessOwnershipError("Late private cleanup acknowledgement")


def main(arguments: Sequence[str] | None = None) -> int:
    stage = "argument-parsing"
    guard: _PrivateDirectory | None = None
    namespace: _PrivateDirectory | None = None
    channel: socket.socket | None = None
    try:
        guard, deadline, claim_path, namespace, confirmed, channel_handle = (
            _parse_arguments(tuple(sys.argv[1:] if arguments is None else arguments))
        )
        _PROCESS_LIFETIME_PINS.append(guard._pin)
        if namespace is not None:
            _PROCESS_LIFETIME_PINS.append(namespace._pin)
        if channel_handle != -1:
            channel = socket.socket(fileno=channel_handle)
            channel.set_inheritable(False)
        stage = "start-gate"
        if _read_start_gate(sys.stdin.fileno()) != b"start\n":
            return 4
        claim_allowed = threading.Event()
        removal_allowed = threading.Event()
        root_released = threading.Event() if os.name == "nt" else None
        namespace_released = threading.Event() if os.name == "nt" else None
        threading.Thread(
            target=_watch_parent_channel,
            args=(
                sys.stdin.fileno(),
                claim_allowed,
                removal_allowed,
                root_released,
                namespace_released,
            ),
            name="hbrowser-private-cleanup-parent-watch",
            daemon=True,
        ).start()
        stage = "namespace-setup"
        if time.monotonic() >= deadline:
            raise ProcessOwnershipError("Private cleanup setup deadline expired")
        if namespace is None:
            # Creation itself stays in this killable worker. An interrupted setup
            # without a complete receipt remains an unverified parent intent.
            claim_path.mkdir(mode=0o700)
            namespace = _PrivateDirectory.capture(
                claim_path, platform_name=guard.platform_name
            )
            _PROCESS_LIFETIME_PINS.append(namespace._pin)
        elif not namespace._assert_identity_or_absence():
            raise ProcessOwnershipError("Private cleanup namespace is absent")
        native_suffix = ""
        if channel is not None:
            send_pin(channel, namespace._pin)
        else:
            native_suffix = f" {namespace._pin.native_handle()}"
        sys.stdout.write(f"ready {namespace.device} {namespace.inode}{native_suffix}\n")
        sys.stdout.flush()
        _wait_permission(claim_allowed, deadline=deadline)
        stage = "directory-claim"
        guard._claim_inline(namespace, confirmed=confirmed, deadline=deadline)
        sys.stdout.write("claimed\n")
        sys.stdout.flush()
        _wait_permission(removal_allowed, deadline=deadline)
        stage = "directory-removal"

        def release_deleted_root() -> None:
            assert guard is not None and root_released is not None
            if not guard._pin.deleted():
                raise ProcessOwnershipError("Private root lacks native deletion proof")
            sys.stdout.write("removed\n")
            sys.stdout.flush()
            _wait_permission(root_released, deadline=deadline)
            guard._retire()

        def release_deleted_namespace() -> None:
            assert namespace is not None and namespace_released is not None
            if not namespace._pin.deleted():
                raise ProcessOwnershipError(
                    "Private namespace lacks native deletion proof"
                )
            sys.stdout.write("finished\n")
            sys.stdout.flush()
            _wait_permission(namespace_released, deadline=deadline)
            namespace._retire()

        guard._remove_inline(
            namespace=namespace,
            deadline=deadline,
            release_root=release_deleted_root if os.name == "nt" else None,
            release_namespace=release_deleted_namespace if os.name == "nt" else None,
        )
    except BaseException as error:
        _report_failure(stage, error)
        return 1
    finally:
        if channel is not None:
            channel.close()
    return 0


if __name__ == "__main__":
    # Frames are explicitly flushed and failures use os.write. Python finalizers
    # must not close published handles while the process can still be queried.
    os._exit(main())
