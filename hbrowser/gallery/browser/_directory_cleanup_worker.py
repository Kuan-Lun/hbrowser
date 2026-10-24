"""Start-gated worker for killable private-directory removal."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# Production executes this file directly to keep startup dependency-light;
# tests import it through the package so both module contexts are intentional.
if TYPE_CHECKING or __package__:
    from .process import ProcessOwnershipError, _PrivateDirectory
else:
    from process import ProcessOwnershipError, _PrivateDirectory


_DIAGNOSTIC_LIMIT = 768


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
) -> tuple[_PrivateDirectory, float, Path, _PrivateDirectory | None, bool]:
    if len(arguments) != 9:
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
    ) = arguments
    parsed_path = Path(path)
    device = int(raw_device)
    inode = int(raw_inode)
    deadline = float(raw_deadline)
    parsed_claim_path = Path(claim_path)
    claim_device = int(raw_claim_device)
    claim_inode = int(raw_claim_inode)
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
    ):
        raise ValueError("invalid private cleanup worker identity")
    return (
        _PrivateDirectory(
            path=parsed_path,
            device=device,
            inode=inode,
            platform_name=platform_name,
        ),
        deadline,
        parsed_claim_path,
        None
        if claim_device == -1
        else _PrivateDirectory(
            parsed_claim_path, claim_device, claim_inode, platform_name
        ),
        raw_confirmed == "1",
    )


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
) -> None:
    """Acknowledge phases in order; parent death always ends filesystem work."""

    try:
        if _read_start_gate(file_descriptor) != b"claim\n":
            os._exit(5)
        claim_allowed.set()
        if _read_start_gate(file_descriptor) != b"remove\n":
            os._exit(5)
        removal_allowed.set()
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
    try:
        guard, deadline, claim_path, namespace, confirmed = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
        stage = "start-gate"
        if _read_start_gate(sys.stdin.fileno()) != b"start\n":
            return 4
        claim_allowed = threading.Event()
        removal_allowed = threading.Event()
        threading.Thread(
            target=_watch_parent_channel,
            args=(sys.stdin.fileno(), claim_allowed, removal_allowed),
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
        elif not namespace._assert_identity_or_absence():
            raise ProcessOwnershipError("Private cleanup namespace is absent")
        sys.stdout.write(f"ready {namespace.device} {namespace.inode}\n")
        sys.stdout.flush()
        _wait_permission(claim_allowed, deadline=deadline)
        stage = "directory-claim"
        guard._claim_inline(namespace, confirmed=confirmed, deadline=deadline)
        sys.stdout.write("claimed\n")
        sys.stdout.flush()
        _wait_permission(removal_allowed, deadline=deadline)
        stage = "directory-removal"
        guard._remove_inline(namespace=namespace, deadline=deadline)
    except BaseException as error:
        _report_failure(stage, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
