"""Start-gated worker for killable private-directory removal."""

from __future__ import annotations

import math
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# Production executes this file directly to keep startup dependency-light;
# tests import it through the package so both module contexts are intentional.
if TYPE_CHECKING:
    from .process import _PrivateDirectory
elif __package__:
    from .process import _PrivateDirectory
else:
    from process import _PrivateDirectory


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


def _parse_arguments(arguments: Sequence[str]) -> tuple[_PrivateDirectory, float]:
    if len(arguments) != 5:
        raise ValueError("invalid private cleanup worker arguments")
    path, raw_device, raw_inode, platform_name, raw_deadline = arguments
    parsed_path = Path(path)
    device = int(raw_device)
    inode = int(raw_inode)
    deadline = float(raw_deadline)
    if (
        not parsed_path.is_absolute()
        or device < 0
        or inode < 0
        or platform_name not in {"nt", "posix"}
        or not math.isfinite(deadline)
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


def _exit_when_parent_channel_closes(file_descriptor: int) -> None:
    """Make parent death terminate an in-flight filesystem mutation."""

    try:
        while os.read(file_descriptor, 64 * 1024):
            pass
    except OSError:
        pass
    os._exit(5)


def main(arguments: Sequence[str] | None = None) -> int:
    stage = "argument-parsing"
    try:
        guard, deadline = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
        stage = "start-gate"
        if _read_start_gate(sys.stdin.fileno()) != b"start\n":
            return 4
        threading.Thread(
            target=_exit_when_parent_channel_closes,
            args=(sys.stdin.fileno(),),
            name="hbrowser-private-cleanup-parent-watch",
            daemon=True,
        ).start()
        stage = "directory-removal"
        guard._remove_inline(deadline=deadline)
    except BaseException as error:
        _report_failure(stage, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
