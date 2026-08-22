"""Start-gated worker for killable private-directory removal."""

from __future__ import annotations

import math
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from .process import _PrivateDirectory


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
    try:
        guard, deadline = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
        if _read_start_gate(sys.stdin.fileno()) != b"start\n":
            return 4
        threading.Thread(
            target=_exit_when_parent_channel_closes,
            args=(sys.stdin.fileno(),),
            name="hbrowser-private-cleanup-parent-watch",
            daemon=True,
        ).start()
        guard._remove_inline(deadline=deadline)
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
