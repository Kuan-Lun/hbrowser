"""Owned Chrome artifact installer with an atomic, nonce-bound receipt."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ..utils import Deadline
from .chrome_manager import ensure_chrome_installed

_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}")
_RECEIPT_SCHEMA = 1


def _parse_arguments(arguments: Sequence[str]) -> tuple[Path, str, float, Path]:
    if len(arguments) != 4:
        raise ValueError("invalid Chrome install worker arguments")
    raw_receipt, nonce, raw_deadline, raw_staging_root = arguments
    receipt_path = Path(raw_receipt)
    staging_root = Path(raw_staging_root)
    deadline = float(raw_deadline)
    if (
        not receipt_path.is_absolute()
        or receipt_path.name != "receipt.json"
        or not staging_root.is_absolute()
        or _NONCE_PATTERN.fullmatch(nonce) is None
        or not math.isfinite(deadline)
    ):
        raise ValueError("invalid Chrome install worker arguments")
    return receipt_path, nonce, deadline, staging_root


def _publish_receipt(
    receipt_path: Path,
    payload: dict[str, object],
    *,
    deadline: float,
) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Chrome install receipt deadline expired before write")
    temporary_path = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    if len(encoded) > 16 * 1024:
        raise RuntimeError("Chrome install receipt exceeded its size limit")
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    if time.monotonic() >= deadline:
        temporary_path.unlink(missing_ok=True)
        raise TimeoutError("Chrome install receipt completed after its deadline")
    os.replace(temporary_path, receipt_path)
    if time.monotonic() >= deadline:
        receipt_path.unlink(missing_ok=True)
        raise TimeoutError("Chrome install receipt published after its deadline")


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        receipt_path, nonce, deadline, staging_root = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
        result = ensure_chrome_installed(
            deadline=Deadline(deadline),
            staging_root=staging_root,
        )
        _publish_receipt(
            receipt_path,
            {
                "schema": _RECEIPT_SCHEMA,
                "nonce": nonce,
                "chrome": result.chrome,
                "version": result.version,
            },
            deadline=deadline,
        )
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
