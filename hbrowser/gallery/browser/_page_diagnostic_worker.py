"""Write one page diagnostic inside an owned, killable worker process."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

from ..utils.diagnostic import (
    _PAGE_DIAGNOSTIC_KIND_PATTERN,
    _PAGE_DIAGNOSTIC_MAX_FILE_BYTES,
    _PAGE_DIAGNOSTIC_NONCE_PATTERN,
    _PAGE_DIAGNOSTIC_TOTAL_BYTES,
    _write_prepared_page_diagnostic,
)

_RECEIPT_SCHEMA = 1
_RECEIPT_MAX_BYTES = 4096


def _parse_arguments(
    arguments: Sequence[str],
) -> tuple[Path, str, str, float, str, int]:
    if len(arguments) != 6:
        raise ValueError("invalid page diagnostic worker arguments")
    raw_directory, kind, nonce, raw_deadline, shared_memory_name, raw_size = arguments
    directory = Path(raw_directory)
    deadline = float(raw_deadline)
    payload_size = int(raw_size)
    maximum_payload_size = min(
        _PAGE_DIAGNOSTIC_MAX_FILE_BYTES,
        _PAGE_DIAGNOSTIC_TOTAL_BYTES,
    )
    if (
        not directory.is_absolute()
        or _PAGE_DIAGNOSTIC_KIND_PATTERN.fullmatch(kind) is None
        or _PAGE_DIAGNOSTIC_NONCE_PATTERN.fullmatch(nonce) is None
        or not math.isfinite(deadline)
        or not shared_memory_name
        or not 0 <= payload_size <= maximum_payload_size
    ):
        raise ValueError("invalid page diagnostic worker arguments")
    return directory, kind, nonce, deadline, shared_memory_name, payload_size


def _require_budget(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Page diagnostic worker deadline expired before {phase}")


def _read_payload(
    shared_memory_name: str,
    payload_size: int,
    *,
    deadline: float,
) -> bytes:
    _require_budget(deadline, "shared-memory attachment")
    shared_memory = SharedMemory(name=shared_memory_name, track=False)
    try:
        if payload_size > shared_memory.size:
            raise ValueError("page diagnostic payload exceeded shared memory")
        payload_buffer = shared_memory.buf
        if payload_buffer is None:
            raise RuntimeError("page diagnostic shared memory is unavailable")
        try:
            payload = bytes(payload_buffer[:payload_size])
        finally:
            del payload_buffer
    finally:
        shared_memory.close()
    _require_budget(deadline, "shared-memory copy")
    return payload


def _publish_receipt(
    *,
    nonce: str,
    filename: str,
    deadline: float,
) -> None:
    encoded_receipt = (
        json.dumps(
            {
                "schema": _RECEIPT_SCHEMA,
                "nonce": nonce,
                "filename": filename,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded_receipt) > _RECEIPT_MAX_BYTES:
        raise RuntimeError("page diagnostic receipt exceeded its size limit")
    _require_budget(deadline, "receipt publication")
    offset = 0
    while offset < len(encoded_receipt):
        written = os.write(sys.stdout.fileno(), encoded_receipt[offset:])
        if written <= 0:
            raise BrokenPipeError("page diagnostic receipt pipe closed")
        offset += written
        _require_budget(deadline, "receipt write completion")


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        directory, kind, nonce, deadline, shared_memory_name, payload_size = (
            _parse_arguments(tuple(sys.argv[1:] if arguments is None else arguments))
        )
        _require_budget(deadline, "diagnostic directory resolution")
        directory = directory.resolve()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_budget(deadline, "diagnostic directory creation")
        payload = _read_payload(
            shared_memory_name,
            payload_size,
            deadline=deadline,
        )
        path = _write_prepared_page_diagnostic(
            directory,
            kind,
            payload,
            deadline=deadline,
            nonce=nonce,
        )
        _publish_receipt(
            nonce=nonce,
            filename=path.name,
            deadline=deadline,
        )
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
