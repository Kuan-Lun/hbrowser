"""Write one page diagnostic inside an owned, killable worker process."""

from __future__ import annotations

import json
import math
import os
import secrets
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ..utils.diagnostic import (
    _PAGE_DIAGNOSTIC_KIND_PATTERN,
    _PAGE_DIAGNOSTIC_MAX_FILE_BYTES,
    _PAGE_DIAGNOSTIC_NONCE_PATTERN,
    _PAGE_DIAGNOSTIC_TOTAL_BYTES,
    _write_prepared_page_diagnostic,
)
from ._page_diagnostic_transport import receive_payload

_RECEIPT_SCHEMA = 1
_RECEIPT_MAX_BYTES = 4096


def _parse_arguments(
    arguments: Sequence[str],
) -> tuple[Path, str, str, float, int]:
    if len(arguments) != 5:
        raise ValueError("invalid page diagnostic worker arguments")
    raw_directory, kind, nonce, raw_deadline, raw_size = arguments
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
        or not 0 <= payload_size <= maximum_payload_size
    ):
        raise ValueError("invalid page diagnostic worker arguments")
    return directory, kind, nonce, deadline, payload_size


def _require_budget(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Page diagnostic worker deadline expired before {phase}")


def _read_payload(payload_size: int, *, nonce: str, deadline: float) -> bytes:
    _require_budget(deadline, "payload listener creation")
    capability = secrets.token_bytes(32)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        _publish_frame(
            {
                "schema": _RECEIPT_SCHEMA,
                "phase": "ready",
                "nonce": nonce,
                "size": payload_size,
                "port": listener.getsockname()[1],
                "capability": capability.hex(),
            },
            deadline=deadline,
        )
        return receive_payload(
            listener, capability, nonce, payload_size, deadline=deadline
        )


def _publish_frame(frame: dict[str, str | int], *, deadline: float) -> None:
    encoded = (
        json.dumps(frame, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    if len(encoded) > _RECEIPT_MAX_BYTES:
        raise RuntimeError("page diagnostic frame exceeded its size limit")
    _require_budget(deadline, "frame publication")
    offset = 0
    while offset < len(encoded):
        written = os.write(sys.stdout.fileno(), encoded[offset:])
        if written <= 0:
            raise BrokenPipeError("page diagnostic frame pipe closed")
        offset += written
        _require_budget(deadline, "frame write completion")


def _publish_receipt(
    *,
    nonce: str,
    filename: str,
    deadline: float,
) -> None:
    _publish_frame(
        {"schema": _RECEIPT_SCHEMA, "nonce": nonce, "filename": filename},
        deadline=deadline,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        directory, kind, nonce, deadline, payload_size = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
        payload = _read_payload(payload_size, nonce=nonce, deadline=deadline)
        _require_budget(deadline, "diagnostic directory resolution")
        directory = directory.resolve()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_budget(deadline, "diagnostic directory creation")
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
