"""Machine receipts retain canonical bytes under Windows text translation."""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from hbrowser.gallery.browser import _directory_cleanup_worker as cleanup_worker
from hbrowser.gallery.browser import _process_supervisor as supervisor
from hbrowser.gallery.browser._process_control import ReadyReceipt


@pytest.mark.parametrize("value", ["hbrowser-ready-v1 11 22 33 44", "error OSError"])
def test_supervisor_status_uses_lf_with_windows_text_defaults(
    tmp_path: Path, value: str
) -> None:
    status = tmp_path / "status"
    temporary = status.with_suffix(".tmp")
    original_open = io.open

    def windows_text_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if file == temporary and "b" not in mode:
            if len(args) >= 4:
                args = (*args[:3], "\r\n", *args[4:])
            else:
                kwargs["newline"] = "\r\n"
        return original_open(file, mode, *args, **kwargs)

    with patch.object(io, "open", windows_text_open):
        supervisor._write_status(status, value)
    frame = status.read_bytes()
    assert frame == value.encode("ascii") + b"\n"
    assert not temporary.exists()
    if value.startswith("hbrowser-ready-v1"):
        assert ReadyReceipt.parse(frame) == ReadyReceipt(11, 22, 33, 44)


def test_cleanup_worker_emits_all_windows_handshake_phases_as_lf() -> None:
    guard = Mock()
    namespace = Mock(device=22, inode=33)
    namespace._pin.native_handle.return_value = 44
    guard._pin.deleted.return_value = True
    namespace._pin.deleted.return_value = True
    claim_path = Path("unused-namespace")
    stdin = Mock()
    stdin.fileno.return_value = 3
    deadline = time.monotonic() + 5

    def remove_inline(**kwargs: Any) -> None:
        kwargs["release_root"]()
        kwargs["release_namespace"]()

    guard._remove_inline.side_effect = remove_inline
    output = io.BytesIO()
    with (
        io.TextIOWrapper(output, encoding="ascii", newline="\r\n") as text_output,
        patch.object(sys, "stdout", text_output),
        patch.object(sys, "stdin", stdin),
        patch.object(cleanup_worker, "os", SimpleNamespace(name="nt")),
        patch.object(cleanup_worker, "_PROCESS_LIFETIME_PINS", []),
        patch.object(
            cleanup_worker,
            "_parse_arguments",
            return_value=(guard, deadline, claim_path, namespace, True, -1),
        ),
        patch.object(cleanup_worker, "_read_start_gate", return_value=b"start\n"),
        patch.object(threading, "Thread") as thread,
        patch.object(cleanup_worker, "_wait_permission") as permission,
    ):
        assert cleanup_worker.main([]) == 0
        assert output.getvalue() == b"ready 22 33 44\nclaimed\nremoved\nfinished\n"
        thread.return_value.start.assert_called_once_with()
        assert permission.call_count == 4
        guard._claim_inline.assert_called_once_with(
            namespace, confirmed=True, deadline=deadline
        )
        guard._retire.assert_called_once_with()
        namespace._retire.assert_called_once_with()


def test_ready_parser_still_rejects_translated_crlf() -> None:
    with pytest.raises(ValueError, match="invalid READY receipt schema"):
        ReadyReceipt.parse(b"hbrowser-ready-v1 11 22 33 44\r\n")
