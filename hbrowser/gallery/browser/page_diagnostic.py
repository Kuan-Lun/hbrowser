"""Own the hard-deadline worker used for page diagnostic persistence."""

from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import subprocess
import sys
from asyncio import Task
from pathlib import Path

from ..utils.deadline import Deadline
from ..utils.diagnostic import (
    _PAGE_DIAGNOSTIC_FILENAME_PATTERN,
    _PAGE_DIAGNOSTIC_KIND_PATTERN,
    _bounded_page_diagnostic_content,
)
from ._page_diagnostic_transport import send_payload
from ._task_settlement import wait_owned_task_completion
from .process import OwnedProcess, ProcessOwnershipError, start_owned_process

_DIAGNOSTIC_WORKER_MODULE = "hbrowser.gallery.browser._page_diagnostic_worker"
_DIAGNOSTIC_WORKER_STARTUP_SECONDS = 1.0
_DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS = 2.0
_DIAGNOSTIC_CLEANUP_RESERVE_SECONDS = 1.0
_DIAGNOSTIC_RECEIPT_MAX_BYTES = 4096
_DIAGNOSTIC_POLL_SECONDS = 0.01
_RECEIPT_SCHEMA = 1
_LOG_DIRECTORY_ENVIRONMENT_VARIABLE = "HBROWSER_LOG_DIR"


def _page_diagnostic_directory_hint() -> Path:
    """Derive the target lexically without resolving or touching its filesystem."""

    configured_directory = os.getenv(_LOG_DIRECTORY_ENVIRONMENT_VARIABLE)
    if configured_directory:
        directory = Path(configured_directory).expanduser()
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        return directory.absolute()

    script_name = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if script_name and not script_name.startswith("-"):
        script_path = Path(script_name).expanduser()
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        return script_path.absolute().parent / "log"
    return Path.cwd().absolute() / "log"


async def _settle_task[T](
    task: Task[T],
) -> tuple[T | None, BaseException | None, asyncio.CancelledError | None]:
    """Wait through caller cancellation so no ownership thread is detached."""

    cancellation = await wait_owned_task_completion(task)
    try:
        return task.result(), None, cancellation
    except BaseException as error:
        return None, error, cancellation


def _shutdown_owned_worker(
    owner: OwnedProcess,
    *,
    deadline: Deadline,
    force: bool,
) -> int:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise ProcessOwnershipError(
            "Page diagnostic worker ownership deadline expired before cleanup"
        )
    cleanup_timeout = min(
        _DIAGNOSTIC_CLEANUP_RESERVE_SECONDS,
        remaining / 2.0,
    )
    process_timeout = max(0.0, remaining - cleanup_timeout)
    if force:
        owner.kill(deadline=deadline.expires_at)
        return owner.shutdown(
            graceful_timeout=0,
            terminate_timeout=0,
            kill_timeout=min(5.0, process_timeout),
            cleanup_timeout=min(5.0, cleanup_timeout),
            deadline=deadline.expires_at,
        )
    return owner.shutdown(
        graceful_timeout=min(5.0, process_timeout),
        terminate_timeout=0,
        kill_timeout=0,
        cleanup_timeout=min(5.0, cleanup_timeout),
        deadline=deadline.expires_at,
    )


async def _settle_owned_worker(
    owner: OwnedProcess,
    *,
    deadline: Deadline,
    force: bool,
) -> asyncio.CancelledError | None:
    shutdown_task = asyncio.create_task(
        asyncio.to_thread(
            _shutdown_owned_worker,
            owner,
            deadline=deadline,
            force=force,
        )
    )
    _, shutdown_error, cancellation = await _settle_task(shutdown_task)
    if shutdown_error is not None:
        if cancellation is not None:
            shutdown_error.add_note(
                "Worker shutdown was also interrupted by caller cancellation"
            )
        raise shutdown_error.with_traceback(shutdown_error.__traceback__)
    return cancellation


async def _read_frame(
    owner: OwnedProcess, *, deadline: Deadline, require_eof: bool = False
) -> bytes:
    """Read bounded raw pipe bytes without a blocking thread or buffered read."""
    output = owner.stdout
    if output is None:
        raise RuntimeError("Page diagnostic worker receipt pipe is unavailable")
    descriptor = output.fileno()
    os.set_blocking(descriptor, False)
    encoded = bytearray()
    while True:
        if deadline.expired:
            raise TimeoutError(
                "Page diagnostic worker frame exceeded its work deadline"
            )
        try:
            chunk = os.read(
                descriptor, _DIAGNOSTIC_RECEIPT_MAX_BYTES + 1 - len(encoded)
            )
        except BlockingIOError:
            await asyncio.sleep(min(_DIAGNOSTIC_POLL_SECONDS, deadline.remaining()))
            continue
        if deadline.expired:
            raise TimeoutError("Page diagnostic frame arrived after its deadline")
        if not chunk:
            raise RuntimeError("Page diagnostic worker frame was truncated")
        encoded.extend(chunk)
        if len(encoded) > _DIAGNOSTIC_RECEIPT_MAX_BYTES:
            raise RuntimeError("Page diagnostic worker frame exceeded its size limit")
        if b"\n" in encoded:
            if not encoded.endswith(b"\n") or encoded.count(b"\n") != 1:
                raise RuntimeError("Page diagnostic worker returned excess frame bytes")
            if require_eof:
                try:
                    trailing = os.read(descriptor, 1)
                except BlockingIOError:
                    raise RuntimeError(
                        "Page diagnostic final receipt pipe remains open"
                    ) from None
                if trailing:
                    raise RuntimeError("Page diagnostic receipt has trailing bytes")
                if deadline.expired:
                    raise TimeoutError("Page diagnostic receipt EOF arrived too late")
            return bytes(encoded)


async def _transfer_payload(
    owner: OwnedProcess, payload: bytes, *, nonce: str, deadline: Deadline
) -> None:
    encoded = await _read_frame(owner, deadline=deadline)
    try:
        ready: object = json.loads(encoded)
    except json.JSONDecodeError, UnicodeDecodeError:
        raise RuntimeError(
            "Page diagnostic worker returned invalid READY JSON"
        ) from None
    if (
        not isinstance(ready, dict)
        or set(ready) != {"schema", "phase", "nonce", "size", "port", "capability"}
        or type(ready["schema"]) is not int
        or ready["schema"] != _RECEIPT_SCHEMA
        or ready["phase"] != "ready"
        or ready["nonce"] != nonce
        or type(ready["size"]) is not int
        or ready["size"] != len(payload)
        or type(ready["port"]) is not int
        or not 1 <= ready["port"] <= 65535
        or not isinstance(ready["capability"], str)
        or len(ready["capability"]) != 64
        or any(value not in "0123456789abcdef" for value in ready["capability"])
    ):
        raise RuntimeError("Page diagnostic worker returned invalid READY authority")
    await send_payload(
        ready["port"],
        bytes.fromhex(ready["capability"]),
        nonce,
        payload,
        deadline=deadline.expires_at,
    )


async def _read_and_validate_receipt(
    owner: OwnedProcess,
    *,
    directory: Path,
    kind: str,
    nonce: str,
    deadline: Deadline,
) -> Path:
    if deadline.expired:
        raise TimeoutError("Page diagnostic receipt arrived after its deadline")
    encoded_receipt = await _read_frame(owner, deadline=deadline, require_eof=True)
    try:
        receipt: object = json.loads(encoded_receipt)
    except json.JSONDecodeError, UnicodeDecodeError:
        raise RuntimeError("Page diagnostic worker returned invalid JSON") from None
    if deadline.expired:
        raise TimeoutError(
            "Page diagnostic receipt parsing completed after its deadline"
        )
    if not isinstance(receipt, dict) or receipt.get("schema") != _RECEIPT_SCHEMA:
        raise RuntimeError("Page diagnostic worker returned an invalid receipt")
    if receipt.get("nonce") != nonce:
        raise RuntimeError("Page diagnostic receipt identity did not match")
    filename = receipt.get("filename")
    if not isinstance(filename, str):
        raise RuntimeError("Page diagnostic receipt filename was invalid")
    match = _PAGE_DIAGNOSTIC_FILENAME_PATTERN.fullmatch(filename)
    if match is None or match.group("kind") != kind or match.group("nonce") != nonce:
        raise RuntimeError("Page diagnostic receipt path was not trustworthy")
    path = directory / filename
    if deadline.expired:
        raise TimeoutError(
            "Page diagnostic receipt validation completed after its deadline"
        )
    return path


def _validate_operation_deadline(deadline: Deadline) -> None:
    if not isinstance(deadline, Deadline) or not math.isfinite(deadline.expires_at):
        raise ValueError("Page diagnostic owner deadline must be finite")
    if deadline.expired:
        raise TimeoutError("Page diagnostic owner deadline already expired")


async def write_page_diagnostic_owned(
    kind: str,
    content: str,
    *,
    deadline: Deadline,
) -> Path:
    """Write via one start-gated child and prove it is reaped before returning."""

    _validate_operation_deadline(deadline)
    if _PAGE_DIAGNOSTIC_KIND_PATTERN.fullmatch(kind) is None:
        raise ValueError(f"Invalid page diagnostic kind: {kind!r}")
    directory = _page_diagnostic_directory_hint()
    prepared_content = _bounded_page_diagnostic_content(content)
    if deadline.expired:
        raise TimeoutError(
            "Page diagnostic deadline expired during content preparation"
        )
    if deadline.remaining() <= _DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS:
        raise TimeoutError(
            "Page diagnostic deadline has no safe worker ownership budget"
        )

    nonce = secrets.token_hex(16)
    owner: OwnedProcess | None = None
    primary_error: BaseException | None = None
    result: Path | None = None
    receipt_pipe_owner: OwnedProcess | None = None
    try:
        work_deadline = Deadline(
            deadline.expires_at - _DIAGNOSTIC_OWNERSHIP_RESERVE_SECONDS
        )
        if work_deadline.expired:
            raise TimeoutError(
                "Page diagnostic deadline has no budget before worker cleanup"
            )
        startup_task = asyncio.create_task(
            asyncio.to_thread(
                start_owned_process,
                sys.executable,
                (
                    "-m",
                    _DIAGNOSTIC_WORKER_MODULE,
                    str(directory),
                    kind,
                    nonce,
                    repr(work_deadline.expires_at),
                    str(len(prepared_content)),
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                startup_timeout=min(
                    _DIAGNOSTIC_WORKER_STARTUP_SECONDS,
                    work_deadline.remaining(),
                ),
                deadline=deadline.expires_at,
            )
        )
        owner, startup_error, startup_cancellation = await _settle_task(startup_task)
        if startup_error is not None:
            if isinstance(startup_error, ProcessOwnershipError):
                if startup_cancellation is not None:
                    startup_error.add_note(
                        "Diagnostic worker startup was also interrupted by "
                        "caller cancellation"
                    )
                primary_error = startup_error
            elif startup_cancellation is not None:
                startup_cancellation.add_note(
                    "Diagnostic worker startup also failed: "
                    f"{type(startup_error).__name__}"
                )
                primary_error = startup_cancellation
            else:
                primary_error = startup_error
        else:
            assert owner is not None
            receipt_pipe_owner = owner
            if startup_cancellation is not None:
                primary_error = startup_cancellation

        if primary_error is None:
            assert owner is not None
            try:
                await _transfer_payload(
                    owner, prepared_content, nonce=nonce, deadline=work_deadline
                )
                while owner.poll() is None:
                    remaining_work = work_deadline.remaining()
                    if remaining_work <= 0:
                        raise TimeoutError(
                            "Page diagnostic worker exceeded its work deadline"
                        )
                    await asyncio.sleep(min(_DIAGNOSTIC_POLL_SECONDS, remaining_work))
                shutdown_cancellation = await _settle_owned_worker(
                    owner,
                    deadline=deadline,
                    force=False,
                )
                owner = None
                if shutdown_cancellation is not None:
                    raise shutdown_cancellation
                assert receipt_pipe_owner is not None
                result = await _read_and_validate_receipt(
                    receipt_pipe_owner,
                    directory=directory,
                    kind=kind,
                    nonce=nonce,
                    deadline=deadline,
                )
            except BaseException as error:
                primary_error = error
    finally:
        ownership_error: BaseException | None = None
        if owner is not None:
            try:
                shutdown_cancellation = await _settle_owned_worker(
                    owner,
                    deadline=deadline,
                    force=True,
                )
                if shutdown_cancellation is not None and primary_error is None:
                    primary_error = shutdown_cancellation
                owner = None
            except BaseException as error:
                ownership_error = error
        if receipt_pipe_owner is not None and receipt_pipe_owner.stdout is not None:
            try:
                receipt_pipe_owner.stdout.close()
            except OSError:
                pass

        if ownership_error is not None:
            unresolved = ProcessOwnershipError(
                "Page diagnostic worker or artifact ownership remains unresolved"
            )
            if primary_error is not None:
                unresolved.add_note(
                    f"Diagnostic failure type: {type(primary_error).__name__}"
                )
            raise unresolved from ownership_error

    if primary_error is not None:
        # Do not start a fresh target-filesystem scan or unlink after the
        # deadline. A worker may have atomically published a valid, already
        # redacted/private diagnostic just before its receipt became late; the
        # caller rejects that receipt, while a killed pre-publication write is
        # confined to the one bounded partial cleaned by the next locked writer.
        raise primary_error.with_traceback(primary_error.__traceback__)
    if result is None:
        raise RuntimeError("Page diagnostic worker produced no result")
    if deadline.expired:
        raise TimeoutError("Page diagnostic result arrived after its total deadline")
    return result


__all__ = ["write_page_diagnostic_owned"]
