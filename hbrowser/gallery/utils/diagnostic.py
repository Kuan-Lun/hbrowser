"""Bounded, private HTML diagnostics for browser failures."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

_PAGE_DIAGNOSTIC_FILE_LIMIT = 20
_PAGE_DIAGNOSTIC_MAX_FILE_BYTES = 2 * 1024 * 1024
_PAGE_DIAGNOSTIC_TOTAL_BYTES = 20 * 1024 * 1024
_PAGE_DIAGNOSTIC_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_PAGE_DIAGNOSTIC_FILENAME_PATTERN = re.compile(
    r"[a-z][a-z0-9_]{0,63}_(?P<sequence>[0-9a-f]{16})_[0-9a-f]{32}\.html\Z"
)
_ENCOUNTER_QUERY_SECRET_PATTERN = re.compile(
    r"(?P<prefix>"
    r"(?:\?|&(?:amp;|#0*38;|#x0*26;)?)"
    r"encounter="
    r")"
    r"[^&#\s\"'<>]*",
    flags=re.IGNORECASE,
)
_REDACTED_QUERY_VALUE = "REDACTED"
_PAGE_DIAGNOSTIC_LOCK_FILENAME = ".hbrowser-page-diagnostics.lock"
_PAGE_DIAGNOSTIC_MAX_SEQUENCE = (1 << 64) - 1
_PAGE_DIAGNOSTIC_THREAD_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class _PageDiagnosticCandidate:
    path: Path
    size: int
    sequence: int


def _reset_page_diagnostic_thread_lock_after_fork() -> None:
    global _PAGE_DIAGNOSTIC_THREAD_LOCK

    _PAGE_DIAGNOSTIC_THREAD_LOCK = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_page_diagnostic_thread_lock_after_fork)


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.lockf(descriptor, fcntl.LOCK_EX, 0, 0, os.SEEK_SET)
    elif os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_LOCK"), 1)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.lockf(descriptor, fcntl.LOCK_UN, 0, 0, os.SEEK_SET)
    elif os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_UNLCK"), 1)


@contextmanager
def _locked_page_diagnostic_directory(directory: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    lock_path = directory / _PAGE_DIAGNOSTIC_LOCK_FILENAME

    with _PAGE_DIAGNOSTIC_THREAD_LOCK:
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            _lock_descriptor(descriptor)
            try:
                yield
            finally:
                _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)


def _redact_page_diagnostic_secrets(content: str) -> str:
    """Hide short-lived encounter query values while retaining HTML context."""
    return _ENCOUNTER_QUERY_SECRET_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED_QUERY_VALUE}",
        content,
    )


def _bounded_page_diagnostic_content(content: str) -> bytes:
    maximum_bytes = min(
        _PAGE_DIAGNOSTIC_MAX_FILE_BYTES,
        _PAGE_DIAGNOSTIC_TOTAL_BYTES,
    )
    if maximum_bytes <= 0:
        raise OSError("Page diagnostic byte limits must be positive")

    redacted_content = _redact_page_diagnostic_secrets(content)
    content_bytes = redacted_content.encode("utf-8", errors="ignore")
    if len(content_bytes) <= maximum_bytes:
        return content_bytes

    marker = (
        "\n<!-- hbrowser page diagnostic truncated at " f"{maximum_bytes} bytes -->\n"
    ).encode()
    if len(marker) >= maximum_bytes:
        return marker[:maximum_bytes]
    prefix = content_bytes[: maximum_bytes - len(marker)]
    return prefix.decode("utf-8", errors="ignore").encode("utf-8") + marker


def _write_private_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        file = os.fdopen(descriptor, "wb")
        descriptor = None
        with file:
            written = file.write(content)
            if written != len(content):
                raise OSError(
                    f"Short page diagnostic write: {written}/{len(content)} bytes"
                )
    except BaseException as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as close_error:
                error.add_note(f"Could not close diagnostic file: {close_error!r}")
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                error.add_note(
                    f"Could not remove partial diagnostic file: {cleanup_error!r}"
                )
        raise


def write_page_diagnostic(directory: Path, kind: str, content: str) -> Path:
    """Write one uniquely named diagnostic and enforce global retention bounds."""
    if _PAGE_DIAGNOSTIC_KIND_PATTERN.fullmatch(kind) is None:
        raise ValueError(f"Invalid page diagnostic kind: {kind!r}")
    if _PAGE_DIAGNOSTIC_FILE_LIMIT <= 0:
        raise OSError("Page diagnostic file limit must be positive")

    content_bytes = _bounded_page_diagnostic_content(content)
    with _locked_page_diagnostic_directory(directory):
        candidates = list[_PageDiagnosticCandidate]()
        maximum_sequence = -1
        for directory_entry in directory.iterdir():
            match = _PAGE_DIAGNOSTIC_FILENAME_PATTERN.fullmatch(directory_entry.name)
            if match is None:
                continue
            try:
                candidate_stat = directory_entry.lstat()
            except OSError as error:
                raise OSError(
                    f"Could not inspect page diagnostic {directory_entry}: {error!r}"
                ) from error
            if not stat.S_ISREG(candidate_stat.st_mode):
                continue

            sequence = int(match.group("sequence"), 16)
            maximum_sequence = max(maximum_sequence, sequence)
            candidates.append(
                _PageDiagnosticCandidate(
                    path=directory_entry,
                    size=candidate_stat.st_size,
                    sequence=sequence,
                )
            )

        candidates.sort(key=lambda candidate: (candidate.sequence, candidate.path.name))
        allowed_count = _PAGE_DIAGNOSTIC_FILE_LIMIT - 1
        allowed_bytes = _PAGE_DIAGNOSTIC_TOTAL_BYTES - len(content_bytes)
        remaining_count = len(candidates)
        remaining_bytes = sum(candidate.size for candidate in candidates)
        for diagnostic_candidate in candidates:
            if remaining_count <= allowed_count and remaining_bytes <= allowed_bytes:
                break
            try:
                diagnostic_candidate.path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                raise OSError(
                    "Could not enforce page diagnostic retention while removing "
                    f"{diagnostic_candidate.path}: {error!r}"
                ) from error
            remaining_count -= 1
            remaining_bytes -= diagnostic_candidate.size

        if remaining_count > allowed_count or remaining_bytes > allowed_bytes:
            raise OSError(
                "Could not make room within page diagnostic retention bounds: "
                f"files={remaining_count}, bytes={remaining_bytes}"
            )
        if maximum_sequence >= _PAGE_DIAGNOSTIC_MAX_SEQUENCE:
            raise OSError("Page diagnostic sequence is exhausted")

        sequence = maximum_sequence + 1
        path = directory / f"{kind}_{sequence:016x}_{uuid4().hex}.html"
        _write_private_file(path, content_bytes)
        return path
