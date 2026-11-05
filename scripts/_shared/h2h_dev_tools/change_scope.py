#!/usr/bin/env python3
"""Classify an exact Git tree difference for documentation-only checks."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

ChangeScope = Literal["documentation", "full"]
_DOCUMENT_PATHS = frozenset(
    {"README.md", "benchmarks/README.md", "verification/README.md"}
)
_OBJECT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class ChangeScopeError(RuntimeError):
    """The exact change set could not be established safely."""


class GitReader(Protocol):
    """A caller may supply its existing owned Git execution boundary."""

    def __call__(
        self, root: Path, /, *arguments: str, deadline: float | None = None
    ) -> bytes: ...


def _git_timeout(deadline: float | None) -> float:
    if deadline is None:
        return 60
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ChangeScopeError("Change classification deadline expired or is invalid")
    return min(60, remaining)


def git_output(root: Path, *arguments: str, deadline: float | None = None) -> bytes:
    """Read Git output without interpreting filenames as lines or shell code."""

    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            timeout=_git_timeout(deadline),
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = (
            (error.stderr or error.stdout).decode("utf-8", errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise ChangeScopeError(f"Git operation failed: {detail}") from error
    _git_timeout(deadline)
    return result.stdout


def resolve_tree(
    root: Path,
    revision: str,
    *,
    deadline: float | None = None,
    git_reader: GitReader | None = None,
) -> str:
    """Resolve a commit or tree to one immutable tree object ID."""

    read = git_output if git_reader is None else git_reader
    output = read(
        root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{revision}^{{tree}}",
        deadline=deadline,
    )
    try:
        tree = output.decode("ascii").strip()
    except UnicodeError as error:
        raise ChangeScopeError("Git returned an invalid tree object ID") from error
    if _OBJECT_ID.fullmatch(tree) is None:
        raise ChangeScopeError("Git returned an invalid tree object ID")
    return tree


def capture_candidate(
    root: Path, revision: str | None, *, deadline: float | None = None
) -> str:
    """Capture either the index or an explicitly requested revision."""

    if revision is not None:
        return resolve_tree(root, revision, deadline=deadline)
    try:
        tree = git_output(root, "write-tree", deadline=deadline).decode("ascii").strip()
    except UnicodeError as error:
        raise ChangeScopeError("Git returned an invalid index tree") from error
    return resolve_tree(root, tree, deadline=deadline)


def _is_document(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if parts and parts[-1].casefold() in {"agents.md", "claude.md"}:
        return False
    return path in _DOCUMENT_PATHS or (
        len(parts) >= 2 and parts[0] == "docs" and path.endswith(".md")
    )


def _is_text(
    root: Path, object_id: str, *, deadline: float | None, git_reader: GitReader
) -> bool:
    content = git_reader(root, "cat-file", "blob", object_id, deadline=deadline)
    if b"\0" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeError:
        return False
    return True


def classify_changes(
    base: str,
    candidate: str,
    root: Path,
    *,
    deadline: float | None = None,
    git_reader: GitReader | None = None,
) -> ChangeScope:
    """Allow only registered, non-executable text documents into the light gate."""

    read = git_output if git_reader is None else git_reader
    base_tree = resolve_tree(root, base, deadline=deadline, git_reader=read)
    candidate_tree = resolve_tree(root, candidate, deadline=deadline, git_reader=read)
    raw = read(
        root,
        "diff",
        "--raw",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "--ignore-submodules=none",
        "--no-abbrev",
        "-z",
        base_tree,
        candidate_tree,
        "--",
        deadline=deadline,
    )
    if not raw:
        return "full"
    fields = raw.split(b"\0")
    if fields.pop() != b"" or len(fields) % 2:
        raise ChangeScopeError("Malformed NUL-delimited Git difference")
    for index in range(0, len(fields), 2):
        _git_timeout(deadline)
        try:
            header = fields[index].decode("ascii").split()
            path = fields[index + 1].decode("utf-8")
        except UnicodeError:
            return "full"
        if len(header) != 5 or not header[0].startswith(":"):
            raise ChangeScopeError("Malformed raw Git difference entry")
        old_mode, new_mode, old_id, new_id, status = header
        modes = (old_mode[1:], new_mode)
        expected = {
            "A": ("000000", "100644"),
            "M": ("100644", "100644"),
            "D": ("100644", "000000"),
        }
        if not _is_document(path) or modes != expected.get(status):
            return "full"
        for mode, object_id in zip(modes, (old_id, new_id), strict=True):
            if mode != "000000" and not _is_text(
                root, object_id, deadline=deadline, git_reader=read
            ):
                return "full"
    _git_timeout(deadline)
    return "documentation"


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--index", action="store_true", help="check the staged tree")
    targets.add_argument("--candidate", help="check this commit or tree")
    parser.add_argument("--base", default="HEAD", help="comparison commit or tree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser)
    arguments = parser.parse_args()
    root = Path.cwd()
    base = resolve_tree(root, arguments.base)
    candidate = capture_candidate(root, arguments.candidate)
    scope = classify_changes(base, candidate, root)
    if capture_candidate(root, arguments.candidate) != candidate:
        raise ChangeScopeError("Candidate changed while classifying its scope")
    if resolve_tree(root, arguments.base) != base:
        raise ChangeScopeError("Base changed while classifying the candidate scope")
    print(scope)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChangeScopeError as error:
        print(f"check-change-scope: {error}", file=sys.stderr)
        raise SystemExit(1) from error
