"""Reject index flags which can conceal candidate working-tree changes."""

from __future__ import annotations

from collections.abc import Callable


class GitIndexError(ValueError):
    """The index cannot support ordinary Git working-tree verification."""


def assert_index_visible(git_bytes: Callable[..., bytes]) -> None:
    """Check the caller's exact Git authority without changing its index.

    ``-v`` lowercases assume-unchanged entries; ``S`` denotes skip-worktree.
    NUL records preserve arbitrary filename bytes, including newlines. The
    caller retains its environment, process owner and deadline for this query.
    """
    output = git_bytes("ls-files", "-v", "-z")
    if not output:
        return
    if not output.endswith(b"\0"):
        raise GitIndexError("Git index flag inventory is not NUL terminated")
    for record in output[:-1].split(b"\0"):
        if len(record) < 3 or record[1:2] != b" ":
            raise GitIndexError("Git index flag inventory has a malformed record")
        tag = record[:1]
        if tag.islower() or tag == b"S":
            raise GitIndexError(
                "Candidate index must not use assume-unchanged or skip-worktree: "
                f"{record[2:]!r}"
            )
        if tag not in (b"H", b"M", b"R", b"C", b"K", b"?"):
            raise GitIndexError("Git index flag inventory has an unknown state")
