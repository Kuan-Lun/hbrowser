"""Shared bounds for challenge work sent to local or external solvers."""

from __future__ import annotations

MAX_TURNSTILE_TABS = 30


def validate_turnstile_tabs(value: object) -> int:
    """Reject unbounded or non-integral solver work before it reaches a service."""

    if type(value) is not int or not 1 <= value <= MAX_TURNSTILE_TABS:
        raise ValueError(
            "turnstile_tabs must be an integer in " f"[1, {MAX_TURNSTILE_TABS}]"
        )
    return value


__all__ = ["MAX_TURNSTILE_TABS", "validate_turnstile_tabs"]
