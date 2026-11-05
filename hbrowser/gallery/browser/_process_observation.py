"""Optional lifecycle observations; runtime ownership remains authoritative.

Observers are installed by an explicit composition boundary. They do not launch,
signal or discover processes, and the runtime does not import any development
runner. A token spans pre-spawn admission, provisional ownership and final
ownership without transferring responsibility between separate tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class ProcessObservation(Protocol):
    def bind(self, cleanup: Callable[[float], None], *, deadline: float) -> None: ...
    def finish_start(self, *, deadline: float, failed: bool = False) -> None: ...
    def complete(self, *, deadline: float) -> None: ...


class ProcessObserver(Protocol):
    def cleanup_deadline(self, deadline: float) -> float: ...
    def admit(self, *, recovery: bool, deadline: float) -> ProcessObservation: ...


observer: ProcessObserver | None = None


def admit(*, recovery: bool, deadline: float) -> ProcessObservation | None:
    current = observer
    return (
        None if current is None else current.admit(recovery=recovery, deadline=deadline)
    )


def cleanup_deadline(deadline: float) -> float:
    current = observer
    return deadline if current is None else current.cleanup_deadline(deadline)
