"""Monotonic absolute deadlines shared by multi-phase browser operations."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Deadline:
    """One monotonic deadline that cannot be reset by a later phase."""

    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise TypeError("deadline seconds must be a real number")
        duration = float(seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("deadline seconds must be finite and non-negative")
        return cls(time.monotonic() + duration)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def bounded(self, seconds: float) -> Deadline:
        """Return a child deadline that can only shorten this deadline."""

        child = Deadline.after(seconds)
        return Deadline(min(self.expires_at, child.expires_at))

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0


__all__ = ["Deadline"]
