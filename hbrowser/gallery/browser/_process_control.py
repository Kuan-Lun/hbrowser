"""Private, bounded control protocol for the owned-process supervisor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

_START_PROTOCOL_TAG = b"hbrowser-start-v1"
_CONTROL_PROTOCOL_TAG = b"hbrowser-control-v1"
_MAX_DEADLINE_NS = (1 << 63) - 1
MAX_START_LINE_BYTES = 64
MAX_CONTROL_LINE_BYTES = 96
PROVEN_PROTOCOL_FAILURE_EXIT_CODE = 4
PROVEN_CLEANUP_FAILURE_EXIT_CODE = 5
PROVEN_TARGET_NOT_STARTED_EXIT_CODE = 6


class ControlIntent(IntEnum):
    """Monotonic shutdown intent; larger values are strictly stronger."""

    TERMINATE = 1
    KILL = 2


@dataclass(frozen=True, slots=True)
class StartRequest:
    """One versioned authorization to start before an absolute deadline."""

    deadline_ns: int

    def __post_init__(self) -> None:
        if isinstance(self.deadline_ns, bool) or not isinstance(self.deadline_ns, int):
            raise TypeError("start deadline must be integer nanoseconds")
        if not 0 <= self.deadline_ns <= _MAX_DEADLINE_NS:
            raise ValueError("start deadline is outside the supported range")

    def encode(self) -> bytes:
        return (
            b" ".join((_START_PROTOCOL_TAG, str(self.deadline_ns).encode("ascii")))
            + b"\n"
        )

    @classmethod
    def parse(cls, frame: bytes) -> StartRequest:
        if len(frame) > MAX_START_LINE_BYTES:
            raise ValueError("start request exceeds its size limit")
        if not frame.endswith(b"\n") or frame.count(b"\n") != 1 or b"\r" in frame:
            raise ValueError("invalid start request framing")
        line = frame[:-1]
        fields = line.split(b" ")
        if (
            len(fields) != 2
            or fields[0] != _START_PROTOCOL_TAG
            or not fields[1].isdigit()
        ):
            raise ValueError("invalid start request schema")
        request = cls(deadline_ns=int(fields[1]))
        if request.encode() != frame:
            raise ValueError("non-canonical start request")
        return request


@dataclass(frozen=True, slots=True)
class ControlRequest:
    """One shutdown intent bounded by phase and overall monotonic deadlines."""

    intent: ControlIntent
    phase_deadline_ns: int
    overall_deadline_ns: int | None
    allow_immediate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ControlIntent):
            raise TypeError("control intent must be a ControlIntent")
        for value in (self.phase_deadline_ns, self.overall_deadline_ns):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("control deadlines must be integer nanoseconds")
            if not 0 <= value <= _MAX_DEADLINE_NS:
                raise ValueError("control deadline is outside the supported range")
        if (
            self.overall_deadline_ns is not None
            and self.phase_deadline_ns > self.overall_deadline_ns
        ):
            raise ValueError("control phase deadline exceeds its overall deadline")
        if not isinstance(self.allow_immediate, bool):
            raise TypeError("allow_immediate must be a bool")
        if self.intent is ControlIntent.KILL and self.overall_deadline_ns is None:
            raise ValueError("KILL control requires a finite overall deadline")

    def merge(self, later: ControlRequest) -> ControlRequest:
        """Escalate intent; only a stronger phase receives its own deadline."""

        intent = max(self.intent, later.intent)
        if self.overall_deadline_ns is None:
            overall_deadline_ns = later.overall_deadline_ns
        elif later.overall_deadline_ns is None:
            overall_deadline_ns = self.overall_deadline_ns
        else:
            overall_deadline_ns = min(
                self.overall_deadline_ns,
                later.overall_deadline_ns,
            )

        def bounded_phase(deadline_ns: int) -> int:
            if overall_deadline_ns is None:
                return deadline_ns
            return min(deadline_ns, overall_deadline_ns)

        if later.intent > self.intent:
            phase_deadline_ns = bounded_phase(later.phase_deadline_ns)
            allow_immediate = later.allow_immediate
        elif later.intent is self.intent:
            phase_deadline_ns = min(
                self.phase_deadline_ns,
                later.phase_deadline_ns,
            )
            phase_deadline_ns = bounded_phase(phase_deadline_ns)
            allow_immediate = self.allow_immediate or later.allow_immediate
        else:
            phase_deadline_ns = bounded_phase(self.phase_deadline_ns)
            allow_immediate = self.allow_immediate
        return ControlRequest(
            intent=intent,
            phase_deadline_ns=phase_deadline_ns,
            overall_deadline_ns=overall_deadline_ns,
            allow_immediate=allow_immediate,
        )

    def encode(self) -> bytes:
        intent = self.intent.name.lower().encode("ascii")
        return (
            b" ".join(
                (
                    _CONTROL_PROTOCOL_TAG,
                    intent,
                    str(self.phase_deadline_ns).encode("ascii"),
                    (
                        b"none"
                        if self.overall_deadline_ns is None
                        else str(self.overall_deadline_ns).encode("ascii")
                    ),
                    b"1" if self.allow_immediate else b"0",
                )
            )
            + b"\n"
        )

    @classmethod
    def parse(cls, line: bytes) -> ControlRequest:
        if len(line) > MAX_CONTROL_LINE_BYTES:
            raise ValueError("control request exceeds its size limit")
        if line.endswith(b"\n"):
            line = line[:-1]
        if b"\n" in line or b"\r" in line:
            raise ValueError("invalid control request framing")
        fields = line.split(b" ")
        if len(fields) != 5 or fields[0] != _CONTROL_PROTOCOL_TAG:
            raise ValueError("invalid control request schema")
        try:
            intent = ControlIntent[fields[1].decode("ascii").upper()]
        except KeyError, UnicodeDecodeError:
            raise ValueError("invalid control intent") from None
        if not fields[2].isdigit() or (
            fields[3] != b"none" and not fields[3].isdigit()
        ):
            raise ValueError("invalid control deadline")
        if fields[4] not in {b"0", b"1"}:
            raise ValueError("invalid immediate-control flag")
        request = cls(
            intent=intent,
            phase_deadline_ns=int(fields[2]),
            overall_deadline_ns=(None if fields[3] == b"none" else int(fields[3])),
            allow_immediate=fields[4] == b"1",
        )
        canonical = request.encode()
        if canonical[:-1] != line:
            raise ValueError("non-canonical control request")
        return request


def deadline_to_monotonic_ns(deadline: int | float) -> int:
    """Convert a finite monotonic-seconds deadline without extending it."""

    if isinstance(deadline, bool) or not isinstance(deadline, int | float):
        raise TypeError("control deadline must be a real number")
    if not math.isfinite(float(deadline)):
        raise ValueError("control deadline must be finite")
    if isinstance(deadline, int):
        deadline_ns = deadline * 1_000_000_000
    else:
        numerator, denominator = deadline.as_integer_ratio()
        deadline_ns = numerator * 1_000_000_000 // denominator
    if not 0 <= deadline_ns <= _MAX_DEADLINE_NS:
        raise ValueError("control deadline is outside the supported range")
    return deadline_ns
