from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatThreshold:
    hp_low: int = 30
    hp_high: int = 60
    mp_low: int = 30
    mp_high: int = 60
    sp_low: int = 50
    sp_high: int = 60
    overcharge_low: int = 100
    overcharge_high: int = 240
    countmonster_low: int = 0
    countmonster_high: int = 5


DEFAULT_STATTHRESHOLD = StatThreshold()

DEFAULT_FORBIDDEN_SKILLS: list[str] = [
    "blind",
    "confuse",
    "drain",
    "magnet",
    "silence",
    "sleep",
]
