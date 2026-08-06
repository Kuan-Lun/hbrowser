"""Public outcomes produced by the E-Hentai daily check-in."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PunchInComplete:
    """The daily check-in completed without a random encounter."""


@dataclass(frozen=True, slots=True)
class RandomEncounterFound:
    """The daily check-in advertised one trusted HentaiVerse encounter."""

    url: str


type PunchInResult = PunchInComplete | RandomEncounterFound
