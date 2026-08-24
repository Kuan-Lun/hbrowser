"""Environment hygiene shared by non-owned external helper commands."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

RESERVED_LOGGING_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "HBROWSER_LOG_DIR",
        "HBROWSER_PROCESS_LOG_FILE",
        "HBROWSER_LOG_FORWARD_ENDPOINT",
        "HBROWSER_LOG_FORWARD_TOKEN",
    }
)


def environment_without_logging_capabilities(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an ordinary command environment without parent log ownership."""
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if key.upper() not in RESERVED_LOGGING_ENVIRONMENT_KEYS
    }


__all__ = [
    "RESERVED_LOGGING_ENVIRONMENT_KEYS",
    "environment_without_logging_capabilities",
]
