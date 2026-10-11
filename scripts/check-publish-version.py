"""Preserve hbrowser's registry-based publication/retry decision."""

from __future__ import annotations

import json
import tomllib
import urllib.request
from pathlib import Path

from packaging.version import Version


def check_version(repository: Path) -> None:
    """Emit GitHub outputs; an equal published version needs no upload."""
    with (repository / "pyproject.toml").open("rb") as project_file:
        current = Version(tomllib.load(project_file)["project"]["version"])
    with urllib.request.urlopen(
        "https://pypi.org/pypi/hbrowser/json", timeout=30
    ) as response:
        published = Version(json.load(response)["info"]["version"])
    if current < published:
        raise SystemExit(
            f"Project version {current} is older than published version {published}"
        )
    print(f"current={current}")
    print(f"published={published}")
    print(f"publish={'true' if current > published else 'false'}")


if __name__ == "__main__":
    check_version(Path(__file__).resolve().parents[1])
