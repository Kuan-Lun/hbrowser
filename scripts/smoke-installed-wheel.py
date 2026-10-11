"""Check the isolated installed hbrowser wheel without launching a browser."""

from __future__ import annotations

import sys
from importlib.metadata import distribution
from pathlib import Path

# Isolated execution imports only tool sources, never the repository runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared.h2h_dev_tools.distribution_check import (
    check_runtime_requirements,
    check_wheel_origin,
)


def main() -> None:
    if not sys.flags.isolated or sys.flags.optimize:
        raise RuntimeError("Run the installed-wheel smoke with python -I, without -O")
    origin = check_wheel_origin("hbrowser", "hbrowser", local=True)
    check_runtime_requirements(distribution("hbrowser"))
    print(f"Installed hbrowser wheel and runtime requirements verified: {origin}")


if __name__ == "__main__":
    main()
