"""Check the isolated installed hbrowser wheel without launching a browser."""

from __future__ import annotations

import runpy
import sys
from importlib.metadata import distribution
from pathlib import Path

# Isolated execution loads only verified tool sources, not repository runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent))
shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)

from _shared.h2h_dev_tools.distribution_check import (  # noqa: E402 - verify snapshot source first.
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
