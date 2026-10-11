#!/usr/bin/env python3
"""Run pytest against this repository's verified shared tool source."""

import runpy
from pathlib import Path

shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)

from _shared.h2h_dev_tools.pytest_source import main  # noqa: E402 - verify snapshot source before importing it.

if __name__ == "__main__":
    main()
