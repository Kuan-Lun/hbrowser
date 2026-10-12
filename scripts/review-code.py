#!/usr/bin/env python3
"""Review this repository's exact candidate with the shared review implementation."""

import runpy
from pathlib import Path

shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)

from _shared.h2h_dev_tools.code_review import main  # noqa: E402 - verify snapshot source before importing it.

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
