#!/usr/bin/env python3
"""Run the committed h2h-dev-tools documentation entry point."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
    runpy.run_path(str(shared / "source_loader.py"))["load_package"](
        shared, "_shared.h2h_dev_tools"
    )
    runpy.run_module("_shared.h2h_dev_tools.documentation", run_name="__main__")
