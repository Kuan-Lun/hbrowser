#!/usr/bin/env python3
"""Run the shared dependency audit implementation."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
    runpy.run_path(str(shared / "source_loader.py"))["load_package"](
        shared, "_shared.h2h_dev_tools"
    )
    runpy.run_module("_shared.h2h_dev_tools.dependency_audit", run_name="__main__")
