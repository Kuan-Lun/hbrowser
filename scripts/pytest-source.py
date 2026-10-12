#!/usr/bin/env python3
"""Run pytest against this repository's verified shared tool source."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parent),
]

shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)

from _shared.h2h_dev_tools.pytest_source import main  # noqa: E402 - verify snapshot source before importing it.

adapter_path = Path(__file__).resolve().with_name("runtime_test_owner.py")
adapter = ModuleType("runtime_test_owner")
adapter.__file__ = str(adapter_path)
sys.modules[adapter.__name__] = adapter
exec(compile(adapter_path.read_bytes(), str(adapter_path), "exec"), adapter.__dict__)

if __name__ == "__main__":
    adapter.run(main, pytest_entry=True)
