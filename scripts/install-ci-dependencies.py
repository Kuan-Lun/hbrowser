"""Install CI tools using this checkout's declared requirements."""

import runpy
from pathlib import Path

shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)

from _shared.h2h_dev_tools.ci_dependencies import main  # noqa: E402 - verify snapshot source before importing it.

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
