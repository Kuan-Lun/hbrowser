"""Private stdin bridge: its canonical parent owns this group and all children."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    # Do not detach the command, replace its configuration, or interpret input.
    prompt = Path(sys.argv[1]).read_bytes()
    return subprocess.run(sys.argv[2:], input=prompt, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
