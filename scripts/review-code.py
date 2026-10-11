#!/usr/bin/env python3
"""Review this repository's exact candidate with the shared review implementation."""

from pathlib import Path

from _shared.h2h_dev_tools.code_review import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parents[1]))
