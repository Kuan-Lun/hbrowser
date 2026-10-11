#!/usr/bin/env python3
"""Run the shared version policy implementation."""

import runpy

if __name__ == "__main__":
    runpy.run_module("_shared.h2h_dev_tools.version_policy", run_name="__main__")
