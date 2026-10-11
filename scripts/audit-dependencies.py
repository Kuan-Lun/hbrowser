#!/usr/bin/env python3
"""Run the shared dependency audit implementation."""

import runpy

if __name__ == "__main__":
    runpy.run_module("_shared.h2h_dev_tools.dependency_audit", run_name="__main__")
