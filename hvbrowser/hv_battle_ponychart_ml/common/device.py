"""Device detection utilities."""

from __future__ import annotations

import os
import platform
import subprocess

import torch


def get_performance_cpu_count() -> int:
    """Return performance core count (macOS Apple Silicon) or total cores."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return max((os.cpu_count() or 1) - 2, 1)


def get_device(device_str: str = "auto") -> torch.device:
    """Detect and return the best available device."""
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
