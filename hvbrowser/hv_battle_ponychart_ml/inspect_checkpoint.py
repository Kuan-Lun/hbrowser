"""Inspect checkpoint.pt metadata to review training context."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from common.constants import CLASS_NAMES, OUTPUT_CHECKPOINT


def inspect(path: Path = OUTPUT_CHECKPOINT) -> None:
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # --- basic info ---
    print(f"Checkpoint : {path}")
    print(f"File size  : {path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # --- training sample count ---
    n_samples = ckpt.get("n_samples")
    print(f"Training samples : {n_samples}")

    # --- created_at timestamp ---
    created_at = ckpt.get("created_at")
    print(f"Latest image ts  : {created_at}")

    # --- per-class positive rates ---
    class_rates = ckpt.get("class_rates")
    if class_rates is not None:
        print()
        print("Per-class positive rates:")
        for name, rate in zip(CLASS_NAMES, class_rates):
            print(f"  {name:<18s} {rate:.4f}  ({rate * 100:.1f}%)")

    # --- model architecture summary ---
    state_dict = ckpt.get("state_dict", {})
    n_params = sum(p.numel() for p in (torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in state_dict.values()))
    print()
    print(f"Model parameters : {n_params:,}")
    print(f"State dict keys  : {len(state_dict)}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_CHECKPOINT
    inspect(path)
