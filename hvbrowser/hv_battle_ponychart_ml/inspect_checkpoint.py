"""Inspect checkpoint.pt metadata to review training context."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .common.constants import OUTPUT_CHECKPOINT
from .common.data import load_samples, separate_orig_crop
from .common.model import BACKBONE_REGISTRY, build_model


def _detect_backbone(state_dict: dict[str, object]) -> str:
    """Try loading state_dict into each registered backbone to identify it."""
    for name in BACKBONE_REGISTRY:
        model = build_model(backbone=name, pretrained=False)
        try:
            model.load_state_dict(state_dict)
            return name
        except RuntimeError:
            continue
    return "unknown"


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
    print(f"Training samples : {n_samples:,}")

    # --- created_at timestamp ---
    created_at = ckpt.get("created_at")
    print(f"Latest image ts  : {created_at}")

    # --- current rawimage data ---
    samples = load_samples()
    orig, crop = separate_orig_crop(samples)
    print()
    print(f"Current rawimage total : {len(samples):,}")
    print(f"  Originals            : {len(orig):,}")
    print(f"  Crops                : {len(crop):,}")
    if n_samples is not None:
        new = len(samples) - n_samples
        ratio = new / n_samples if n_samples else 0
        print(f"  New since checkpoint : {new:+,d} ({ratio:+.1%})")

    # --- model architecture ---
    state_dict = ckpt.get("state_dict", {})
    n_params = sum(
        p.numel()
        for p in (
            torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for v in state_dict.values()
        )
    )
    backbone_name = ckpt.get("backbone") or _detect_backbone(state_dict)
    print()
    print(f"Backbone         : {backbone_name}")
    print(f"Input size       : {ckpt.get('input_size', 'N/A')}")
    print(f"Pre-resize       : {ckpt.get('pre_resize', 'N/A')}")
    print(f"Num classes      : {ckpt.get('num_classes', 'N/A')}")
    print(f"Model parameters : {n_params:,}")
    print(f"State dict keys  : {len(state_dict):,}")

    # --- training hyperparameters ---
    hp_keys = [
        ("seed", "Seed"),
        ("batch_size", "Batch size"),
        ("lr_head", "LR head"),
        ("lr_features", "LR features"),
        ("lr_classifier", "LR classifier"),
        ("weight_decay", "Weight decay"),
        ("label_smoothing", "Label smoothing"),
    ]
    has_hp = any(ckpt.get(k) is not None for k, _ in hp_keys)
    if has_hp:
        print()
        print("Training hyperparameters:")
        for key, label in hp_keys:
            val = ckpt.get(key)
            if val is not None:
                print(f"  {label:<18s} {val}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_CHECKPOINT
    inspect(path)
