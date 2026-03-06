"""Inspect checkpoint.pt metadata to review training context."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from .common.constants import OUTPUT_CHECKPOINT
from .common.model import BACKBONE_REGISTRY, build_model
from .common.sampling import load_samples, separate_orig_crop

logger = logging.getLogger(__name__)


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
        logger.error("Checkpoint not found: %s", path)
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # --- basic info ---
    logger.info("Checkpoint : %s", path)
    logger.info("File size  : %.2f MB", path.stat().st_size / 1024 / 1024)
    logger.info("")

    # --- sample counts ---
    n_orig = ckpt.get("n_orig")
    n_crop = ckpt.get("n_crop")
    n_orig_full = ckpt.get("n_orig_at_full_train")

    created_at = ckpt.get("created_at")
    logger.info("Latest image ts : %s", created_at)

    samples = load_samples()
    orig, crop = separate_orig_crop(samples)

    # Use full train baseline; fall back to last save if same (first train)
    if n_orig_full is None:
        n_orig_full = n_orig

    def _fmt_diff(cur: int, base: int | None) -> str:
        if base is None:
            return ""
        diff = cur - base
        ratio = diff / base if base else 0
        return f"{diff:+,d} ({ratio:+.1%})"

    header = (
        f"{'':14s} {'Full train':>12s} {'Last save':>12s}"
        f" {'Current':>10s}   {'Since last':>16s}"
        f"   {'Since full':>16s}"
    )
    logger.info("")
    logger.info(header)

    def _row(
        label: str,
        val_full: int | None,
        val_last: int | None,
        val_cur: int,
    ) -> str:
        full_s = f"{val_full:>12,d}" if val_full is not None else f"{'-':>12s}"
        last_s = f"{val_last:>12,d}" if val_last is not None else f"{'-':>12s}"
        return (
            f"{label:14s} {full_s} {last_s}"
            f" {val_cur:>10,d}   {_fmt_diff(val_cur, val_last):>16s}"
            f"   {_fmt_diff(val_cur, val_full):>16s}"
        )

    logger.info(_row("Originals", n_orig_full, n_orig, len(orig)))
    logger.info(_row("Crops", None, n_crop, len(crop)))
    n_total_last = (n_orig or 0) + (n_crop or 0) if n_orig is not None else None
    logger.info(_row("Total", None, n_total_last, len(orig) + len(crop)))

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
    logger.info("")
    logger.info("Backbone         : %s", backbone_name)
    logger.info("Input size       : %s", ckpt.get("input_size", "N/A"))
    logger.info("Pre-resize       : %s", ckpt.get("pre_resize", "N/A"))
    logger.info("Num classes      : %s", ckpt.get("num_classes", "N/A"))
    logger.info("Model parameters : %s", f"{n_params:,}")
    logger.info("State dict keys  : %s", f"{len(state_dict):,}")

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
        logger.info("")
        logger.info("Training hyperparameters:")
        for key, label in hp_keys:
            val = ckpt.get(key)
            if val is not None:
                logger.info("  %s %s", f"{label:<18s}", val)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_CHECKPOINT
    inspect(path)
