"""Inspect checkpoint.pt metadata to review training context."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

from .common.constants import OUTPUT_CHECKPOINT
from .common.model import BACKBONE_REGISTRY, build_model
from .common.sampling import is_original, load_samples, separate_orig_crop

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
    labels_full: dict[str, list[int]] = ckpt["labels_at_full_train"]
    labels_last: dict[str, list[int]] = ckpt["labels_at_last_save"]

    created_at = ckpt.get("created_at")
    logger.info("Latest image ts : %s", created_at)

    samples = load_samples()
    orig, crop = separate_orig_crop(samples)
    labels_current = {
        f"rawimage/{os.path.basename(p)}": labels for p, labels in samples
    }

    def _count_orig_crop(labels: dict[str, list[int]]) -> tuple[int, int]:
        n_o = sum(1 for k in labels if is_original(k.split("/")[-1]))
        return n_o, len(labels) - n_o

    n_orig_full, n_crop_full = _count_orig_crop(labels_full)

    def _fmt_diff(cur: int, base: int) -> str:
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
        val_full: int,
        val_last: int | None,
        val_cur: int,
    ) -> str:
        full_s = f"{val_full:>12,d}"
        last_s = f"{val_last:>12,d}" if val_last is not None else f"{'-':>12s}"
        since_last = _fmt_diff(val_cur, val_last) if val_last is not None else ""
        return (
            f"{label:14s} {full_s} {last_s}"
            f" {val_cur:>10,d}   {since_last:>16s}"
            f"   {_fmt_diff(val_cur, val_full):>16s}"
        )

    logger.info(_row("Originals", n_orig_full, n_orig, len(orig)))
    logger.info(_row("Crops", n_crop_full, n_crop, len(crop)))
    n_total_last = (n_orig or 0) + (n_crop or 0) if n_orig is not None else None
    logger.info(_row("Total", len(labels_full), n_total_last, len(orig) + len(crop)))

    # --- changes detail ---
    def _diff_labels(
        baseline: dict[str, list[int]], current: dict[str, list[int]]
    ) -> tuple[set[str], set[str], set[str]]:
        base_keys = set(baseline)
        cur_keys = set(current)
        added = cur_keys - base_keys
        removed = base_keys - cur_keys
        relabeled = {k for k in base_keys & cur_keys if baseline[k] != current[k]}
        return added, removed, relabeled

    def _split_orig_crop(keys: set[str]) -> tuple[int, int]:
        n_o = sum(1 for k in keys if is_original(k.split("/")[-1]))
        return n_o, len(keys) - n_o

    def _log_changes(title: str, baseline: dict[str, list[int]]) -> None:
        added, removed, relabeled = _diff_labels(baseline, labels_current)
        if not added and not removed and not relabeled:
            logger.info("%s: no changes", title)
            return
        logger.info("%s:", title)
        fmt = "  %-11s %4d images (%d orig, %d crop)"
        if added:
            ao, ac = _split_orig_crop(added)
            logger.info(fmt, "Added", len(added), ao, ac)
        if removed:
            ro, rc = _split_orig_crop(removed)
            logger.info(fmt, "Removed", len(removed), ro, rc)
        if relabeled:
            lo, lc = _split_orig_crop(relabeled)
            logger.info(fmt, "Relabeled", len(relabeled), lo, lc)

    logger.info("")
    _log_changes("Changes since full train", labels_full)
    _log_changes("Changes since last save", labels_last)

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
    logger.info("Val size         : %s", ckpt.get("val_size", "N/A"))
    val_f1 = ckpt.get("val_f1")
    val_f1_str = f"{val_f1:.4f}" if val_f1 is not None else "N/A"
    logger.info("Val F1           : %s", val_f1_str)

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
