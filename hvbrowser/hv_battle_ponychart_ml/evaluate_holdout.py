"""
Holdout 評估：在僅原圖的 test set 上測量真實 F1。

80% timestamp groups 用於訓練（原圖 + balanced crops），
20% groups 的原圖作為 holdout test set，模擬實際推論場景。

Thresholds 在 val set 上 optimize，再套用到 test set 評估。

支援 learning curve 模式（預設）：以不同數量的訓練原圖 group 訓練模型，
觀察資料量對 F1 的影響。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.evaluate_holdout
  uv run python -m hvbrowser.hv_battle_ponychart_ml.evaluate_holdout --steps 1 10 50 100
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    SEED,
    balance_crop_samples,
    compute_class_rates,
    evaluate,
    get_base_timestamp,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    is_original,
    load_samples,
    make_dataloader,
    separate_orig_crop,
    split_by_groups,
    train_model,
)
from .common.data import PonyChartDataset

logger = logging.getLogger(__name__)

DEFAULT_STEPS = [1, 5, 10, 20, 50, 100, 200, 400]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model F1 on originals-only holdout test set"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=DEFAULT_STEPS,
        help=(
            "Number of original-image groups to include for each training run. "
            "The full training set is always appended as the last step. "
            f"(default: {DEFAULT_STEPS})"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # ── Load all samples ──
    all_samples = load_samples()
    if not all_samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        return
    logger.info("Total samples loaded: %d", len(all_samples))

    # ── Split groups: 80% train+val, 20% test ──
    train_val_groups, test_groups = split_by_groups(
        all_samples, test_size=0.20, seed=SEED
    )

    # Build group index
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        base = get_base_timestamp(os.path.basename(path))
        groups[base].append(idx)

    # ── Test set: only originals from test groups ──
    test_samples = [
        all_samples[idx]
        for gk in test_groups
        for idx in groups[gk]
        if is_original(os.path.basename(all_samples[idx][0]))
    ]
    logger.info("Test set (originals only): %d images", len(test_samples))

    # ── Train+val pool from train groups ──
    train_val_all = [
        all_samples[idx] for gk in train_val_groups for idx in groups[gk]
    ]
    train_val_orig, train_val_crop = separate_orig_crop(train_val_all)
    logger.info(
        "Train+val pool: %d orig, %d crops (raw)",
        len(train_val_orig),
        len(train_val_crop),
    )

    # ── Group originals and raw crops by base timestamp ──
    orig_by_group: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for sample in train_val_orig:
        orig_by_group[get_base_timestamp(os.path.basename(sample[0]))].append(sample)

    crop_by_group: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for sample in train_val_crop:
        crop_by_group[get_base_timestamp(os.path.basename(sample[0]))].append(sample)

    # ── Split original groups into train / val (85 / 15) ──
    train_orig_gk, val_orig_gk = split_by_groups(
        train_val_orig, test_size=0.15, seed=SEED
    )

    # ── Fixed val set (originals + balanced crops from val groups) ──
    val_orig = [s for gk in val_orig_gk for s in orig_by_group.get(gk, [])]
    val_crop = [s for gk in val_orig_gk for s in crop_by_group.get(gk, [])]
    if val_orig and val_crop:
        val_rates = compute_class_rates(val_orig)
        val_balanced_crop = balance_crop_samples(val_crop, val_rates, rng)
    else:
        val_balanced_crop = []
    val_samples = val_orig + val_balanced_crop
    logger.info(
        "Val set: %d orig + %d crops = %d total",
        len(val_orig),
        len(val_balanced_crop),
        len(val_samples),
    )

    # ── Shuffle training groups (cumulative subsets for learning curve) ──
    train_orig_gk_list = list(train_orig_gk)
    rng.shuffle(train_orig_gk_list)
    total_train_groups = len(train_orig_gk_list)
    logger.info("Train original groups: %d", total_train_groups)

    # ── Determine step sizes ──
    step_sizes = sorted(set(s for s in args.steps if 0 < s <= total_train_groups))
    if total_train_groups not in step_sizes:
        step_sizes.append(total_train_groups)
    logger.info("Steps: %s", step_sizes)

    # ── Prepare test loader (reused across steps) ──
    criterion = nn.BCEWithLogitsLoss()
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    # ── Run evaluation for each step ──
    all_results: list[dict[str, Any]] = []
    for n in step_sizes:
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "Training with %d / %d original groups …", n, total_train_groups
        )
        logger.info("=" * 60)

        # Per-step reproducibility
        step_rng = np.random.RandomState(SEED + n)
        torch.manual_seed(SEED + n)
        np.random.seed(SEED + n)

        selected_gk = train_orig_gk_list[:n]
        train_orig = [s for gk in selected_gk for s in orig_by_group.get(gk, [])]
        train_crop = [s for gk in selected_gk for s in crop_by_group.get(gk, [])]

        if train_orig and train_crop:
            rates = compute_class_rates(train_orig)
            balanced = balance_crop_samples(train_crop, rates, step_rng)
        else:
            balanced = []
        train_data = train_orig + balanced

        logger.info(
            "  %d orig + %d crops = %d training samples",
            len(train_orig),
            len(balanced),
            len(train_data),
        )

        model, thresholds = train_model(
            train_data,
            val_samples,
            device,
            num_workers,
            f"N={n}",
            backbone=BACKBONE,
            verbose=len(step_sizes) == 1,
        )

        result = evaluate(model, test_loader, criterion, device, thresholds)
        all_results.append(
            {
                "n_groups": n,
                "n_orig": len(train_orig),
                "n_total": len(train_data),
                "macro_f1": result["macro_f1"],
                "loss": result["loss"],
                "per_class_f1": result["per_class_f1"],
                "per_class_precision": result["per_class_precision"],
                "per_class_recall": result["per_class_recall"],
            }
        )

    # ── Summary table ──
    _print_summary(test_samples, all_results)


def _print_summary(
    test_samples: list[tuple[str, list[int]]],
    all_results: list[dict[str, Any]],
) -> None:
    logger.info("")
    logger.info("=" * 90)
    logger.info(
        "LEARNING CURVE SUMMARY  (test set: %d original images)",
        len(test_samples),
    )
    logger.info("=" * 90)

    # Header
    fixed_hdr = (
        f"  {'Groups':>6}  {'Orig':>5}  {'Total':>6}"
        f"  {'MacroF1':>7}  {'Loss':>7}"
    )
    cls_hdr = "".join(f"  {n[:10]:>10}" for n in CLASS_NAMES)
    logger.info("%s%s", fixed_hdr, cls_hdr)
    logger.info(
        "  %s",
        "-" * (6 + 2 + 5 + 2 + 6 + 2 + 7 + 2 + 7 + (2 + 10) * len(CLASS_NAMES)),
    )

    for r in all_results:
        fixed_cols = (
            f"  {r['n_groups']:>6}  {r['n_orig']:>5}  {r['n_total']:>6}"
            f"  {r['macro_f1']:>7.4f}  {r['loss']:>7.4f}"
        )
        cls_cols = "".join(f"  {f:>10.4f}" for f in r["per_class_f1"])
        logger.info("%s%s", fixed_cols, cls_cols)

    logger.info("=" * 90)

    # Detailed report for the last (full training set) result
    r = all_results[-1]
    logger.info("")
    logger.info("DETAILED REPORT (full training set, %d groups)", r["n_groups"])
    logger.info("-" * 60)
    logger.info(
        "  %-20s  %-10s  %-10s  %-10s",
        "Class",
        "Precision",
        "Recall",
        "F1",
    )
    logger.info("  " + "-" * 55)
    for i, name in enumerate(CLASS_NAMES):
        logger.info(
            "  %-20s  %-10.4f  %-10.4f  %-10.4f",
            name,
            r["per_class_precision"][i],
            r["per_class_recall"][i],
            r["per_class_f1"][i],
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
