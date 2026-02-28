"""
Holdout 評估：在僅原圖的 test set 上測量真實 F1。

80% timestamp groups 用於訓練（原圖 + balanced crops），
20% groups 的原圖作為 holdout test set，模擬實際推論場景。

Thresholds 在 val set 上 optimize，再套用到 test set 評估。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.evaluate_holdout
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

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


def main() -> None:
    argparse.ArgumentParser(
        description="Evaluate model F1 on originals-only holdout test set"
    ).parse_args()

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

    # ── Train+val pool: originals + balanced crops from train groups ──
    train_val_all = [
        all_samples[idx] for gk in train_val_groups for idx in groups[gk]
    ]
    train_val_orig, train_val_crop = separate_orig_crop(train_val_all)
    orig_rates = compute_class_rates(train_val_orig)
    balanced_crops = balance_crop_samples(train_val_crop, orig_rates, rng)
    train_val_balanced = train_val_orig + balanced_crops
    logger.info(
        "Train+val pool: %d orig + %d crops (raw %d) = %d total",
        len(train_val_orig),
        len(balanced_crops),
        len(train_val_crop),
        len(train_val_balanced),
    )

    # ── Split train/val within 90% pool ──
    tv_groups_inner: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_balanced):
        base = get_base_timestamp(os.path.basename(path))
        tv_groups_inner[base].append(idx)

    train_gk, val_gk = split_by_groups(
        train_val_balanced, test_size=0.15, seed=SEED
    )
    train_samples = [
        train_val_balanced[idx] for gk in train_gk for idx in tv_groups_inner[gk]
    ]
    val_samples = [
        train_val_balanced[idx] for gk in val_gk for idx in tv_groups_inner[gk]
    ]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    # ── Train from scratch (never resume: different split → data leakage) ──
    model, thresholds = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "Holdout Evaluation",
        backbone=BACKBONE,
        verbose=True,
    )

    # ── Evaluate on holdout test set (originals only) ──
    criterion = nn.BCEWithLogitsLoss()
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    result = evaluate(model, test_loader, criterion, device, thresholds)

    # ── Report ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("HOLDOUT TEST SET EVALUATION (%d original images)", len(test_samples))
    logger.info("=" * 70)
    logger.info("Thresholds (from val set): %s", dict(zip(CLASS_NAMES, thresholds)))
    logger.info("")
    logger.info("  Macro F1: %.4f", result["macro_f1"])
    logger.info("  Loss:     %.4f", result["loss"])
    logger.info("")
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
            result["per_class_precision"][i],
            result["per_class_recall"][i],
            result["per_class_f1"][i],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
