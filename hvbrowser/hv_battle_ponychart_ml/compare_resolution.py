"""
比較不同輸入解析度對訓練效果的影響。

測試三種 (PRE_RESIZE, INPUT_SIZE) 組合：
  224: PRE_RESIZE=256, INPUT_SIZE=224  (current baseline)
  288: PRE_RESIZE=320, INPUT_SIZE=288
  320: PRE_RESIZE=384, INPUT_SIZE=320

共用 20% groups 原圖作為 holdout test set（各解析度使用各自的 test transforms）。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_resolution
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    HOLDOUT_TEST_SIZE,
    SEED,
    VAL_SIZE,
    balance_crop_samples,
    compute_class_rates,
    evaluate,
    get_base_timestamp,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    is_original,
    load_samples,
    log_section,
    make_dataloader,
    separate_orig_crop,
    split_by_groups,
    train_model,
)
from .common.dataset import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESOLUTIONS: list[tuple[int, int]] = [
    (256, 224),
    (320, 288),
    (384, 320),
]


def main() -> None:
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
        all_samples, test_size=HOLDOUT_TEST_SIZE, seed=SEED
    )

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        base = get_base_timestamp(os.path.basename(path))
        groups[base].append(idx)

    test_samples = [
        all_samples[idx]
        for gk in test_groups
        for idx in groups[gk]
        if is_original(os.path.basename(all_samples[idx][0]))
    ]
    logger.info("Test set (originals only): %d images", len(test_samples))

    # ── Train+val pool: originals + balanced crops ──
    train_val_all = [all_samples[idx] for gk in train_val_groups for idx in groups[gk]]
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

    # ── Split train/val ──
    tv_groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_balanced):
        base = get_base_timestamp(os.path.basename(path))
        tv_groups[base].append(idx)

    train_gk, val_gk = split_by_groups(
        train_val_balanced, test_size=VAL_SIZE, seed=SEED
    )
    train_samples = [train_val_balanced[i] for gk in train_gk for i in tv_groups[gk]]
    val_samples = [train_val_balanced[i] for gk in val_gk for i in tv_groups[gk]]
    logger.info(
        "Train: %s  Val: %s", f"{len(train_samples):,}", f"{len(val_samples):,}"
    )

    criterion = nn.BCEWithLogitsLoss()

    # ── Run experiments ──
    results: dict[str, dict[str, Any]] = {}

    for pre_resize, input_size in RESOLUTIONS:
        label = f"{input_size}px"
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_section(
            logger,
            "RESOLUTION: PRE_RESIZE=%d  INPUT_SIZE=%d",
            pre_resize,
            input_size,
            width=60,
        )

        t0 = time.monotonic()
        model, thresholds = train_model(
            train_samples,
            val_samples,
            device,
            num_workers,
            label,
            backbone=BACKBONE,
            pre_resize=pre_resize,
            input_size=input_size,
        )
        train_time = time.monotonic() - t0

        # Each resolution needs its own test transforms and dataset
        test_tf = get_transforms(is_train=False, input_size=input_size)
        test_ds = PonyChartDataset(test_samples, test_tf, pre_resize=pre_resize)
        test_loader = make_dataloader(
            test_ds,
            BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        )

        test_result = evaluate(model, test_loader, criterion, device, thresholds)

        results[label] = {
            "pre_resize": pre_resize,
            "input_size": input_size,
            "test_result": test_result,
            "thresholds": thresholds,
            "train_time_s": train_time,
        }

        logger.info(
            ">> %s: test Macro F1=%.4f  time=%.0fs",
            label,
            test_result["macro_f1"],
            train_time,
        )

    # ── Comparison table ──
    labels = [f"{s}px" for _, s in RESOLUTIONS]

    log_section(logger, "RESOLUTION COMPARISON RESULTS", width=80)
    logger.info("")

    baseline_f1 = results[labels[0]]["test_result"]["macro_f1"]
    logger.info(
        "  %-10s  %-12s  %-12s  %-10s  %-10s",
        "Resolution",
        "PRE_RESIZE",
        "INPUT_SIZE",
        "Macro F1",
        "Delta",
    )
    logger.info("  " + "-" * 58)
    for lbl in labels:
        r = results[lbl]
        f1 = r["test_result"]["macro_f1"]
        delta = f1 - baseline_f1
        marker = " (baseline)" if lbl == labels[0] else f"  {delta:+.4f}"
        logger.info(
            "  %-10s  %-12d  %-12d  %-10.4f  %s",
            lbl,
            r["pre_resize"],
            r["input_size"],
            f1,
            marker,
        )

    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for lbl in labels:
        header += f"  {lbl:<12s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 14 * len(labels)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for lbl in labels:
            f1 = results[lbl]["test_result"]["per_class_f1"][i]
            row += f"  {f1:<12.4f}"
        logger.info(row)

    logger.info("")
    logger.info("Training time:")
    for lbl in labels:
        r = results[lbl]
        logger.info(
            "  %s: %.0fs (%.1fx baseline)",
            lbl,
            r["train_time_s"],
            r["train_time_s"] / results[labels[0]]["train_time_s"],
        )

    # ── Summary ──
    log_section(logger, "SUMMARY", width=80)
    best_lbl = max(labels, key=lambda lbl: results[lbl]["test_result"]["macro_f1"])
    best_f1 = results[best_lbl]["test_result"]["macro_f1"]
    logger.info("  Best resolution: %s (Macro F1=%.4f)", best_lbl, best_f1)
    delta_vs_baseline = best_f1 - baseline_f1
    if best_lbl == labels[0]:
        logger.info("  結論: 提高解析度沒有帶來改善，維持 %s", labels[0])
    elif delta_vs_baseline > 0.005:
        logger.info(
            "  結論: %s 有顯著改善 (%+.4f F1)，建議更新 constants.py",
            best_lbl,
            delta_vs_baseline,
        )
    else:
        logger.info(
            "  結論: %s 改善有限 (%+.4f F1)，考量訓練時間後不建議更換",
            best_lbl,
            delta_vs_baseline,
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
