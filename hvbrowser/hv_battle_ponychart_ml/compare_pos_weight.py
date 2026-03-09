"""
比較有無 pos_weight 對訓練效果的影響。

Experiment A (Baseline): BCEWithLogitsLoss() 無 pos_weight
Experiment B (pos_weight): BCEWithLogitsLoss(pos_weight=w)
  w[cls] = (N - pos_count) / pos_count

共用 20% groups 原圖作為 holdout test set。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_pos_weight
"""

from __future__ import annotations

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
    HOLDOUT_TEST_SIZE,
    NUM_CLASSES,
    SEED,
    VAL_SIZE,
    balance_crop_samples,
    compute_class_rates,
    compute_pos_weight,
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
        all_samples, test_size=HOLDOUT_TEST_SIZE
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

    train_gk, val_gk = split_by_groups(train_val_balanced, test_size=VAL_SIZE)
    train_samples = [train_val_balanced[i] for gk in train_gk for i in tv_groups[gk]]
    val_samples = [train_val_balanced[i] for gk in val_gk for i in tv_groups[gk]]
    logger.info(
        "Train: %s  Val: %s", f"{len(train_samples):,}", f"{len(val_samples):,}"
    )

    # ── Compute pos_weight from training data ──
    pw = compute_pos_weight(train_samples)
    logger.info("pos_weight: %s", dict(zip(CLASS_NAMES, pw.tolist())))

    # ── Experiment A: Baseline (no pos_weight) ──
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    result_a = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "A: Baseline (no pos_weight)",
        backbone=BACKBONE,
        verbose=True,
    )
    model_a, thresholds_a = result_a.model, result_a.thresholds

    # ── Experiment B: With pos_weight ──
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_result_b = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "B: With pos_weight",
        backbone=BACKBONE,
        pos_weight=pw,
        verbose=True,
    )
    model_b, thresholds_b = train_result_b.model, train_result_b.thresholds

    # ── Evaluate both on holdout test set ──
    criterion = nn.BCEWithLogitsLoss()
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    result_a = evaluate(model_a, test_loader, criterion, device, thresholds_a)
    result_b = evaluate(model_b, test_loader, criterion, device, thresholds_b)

    # ── Report ──
    log_section(
        logger,
        "HOLDOUT TEST SET EVALUATION (%d original images)",
        len(test_samples),
        width=80,
    )
    logger.info("  A thresholds: %s", dict(zip(CLASS_NAMES, thresholds_a)))
    logger.info("  B thresholds: %s", dict(zip(CLASS_NAMES, thresholds_b)))
    logger.info("")

    logger.info(
        "%-20s  %-18s  %-18s  %-10s",
        "Metric",
        "A (Baseline)",
        "B (pos_weight)",
        "Delta",
    )
    logger.info("-" * 70)
    delta_f1 = result_b["macro_f1"] - result_a["macro_f1"]
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Macro F1",
        result_a["macro_f1"],
        result_b["macro_f1"],
        delta_f1,
    )
    delta_loss = result_b["loss"] - result_a["loss"]
    logger.info(
        "%-20s  %-18.4f  %-18.4f  %+.4f",
        "Loss",
        result_a["loss"],
        result_b["loss"],
        delta_loss,
    )

    logger.info("")
    logger.info("Per-class F1 comparison:")
    logger.info(
        "  %-20s  %-7s %-7s %-7s | %-7s %-7s %-7s | %-7s",
        "Class",
        "A_P",
        "A_R",
        "A_F1",
        "B_P",
        "B_R",
        "B_F1",
        "Delta",
    )
    logger.info("  " + "-" * 75)
    deltas = []
    for i, name in enumerate(CLASS_NAMES):
        d = result_b["per_class_f1"][i] - result_a["per_class_f1"][i]
        deltas.append(d)
        logger.info(
            "  %-20s  %-7.4f %-7.4f %-7.4f | %-7.4f %-7.4f %-7.4f | %+.4f",
            name,
            result_a["per_class_precision"][i],
            result_a["per_class_recall"][i],
            result_a["per_class_f1"][i],
            result_b["per_class_precision"][i],
            result_b["per_class_recall"][i],
            result_b["per_class_f1"][i],
            d,
        )

    # ── Summary ──
    log_section(logger, "SUMMARY", width=80)
    logger.info(
        "  Macro F1:  A (Baseline)=%.4f  B (pos_weight)=%.4f  Delta=%+.4f",
        result_a["macro_f1"],
        result_b["macro_f1"],
        delta_f1,
    )

    improved = sum(1 for d in deltas if d > 0)
    degraded = sum(1 for d in deltas if d < 0)
    logger.info(
        "  Per-class: %d improved, %d degraded (of %d)",
        improved,
        degraded,
        NUM_CLASSES,
    )

    if delta_f1 > 0.005:
        logger.info("  結論: pos_weight 有正面效果 (%+.4f F1)", delta_f1)
    elif delta_f1 < -0.005:
        logger.info("  結論: pos_weight 有負面效果 (%+.4f F1)", delta_f1)
    else:
        logger.info("  結論: pos_weight 影響不大 (delta=%.4f)", delta_f1)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
