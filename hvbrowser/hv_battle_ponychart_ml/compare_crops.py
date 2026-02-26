"""
比較裁切圖片對訓練效果的影響，分離「資料增量」與「分佈偏差」兩個因素。

三組實驗：
  A: 原圖 + 所有 crop（現有偏差分佈）
  B: 僅原圖（baseline）
  C: 原圖 + 平衡 resample 後的 crop（移除偏差）

- B vs A = 總效應（增量 + 偏差）
- C vs B = 純增量效應
- A vs C = 純偏差效應

共用 10% 原始圖片作為測試集，確保評估基準一致。
最後印出 per-class 分佈偏差與 F1 差異的 Pearson 相關性，量化偏差的因果影響。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_crops
"""

from __future__ import annotations

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
    NUM_CLASSES,
    SEED,
    evaluate,
    get_base_timestamp,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    is_original,
    load_samples,
    make_dataloader,
    split_by_groups,
    train_model,
)
from .common.data import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Unique helpers for crop analysis
# ---------------------------------------------------------------------------
def compute_class_rates(
    samples: list[tuple[str, list[int]]],
) -> list[float]:
    """計算每個 class 的出現比例 (positive rate)。"""
    counts = [0] * NUM_CLASSES
    for _, labels in samples:
        for lbl in labels:
            counts[lbl - 1] += 1
    n = max(len(samples), 1)
    return [c / n for c in counts]


def balance_crop_samples(
    crop_samples: list[tuple[str, list[int]]],
    target_rates: list[float],
    rng: np.random.RandomState,
) -> list[tuple[str, list[int]]]:
    """Oversample crop 圖片使 per-class 出現比例接近 target_rates。"""
    if not crop_samples:
        return []

    current_rates = compute_class_rates(crop_samples)
    n = len(crop_samples)

    target_counts = [max(int(round(tr * n)), 0) for tr in target_rates]
    current_counts = [int(round(cr * n)) for cr in current_rates]

    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, labels) in enumerate(crop_samples):
        for lbl in labels:
            class_to_indices[lbl - 1].append(idx)

    extra_indices: set[int] = set()
    extra_samples: list[tuple[str, list[int]]] = []

    for cls in range(NUM_CLASSES):
        deficit = target_counts[cls] - current_counts[cls]
        if deficit <= 0 or not class_to_indices[cls]:
            continue
        sampled = rng.choice(class_to_indices[cls], size=deficit, replace=True)
        for idx in sampled:
            if idx not in extra_indices:
                extra_indices.add(idx)
                extra_samples.append(crop_samples[idx])

    return list(crop_samples) + extra_samples


def log_distribution(
    label: str,
    samples: list[tuple[str, list[int]]],
) -> list[float]:
    """印出並回傳 per-class positive rate。"""
    rates = compute_class_rates(samples)
    logger.info("  %s (%d samples):", label, len(samples))
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for _, lbls in samples if (i + 1) in lbls)
        logger.info(
            "    %-20s  %4d  (%.1f%%)", name, count, rates[i] * 100
        )
    return rates


def _pearson_r(x: list[float], y: list[float]) -> float:
    """計算 Pearson 相關係數（不依賴 scipy）。"""
    xa = np.array(x)
    ya = np.array(y)
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.sqrt((xa**2).sum() * (ya**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xa * ya).sum() / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load all samples
    all_samples = load_samples()
    logger.info("Total samples loaded: %d", len(all_samples))

    # Split groups: 10% test, 90% train+val
    train_val_groups, test_groups = split_by_groups(
        all_samples, test_size=0.10, seed=SEED
    )

    # Build index from base timestamp to sample indices
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    # Test set: only original images from test groups
    test_indices = []
    for gk in test_groups:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                test_indices.append(idx)
    test_samples = [all_samples[i] for i in test_indices]

    # Collect train+val indices, separate originals and crops
    train_val_indices_orig = []
    train_val_indices_crop = []
    for gk in train_val_groups:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                train_val_indices_orig.append(idx)
            else:
                train_val_indices_crop.append(idx)

    train_val_all = [
        all_samples[i]
        for gk in train_val_groups
        for i in groups[gk]
    ]
    train_val_orig = [all_samples[i] for i in train_val_indices_orig]
    train_val_crop = [all_samples[i] for i in train_val_indices_crop]

    # ── Experiment C: balance crop samples to match original distribution ──
    orig_rates = compute_class_rates(train_val_orig)
    balanced_crops = balance_crop_samples(train_val_crop, orig_rates, rng)
    train_val_balanced = train_val_orig + balanced_crops

    # ── Split train/val for each experiment ──
    # Experiment A: originals + all crops (biased)
    train_gk_a, val_gk_a = split_by_groups(
        train_val_all, test_size=0.15, seed=SEED
    )
    groups_a: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_all):
        base = get_base_timestamp(os.path.basename(path))
        groups_a[base].append(idx)
    train_a = [
        train_val_all[i] for gk in train_gk_a for i in groups_a[gk]
    ]
    val_a = [train_val_all[i] for gk in val_gk_a for i in groups_a[gk]]

    # Experiment B: originals only
    train_gk_b, val_gk_b = split_by_groups(
        train_val_orig, test_size=0.15, seed=SEED
    )
    groups_b: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_orig):
        base = get_base_timestamp(os.path.basename(path))
        groups_b[base].append(idx)
    train_b = [
        train_val_orig[i] for gk in train_gk_b for i in groups_b[gk]
    ]
    val_b = [train_val_orig[i] for gk in val_gk_b for i in groups_b[gk]]

    # Experiment C: originals + balanced crops
    train_gk_c, val_gk_c = split_by_groups(
        train_val_balanced, test_size=0.15, seed=SEED
    )
    groups_c: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_balanced):
        base = get_base_timestamp(os.path.basename(path))
        groups_c[base].append(idx)
    train_c = [
        train_val_balanced[i] for gk in train_gk_c for i in groups_c[gk]
    ]
    val_c = [
        train_val_balanced[i] for gk in val_gk_c for i in groups_c[gk]
    ]

    criterion = nn.BCEWithLogitsLoss()

    # ---- Train experiments ----
    model_a, thresholds_a = train_model(
        train_a, val_a, device, num_workers,
        "A: Originals + biased crops",
        backbone=BACKBONE,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_b, thresholds_b = train_model(
        train_b, val_b, device, num_workers,
        "B: Originals only (baseline)",
        backbone=BACKBONE,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_c, thresholds_c = train_model(
        train_c, val_c, device, num_workers,
        "C: Originals + balanced crops",
        backbone=BACKBONE,
    )

    # ---- Evaluate all on test set ----
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=num_workers, device=device,
    )

    result_a = evaluate(
        model_a, test_loader, criterion, device, thresholds_a
    )
    result_b = evaluate(
        model_b, test_loader, criterion, device, thresholds_b
    )
    result_c = evaluate(
        model_c, test_loader, criterion, device, thresholds_c
    )

    # ── Data split summary ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA SPLIT SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "Test set (shared, originals only): %d images", len(test_samples)
    )
    logger.info("")
    logger.info("Experiment A (orig + biased crops):")
    logger.info("  Train: %d  Val: %d", len(train_a), len(val_a))
    logger.info("Experiment B (originals only):")
    logger.info("  Train: %d  Val: %d", len(train_b), len(val_b))
    logger.info("Experiment C (orig + balanced crops):")
    logger.info("  Train: %d  Val: %d", len(train_c), len(val_c))

    # ── Distribution analysis ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)
    log_distribution("Original images (train+val)", train_val_orig)
    crop_rates = log_distribution("Crop images (raw)", train_val_crop)
    log_distribution("Crop images (balanced)", balanced_crops)
    logger.info("")
    logger.info("  Per-class bias (crop_rate - orig_rate):")
    bias_per_class = []
    for i, name in enumerate(CLASS_NAMES):
        bias = crop_rates[i] - orig_rates[i]
        bias_per_class.append(bias)
        logger.info("    %-20s  %+.1f%%", name, bias * 100)

    # ── Print comparison ──
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "TEST SET EVALUATION (on %d original images)", len(test_samples)
    )
    logger.info("=" * 80)
    logger.info(
        "  A thresholds: %s", dict(zip(CLASS_NAMES, thresholds_a))
    )
    logger.info(
        "  B thresholds: %s", dict(zip(CLASS_NAMES, thresholds_b))
    )
    logger.info(
        "  C thresholds: %s", dict(zip(CLASS_NAMES, thresholds_c))
    )
    logger.info("")
    logger.info(
        "%-20s  %-14s  %-14s  %-14s",
        "Metric",
        "A (biased)",
        "B (orig only)",
        "C (balanced)",
    )
    logger.info("-" * 80)
    logger.info(
        "%-20s  %-14.4f  %-14.4f  %-14.4f",
        "Macro F1",
        result_a["macro_f1"],
        result_b["macro_f1"],
        result_c["macro_f1"],
    )
    logger.info(
        "%-20s  %-14.4f  %-14.4f  %-14.4f",
        "Loss",
        result_a["loss"],
        result_b["loss"],
        result_c["loss"],
    )

    logger.info("")
    logger.info("Per-class detail (optimized thresholds):")
    logger.info(
        "  %-20s  %-7s %-7s %-7s | %-7s %-7s %-7s | %-7s %-7s %-7s",
        "Class",
        "A_P",
        "A_R",
        "A_F1",
        "B_P",
        "B_R",
        "B_F1",
        "C_P",
        "C_R",
        "C_F1",
    )
    for i, name in enumerate(CLASS_NAMES):
        logger.info(
            "  %-20s  %-7.4f %-7.4f %-7.4f | %-7.4f %-7.4f %-7.4f"
            " | %-7.4f %-7.4f %-7.4f",
            name,
            result_a["per_class_precision"][i],
            result_a["per_class_recall"][i],
            result_a["per_class_f1"][i],
            result_b["per_class_precision"][i],
            result_b["per_class_recall"][i],
            result_b["per_class_f1"][i],
            result_c["per_class_precision"][i],
            result_c["per_class_recall"][i],
            result_c["per_class_f1"][i],
        )

    # ── Effect decomposition ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("EFFECT DECOMPOSITION")
    logger.info("=" * 80)
    total_effect = result_a["macro_f1"] - result_b["macro_f1"]
    augment_effect = result_c["macro_f1"] - result_b["macro_f1"]
    bias_effect = result_a["macro_f1"] - result_c["macro_f1"]
    logger.info(
        "  A vs B (total effect = augmentation + bias): %+.4f",
        total_effect,
    )
    logger.info(
        "  C vs B (pure augmentation effect):           %+.4f",
        augment_effect,
    )
    logger.info(
        "  A vs C (pure bias effect):                   %+.4f", bias_effect
    )

    logger.info("")
    logger.info("Per-class effect decomposition:")
    logger.info(
        "  %-20s  %-10s  %-10s  %-10s  %-10s",
        "Class",
        "Bias",
        "A-B total",
        "C-B augment",
        "A-C bias",
    )
    f1_diff_ab = []
    for i, name in enumerate(CLASS_NAMES):
        ab = result_a["per_class_f1"][i] - result_b["per_class_f1"][i]
        cb = result_c["per_class_f1"][i] - result_b["per_class_f1"][i]
        ac = result_a["per_class_f1"][i] - result_c["per_class_f1"][i]
        f1_diff_ab.append(ab)
        logger.info(
            "  %-20s  %+.1f%%     %+.4f     %+.4f     %+.4f",
            name,
            bias_per_class[i] * 100,
            ab,
            cb,
            ac,
        )

    # ── Correlation: distribution bias vs F1 impact ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 80)
    r_ab = _pearson_r(bias_per_class, f1_diff_ab)
    f1_diff_ac = [
        result_a["per_class_f1"][i] - result_c["per_class_f1"][i]
        for i in range(NUM_CLASSES)
    ]
    r_ac = _pearson_r(bias_per_class, f1_diff_ac)
    ab_hint = (
        "(偏差越大 -> 該 class 在 A 中表現越好)"
        if r_ab > 0
        else "(偏差越大 -> 反而越差)"
    )
    ac_hint = (
        "(正相關 = 偏差確實影響 F1)"
        if abs(r_ac) > 0.3
        else "(弱相關 = 偏差對 F1 影響有限)"
    )
    logger.info(
        "  Pearson r (bias vs A-B F1 diff): %.4f  %s", r_ab, ab_hint
    )
    logger.info(
        "  Pearson r (bias vs A-C F1 diff): %.4f  %s", r_ac, ac_hint
    )

    # ── Summary ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "  Macro F1:  A=%.4f  B=%.4f  C=%.4f",
        result_a["macro_f1"],
        result_b["macro_f1"],
        result_c["macro_f1"],
    )
    logger.info("  Total effect   (A-B): %+.4f", total_effect)
    logger.info("  Augment effect (C-B): %+.4f", augment_effect)
    logger.info("  Bias effect    (A-C): %+.4f", bias_effect)
    logger.info("  Bias-F1 correlation:  r=%.4f", r_ac)
    logger.info("")
    if abs(bias_effect) < 0.005:
        logger.info(
            "  結論: 裁切偏差對整體 F1 影響有限 (%.4f)", bias_effect
        )
    elif bias_effect < -0.005:
        logger.info(
            "  結論: 裁切偏差降低效果 (%.4f F1)，建議使用平衡後的 crop",
            bias_effect,
        )
    else:
        logger.info(
            "  結論: 裁切偏差反而有正面效果 (+%.4f F1)", bias_effect
        )
    if abs(r_ac) > 0.5:
        logger.info(
            "  注意: 偏差與 per-class F1 有強相關"
            " (r=%.2f)，特定角色受影響顯著",
            r_ac,
        )
    logger.info("=" * 80)

    # ── Crop recommendation ──
    crop_counts_per_class = [0] * NUM_CLASSES
    for _, labels in train_val_crop:
        for lbl in labels:
            crop_counts_per_class[lbl - 1] += 1

    total_crops = len(train_val_crop)

    orig_rate_sum = sum(orig_rates)
    target_per_class = [
        int(round(total_crops * (orig_rates[i] / orig_rate_sum)))
        for i in range(NUM_CLASSES)
    ]

    recommendations: list[dict[str, Any]] = []
    max_crop = max(crop_counts_per_class) if crop_counts_per_class else 1
    for i in range(NUM_CLASSES):
        cb = result_c["per_class_f1"][i] - result_b["per_class_f1"][i]
        ab = result_a["per_class_f1"][i] - result_b["per_class_f1"][i]
        b_f1 = result_b["per_class_f1"][i]
        crop_n = crop_counts_per_class[i]
        target_n = target_per_class[i]
        deficit = max(target_n - crop_n, 0)

        is_beneficial = cb > 0.01 or ab > 0.01
        suggested = deficit if is_beneficial else 0

        scarcity = 1.0 - (crop_n / max(max_crop, 1))
        room = 1.0 - b_f1
        score = max(cb, ab) * 0.4 + room * 0.3 + scarcity * 0.3
        recommendations.append(
            {
                "idx": i,
                "name": CLASS_NAMES[i],
                "crop_n": crop_n,
                "target_n": target_n,
                "deficit": deficit,
                "suggested": suggested,
                "b_f1": b_f1,
                "cb": cb,
                "ab": ab,
                "score": score,
                "beneficial": is_beneficial,
            }
        )
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("CROP RECOMMENDATION")
    logger.info("=" * 80)
    logger.info(
        "  估算方式: 讓各 class 的 crop 數量比例"
        "對齊原圖出現比例 (total crops=%d)",
        total_crops,
    )
    logger.info("")
    logger.info(
        "  %-4s %-18s  %-6s %-6s %-6s  %-8s  %-9s %-9s",
        "Rank",
        "Class",
        "Crops",
        "Target",
        "+Need",
        "B F1",
        "C-B",
        "A-B",
    )
    logger.info("  " + "-" * 74)
    for rank, r in enumerate(recommendations, 1):
        if r["beneficial"]:
            if r["suggested"] > 0:
                advice = "<- 建議再裁 {} 張".format(r["suggested"])
            else:
                advice = "<- 已達標，可裁可不裁"
        elif r["cb"] < -0.03 and r["ab"] < -0.03:
            advice = "  (crop 有害，暫不裁切)"
        else:
            advice = "  (效果有限)"
        logger.info(
            "  #%-3d %-18s  %-6d %-6d %-6d  %-8.4f"
            "  %+-9.4f %+-9.4f %s",
            rank,
            r["name"],
            r["crop_n"],
            r["target_n"],
            r["deficit"],
            r["b_f1"],
            r["cb"],
            r["ab"],
            advice,
        )

    # 摘要
    logger.info("")
    to_crop = [
        r for r in recommendations if r["beneficial"] and r["suggested"] > 0
    ]
    if to_crop:
        logger.info("  具體行動:")
        total_suggested = 0
        for r in to_crop:
            logger.info(
                "    %s: 再裁切 ~%d 張 (目前 %d -> 目標 %d)",
                r["name"],
                r["suggested"],
                r["crop_n"],
                r["target_n"],
            )
            total_suggested += r["suggested"]
        logger.info("    共計約 %d 張新 crop", total_suggested)
    else:
        logger.info("  各 class crop 數量已足夠或 crop 無正面效果。")

    saturated = [
        r
        for r in recommendations
        if r["beneficial"] and r["suggested"] == 0
    ]
    if saturated:
        names = ", ".join(r["name"] for r in saturated)
        logger.info(
            "  已達標 (crop 有效但數量足夠): %s", names
        )

    harmful = [
        r
        for r in recommendations
        if r["cb"] < -0.03 and r["ab"] < -0.03
    ]
    if harmful:
        names = ", ".join(r["name"] for r in harmful)
        logger.info("  暫不裁切 (crop 反而有害): %s", names)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
