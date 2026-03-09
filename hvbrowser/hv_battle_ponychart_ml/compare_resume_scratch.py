"""
比較 resume training vs from-scratch training，找出 checkpoint 過時的臨界點。

以時間順序將資料分為「舊資料」與「新資料」，模擬不同 checkpoint 年齡下的
resume 效果，與同資料量的 from-scratch 比較，找出 crossover point。

實驗設計：
  - 20% holdout test set（僅原圖）
  - 對每個 base_fraction（模擬 checkpoint 只看過前 F% 的資料）：
    1. 用前 F% 資料 from scratch 訓練 -> 取得 checkpoint
    2. 用該 checkpoint resume，在 100% 資料上 fine-tune -> resume_f1
  - Baseline：100% 資料 from scratch -> scratch_f1
  - 比較 delta = scratch_f1 - resume_f1，找出何時 from-scratch 開始勝出

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_resume_scratch
"""

from __future__ import annotations

import copy
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

BASE_FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]
SAFETY_MARGIN = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_train_val(
    samples: list[tuple[str, list[int]]],
    seed: int,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Apply orig/crop balance + train/val split (same pipeline as train.py)."""
    orig, crop = separate_orig_crop(samples)
    orig_rates = compute_class_rates(orig)
    rng = np.random.RandomState(seed)
    balanced_crops = balance_crop_samples(crop, orig_rates, rng)
    balanced = orig + balanced_crops

    sub_groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(balanced):
        base = get_base_timestamp(os.path.basename(path))
        sub_groups[base].append(idx)

    train_gk, val_gk = split_by_groups(balanced, test_size=VAL_SIZE)
    train_samples = [balanced[i] for gk in train_gk for i in sub_groups[gk]]
    val_samples = [balanced[i] for gk in val_gk for i in sub_groups[gk]]
    return train_samples, val_samples


def find_crossover(
    ratios: list[float],
    deltas: list[float],
) -> float | None:
    """Find the new_data_ratio where delta crosses from negative to positive.

    Uses linear interpolation between adjacent points.
    Returns None if no crossover is found.
    """
    for i in range(len(deltas) - 1):
        if deltas[i] <= 0 < deltas[i + 1]:
            # Linear interpolation
            t = -deltas[i] / (deltas[i + 1] - deltas[i])
            return ratios[i] + t * (ratios[i + 1] - ratios[i])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # ── Load all samples ──
    all_samples = load_samples()
    logger.info("Total samples loaded: %d", len(all_samples))

    # ── Build group index ──
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    # ── Split: 20% test (originals only), 80% train+val pool ──
    train_val_group_keys, test_group_keys = split_by_groups(
        all_samples, test_size=HOLDOUT_TEST_SIZE
    )

    test_indices = []
    for gk in test_group_keys:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                test_indices.append(idx)
    test_samples = [all_samples[i] for i in test_indices]
    logger.info("Test set (originals only): %d images", len(test_samples))

    # Prepare shared test loader
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds, BATCH_SIZE, shuffle=False, num_workers=num_workers, device=device
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Collect pool samples ──
    pool_indices = [idx for gk in train_val_group_keys for idx in groups[gk]]
    pool_samples = [all_samples[i] for i in pool_indices]

    # Sort pool group keys chronologically
    sorted_pool_groups = sorted(train_val_group_keys)
    logger.info("Pool groups (chronological): %d", len(sorted_pool_groups))

    # ── Full train/val split for baseline and resume targets ──
    full_train, full_val = prepare_train_val(pool_samples, SEED)
    logger.info("Full pool: train=%d  val=%d", len(full_train), len(full_val))

    # ── Baseline: 100% from scratch ──
    log_section(logger, "BASELINE: %d%% data from scratch", 100, width=70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    scratch_model, scratch_thresholds = train_model(
        full_train,
        full_val,
        device,
        num_workers,
        "Baseline (100% from scratch)",
        backbone=BACKBONE,
    )
    scratch_result = evaluate(
        scratch_model, test_loader, criterion, device, scratch_thresholds
    )
    scratch_f1 = scratch_result["macro_f1"]
    logger.info(">> Baseline scratch F1: %.4f", scratch_f1)

    # ── Per-fraction experiments ──
    experiment_results: list[dict[str, Any]] = []

    for frac in BASE_FRACTIONS:
        n_groups = max(1, int(np.ceil(frac * len(sorted_pool_groups))))
        base_group_keys = sorted_pool_groups[:n_groups]

        # Collect base samples
        base_indices = []
        for gk in base_group_keys:
            base_indices.extend(groups[gk])
        base_samples = [all_samples[i] for i in base_indices]

        new_n = len(pool_samples) - len(base_samples)
        new_ratio = new_n / max(len(base_samples), 1)
        pct = int(frac * 100)

        # Step 1: Train from scratch on base data -> checkpoint
        log_section(
            logger,
            "BASE %d%% (%d samples) -> new_data_ratio=%.1f%%",
            pct,
            len(base_samples),
            new_ratio * 100,
            width=70,
        )

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        base_train, base_val = prepare_train_val(base_samples, SEED)
        logger.info("  Base train=%d  val=%d", len(base_train), len(base_val))

        base_model, _ = train_model(
            base_train,
            base_val,
            device,
            num_workers,
            f"Base {pct}% from scratch",
            backbone=BACKBONE,
        )
        base_state_dict = copy.deepcopy(base_model.state_dict())

        # Step 2: Resume from base checkpoint with 100% data
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        resume_model, resume_thresholds = train_model(
            full_train,
            full_val,
            device,
            num_workers,
            f"Resume from {pct}% checkpoint",
            backbone=BACKBONE,
            resume_state_dict=base_state_dict,
        )
        resume_result = evaluate(
            resume_model, test_loader, criterion, device, resume_thresholds
        )

        delta = scratch_f1 - resume_result["macro_f1"]
        experiment_results.append(
            {
                "fraction": frac,
                "base_n": len(base_samples),
                "new_n": new_n,
                "new_ratio": new_ratio,
                "resume_f1": resume_result["macro_f1"],
                "resume_per_class_f1": resume_result["per_class_f1"],
                "delta": delta,
            }
        )
        logger.info(
            ">> Base %d%%: resume_f1=%.4f  scratch_f1=%.4f  delta=%+.4f",
            pct,
            resume_result["macro_f1"],
            scratch_f1,
            delta,
        )

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    log_section(logger, "COMPARISON TABLE (scratch F1=%.4f)", scratch_f1)
    logger.info("")
    logger.info(
        "  %-6s  %-8s  %-8s  %-12s  %-10s  %-10s  %-10s  %-10s",
        "Base%",
        "Base N",
        "New N",
        "New Ratio",
        "Resume F1",
        "Scratch F1",
        "Delta",
        "Winner",
    )
    logger.info("  " + "-" * 88)

    for r in experiment_results:
        winner = "SCRATCH" if r["delta"] > 0 else "RESUME"
        logger.info(
            "  %-6d  %-8d  %-8d  %-12.1f%%  %-10.4f  %-10.4f  %+-10.4f  %-10s",
            int(r["fraction"] * 100),
            r["base_n"],
            r["new_n"],
            r["new_ratio"] * 100,
            r["resume_f1"],
            scratch_f1,
            r["delta"],
            winner,
        )

    # ── Per-class breakdown ──
    log_section(
        logger,
        "PER-CLASS DELTA (scratch_f1 - resume_f1, positive = scratch better)",
    )
    logger.info("")

    header = f"  {'Class':<20s}"
    for r in experiment_results:
        header += f"  {int(r['fraction'] * 100)}%".ljust(10)
    logger.info(header)
    logger.info("  " + "-" * (20 + 10 * len(experiment_results)))

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<20s}"
        for r in experiment_results:
            class_delta = (
                scratch_result["per_class_f1"][i] - r["resume_per_class_f1"][i]
            )
            row += f"  {class_delta:+.4f}  "
        logger.info(row)

    # ── Crossover analysis ──
    log_section(logger, "CROSSOVER ANALYSIS")

    ratios = [r["new_ratio"] for r in experiment_results]
    deltas = [r["delta"] for r in experiment_results]
    crossover = find_crossover(ratios, deltas)

    if crossover is not None:
        logger.info("  Crossover point: new_data_ratio ≈ %.1f%%", crossover * 100)
        logger.info(
            "  (當新增資料超過 checkpoint 訓練量的 %.1f%% 時，"
            "from-scratch 開始優於 resume)",
            crossover * 100,
        )
    elif all(d <= 0 for d in deltas):
        logger.info("  在所有測試的 base fractions 下，resume 始終優於或等於 scratch。")
        logger.info("  未找到 crossover point。")
    elif all(d > 0 for d in deltas):
        logger.info("  在所有測試的 base fractions 下，scratch 始終優於 resume。")
        logger.info("  建議始終使用 from-scratch 訓練。")

    # ── Per-class warnings ──
    logger.info("")
    logger.info("Per-class 警示 (在最保守的 base fraction 下):")
    worst_idx = -1  # Use the largest base fraction (smallest new_ratio)
    worst_r = experiment_results[worst_idx]
    warned = False
    for i, name in enumerate(CLASS_NAMES):
        class_delta = (
            scratch_result["per_class_f1"][i] - worst_r["resume_per_class_f1"][i]
        )
        if class_delta > 0.03:
            logger.info(
                "  %s: 即使 base=%d%%, resume 仍比 scratch 差 %.4f F1",
                name,
                int(worst_r["fraction"] * 100),
                class_delta,
            )
            warned = True
    if not warned:
        logger.info(
            "  無異常。在 base=%d%% 下各 class 的 resume 表現均正常。",
            int(worst_r["fraction"] * 100),
        )

    # ══════════════════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════
    log_section(logger, "RECOMMENDATION")

    if crossover is not None:
        recommended = crossover * SAFETY_MARGIN
        logger.info("  Crossover: new_data_ratio ≈ %.1f%%", crossover * 100)
        logger.info(
            "  安全邊界 (×%.0f%%): %.1f%%",
            SAFETY_MARGIN * 100,
            recommended * 100,
        )
        logger.info("")
        logger.info("  具體行動:")
        logger.info(
            "    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = %.2f",
            recommended,
        )
        logger.info("")
        logger.info(
            "  效果: 當 load_samples() 的樣本數超過 checkpoint 訓練量的 %.1f%% 時，",
            recommended * 100,
        )
        logger.info("        train.py 會自動切換為 from-scratch 訓練。")
    elif all(d <= 0 for d in deltas):
        max_ratio = max(ratios)
        recommended = max_ratio * 2
        logger.info(
            "  Resume 在所有測試範圍內皆優於 scratch (最大 new_ratio=%.1f%%)。",
            max_ratio * 100,
        )
        logger.info("")
        logger.info("  具體行動:")
        logger.info(
            "    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = %.2f",
            recommended,
        )
        logger.info("")
        logger.info(
            "  說明: 設為 %.2f (測試範圍 %.1f%% 的兩倍)，" "保留充足 resume 空間。",
            recommended,
            max_ratio * 100,
        )
        logger.info("  建議: 當資料量大幅增加後重新執行此分析工具。")
    else:
        logger.info("  Scratch 在所有測試範圍內皆優於 resume。")
        logger.info("")
        logger.info("  具體行動:")
        logger.info("    修改 common/constants.py 中的 RETRAIN_NEW_DATA_RATIO = 0.0")
        logger.info("")
        logger.info("  效果: train.py 將永遠使用 from-scratch 訓練。")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
