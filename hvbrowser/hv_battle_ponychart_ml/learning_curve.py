"""
Learning Curve 分析腳本：估算增加 raw 資料對模型 F1 的邊際效益。

以不同比例的訓練 groups 進行巢狀、接續訓練（模擬 train.py 的增量流程），
在共用 test set 上評估，以 power-law 外推預測加 N 張新 raw 圖的預期 F1 提升。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.learning_curve
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
    split_by_groups,
    train_model,
)
from .common.data import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_FRACTIONS = [0.40, 0.55, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]


# ---------------------------------------------------------------------------
# Unique helpers for learning curve
# ---------------------------------------------------------------------------
def nested_subsample_groups(
    group_keys: list[str],
    fraction: float,
    shuffled_order: list[int],
) -> list[str]:
    """從 group_keys 中取前 fraction 比例的 groups（巢狀抽樣）。

    使用預先洗牌的索引順序，保證較小 fraction 是較大 fraction 的子集。
    """
    n = max(1, int(np.ceil(fraction * len(group_keys))))
    return [group_keys[i] for i in shuffled_order[:n]]


def fit_power_law(
    ns: list[int],
    f1s: list[float],
    max_asymptote: float = 1.0,
) -> tuple[float, float, float] | None:
    """Fit F1 ~ a - b * N^(-c) using least-squares grid search on c.

    Constraints:
      - a <= max_asymptote (F1 cannot exceed 1.0)
      - b > 0 (F1 increases with N)

    Returns (a, b, c) or None if fitting fails.
    """
    if len(ns) < 3:
        return None

    x = np.array(ns, dtype=float)
    y = np.array(f1s, dtype=float)

    best_residual = float("inf")
    best_params: tuple[float, float, float] | None = None

    for c in np.arange(0.1, 2.01, 0.05):
        # F1 = a - b * N^(-c); let z = -N^(-c), then F1 = a + b*z
        z = -(x ** (-c))
        design = np.column_stack([np.ones_like(z), z])
        try:
            params, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a_hat, b_hat = params

        if b_hat < 0 or a_hat > max_asymptote:
            continue

        pred = a_hat - b_hat * (x ** (-c))
        res = float(np.sum((y - pred) ** 2))
        if res < best_residual:
            best_residual = res
            best_params = (float(a_hat), float(b_hat), float(c))

    return best_params


def extrapolate_f1(
    params: tuple[float, float, float], n: int
) -> float:
    """Predict F1 at sample count n using fitted power-law parameters."""
    a, b, c = params
    return float(a - b * (n ** (-c)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load all samples
    all_samples = load_samples()
    logger.info("Total samples loaded: %d", len(all_samples))

    # Build group index
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    # Split: 15% test (original images only), 85% train+val pool
    train_val_group_keys, test_group_keys = split_by_groups(
        all_samples, test_size=0.15, seed=SEED
    )

    test_indices = []
    for gk in test_group_keys:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                test_indices.append(idx)
    test_samples = [all_samples[i] for i in test_indices]
    logger.info("Test set (originals only): %d images", len(test_samples))

    # Prepare test loader (shared)
    test_ds = PonyChartDataset(
        test_samples, get_transforms(is_train=False)
    )
    test_loader = make_dataloader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=num_workers, device=device,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Prepare nested sampling order (shuffle once) ──
    rng = np.random.RandomState(SEED)
    shuffled_order = list(rng.permutation(len(train_val_group_keys)))

    # ── Run experiments at different data fractions (sequential resume) ──
    experiment_results: list[dict[str, Any]] = []
    prev_state_dict: dict[str, Any] | None = None

    for frac in DATA_FRACTIONS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Nested subsample: smaller fractions are always subsets of larger ones
        selected_groups = nested_subsample_groups(
            train_val_group_keys, frac, shuffled_order
        )

        # Collect samples for selected groups
        selected_indices = []
        for gk in selected_groups:
            selected_indices.extend(groups[gk])
        selected_samples = [all_samples[i] for i in selected_indices]

        # Split into train / val (15% val for early stopping)
        sub_train_gk, sub_val_gk = split_by_groups(
            selected_samples, test_size=0.15, seed=SEED
        )
        sub_groups: dict[str, list[int]] = defaultdict(list)
        for idx, (path, _) in enumerate(selected_samples):
            base = get_base_timestamp(os.path.basename(path))
            sub_groups[base].append(idx)

        raw_train_samples = [
            selected_samples[i]
            for gk in sub_train_gk
            for i in sub_groups[gk]
        ]
        val_samples = [
            selected_samples[i]
            for gk in sub_val_gk
            for i in sub_groups[gk]
        ]

        # Crop balancing (same as train.py)
        orig_train = [
            s for s in raw_train_samples
            if is_original(os.path.basename(s[0]))
        ]
        crop_train = [
            s for s in raw_train_samples
            if not is_original(os.path.basename(s[0]))
        ]
        orig_rates = compute_class_rates(orig_train)
        balanced_crops = balance_crop_samples(
            crop_train, orig_rates, np.random.RandomState(SEED)
        )
        train_samples = orig_train + balanced_crops

        pct = int(frac * 100)
        name = f"{pct}% data ({len(selected_samples)} samples)"

        model, thresholds = train_model(
            train_samples,
            val_samples,
            device,
            num_workers,
            name,
            backbone=BACKBONE,
            resume_state_dict=prev_state_dict,
        )
        prev_state_dict = copy.deepcopy(model.state_dict())

        # Evaluate on shared test set
        result = evaluate(
            model, test_loader, criterion, device, thresholds
        )
        result["fraction"] = frac
        result["n_samples"] = len(selected_samples)
        result["n_train"] = len(train_samples)
        result["n_val"] = len(val_samples)
        result["thresholds"] = thresholds
        experiment_results.append(result)

        logger.info(
            ">> %s: test Macro F1=%.4f  thresholds=%s",
            name,
            result["macro_f1"],
            dict(zip(CLASS_NAMES, thresholds)),
        )

    # ── Comparison table ──
    logger.info("")
    logger.info("=" * 90)
    logger.info(
        "LEARNING CURVE RESULTS (test set: %d original images)",
        len(test_samples),
    )
    logger.info("=" * 90)

    logger.info("")
    logger.info(
        "%-8s  %-10s  %-8s  %-8s  %-10s  %-10s",
        "Frac",
        "Samples",
        "Train",
        "Val",
        "Macro F1",
        "Delta",
    )
    logger.info("-" * 65)
    prev_f1 = 0.0
    for r in experiment_results:
        pct_str = f"{int(r['fraction'] * 100)}%"
        delta = r["macro_f1"] - prev_f1 if prev_f1 > 0 else 0.0
        delta_str = f"{delta:+.4f}" if prev_f1 > 0 else "---"
        logger.info(
            "%-8s  %-10d  %-8d  %-8d  %-10.4f  %-10s",
            pct_str,
            r["n_samples"],
            r["n_train"],
            r["n_val"],
            r["macro_f1"],
            delta_str,
        )
        prev_f1 = r["macro_f1"]

    # Per-class F1 table
    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for r in experiment_results:
        header += f"  {int(r['fraction'] * 100)}%".ljust(14)
    logger.info(header)
    logger.info(
        "  " + "-" * (20 + 14 * len(experiment_results))
    )

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<20s}"
        for r in experiment_results:
            row += f"  {r['per_class_f1'][i]:<12.4f}"
        logger.info(row)

    # Per-class threshold table
    logger.info("")
    logger.info("Per-class optimized thresholds:")
    header = f"  {'Class':<20s}"
    for r in experiment_results:
        header += f"  {int(r['fraction'] * 100)}%".ljust(14)
    logger.info(header)
    logger.info(
        "  " + "-" * (20 + 14 * len(experiment_results))
    )

    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<20s}"
        for r in experiment_results:
            row += f"  {r['thresholds'][i]:<12.4f}"
        logger.info(row)

    # ── Power-law extrapolation ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("POWER-LAW EXTRAPOLATION")
    logger.info("=" * 90)

    ns = [r["n_samples"] for r in experiment_results]
    macro_f1s = [r["macro_f1"] for r in experiment_results]

    # Macro F1 extrapolation
    params = fit_power_law(ns, macro_f1s)
    if params is not None:
        a, b, c = params
        logger.info(
            "  Fitted: F1 ~ %.4f - %.4f * N^(-%.2f)", a, b, c
        )
        logger.info("  Asymptotic F1 (N->inf): %.4f", a)
        logger.info("")

        current_n = ns[-1]
        current_f1 = macro_f1s[-1]
        logger.info(
            "  %-20s  %-12s  %-12s",
            "Additional samples",
            "Predicted F1",
            "Delta",
        )
        logger.info("  " + "-" * 50)
        for extra in [200, 400, 600, 1000, 2000]:
            pred = extrapolate_f1(params, current_n + extra)
            delta = pred - current_f1
            logger.info(
                "  +%-19d  %-12.4f  %+.4f",
                extra,
                min(pred, a),
                min(delta, a - current_f1),
            )

        # 計算達到特定 F1 所需的資料量
        logger.info("")
        logger.info("  Estimated samples needed for target F1:")
        gap = a - current_f1
        for target_pct in [0.25, 0.50, 0.75, 0.90]:
            target_f1 = current_f1 + gap * target_pct
            if target_f1 >= a:
                logger.info(
                    "    %.0f%% of gap (F1=%.4f): "
                    "unreachable (asymptote=%.4f)",
                    target_pct * 100,
                    target_f1,
                    a,
                )
                continue
            # a - b * N^(-c) = target_f1
            # => N = (b / (a - target_f1))^(1/c)
            needed_n = (b / (a - target_f1)) ** (1 / c)
            extra_needed = max(0, int(needed_n) - current_n)
            logger.info(
                "    %.0f%% of gap (F1=%.4f): "
                "~%d total samples (+%d new)",
                target_pct * 100,
                target_f1,
                int(needed_n),
                extra_needed,
            )
    else:
        logger.info(
            "  Power-law fitting failed (insufficient data points)."
        )

    # ── Per-class saturation analysis ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("PER-CLASS SATURATION ANALYSIS")
    logger.info("=" * 90)

    logger.info("")
    logger.info(
        "  %-20s  %-8s  %-8s  %-10s  %-10s  %s",
        "Class",
        f"F1@{int(DATA_FRACTIONS[0] * 100)}%",
        "F1@100%",
        "Gain",
        "Slope",
        "Status",
    )
    logger.info("  " + "-" * 80)

    for i, name in enumerate(CLASS_NAMES):
        f1_first = experiment_results[0]["per_class_f1"][i]
        f1_last = experiment_results[-1]["per_class_f1"][i]
        gain = f1_last - f1_first

        # Slope between last two points (marginal gain)
        if len(experiment_results) >= 2:
            f1_prev = experiment_results[-2]["per_class_f1"][i]
            n_prev = experiment_results[-2]["n_samples"]
            n_last = experiment_results[-1]["n_samples"]
            slope = (
                (f1_last - f1_prev) / max(n_last - n_prev, 1) * 100
            )
        else:
            slope = 0.0

        if gain < 0.01 and f1_last > 0.85:
            status = "SATURATED"
        elif slope > 0.005:
            status = "GROWING"
        elif slope > 0.001:
            status = "SLOWING"
        elif gain < 0.02:
            status = "FLAT"
        else:
            status = "SATURATING"

        logger.info(
            "  %-20s  %-8.4f  %-8.4f  %+-.4f     %.5f/100  %s",
            name,
            f1_first,
            f1_last,
            gain,
            slope,
            status,
        )

    # ── Per-class extrapolation ──
    logger.info("")
    logger.info("Per-class extrapolation (+400 samples):")
    current_n = ns[-1]
    for i, name in enumerate(CLASS_NAMES):
        class_f1s = [r["per_class_f1"][i] for r in experiment_results]
        class_params = fit_power_law(ns, class_f1s)
        if class_params is not None:
            pred = extrapolate_f1(class_params, current_n + 400)
            delta = pred - class_f1s[-1]
            asymptote = class_params[0]
            logger.info(
                "  %-20s  now=%.4f  +400->%.4f (%+.4f)"
                "  asymptote=%.4f",
                name,
                class_f1s[-1],
                min(pred, asymptote),
                min(delta, asymptote - class_f1s[-1]),
                asymptote,
            )
        else:
            logger.info("  %-20s  fitting failed", name)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("SUMMARY")
    logger.info("=" * 90)
    logger.info(
        "  Current: %d samples -> Macro F1=%.4f",
        ns[-1],
        macro_f1s[-1],
    )

    if params is not None:
        asymptote = params[0]
        headroom = asymptote - macro_f1s[-1]
        logger.info(
            "  Asymptotic Macro F1: %.4f (headroom: %.4f)",
            asymptote,
            headroom,
        )
        if headroom < 0.01:
            logger.info(
                "  結論: 模型在目前架構下已接近飽和 (headroom < 0.01)，"
                "加更多資料幾乎無法提升 F1。"
            )
            logger.info(
                "  建議: 考慮更大的模型架構、更好的特徵工程、或 ensemble。"
            )
        elif headroom < 0.03:
            logger.info(
                "  結論: 剩餘提升空間有限 (headroom=%.4f)，"
                "加資料的邊際效益已明顯遞減。",
                headroom,
            )
            pred_400 = extrapolate_f1(params, ns[-1] + 400)
            logger.info(
                "  預估再加 400 張 raw 可提升 F1 約 %+.4f",
                min(pred_400 - macro_f1s[-1], headroom),
            )
        else:
            logger.info(
                "  結論: 仍有提升空間 (headroom=%.4f)，"
                "加更多資料可能有效。",
                headroom,
            )
            pred_400 = extrapolate_f1(params, ns[-1] + 400)
            logger.info(
                "  預估再加 400 張 raw 可提升 F1 約 %+.4f",
                min(pred_400 - macro_f1s[-1], headroom),
            )

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
