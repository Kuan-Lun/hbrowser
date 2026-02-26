"""
Batch size x Learning rate 超參數搜尋。

以縮短的訓練流程測試不同組合，
找出最佳配置後再套用至 train.py 做完整訓練。

搜尋策略：
  - batch_sizes: [32, 64, 128, 256]
  - lr_scales: [0.5, 1.0, 2.0]（相對於 linear scaling rule 的倍率）
  - linear scaling rule: lr = base_lr * (batch_size / 32)

共 4 x 3 = 12 組實驗，使用相同的 train/val split 確保公平比較。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.search_batch_lr
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE,
    CLASS_NAMES,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    MIN_DELTA,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEARCH_PATIENCE,
    SEARCH_PHASE1_EPOCHS,
    SEARCH_PHASE2_EPOCHS,
    SEED,
    WEIGHT_DECAY,
    build_model,
    evaluate,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    group_stratified_split,
    load_samples,
    make_dataloader,
    train_one_epoch,
)
from .common.data import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Search grid
BATCH_SIZES = [32, 64, 128, 256]
LR_SCALES = [0.5, 1.0, 2.0]

# Base LRs from constants (single source of truth with train.py)
BASE_LR_HEAD = LR_HEAD
BASE_LR_FEATURES = LR_FEATURES
BASE_LR_CLASSIFIER = LR_CLASSIFIER


def run_experiment(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    lr_head: float,
    lr_features: float,
    lr_classifier: float,
    backbone: str,
) -> dict[str, Any]:
    """Run one training experiment, return results dict."""
    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    val_ds = PonyChartDataset(val_samples, get_transforms(is_train=False))
    train_loader = make_dataloader(
        train_ds, batch_size, shuffle=True,
        num_workers=num_workers, device=device,
    )
    val_loader = make_dataloader(
        val_ds, batch_size, shuffle=False,
        num_workers=num_workers, device=device,
    )

    model = build_model(backbone=backbone, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: Head only
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=lr_head, weight_decay=WEIGHT_DECAY
    )
    for _epoch in range(1, SEARCH_PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Phase 2: Full fine-tuning
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_features},
            {"params": model.classifier.parameters(), "lr": lr_classifier},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR,
    )

    best_f1 = 0.0
    best_per_class: list[float] = []
    patience_counter = 0
    stopped_epoch = SEARCH_PHASE2_EPOCHS

    for epoch in range(1, SEARCH_PHASE2_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = val_result["macro_f1"]
        scheduler.step(val_f1)

        if val_f1 > best_f1 + MIN_DELTA:
            best_f1 = val_f1
            best_per_class = list(val_result["per_class_f1"])
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= SEARCH_PATIENCE:
            stopped_epoch = epoch
            break

    return {
        "best_f1": best_f1,
        "per_class_f1": best_per_class,
        "stopped_epoch": stopped_epoch,
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load data (same split for all experiments)
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        return
    train_idx, val_idx = group_stratified_split(
        samples, test_size=0.15, seed=SEED
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    total_combos = len(BATCH_SIZES) * len(LR_SCALES)
    logger.info("")
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER SEARCH: %d combinations", total_combos)
    logger.info("  Backbone:    %s", BACKBONE)
    logger.info("  Batch sizes: %s", BATCH_SIZES)
    logger.info("  LR scales:   %s (x linear scaling rule)", LR_SCALES)
    logger.info(
        "  Phase 1: %d epochs, Phase 2: %d epochs (patience=%d)",
        SEARCH_PHASE1_EPOCHS,
        SEARCH_PHASE2_EPOCHS,
        SEARCH_PATIENCE,
    )
    logger.info("=" * 70)
    logger.info("")

    results: list[dict[str, Any]] = []
    run_idx = 0

    for batch_size in BATCH_SIZES:
        for lr_scale in LR_SCALES:
            run_idx += 1
            linear_factor = batch_size / 32.0
            lr_head = BASE_LR_HEAD * linear_factor * lr_scale
            lr_features = BASE_LR_FEATURES * linear_factor * lr_scale
            lr_classifier = BASE_LR_CLASSIFIER * linear_factor * lr_scale

            label = (
                f"[{run_idx}/{total_combos}] "
                f"batch={batch_size}  lr_scale={lr_scale}  "
                f"(head={lr_head:.1e}  feat={lr_features:.1e}"
                f"  cls={lr_classifier:.1e})"
            )
            logger.info("--- %s ---", label)

            # Reset seeds for reproducibility
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            t0 = time.monotonic()
            result = run_experiment(
                train_samples,
                val_samples,
                device,
                num_workers,
                batch_size,
                lr_head,
                lr_features,
                lr_classifier,
                BACKBONE,
            )
            elapsed = time.monotonic() - t0

            result.update(
                {
                    "batch_size": batch_size,
                    "lr_scale": lr_scale,
                    "lr_head": lr_head,
                    "lr_features": lr_features,
                    "lr_classifier": lr_classifier,
                    "time_s": elapsed,
                }
            )
            results.append(result)

            logger.info(
                "    -> F1=%.4f  stopped_epoch=%d  time=%.1fs",
                result["best_f1"],
                result["stopped_epoch"],
                elapsed,
            )
            logger.info("")

    # ── Results table sorted by F1 ──
    results.sort(key=lambda r: r["best_f1"], reverse=True)

    logger.info("=" * 90)
    logger.info("RESULTS (sorted by best val Macro F1)")
    logger.info("=" * 90)
    logger.info(
        "  %-4s  %-6s  %-8s  %-10s  %-10s  %-10s  %-8s  %-6s  %-7s",
        "Rank",
        "Batch",
        "LR scale",
        "LR head",
        "LR feat",
        "LR cls",
        "Macro F1",
        "Epoch",
        "Time",
    )
    logger.info("  " + "-" * 85)
    for rank, r in enumerate(results, 1):
        logger.info(
            "  #%-3d  %-6d  %-8s  %-10.1e  %-10.1e  %-10.1e"
            "  %-8.4f  %-6d  %-7.1fs",
            rank,
            r["batch_size"],
            f"{r['lr_scale']:.1f}x",
            r["lr_head"],
            r["lr_features"],
            r["lr_classifier"],
            r["best_f1"],
            r["stopped_epoch"],
            r["time_s"],
        )

    # ── Per-class detail for top 3 ──
    logger.info("")
    logger.info("Per-class F1 for top 3:")
    for rank, r in enumerate(results[:3], 1):
        logger.info(
            "  #%d (batch=%d, scale=%.1fx, F1=%.4f):",
            rank,
            r["batch_size"],
            r["lr_scale"],
            r["best_f1"],
        )
        for i, name in enumerate(CLASS_NAMES):
            logger.info("    %-20s  %.4f", name, r["per_class_f1"][i])

    # ── Recommendation ──
    best = results[0]
    logger.info("")
    logger.info("=" * 90)
    logger.info("RECOMMENDATION")
    logger.info("=" * 90)
    logger.info("  Best config:")
    logger.info("    --batch-size %d", best["batch_size"])
    logger.info(
        "    Phase 1 lr: %.1e  (train.py default: 1e-3)", best["lr_head"]
    )
    logger.info(
        "    Phase 2 lr_features: %.1e  (train.py default: 3e-5)",
        best["lr_features"],
    )
    logger.info(
        "    Phase 2 lr_classifier: %.1e  (train.py default: 3e-4)",
        best["lr_classifier"],
    )
    logger.info("")

    # Compare with baseline (batch=32, scale=1.0)
    baseline = next(
        (
            r
            for r in results
            if r["batch_size"] == 32 and r["lr_scale"] == 1.0
        ),
        None,
    )
    if baseline and best is not baseline:
        diff = best["best_f1"] - baseline["best_f1"]
        speedup = (
            baseline["time_s"] / best["time_s"] if best["time_s"] > 0 else 0
        )
        logger.info(
            "  vs baseline (batch=32, 1.0x): F1 %+.4f, %.1fx speed",
            diff,
            speedup,
        )
    elif baseline:
        logger.info(
            "  Baseline (batch=32, 1.0x) is already the best config."
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
