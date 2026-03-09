"""
Learning rate 超參數搜尋。

固定 batch_size=64，測試不同 LR 倍率，
找出能加速收斂且不降低 F1 的最佳 LR。

搜尋策略：
  - batch_size: 64 (固定)
  - lr_scale: [0.5, 1.0, 1.5, 2.0, 3.0]

共 5 組實驗，使用相同的 train/val split 確保公平比較。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.search_batch_lr
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    INPUT_SIZE,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    MIN_DELTA,
    PRE_RESIZE,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEARCH_PATIENCE,
    SEARCH_PHASE1_EPOCHS,
    SEARCH_PHASE2_EPOCHS,
    SEED,
    VAL_SIZE,
    WEIGHT_DECAY,
    build_model,
    evaluate,
    get_device,
    get_performance_cpu_count,
    group_hash_split,
    load_samples,
    log_section,
    make_dataloader,
    measure_training_memory,
    train_one_epoch,
)
from .common.dataset import PonyChartDataset, compute_cache_budget, get_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Search grid – (batch_size, lr_scale) pairs
# 固定 batch_size=64，搜尋不同 LR 倍率
SEARCH_GRID: list[tuple[int, float]] = [
    (64, 0.5),  # 0.5x LR (more conservative)
    (64, 1.0),  # baseline
    (64, 1.5),  # 1.5x LR
    (64, 2.0),  # 2x LR
    (64, 3.0),  # 3x LR (aggressive)
]

# Base LRs from constants (single source of truth with train.py)
BASE_LR_HEAD = LR_HEAD
BASE_LR_FEATURES = LR_FEATURES
BASE_LR_CLASSIFIER = LR_CLASSIFIER


def run_experiment(
    train_ds: PonyChartDataset,
    val_ds: PonyChartDataset,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    lr_head: float,
    lr_features: float,
    lr_classifier: float,
    backbone: str,
) -> dict[str, Any]:
    """Run one training experiment, return results dict."""
    train_loader = make_dataloader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
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
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
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
    logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

    # Load data (same split for all experiments)
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        return
    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info(
        "Train: %s  Val: %s", f"{len(train_samples):,}", f"{len(val_samples):,}"
    )

    total_combos = len(SEARCH_GRID)
    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "HYPERPARAMETER SEARCH: %d combinations",
        total_combos,
    )
    logger.info("  Backbone:    %s", BACKBONE)
    logger.info("  Grid:        %s", SEARCH_GRID)
    logger.info(
        "  Phase 1: %d epochs, Phase 2: %d epochs (patience=%d)",
        SEARCH_PHASE1_EPOCHS,
        SEARCH_PHASE2_EPOCHS,
        SEARCH_PATIENCE,
    )
    logger.info("=" * 70)
    logger.info("")

    # Build datasets once (shared across all experiments)
    max_batch = max(bs for bs, _ in SEARCH_GRID)
    training_reserve = measure_training_memory(
        BACKBONE,
        max_batch,
        INPUT_SIZE,
        device,
    )
    total_budget = compute_cache_budget(
        PRE_RESIZE,
        n_datasets=2,
        training_reserve=training_reserve,
    )
    n_total = len(train_samples) + len(val_samples)
    train_budget = int(total_budget * len(train_samples) / n_total)
    val_budget = total_budget - train_budget

    train_ds = PonyChartDataset(
        train_samples,
        get_transforms(is_train=True),
        max_cached=train_budget,
    )
    val_ds = PonyChartDataset(
        val_samples,
        get_transforms(is_train=False),
        max_cached=val_budget,
    )

    # Run experiments sequentially on device
    results: list[dict[str, Any]] = []
    for run_idx, (batch_size, lr_scale) in enumerate(SEARCH_GRID, 1):
        linear_factor = batch_size / BATCH_SIZE
        lr_head = BASE_LR_HEAD * linear_factor * lr_scale
        lr_features = BASE_LR_FEATURES * linear_factor * lr_scale
        lr_classifier = BASE_LR_CLASSIFIER * linear_factor * lr_scale

        logger.info(
            "  [%d/%d] batch=%d  lr_scale=%s  " "(head=%.1e  feat=%.1e  cls=%.1e)",
            run_idx,
            total_combos,
            batch_size,
            lr_scale,
            lr_head,
            lr_features,
            lr_classifier,
        )

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        t0 = time.monotonic()
        result = run_experiment(
            train_ds,
            val_ds,
            device,
            num_workers,
            batch_size,
            lr_head,
            lr_features,
            lr_classifier,
            BACKBONE,
        )
        elapsed = time.monotonic() - t0

        # Flush MPS allocator cache so experiment memory is returned to OS
        # before the next experiment allocates a new model.
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        result.update(
            {
                "batch_size": batch_size,
                "lr_scale": lr_scale,
                "lr_head": lr_head,
                "lr_features": lr_features,
                "lr_classifier": lr_classifier,
                "time_s": elapsed,
                "run_idx": run_idx,
            }
        )
        results.append(result)
        logger.info(
            "    -> F1=%.4f  stopped_epoch=%d  time=%.1fs",
            result["best_f1"],
            result["stopped_epoch"],
            result["time_s"],
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
            "  #%-3d  %-6d  %-8s  %-10.1e  %-10.1e  %-10.1e" "  %-8.4f  %-6d  %-7.1fs",
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

    # ── Per-class detail for all ──
    logger.info("")
    logger.info("Per-class F1 for all configs:")
    for rank, r in enumerate(results, 1):
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
    log_section(logger, "RECOMMENDATION")
    logger.info("  Best config:")
    logger.info("    --batch-size %d", best["batch_size"])
    logger.info(
        "    Phase 1 lr: %.1e  (train.py default: %.1e)", best["lr_head"], LR_HEAD
    )
    logger.info(
        "    Phase 2 lr_features: %.1e  (train.py default: %.1e)",
        best["lr_features"],
        LR_FEATURES,
    )
    logger.info(
        "    Phase 2 lr_classifier: %.1e  (train.py default: %.1e)",
        best["lr_classifier"],
        LR_CLASSIFIER,
    )
    logger.info("")

    # Compare with baseline (batch=BATCH_SIZE, scale=1.0)
    baseline = next(
        (r for r in results if r["batch_size"] == BATCH_SIZE and r["lr_scale"] == 1.0),
        None,
    )
    if baseline and best is not baseline:
        diff = best["best_f1"] - baseline["best_f1"]
        speedup = baseline["time_s"] / best["time_s"] if best["time_s"] > 0 else 0
        logger.info(
            "  vs baseline (batch=%d, 1.0x): F1 %+.4f, %.1fx speed",
            BATCH_SIZE,
            diff,
            speedup,
        )
    elif baseline:
        logger.info(
            "  Baseline (batch=%d, 1.0x) is already the best config.", BATCH_SIZE
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
