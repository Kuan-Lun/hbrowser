"""Profile data loading vs GPU compute to identify the training bottleneck.

Usage:
  uv run python -m hvbrowser.hv_battle_ponychart_ml.profile_dataloader
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE,
    BATCH_SIZE,
    SEED,
    VAL_SIZE,
    WEIGHT_DECAY,
    build_model,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    group_hash_split,
    load_samples,
    make_dataloader,
)
from .common.dataset import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NUM_EPOCHS = 3


def profile_dataloader_only(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs: int,
) -> dict[str, float]:
    """Measure pure data loading time (iterate loader, move to device, no compute)."""
    times_to_device: list[float] = []
    n_batches = 0

    for _epoch in range(n_epochs):
        for images, targets in loader:
            t0 = time.perf_counter()
            # data is already yielded, measure to-device transfer
            images.to(device)
            targets.to(device)
            t1 = time.perf_counter()
            times_to_device.append(t1 - t0)
            n_batches += 1

    total_batches = n_batches
    avg_to_device = sum(times_to_device) / total_batches
    return {
        "total_batches": total_batches,
        "avg_to_device_ms": avg_to_device * 1000,
    }


def profile_training(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int,
) -> dict[str, float]:
    """Profile one epoch with detailed per-step breakdown."""
    times_data: list[float] = []
    times_forward: list[float] = []
    times_backward: list[float] = []
    times_step: list[float] = []

    for _epoch in range(n_epochs):
        model.train()
        t_data_start = time.perf_counter()
        for images, targets in loader:
            t_data_end = time.perf_counter()
            times_data.append(t_data_end - t_data_start)

            # Forward
            images, targets = images.to(device), targets.to(device)
            t_fwd_start = time.perf_counter()
            logits = model(images)
            loss = criterion(logits, targets)
            t_fwd_end = time.perf_counter()
            times_forward.append(t_fwd_end - t_fwd_start)

            # Backward
            optimizer.zero_grad()
            t_bwd_start = time.perf_counter()
            loss.backward()
            t_bwd_end = time.perf_counter()
            times_backward.append(t_bwd_end - t_bwd_start)

            # Optimizer step
            t_step_start = time.perf_counter()
            optimizer.step()
            t_step_end = time.perf_counter()
            times_step.append(t_step_end - t_step_start)

            t_data_start = time.perf_counter()

    n = len(times_data)
    return {
        "n_steps": n,
        "avg_data_ms": sum(times_data) / n * 1000,
        "avg_forward_ms": sum(times_forward) / n * 1000,
        "avg_backward_ms": sum(times_backward) / n * 1000,
        "avg_optim_ms": sum(times_step) / n * 1000,
        "total_data_s": sum(times_data),
        "total_forward_s": sum(times_forward),
        "total_backward_s": sum(times_backward),
        "total_optim_s": sum(times_step),
        "p50_data_ms": sorted(times_data)[n // 2] * 1000,
        "p95_data_ms": sorted(times_data)[int(n * 0.95)] * 1000,
        "p50_forward_ms": sorted(times_forward)[n // 2] * 1000,
        "p95_forward_ms": sorted(times_forward)[int(n * 0.95)] * 1000,
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d  Batch: %d", device, num_workers, BATCH_SIZE)

    samples = load_samples()
    if not samples:
        logger.error("No samples found.")
        return
    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)
    train_samples = [samples[i] for i in train_idx]
    logger.info("Train samples: %d", len(train_samples))

    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    train_loader = make_dataloader(
        train_ds,
        BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )

    model = build_model(backbone=BACKBONE, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Full fine-tuning (Phase 2 style, which is the long phase)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY
    )

    # Warmup: 1 epoch to stabilize caches, JIT, etc.
    logger.info("Warmup (1 epoch)...")
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        loss = criterion(model(images), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Profile
    logger.info("Profiling %d epochs...", NUM_EPOCHS)
    result = profile_training(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        NUM_EPOCHS,
    )

    total_step = (
        result["total_data_s"]
        + result["total_forward_s"]
        + result["total_backward_s"]
        + result["total_optim_s"]
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "PROFILING RESULTS (%d steps over %d epochs)", result["n_steps"], NUM_EPOCHS
    )
    logger.info("=" * 70)
    logger.info(
        "  %-18s  %8s  %8s  %8s  %8s",
        "Phase",
        "Avg (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "Total %",
    )
    logger.info("  " + "-" * 60)
    logger.info(
        "  %-18s  %8.1f  %8.1f  %8.1f  %7.1f%%",
        "Data loading",
        result["avg_data_ms"],
        result["p50_data_ms"],
        result["p95_data_ms"],
        result["total_data_s"] / total_step * 100,
    )
    logger.info(
        "  %-18s  %8.1f  %8.1f  %8.1f  %7.1f%%",
        "Forward pass",
        result["avg_forward_ms"],
        result["p50_forward_ms"],
        result["p95_forward_ms"],
        result["total_forward_s"] / total_step * 100,
    )
    logger.info(
        "  %-18s  %8.1f  %8s  %8s  %7.1f%%",
        "Backward pass",
        result["avg_backward_ms"],
        "-",
        "-",
        result["total_backward_s"] / total_step * 100,
    )
    logger.info(
        "  %-18s  %8.1f  %8s  %8s  %7.1f%%",
        "Optimizer step",
        result["avg_optim_ms"],
        "-",
        "-",
        result["total_optim_s"] / total_step * 100,
    )
    logger.info("  " + "-" * 60)
    logger.info(
        "  %-18s  %8.1f  %8s  %8s  %7.1f%%",
        "TOTAL per step",
        (
            result["avg_data_ms"]
            + result["avg_forward_ms"]
            + result["avg_backward_ms"]
            + result["avg_optim_ms"]
        ),
        "-",
        "-",
        100.0,
    )
    logger.info("")
    logger.info("  Total wall time: %.1f s", total_step)
    logger.info(
        "  Throughput: %.1f samples/sec",
        result["n_steps"] * BATCH_SIZE / total_step,
    )
    logger.info("")

    # Verdict
    data_pct = result["total_data_s"] / total_step * 100
    compute_pct = (
        (
            result["total_forward_s"]
            + result["total_backward_s"]
            + result["total_optim_s"]
        )
        / total_step
        * 100
    )
    if data_pct > 50:
        logger.info(
            "VERDICT: Data loading is the bottleneck (%.0f%% of step time).", data_pct
        )
        logger.info(
            "  Consider: faster transforms, more workers, or GPU-side augmentation."
        )
    elif data_pct > 30:
        logger.info(
            "VERDICT: Data loading is significant (%.0f%%) but not dominant.",
            data_pct,
        )
        logger.info("  Both data and compute contribute to training time.")
    else:
        logger.info(
            "VERDICT: GPU compute is the bottleneck (%.0f%% of step time).", compute_pct
        )
        logger.info("  Data loading (%.0f%%) is not the issue.", data_pct)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
