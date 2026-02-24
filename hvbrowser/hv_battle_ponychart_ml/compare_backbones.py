"""
比較不同 backbone 架構對 PonyChart 分類效果的影響。

測試 MobileNetV3-Small、MobileNetV3-Large、EfficientNet-B0 三種 backbone，
使用相同的 train/val/test split，比較 Macro F1、per-class F1、模型大小和訓練時間。

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_backbones
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BACKBONE_REGISTRY,
    CLASS_NAMES,
    evaluate,
    export_onnx,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    group_stratified_split,
    load_samples,
    make_dataloader,
    train_model,
)
from .common.data import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
BATCH_SIZE = 32
BACKBONES = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]


def get_onnx_size_mb(model: nn.Module) -> float:
    """Export model to a temp ONNX file and return its size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_onnx(model, tmp_path)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    finally:
        tmp_path.unlink(missing_ok=True)
        # Clean up potential external data file
        data_path = Path(str(tmp_path) + ".data")
        data_path.unlink(missing_ok=True)
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load data
    all_samples = load_samples()
    if not all_samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        return

    # Split: train+val (85%) / test (15%)
    train_val_idx, test_idx = group_stratified_split(
        all_samples, test_size=0.15, seed=SEED
    )
    train_val_samples = [all_samples[i] for i in train_val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    # From train_val, split train (85%) / val (15%)
    train_idx, val_idx = group_stratified_split(
        train_val_samples, test_size=0.15, seed=SEED
    )
    train_samples = [train_val_samples[i] for i in train_idx]
    val_samples = [train_val_samples[i] for i in val_idx]

    logger.info(
        "Train: %d  Val: %d  Test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    # Prepare test loader (shared)
    test_ds = PonyChartDataset(
        test_samples, get_transforms(is_train=False)
    )
    test_loader = make_dataloader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=num_workers, device=device,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Run experiments ──
    results: dict[str, dict[str, Any]] = {}

    for backbone_name in BACKBONES:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        config = BACKBONE_REGISTRY[backbone_name]
        logger.info("")
        logger.info("=" * 70)
        logger.info("BACKBONE: %s", config.description)
        logger.info("=" * 70)

        t0 = time.monotonic()
        model, thresholds = train_model(
            train_samples,
            val_samples,
            device,
            num_workers,
            backbone_name,
            backbone=backbone_name,
        )
        train_time = time.monotonic() - t0

        # Evaluate on test set
        test_result = evaluate(
            model, test_loader, criterion, device, thresholds
        )

        # Model stats
        param_count = count_parameters(model)
        onnx_size = get_onnx_size_mb(model)

        results[backbone_name] = {
            "test_result": test_result,
            "thresholds": thresholds,
            "param_count": param_count,
            "onnx_size_mb": onnx_size,
            "train_time_s": train_time,
            "description": config.description,
        }

        logger.info(
            ">> %s: test Macro F1=%.4f  params=%dK  ONNX=%.1fMB"
            "  time=%.0fs",
            backbone_name,
            test_result["macro_f1"],
            param_count // 1000,
            onnx_size,
            train_time,
        )

    # ── Comparison table ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("BACKBONE COMPARISON RESULTS")
    logger.info("=" * 90)

    logger.info("")
    logger.info(
        "  %-22s  %-10s  %-10s  %-10s  %-10s  %-10s",
        "Backbone",
        "Macro F1",
        "Params",
        "ONNX Size",
        "Time",
        "Thresholds",
    )
    logger.info("  " + "-" * 82)

    for name in BACKBONES:
        r = results[name]
        tr = r["test_result"]
        thr_str = " ".join(
            f"{t:.2f}" for t in r["thresholds"]
        )
        logger.info(
            "  %-22s  %-10.4f  %-10s  %-10s  %-10s  %s",
            name,
            tr["macro_f1"],
            f"{r['param_count'] / 1e6:.1f}M",
            f"{r['onnx_size_mb']:.1f}MB",
            f"{r['train_time_s']:.0f}s",
            thr_str,
        )

    # Per-class F1 table
    logger.info("")
    logger.info("Per-class F1:")
    header = f"  {'Class':<20s}"
    for name in BACKBONES:
        header += f"  {name:<22s}"
    logger.info(header)
    logger.info("  " + "-" * (20 + 24 * len(BACKBONES)))

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"  {cls_name:<20s}"
        for name in BACKBONES:
            f1 = results[name]["test_result"]["per_class_f1"][i]
            row += f"  {f1:<22.4f}"
        logger.info(row)

    # Per-class precision/recall
    logger.info("")
    logger.info("Per-class Precision / Recall:")
    for name in BACKBONES:
        logger.info("  %s:", name)
        tr = results[name]["test_result"]
        for i, cls_name in enumerate(CLASS_NAMES):
            logger.info(
                "    %-20s  P=%.4f  R=%.4f  F1=%.4f",
                cls_name,
                tr["per_class_precision"][i],
                tr["per_class_recall"][i],
                tr["per_class_f1"][i],
            )

    # ── Recommendation ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("RECOMMENDATION")
    logger.info("=" * 90)

    best_name = max(
        BACKBONES,
        key=lambda n: results[n]["test_result"]["macro_f1"],
    )
    best_r = results[best_name]
    best_f1 = best_r["test_result"]["macro_f1"]

    logger.info(
        "  Best backbone: %s (Macro F1=%.4f)", best_name, best_f1
    )
    logger.info("")

    # Compare each to best
    for name in BACKBONES:
        r = results[name]
        f1 = r["test_result"]["macro_f1"]
        diff = f1 - best_f1
        if name == best_name:
            logger.info("  * %s: F1=%.4f (BEST)", name, f1)
        else:
            logger.info(
                "    %s: F1=%.4f (%+.4f vs best)", name, f1, diff
            )

    # Efficiency analysis
    logger.info("")
    logger.info("  Efficiency analysis:")
    small_f1 = results["mobilenet_v3_small"]["test_result"]["macro_f1"]
    for name in BACKBONES:
        r = results[name]
        f1 = r["test_result"]["macro_f1"]
        gain = f1 - small_f1
        size_ratio = (
            r["onnx_size_mb"]
            / results["mobilenet_v3_small"]["onnx_size_mb"]
        )
        time_ratio = (
            r["train_time_s"]
            / results["mobilenet_v3_small"]["train_time_s"]
        )
        logger.info(
            "    %s: F1 %+.4f vs small, %.1fx size, %.1fx time",
            name,
            gain,
            size_ratio,
            time_ratio,
        )

    logger.info("")
    logger.info(
        "  To use the best backbone for production training:"
    )
    logger.info(
        "    uv run python -m hvbrowser.hv_battle_ponychart_ml.train"
        " --backbone %s",
        best_name,
    )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
