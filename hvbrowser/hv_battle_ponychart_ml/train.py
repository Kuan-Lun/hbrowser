"""
PonyChart 多標籤分類訓練腳本。

使用 transfer learning 訓練，匯出 ONNX 供推論。

安裝訓練依賴：
  uv pip install torch torchvision scikit-learn

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.train

訓練超參數集中於 common/constants.py，
可透過分析工具（search_batch_lr, learning_curve 等）決定最佳設定後修改。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

from .common import (
    CLASS_NAMES,
    OUTPUT_CHECKPOINT,
    OUTPUT_ONNX,
    OUTPUT_THRESHOLDS,
    SEED,
    balance_crop_samples,
    compute_class_rates,
    export_onnx,
    get_device,
    get_performance_cpu_count,
    group_stratified_split,
    is_original,
    load_samples,
    train_model,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PonyChart multi-label training")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore existing checkpoint and train from ImageNet weights",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        sys.exit(1)

    # Separate originals and crops, then balance crops to match original distribution
    orig_samples = [s for s in samples if is_original(os.path.basename(s[0]))]
    crop_samples = [s for s in samples if not is_original(os.path.basename(s[0]))]
    orig_rates = compute_class_rates(orig_samples)
    rng = np.random.RandomState(SEED)
    balanced_crops = balance_crop_samples(crop_samples, orig_rates, rng)
    samples = orig_samples + balanced_crops
    logger.info(
        "Orig: %d  Crop: %d -> Balanced: %d  Total: %d",
        len(orig_samples),
        len(crop_samples),
        len(balanced_crops),
        len(samples),
    )

    train_idx, val_idx = group_stratified_split(samples, test_size=0.15, seed=SEED)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    # Auto-detect checkpoint for resume training
    resume_from = None
    if not args.from_scratch and OUTPUT_CHECKPOINT.exists():
        resume_from = OUTPUT_CHECKPOINT
        logger.info("Found checkpoint: %s (use --from-scratch to ignore)", resume_from)

    # Train
    model, thresholds = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "PonyChart Training",
        verbose=True,
        resume_from=resume_from,
    )

    # Save thresholds
    thresholds_dict = dict(zip(CLASS_NAMES, thresholds))
    for name, thr in thresholds_dict.items():
        logger.info("  %s: threshold=%.4f", name, thr)
    with open(OUTPUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, ensure_ascii=False, indent=2)
    logger.info("Thresholds saved: %s", OUTPUT_THRESHOLDS)

    # Save checkpoint for future resume training
    torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
    logger.info("Checkpoint saved: %s", OUTPUT_CHECKPOINT)

    # Export ONNX
    logger.info("Exporting ONNX...")
    export_onnx(model, OUTPUT_ONNX)

    logger.info("Done!")


if __name__ == "__main__":
    main()
