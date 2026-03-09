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
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    INPUT_SIZE,
    LABEL_SMOOTHING,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    NUM_CLASSES,
    OUTPUT_CHECKPOINT,
    OUTPUT_ONNX,
    OUTPUT_THRESHOLDS,
    PRE_RESIZE,
    RETRAIN_NEW_DATA_RATIO,
    SEED,
    VAL_SIZE,
    WEIGHT_DECAY,
    balance_crop_samples,
    compute_class_rates,
    export_onnx,
    get_base_timestamp,
    get_device,
    get_performance_cpu_count,
    group_hash_split,
    is_original,
    load_samples,
    separate_orig_crop,
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
    orig_samples, crop_samples = separate_orig_crop(samples)
    orig_rates = compute_class_rates(orig_samples)
    rng = np.random.RandomState(SEED)
    balanced_crops = balance_crop_samples(crop_samples, orig_rates, rng)
    samples = orig_samples + balanced_crops
    logger.info(
        "Orig: %s  Crop: %s -> Balanced: %s  Total: %s",
        f"{len(orig_samples):,}",
        f"{len(crop_samples):,}",
        f"{len(balanced_crops):,}",
        f"{len(samples):,}",
    )

    # Auto-detect checkpoint for resume training
    resume_from = None
    if not args.from_scratch and OUTPUT_CHECKPOINT.exists():
        ckpt = torch.load(OUTPUT_CHECKPOINT, map_location=device, weights_only=True)

        # Check architecture compatibility
        # (missing keys = legacy checkpoint, treat as incompatible)
        arch_keys = {
            "backbone": BACKBONE,
            "input_size": INPUT_SIZE,
            "pre_resize": PRE_RESIZE,
            "num_classes": NUM_CLASSES,
        }
        required_keys = list(arch_keys) + [
            "labels_at_full_train",
            "val_size",
            "n_orig",
            "created_at",
        ]
        missing = [k for k in required_keys if k not in ckpt]
        mismatches = {
            k: (ckpt[k], v) for k, v in arch_keys.items() if k in ckpt and ckpt[k] != v
        }
        if missing:
            logger.warning(
                "Checkpoint missing keys: %s. " "自動切換為 from-scratch 訓練。",
                ", ".join(missing),
            )
        elif mismatches:
            for k, (old, new) in mismatches.items():
                logger.warning(
                    "Architecture mismatch: %s: %s -> %s",
                    k,
                    old,
                    new,
                )
            logger.warning("自動切換為 from-scratch 訓練。")
        else:
            labels_full = ckpt["labels_at_full_train"]
            n_orig_full_train = sum(
                1 for k in labels_full if is_original(k.split("/")[-1])
            )
            n_orig_current = len(separate_orig_crop(load_samples())[0])
            new_data_ratio = (n_orig_current - n_orig_full_train) / n_orig_full_train
            logger.info(
                "Checkpoint: %s orig at last full train, "
                "%s orig at last save (created_at=%s), current: %s orig, "
                "cumulative new_data_ratio=%.1f%%",
                f"{n_orig_full_train:,}",
                f"{ckpt['n_orig']:,}",
                ckpt["created_at"],
                f"{n_orig_current:,}",
                new_data_ratio * 100,
            )
            should_retrain = False
            if new_data_ratio > RETRAIN_NEW_DATA_RATIO:
                logger.warning(
                    "cumulative new_data_ratio (%.1f%%) 超過閾值 "
                    "RETRAIN_NEW_DATA_RATIO (%.1f%%)，"
                    "自動切換為 from-scratch 訓練。",
                    new_data_ratio * 100,
                    RETRAIN_NEW_DATA_RATIO * 100,
                )
                should_retrain = True
            if VAL_SIZE > ckpt["val_size"]:
                logger.warning(
                    "VAL_SIZE increased (%.2f -> %.2f), "
                    "hash split leakage risk. "
                    "自動切換為 from-scratch 訓練。",
                    ckpt["val_size"],
                    VAL_SIZE,
                )
                should_retrain = True
            if not should_retrain:
                resume_from = OUTPUT_CHECKPOINT
                logger.info(
                    "Found checkpoint: %s (use --from-scratch to ignore)",
                    resume_from,
                )

    train_idx, val_idx = group_hash_split(samples, test_size=VAL_SIZE)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [
        s for s in (samples[i] for i in val_idx) if is_original(os.path.basename(s[0]))
    ]
    logger.info(
        "Train: %s  Val: %s (orig only)",
        f"{len(train_samples):,}",
        f"{len(val_samples):,}",
    )

    # Train
    result = train_model(
        train_samples,
        val_samples,
        device,
        num_workers,
        "PonyChart Training",
        verbose=True,
        resume_from=resume_from,
    )
    model, thresholds, best_f1 = result.model, result.thresholds, result.best_f1

    # Guard: skip overwrite if resume training produced worse val_F1
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        prev_f1 = ckpt.get("val_f1")
        if prev_f1 is not None and best_f1 < prev_f1:
            logger.warning(
                "Resume training val_F1 (%.4f) < previous val_F1 (%.4f). "
                "Skipping checkpoint/ONNX/thresholds overwrite.",
                best_f1,
                prev_f1,
            )
            logger.info("Done! (no files updated)")
            return

    # Save thresholds
    thresholds_dict = dict(zip(CLASS_NAMES, thresholds))
    for name, thr in thresholds_dict.items():
        logger.info("  %s: threshold=%.4f", name, thr)
    with open(OUTPUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, ensure_ascii=False, indent=2)
    logger.info("Thresholds saved: %s", OUTPUT_THRESHOLDS)

    # Save checkpoint with metadata for future resume training
    orig_timestamps = [get_base_timestamp(os.path.basename(p)) for p, _ in orig_samples]
    latest_timestamp = max(orig_timestamps)
    n_orig_current = len(orig_samples)
    n_crop_current = len(crop_samples)

    # Track labels snapshot at full training and last save for drift detection.
    # From-scratch: snapshot current labels. Resume: carry forward from checkpoint.
    # Build labels snapshot from loaded samples (filtered by file existence)
    # to stay consistent with n_orig / n_crop counts.
    current_labels = {
        f"rawimage/{os.path.basename(p)}": labels
        for p, labels in orig_samples + crop_samples
    }
    if resume_from is not None:
        labels_at_full_train = ckpt["labels_at_full_train"]
    else:
        labels_at_full_train = current_labels

    torch.save(
        {
            "state_dict": model.state_dict(),
            "val_f1": best_f1,
            "n_orig": n_orig_current,
            "n_crop": n_crop_current,
            "labels_at_full_train": labels_at_full_train,
            "labels_at_last_save": current_labels,
            "class_rates": orig_rates,
            "created_at": latest_timestamp,
            # Model architecture (mismatch -> from-scratch)
            "backbone": BACKBONE,
            "input_size": INPUT_SIZE,
            "pre_resize": PRE_RESIZE,
            "num_classes": NUM_CLASSES,
            # Split config (val_size increase -> from-scratch)
            "val_size": VAL_SIZE,
            # Training hyperparameters (informational)
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD,
            "lr_features": LR_FEATURES,
            "lr_classifier": LR_CLASSIFIER,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
        },
        OUTPUT_CHECKPOINT,
    )
    logger.info(
        "Checkpoint saved: %s (n_orig=%s, val_f1=%.4f, created_at=%s)",
        OUTPUT_CHECKPOINT,
        f"{n_orig_current:,}",
        best_f1,
        latest_timestamp,
    )

    # Export ONNX
    logger.info("Exporting ONNX...")
    export_onnx(model, OUTPUT_ONNX)

    logger.info("Done!")


if __name__ == "__main__":
    main()
