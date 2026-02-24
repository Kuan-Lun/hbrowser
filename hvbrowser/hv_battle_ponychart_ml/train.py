"""
PonyChart 多標籤分類訓練腳本。

使用 transfer learning 訓練，匯出 ONNX 供推論。

安裝訓練依賴：
  uv pip install torch torchvision scikit-learn

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.train
  uv run python -m hvbrowser.hv_battle_ponychart_ml.train \
    --backbone mobilenet_v3_large
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys

import numpy as np
import torch
import torch.nn as nn

from .common import (
    CLASS_NAMES,
    OUTPUT_ONNX,
    OUTPUT_THRESHOLDS,
    build_model,
    evaluate,
    export_onnx,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    group_stratified_split,
    load_samples,
    make_dataloader,
    optimize_thresholds,
    train_one_epoch,
)
from .common.data import PonyChartDataset

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train PonyChart classifier")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu / cuda / mps / auto",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_large",
        help="mobilenet_v3_small / mobilenet_v3_large / efficientnet_b0",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        sys.exit(1)

    train_idx, val_idx = group_stratified_split(
        samples, test_size=0.15, seed=args.seed
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    val_ds = PonyChartDataset(val_samples, get_transforms(is_train=False))
    train_loader = make_dataloader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=num_workers, device=device,
    )
    val_loader = make_dataloader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=num_workers, device=device,
    )

    # Model
    model = build_model(
        backbone=args.backbone, pretrained=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Phase 1: Head only ----
    phase1_epochs = 10
    logger.info("=== Phase 1: Head-only training (%d epochs) ===", phase1_epochs)
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(1, phase1_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_result = evaluate(model, val_loader, criterion, device)
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f",
            epoch,
            phase1_epochs,
            train_loss,
            val_result["loss"],
            val_result["macro_f1"],
        )

    # ---- Phase 2: Full fine-tuning ----
    logger.info("=== Phase 2: Full fine-tuning (%d epochs) ===", args.epochs)
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 3e-5},
            {"params": model.classifier.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )

    best_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = 12

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = val_result["macro_f1"]
        scheduler.step(val_f1)

        marker = ""
        if val_f1 > best_f1 + 0.0005:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        per_class_str = "  ".join(
            f"{name}={f1:.4f}"
            for name, f1 in zip(CLASS_NAMES, val_result["per_class_f1"])
        )
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f"
            "  val_F1=%.4f%s\n    %s",
            epoch,
            args.epochs,
            train_loss,
            val_result["loss"],
            val_f1,
            marker,
            per_class_str,
        )
        if patience_counter >= patience:
            logger.info(
                "  Early stopping (no improvement for %d epochs)", patience
            )
            break

    # Restore best model
    model.load_state_dict(best_state)
    final_result = evaluate(model, val_loader, criterion, device)
    logger.info("Best val F1: %.4f", final_result["macro_f1"])
    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %s: F1=%.4f", name, final_result["per_class_f1"][i])

    # Optimize thresholds
    logger.info("Optimizing per-class thresholds...")
    thresholds = optimize_thresholds(model, val_loader, device)
    thresholds_dict = dict(zip(CLASS_NAMES, thresholds))
    for name, thr in thresholds_dict.items():
        logger.info("  %s: threshold=%.4f", name, thr)
    with open(OUTPUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, ensure_ascii=False, indent=2)
    logger.info("Thresholds saved: %s", OUTPUT_THRESHOLDS)

    # Export ONNX
    logger.info("Exporting ONNX...")
    export_onnx(model, OUTPUT_ONNX)

    logger.info("Done!")


if __name__ == "__main__":
    main()
