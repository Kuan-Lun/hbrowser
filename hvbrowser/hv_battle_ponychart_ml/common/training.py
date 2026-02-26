"""Training primitives and high-level training pipeline."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

from .constants import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    LR_CLASSIFIER,
    LR_FEATURES,
    LR_HEAD,
    MIN_DELTA,
    NUM_CLASSES,
    PATIENCE,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from .data import PonyChartDataset, get_transforms, make_dataloader
from .model import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()  # type: ignore[untyped-decorator]
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate model, return dict with loss, F1, precision, recall."""
    model.eval()
    total_loss = 0.0
    all_probs: list[np.ndarray[Any, Any]] = []
    all_targets: list[np.ndarray[Any, Any]] = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    all_probs_arr = np.concatenate(all_probs)
    all_targets_arr = np.concatenate(all_targets)

    if thresholds is None:
        thresholds = [0.5] * NUM_CLASSES

    preds = np.zeros_like(all_probs_arr, dtype=int)
    for i in range(NUM_CLASSES):
        preds[:, i] = (all_probs_arr[:, i] >= thresholds[i]).astype(int)

    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(all_targets_arr[:, i], preds[:, i], zero_division=0.0)
        prec = precision_score(all_targets_arr[:, i], preds[:, i], zero_division=0.0)
        rec = recall_score(all_targets_arr[:, i], preds[:, i], zero_division=0.0)
        per_class_f1.append(float(f1))
        per_class_precision.append(float(prec))
        per_class_recall.append(float(rec))

    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": float(np.mean(per_class_f1)),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


@torch.no_grad()  # type: ignore[untyped-decorator]
def optimize_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[float]:
    """Find optimal per-class thresholds by grid search."""
    model.eval()
    all_probs: list[np.ndarray[Any, np.dtype[Any]]] = []
    all_targets: list[np.ndarray[Any, np.dtype[Any]]] = []
    for images, targets in loader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

    all_probs_arr = np.concatenate(all_probs)
    all_targets_arr = np.concatenate(all_targets)

    thresholds: list[float] = []
    for i in range(NUM_CLASSES):
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.20, 0.80, 0.01):
            preds = (all_probs_arr[:, i] >= thr).astype(int)
            f1 = f1_score(all_targets_arr[:, i], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds.append(round(best_thr, 4))
    return thresholds


# ---------------------------------------------------------------------------
# High-level training pipeline
# ---------------------------------------------------------------------------
def train_model(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
    experiment_name: str,
    *,
    backbone: str = BACKBONE,
    train_transform: transforms.Compose | None = None,
    batch_size: int = BATCH_SIZE,
    phase1_epochs: int = PHASE1_EPOCHS,
    phase2_epochs: int = PHASE2_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = False,
) -> tuple[nn.Module, list[float]]:
    """Train a model end-to-end.

    Returns (best_model, optimized_thresholds).
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info(
        "  Train: %d samples, Val: %d samples",
        len(train_samples),
        len(val_samples),
    )
    logger.info("  Backbone: %s", backbone)
    logger.info("=" * 60)

    if train_transform is None:
        train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    train_ds = PonyChartDataset(train_samples, train_transform)
    val_ds = PonyChartDataset(val_samples, val_transform)
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
    logger.info("--- Phase 1: Head-only (%d epochs) ---", phase1_epochs)
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, phase1_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f",
            epoch,
            phase1_epochs,
            train_loss,
            val_result["loss"],
            val_result["macro_f1"],
        )

    # Phase 2: Full fine-tuning
    logger.info("--- Phase 2: Full fine-tuning (%d epochs) ---", phase2_epochs)
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": LR_FEATURES},
            {"params": model.classifier.parameters(), "lr": LR_CLASSIFIER},
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
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, phase2_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = val_result["macro_f1"]
        scheduler.step(val_f1)

        marker = ""
        if val_f1 > best_f1 + MIN_DELTA:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        if verbose:
            per_class_str = "  ".join(
                f"{name}={f1:.4f}"
                for name, f1 in zip(CLASS_NAMES, val_result["per_class_f1"])
            )
            logger.info(
                "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f"
                "  val_F1=%.4f%s\n    %s",
                epoch,
                phase2_epochs,
                train_loss,
                val_result["loss"],
                val_f1,
                marker,
                per_class_str,
            )
        else:
            logger.info(
                "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f%s",
                epoch,
                phase2_epochs,
                train_loss,
                val_result["loss"],
                val_f1,
                marker,
            )
        if patience_counter >= patience:
            logger.info("  Early stopping (no improvement for %d epochs)", patience)
            break

    model.load_state_dict(best_state)

    # Log best model performance
    final_result = evaluate(model, val_loader, criterion, device)
    logger.info("Best val F1: %.4f", final_result["macro_f1"])
    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %s: F1=%.4f", name, final_result["per_class_f1"][i])

    # Optimize thresholds on validation set
    thresholds = optimize_thresholds(model, val_loader, device)
    logger.info("Optimized thresholds: %s", dict(zip(CLASS_NAMES, thresholds)))

    return model, thresholds
