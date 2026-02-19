"""
PonyChart 多標籤分類訓練腳本。

使用 MobileNetV3-Small + transfer learning 訓練，匯出 ONNX 供推論。

安裝訓練依賴：
  uv pip install torch torchvision scikit-learn

使用方式：
  python -m hvbrowser.hv_battle_ponychart_ml.train
  # 或
  python hvbrowser/hv_battle_ponychart_ml/train.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # type: ignore[import-untyped]
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
RAWIMAGE_DIR = _SCRIPT_DIR / "rawimage"
LABELS_FILE = _SCRIPT_DIR / "labels.json"
OUTPUT_ONNX = _SCRIPT_DIR / "model.onnx"
OUTPUT_THRESHOLDS = _SCRIPT_DIR / "thresholds.json"

NUM_CLASSES = 6
CLASS_NAMES = [
    "Twilight Sparkle",
    "Rarity",
    "Fluttershy",
    "Rainbow Dash",
    "Pinkie Pie",
    "Applejack",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _get_performance_cpu_count() -> int:
    """回傳效能核心數（macOS Apple Silicon），其餘平台回傳總核心數。"""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return max((os.cpu_count() or 1) - 2, 1)


def load_samples() -> list[tuple[str, list[int]]]:
    """讀取 labels.json，回傳 [(image_path, [1-indexed labels]), ...]。"""
    with open(LABELS_FILE, encoding="utf-8") as f:
        raw: dict[str, list[int]] = json.load(f)
    samples = []
    for key, label_list in raw.items():
        filename = key.split("/")[-1]
        filepath = str(RAWIMAGE_DIR / filename)
        if os.path.isfile(filepath):
            samples.append((filepath, label_list))
    logger.info("Loaded %d samples (of %d labels.json entries)", len(samples), len(raw))
    return samples


def labels_to_binary(label_list: list[int]) -> torch.Tensor:
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in label_list:
        vec[lbl - 1] = 1.0  # 1-indexed -> 0-indexed
    return vec


def group_stratified_split(
    samples: list[tuple[str, list[int]]],
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """依時間戳分組的分層切分，避免同一組的原圖和裁切圖分到不同集合。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path).replace(".png", "").replace(".jpg", "")
        # 基底：pony_chart_YYYYMMDD_HHMMSS
        parts = fname.split("_")
        base = "_".join(parts[:4])
        groups[base].append(idx)

    group_keys = list(groups.keys())
    # 使用各組中出現頻率最高的單一標籤做分層（避免 powerset 組合太稀疏）
    group_primary_label = []
    for gk in group_keys:
        label_counts: dict[int, int] = defaultdict(int)
        for idx in groups[gk]:
            for lbl in samples[idx][1]:
                label_counts[lbl] += 1
        primary = max(label_counts, key=lambda k: label_counts[k])
        group_primary_label.append(str(primary))

    train_gk, val_gk = train_test_split(
        group_keys,
        test_size=test_size,
        random_state=seed,
        stratify=group_primary_label,
    )
    train_idx = [idx for gk in train_gk for idx in groups[gk]]
    val_idx = [idx for gk in val_gk for idx in groups[gk]]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PonyChartDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        samples: list[tuple[str, list[int]]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label_list = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = labels_to_binary(label_list)
        return image, target  # type: ignore[return-value]


def get_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.BOX),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=90, translate=(0.05, 0.05), scale=(0.9, 1.1)
                ),
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BOX),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features: int = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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
    return total_loss / len(loader.dataset)  # type: ignore[arg-type, no-any-return, operator]


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[float]]:
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
    preds = (all_probs_arr >= 0.5).astype(int)

    per_class_f1 = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(all_targets_arr[:, i], preds[:, i], zero_division=0.0)
        per_class_f1.append(float(f1))

    macro_f1 = float(np.mean(per_class_f1))
    avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type, operator]
    return avg_loss, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------
@torch.no_grad()
def optimize_thresholds(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> dict[str, float]:
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

    thresholds: dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.20, 0.80, 0.01):
            preds = (all_probs_arr[:, i] >= thr).astype(int)
            f1 = f1_score(all_targets_arr[:, i], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds[name] = round(best_thr, 4)
        logger.info("  %s: threshold=%.4f  F1=%.4f", name, best_thr, best_f1)
    return thresholds


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model: nn.Module, output_path: Path) -> None:
    import warnings

    import onnx

    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, 224, 224)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing annotation for parameter")
        torch.onnx.export(
            model_cpu,
            (dummy,),
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
        )
    # 新版 PyTorch 可能把權重存成 external data，需合併成單一檔案
    external_data = Path(str(output_path) + ".data")
    if external_data.exists():
        onnx_model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(
            onnx_model,
            str(output_path),
            save_as_external_data=False,
        )
        external_data.unlink()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported: %s (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    num_workers = _get_performance_cpu_count()
    logger.info("Device: %s  DataLoader workers: %d", device, num_workers)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        sys.exit(1)

    train_idx, val_idx = group_stratified_split(samples, test_size=0.15, seed=args.seed)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    val_ds = PonyChartDataset(val_samples, get_transforms(is_train=False))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Phase 1: Head only ----
    phase1_epochs = 10
    logger.info("=== Phase 1: Head-only training (%d epochs) ===", phase1_epochs)
    for param in model.features.parameters():  # type: ignore[union-attr]
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4  # type: ignore[union-attr]
    )

    for epoch in range(1, phase1_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _ = validate(model, val_loader, criterion, device)
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f",
            epoch,
            phase1_epochs,
            train_loss,
            val_loss,
            val_f1,
        )

    # ---- Phase 2: Full fine-tuning ----
    logger.info("=== Phase 2: Full fine-tuning (%d epochs) ===", args.epochs)
    for param in model.features.parameters():  # type: ignore[union-attr]
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 3e-5},  # type: ignore[union-attr]
            {"params": model.classifier.parameters(), "lr": 3e-4},  # type: ignore[union-attr]
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, per_class = validate(model, val_loader, criterion, device)
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
            f"{name}={f1:.4f}" for name, f1 in zip(CLASS_NAMES, per_class)
        )
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f%s\n    %s",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_f1,
            marker,
            per_class_str,
        )
        if patience_counter >= patience:
            logger.info("  Early stopping (no improvement for %d epochs)", patience)
            break

    # Restore best model
    model.load_state_dict(best_state)
    _, final_f1, final_per_class = validate(model, val_loader, criterion, device)
    logger.info("Best val F1: %.4f", final_f1)
    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %s: F1=%.4f", name, final_per_class[i])

    # Optimize thresholds
    logger.info("Optimizing per-class thresholds...")
    thresholds = optimize_thresholds(model, val_loader, device)
    with open(OUTPUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    logger.info("Thresholds saved: %s", OUTPUT_THRESHOLDS)

    # Export ONNX
    logger.info("Exporting ONNX...")
    export_onnx(model, OUTPUT_ONNX)

    logger.info("Done!")


if __name__ == "__main__":
    main()
