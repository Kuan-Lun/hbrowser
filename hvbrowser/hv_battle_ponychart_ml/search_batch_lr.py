"""
Batch size × Learning rate 超參數搜尋。

以縮短的訓練流程（Phase1=5, Phase2=15, patience=6）測試不同組合，
找出最佳配置後再套用至 train.py 做完整訓練。

搜尋策略：
  - batch_sizes: [32, 64, 128, 256]
  - lr_scales: [0.5, 1.0, 2.0]（相對於 linear scaling rule 的倍率）
  - linear scaling rule: lr = base_lr * (batch_size / 32)

共 4 × 3 = 12 組實驗，使用相同的 train/val split 確保公平比較。

使用方式：
  python -m hvbrowser.hv_battle_ponychart_ml.search_batch_lr
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (same as train.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
RAWIMAGE_DIR = _SCRIPT_DIR / "rawimage"
LABELS_FILE = _SCRIPT_DIR / "labels.json"

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

SEED = 42

# Reduced epochs for faster search
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
PATIENCE = 6

# Search grid
BATCH_SIZES = [32, 64, 128, 256]
LR_SCALES = [0.5, 1.0, 2.0]

# Base LRs (tuned for batch_size=32, from train.py)
BASE_LR_HEAD = 1e-3
BASE_LR_FEATURES = 3e-5
BASE_LR_CLASSIFIER = 3e-4


# ---------------------------------------------------------------------------
# Helpers (same as train.py)
# ---------------------------------------------------------------------------
def _get_performance_cpu_count() -> int:
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
    with open(LABELS_FILE, encoding="utf-8") as f:
        raw: dict[str, list[int]] = json.load(f)
    samples = []
    for key, label_list in raw.items():
        filename = key.split("/")[-1]
        filepath = str(RAWIMAGE_DIR / filename)
        if os.path.isfile(filepath):
            samples.append((filepath, label_list))
    return samples


def labels_to_binary(label_list: list[int]) -> torch.Tensor:
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in label_list:
        vec[lbl - 1] = 1.0
    return vec


def group_stratified_split(
    samples: list[tuple[str, list[int]]],
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path).replace(".png", "").replace(".jpg", "")
        parts = fname.split("_")
        base = "_".join(parts[:4])
        groups[base].append(idx)

    group_keys = list(groups.keys())
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
# Dataset & transforms (same as train.py)
# ---------------------------------------------------------------------------
class PonyChartDataset(Dataset):  # type: ignore[misc]
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
        return image, target


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
# Model (same as train.py)
# ---------------------------------------------------------------------------
def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features: int = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


# ---------------------------------------------------------------------------
# Training & evaluation (same as train.py)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
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
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
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
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, macro_f1, per_class_f1


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_experiment(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    lr_head: float,
    lr_features: float,
    lr_classifier: float,
) -> dict[str, Any]:
    """Run one training experiment, return results dict."""
    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    val_ds = PonyChartDataset(val_samples, get_transforms(is_train=False))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: Head only
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=lr_head, weight_decay=1e-4
    )
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Phase 2: Full fine-tuning
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_features},
            {"params": model.classifier.parameters(), "lr": lr_classifier},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
    )

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_f1, per_class_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        if val_f1 > best_f1 + 0.0005:
            best_f1 = val_f1
            best_per_class = list(per_class_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return {
        "best_f1": best_f1,
        "per_class_f1": best_per_class,
        "stopped_epoch": epoch,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    num_workers = _get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load data (same split for all experiments)
    samples = load_samples()
    if not samples:
        logger.error("No samples found. Check rawimage/ and labels.json.")
        return
    train_idx, val_idx = group_stratified_split(samples, test_size=0.15, seed=SEED)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_samples), len(val_samples))

    total_combos = len(BATCH_SIZES) * len(LR_SCALES)
    logger.info("")
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER SEARCH: %d combinations", total_combos)
    logger.info("  Batch sizes: %s", BATCH_SIZES)
    logger.info("  LR scales:   %s (× linear scaling rule)", LR_SCALES)
    logger.info("  Phase 1: %d epochs, Phase 2: %d epochs (patience=%d)",
                PHASE1_EPOCHS, PHASE2_EPOCHS, PATIENCE)
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
                f"(head={lr_head:.1e}  feat={lr_features:.1e}  cls={lr_classifier:.1e})"
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
            )
            elapsed = time.monotonic() - t0

            result.update({
                "batch_size": batch_size,
                "lr_scale": lr_scale,
                "lr_head": lr_head,
                "lr_features": lr_features,
                "lr_classifier": lr_classifier,
                "time_s": elapsed,
            })
            results.append(result)

            logger.info(
                "    → F1=%.4f  stopped_epoch=%d  time=%.1fs",
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
        "Rank", "Batch", "LR scale", "LR head", "LR feat", "LR cls",
        "Macro F1", "Epoch", "Time",
    )
    logger.info("  " + "-" * 85)
    for rank, r in enumerate(results, 1):
        logger.info(
            "  #%-3d  %-6d  %-8s  %-10.1e  %-10.1e  %-10.1e  %-8.4f  %-6d  %-7.1fs",
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
            rank, r["batch_size"], r["lr_scale"], r["best_f1"],
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
        (r for r in results if r["batch_size"] == 32 and r["lr_scale"] == 1.0),
        None,
    )
    if baseline and best is not baseline:
        diff = best["best_f1"] - baseline["best_f1"]
        speedup = baseline["time_s"] / best["time_s"] if best["time_s"] > 0 else 0
        logger.info(
            "  vs baseline (batch=32, 1.0x): F1 %+.4f, %.1fx speed",
            diff, speedup,
        )
    elif baseline:
        logger.info("  Baseline (batch=32, 1.0x) is already the best config.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
