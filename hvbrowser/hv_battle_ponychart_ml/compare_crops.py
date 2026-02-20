"""
比較裁切圖片對訓練效果的影響，分離「資料增量」與「分佈偏差」兩個因素。

三組實驗：
  A: 原圖 + 所有 crop（現有偏差分佈）
  B: 僅原圖（baseline）
  C: 原圖 + 平衡 resample 後的 crop（移除偏差）

- B vs A = 總效應（增量 + 偏差）
- C vs B = 純增量效應
- A vs C = 純偏差效應

共用 10% 原始圖片作為測試集，確保評估基準一致。
最後印出 per-class 分佈偏差與 F1 差異的 Pearson 相關性，量化偏差的因果影響。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
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

ORIG_PATTERN = re.compile(r"^pony_chart_\d{8}_\d{6}\.png$")

SEED = 42
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 35
PATIENCE = 12


# ---------------------------------------------------------------------------
# Helpers
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


def is_original(filename: str) -> bool:
    return bool(ORIG_PATTERN.match(filename))


def get_base_timestamp(filename: str) -> str:
    """Extract pony_chart_YYYYMMDD_HHMMSS from any variant."""
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    return "_".join(parts[:4])


def compute_class_rates(
    samples: list[tuple[str, list[int]]],
) -> list[float]:
    """計算每個 class 的出現比例 (positive rate)。"""
    counts = [0] * NUM_CLASSES
    for _, labels in samples:
        for lbl in labels:
            counts[lbl - 1] += 1
    n = max(len(samples), 1)
    return [c / n for c in counts]


def balance_crop_samples(
    crop_samples: list[tuple[str, list[int]]],
    target_rates: list[float],
    rng: np.random.RandomState,
) -> list[tuple[str, list[int]]]:
    """Oversample crop 圖片使 per-class 出現比例接近 target_rates。

    策略：為每個 class 計算需要多少額外 sample，從含有該 class 的 crop
    中隨機抽取（有放回）補足差距。最後去重合併。
    """
    if not crop_samples:
        return []

    current_rates = compute_class_rates(crop_samples)
    n = len(crop_samples)

    # 每個 class 目標數量 = target_rate * n，但不低於目前數量
    target_counts = [max(int(round(tr * n)), 0) for tr in target_rates]
    current_counts = [int(round(cr * n)) for cr in current_rates]

    # 為每個 class 收集包含它的 crop index
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, labels) in enumerate(crop_samples):
        for lbl in labels:
            class_to_indices[lbl - 1].append(idx)

    extra_indices: set[int] = set()
    extra_samples: list[tuple[str, list[int]]] = []

    for cls in range(NUM_CLASSES):
        deficit = target_counts[cls] - current_counts[cls]
        if deficit <= 0 or not class_to_indices[cls]:
            continue
        # 從含有此 class 的 crop 中有放回抽樣
        sampled = rng.choice(class_to_indices[cls], size=deficit, replace=True)
        for idx in sampled:
            if idx not in extra_indices:
                extra_indices.add(idx)
                extra_samples.append(crop_samples[idx])

    return list(crop_samples) + extra_samples


def log_distribution(
    label: str,
    samples: list[tuple[str, list[int]]],
) -> list[float]:
    """印出並回傳 per-class positive rate。"""
    rates = compute_class_rates(samples)
    logger.info("  %s (%d samples):", label, len(samples))
    for i, name in enumerate(CLASS_NAMES):
        count = sum(1 for _, lbls in samples if (i + 1) in lbls)
        logger.info("    %-20s  %4d  (%.1f%%)", name, count, rates[i] * 100)
    return rates


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_samples() -> list[tuple[str, list[int]]]:
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


def split_by_groups(
    samples: list[tuple[str, list[int]]],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split timestamp groups into train and test groups."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
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

    train_gk, test_gk = train_test_split(
        group_keys,
        test_size=test_size,
        random_state=seed,
        stratify=group_primary_label,
    )
    return train_gk, test_gk


# ---------------------------------------------------------------------------
# Dataset & transforms
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
# Model
# ---------------------------------------------------------------------------
def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features: int = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


# ---------------------------------------------------------------------------
# Training & evaluation
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate model, return loss, macro F1, per-class metrics."""
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


@torch.no_grad()
def optimize_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[float]:
    model.eval()
    all_probs: list[np.ndarray[Any, Any]] = []
    all_targets: list[np.ndarray[Any, Any]] = []
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
# Full training pipeline
# ---------------------------------------------------------------------------
def train_model(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    device: torch.device,
    num_workers: int,
    experiment_name: str,
) -> tuple[nn.Module, list[float]]:
    """Train a model end-to-end, return (best_model, optimized_thresholds)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info(
        "  Train: %d samples, Val: %d samples", len(train_samples), len(val_samples)
    )
    logger.info("=" * 60)

    train_ds = PonyChartDataset(train_samples, get_transforms(is_train=True))
    val_ds = PonyChartDataset(val_samples, get_transforms(is_train=False))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: Head only
    logger.info("--- Phase 1: Head-only (%d epochs) ---", PHASE1_EPOCHS)
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f",
            epoch,
            PHASE1_EPOCHS,
            train_loss,
            val_result["loss"],
            val_result["macro_f1"],
        )

    # Phase 2: Full fine-tuning
    logger.info("--- Phase 2: Full fine-tuning (%d epochs) ---", PHASE2_EPOCHS)
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

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
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

        logger.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.4f%s",
            epoch,
            PHASE2_EPOCHS,
            train_loss,
            val_result["loss"],
            val_f1,
            marker,
        )
        if patience_counter >= PATIENCE:
            logger.info("  Early stopping (no improvement for %d epochs)", PATIENCE)
            break

    model.load_state_dict(best_state)

    # Optimize thresholds on validation set
    thresholds = optimize_thresholds(model, val_loader, device)
    logger.info("Optimized thresholds: %s", dict(zip(CLASS_NAMES, thresholds)))

    return model, thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _pearson_r(x: list[float], y: list[float]) -> float:
    """計算 Pearson 相關係數（不依賴 scipy）。"""
    xa = np.array(x)
    ya = np.array(y)
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.sqrt((xa**2).sum() * (ya**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xa * ya).sum() / denom)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    num_workers = _get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load all samples
    all_samples = load_all_samples()
    logger.info("Total samples loaded: %d", len(all_samples))

    # Split groups: 10% test, 90% train+val
    train_val_groups, test_groups = split_by_groups(
        all_samples, test_size=0.10, seed=SEED
    )

    # Build index from base timestamp to sample indices
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(all_samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    # Test set: only original images from test groups
    test_indices = []
    for gk in test_groups:
        for idx in groups[gk]:
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                test_indices.append(idx)
    test_samples = [all_samples[i] for i in test_indices]

    # Collect train+val indices, separate originals and crops
    train_val_indices_all = []
    train_val_indices_orig = []
    train_val_indices_crop = []
    for gk in train_val_groups:
        for idx in groups[gk]:
            train_val_indices_all.append(idx)
            fname = os.path.basename(all_samples[idx][0])
            if is_original(fname):
                train_val_indices_orig.append(idx)
            else:
                train_val_indices_crop.append(idx)

    train_val_all = [all_samples[i] for i in train_val_indices_all]
    train_val_orig = [all_samples[i] for i in train_val_indices_orig]
    train_val_crop = [all_samples[i] for i in train_val_indices_crop]

    # ── Experiment C: balance crop samples to match original distribution ──
    orig_rates = compute_class_rates(train_val_orig)
    balanced_crops = balance_crop_samples(train_val_crop, orig_rates, rng)
    train_val_balanced = train_val_orig + balanced_crops

    # ── Distribution analysis ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    log_distribution("Original images (train+val)", train_val_orig)
    crop_rates = log_distribution("Crop images (raw)", train_val_crop)
    log_distribution("Crop images (balanced)", balanced_crops)
    logger.info("")
    logger.info("  Per-class bias (crop_rate - orig_rate):")
    bias_per_class = []
    for i, name in enumerate(CLASS_NAMES):
        bias = crop_rates[i] - orig_rates[i]
        bias_per_class.append(bias)
        logger.info("    %-20s  %+.1f%%", name, bias * 100)

    # ── Split train/val for each experiment ──
    # Experiment A: originals + all crops (biased)
    train_gk_a, val_gk_a = split_by_groups(train_val_all, test_size=0.15, seed=SEED)
    groups_a: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_all):
        base = get_base_timestamp(os.path.basename(path))
        groups_a[base].append(idx)
    train_a = [train_val_all[i] for gk in train_gk_a for i in groups_a[gk]]
    val_a = [train_val_all[i] for gk in val_gk_a for i in groups_a[gk]]

    # Experiment B: originals only
    train_gk_b, val_gk_b = split_by_groups(train_val_orig, test_size=0.15, seed=SEED)
    groups_b: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_orig):
        base = get_base_timestamp(os.path.basename(path))
        groups_b[base].append(idx)
    train_b = [train_val_orig[i] for gk in train_gk_b for i in groups_b[gk]]
    val_b = [train_val_orig[i] for gk in val_gk_b for i in groups_b[gk]]

    # Experiment C: originals + balanced crops
    train_gk_c, val_gk_c = split_by_groups(
        train_val_balanced, test_size=0.15, seed=SEED
    )
    groups_c: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(train_val_balanced):
        base = get_base_timestamp(os.path.basename(path))
        groups_c[base].append(idx)
    train_c = [train_val_balanced[i] for gk in train_gk_c for i in groups_c[gk]]
    val_c = [train_val_balanced[i] for gk in val_gk_c for i in groups_c[gk]]

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info("Test set (shared, originals only): %d images", len(test_samples))
    logger.info("")
    logger.info("Experiment A (orig + biased crops):")
    logger.info("  Train: %d  Val: %d", len(train_a), len(val_a))
    logger.info("Experiment B (originals only):")
    logger.info("  Train: %d  Val: %d", len(train_b), len(val_b))
    logger.info("Experiment C (orig + balanced crops):")
    logger.info("  Train: %d  Val: %d", len(train_c), len(val_c))
    logger.info("=" * 60)
    logger.info("")

    criterion = nn.BCEWithLogitsLoss()
    use_persistent = num_workers > 0

    # ---- Train Experiment A: with biased crops ----
    model_a, thresholds_a = train_model(
        train_a,
        val_a,
        device,
        num_workers,
        "A: Originals + biased crops",
    )

    # ---- Train Experiment B: originals only ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_b, thresholds_b = train_model(
        train_b,
        val_b,
        device,
        num_workers,
        "B: Originals only (baseline)",
    )

    # ---- Train Experiment C: originals + balanced crops ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_c, thresholds_c = train_model(
        train_c,
        val_c,
        device,
        num_workers,
        "C: Originals + balanced crops",
    )

    # ---- Evaluate all on test set ----
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    result_a = evaluate(model_a, test_loader, criterion, device, thresholds_a)
    result_b = evaluate(model_b, test_loader, criterion, device, thresholds_b)
    result_c = evaluate(model_c, test_loader, criterion, device, thresholds_c)

    # ── Print comparison ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SET EVALUATION (on %d original images)", len(test_samples))
    logger.info("=" * 80)
    logger.info("  A thresholds: %s", dict(zip(CLASS_NAMES, thresholds_a)))
    logger.info("  B thresholds: %s", dict(zip(CLASS_NAMES, thresholds_b)))
    logger.info("  C thresholds: %s", dict(zip(CLASS_NAMES, thresholds_c)))
    logger.info("")
    logger.info(
        "%-20s  %-14s  %-14s  %-14s",
        "Metric",
        "A (biased)",
        "B (orig only)",
        "C (balanced)",
    )
    logger.info("-" * 80)
    logger.info(
        "%-20s  %-14.4f  %-14.4f  %-14.4f",
        "Macro F1",
        result_a["macro_f1"],
        result_b["macro_f1"],
        result_c["macro_f1"],
    )
    logger.info(
        "%-20s  %-14.4f  %-14.4f  %-14.4f",
        "Loss",
        result_a["loss"],
        result_b["loss"],
        result_c["loss"],
    )

    logger.info("")
    logger.info("Per-class detail (optimized thresholds):")
    logger.info(
        "  %-20s  %-7s %-7s %-7s | %-7s %-7s %-7s | %-7s %-7s %-7s",
        "Class",
        "A_P",
        "A_R",
        "A_F1",
        "B_P",
        "B_R",
        "B_F1",
        "C_P",
        "C_R",
        "C_F1",
    )
    for i, name in enumerate(CLASS_NAMES):
        logger.info(
            "  %-20s  %-7.4f %-7.4f %-7.4f | %-7.4f %-7.4f %-7.4f"
            " | %-7.4f %-7.4f %-7.4f",
            name,
            result_a["per_class_precision"][i],
            result_a["per_class_recall"][i],
            result_a["per_class_f1"][i],
            result_b["per_class_precision"][i],
            result_b["per_class_recall"][i],
            result_b["per_class_f1"][i],
            result_c["per_class_precision"][i],
            result_c["per_class_recall"][i],
            result_c["per_class_f1"][i],
        )

    # ── Effect decomposition ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("EFFECT DECOMPOSITION")
    logger.info("=" * 80)
    total_effect = result_a["macro_f1"] - result_b["macro_f1"]
    augment_effect = result_c["macro_f1"] - result_b["macro_f1"]
    bias_effect = result_a["macro_f1"] - result_c["macro_f1"]
    logger.info("  A vs B (total effect = augmentation + bias): %+.4f", total_effect)
    logger.info("  C vs B (pure augmentation effect):           %+.4f", augment_effect)
    logger.info("  A vs C (pure bias effect):                   %+.4f", bias_effect)

    logger.info("")
    logger.info("Per-class effect decomposition:")
    logger.info(
        "  %-20s  %-10s  %-10s  %-10s  %-10s",
        "Class",
        "Bias",
        "A-B total",
        "C-B augment",
        "A-C bias",
    )
    f1_diff_ab = []
    for i, name in enumerate(CLASS_NAMES):
        ab = result_a["per_class_f1"][i] - result_b["per_class_f1"][i]
        cb = result_c["per_class_f1"][i] - result_b["per_class_f1"][i]
        ac = result_a["per_class_f1"][i] - result_c["per_class_f1"][i]
        f1_diff_ab.append(ab)
        logger.info(
            "  %-20s  %+.1f%%     %+.4f     %+.4f     %+.4f",
            name,
            bias_per_class[i] * 100,
            ab,
            cb,
            ac,
        )

    # ── Correlation: distribution bias vs F1 impact ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 80)
    r_ab = _pearson_r(bias_per_class, f1_diff_ab)
    f1_diff_ac = [
        result_a["per_class_f1"][i] - result_c["per_class_f1"][i]
        for i in range(NUM_CLASSES)
    ]
    r_ac = _pearson_r(bias_per_class, f1_diff_ac)
    ab_hint = (
        "(偏差越大 → 該 class 在 A 中表現越好)"
        if r_ab > 0
        else "(偏差越大 → 反而越差)"
    )
    ac_hint = (
        "(正相關 = 偏差確實影響 F1)"
        if abs(r_ac) > 0.3
        else "(弱相關 = 偏差對 F1 影響有限)"
    )
    logger.info(
        "  Pearson r (bias vs A-B F1 diff): %.4f  %s", r_ab, ab_hint
    )
    logger.info(
        "  Pearson r (bias vs A-C F1 diff): %.4f  %s", r_ac, ac_hint
    )

    # ── Summary ──
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "  Macro F1:  A=%.4f  B=%.4f  C=%.4f",
        result_a["macro_f1"],
        result_b["macro_f1"],
        result_c["macro_f1"],
    )
    logger.info("  Total effect   (A-B): %+.4f", total_effect)
    logger.info("  Augment effect (C-B): %+.4f", augment_effect)
    logger.info("  Bias effect    (A-C): %+.4f", bias_effect)
    logger.info("  Bias-F1 correlation:  r=%.4f", r_ac)
    logger.info("")
    if abs(bias_effect) < 0.005:
        logger.info("  結論: 裁切偏差對整體 F1 影響有限 (%.4f)", bias_effect)
    elif bias_effect < -0.005:
        logger.info(
            "  結論: 裁切偏差降低效果 (%.4f F1)，建議使用平衡後的 crop",
            bias_effect,
        )
    else:
        logger.info("  結論: 裁切偏差反而有正面效果 (+%.4f F1)", bias_effect)
    if abs(r_ac) > 0.5:
        logger.info(
            "  注意: 偏差與 per-class F1 有強相關"
            " (r=%.2f)，特定角色受影響顯著",
            r_ac,
        )
    logger.info("=" * 80)

    # ── Crop recommendation ──
    crop_counts_per_class = [0] * NUM_CLASSES
    for _, labels in train_val_crop:
        for lbl in labels:
            crop_counts_per_class[lbl - 1] += 1

    total_crops = len(train_val_crop)

    # 計算「讓 crop 分佈對齊原圖分佈」的目標數量
    # target_i = total_crops * (orig_rate_i / sum(orig_rates))
    # 正規化原圖 rate 為比例（因為多標籤，rate 總和 > 1）
    orig_rate_sum = sum(orig_rates)
    target_per_class = [
        int(round(total_crops * (orig_rates[i] / orig_rate_sum)))
        for i in range(NUM_CLASSES)
    ]

    recommendations = []
    max_crop = max(crop_counts_per_class) if crop_counts_per_class else 1
    for i in range(NUM_CLASSES):
        cb = result_c["per_class_f1"][i] - result_b["per_class_f1"][i]
        ab = result_a["per_class_f1"][i] - result_b["per_class_f1"][i]
        b_f1 = result_b["per_class_f1"][i]
        crop_n = crop_counts_per_class[i]
        target_n = target_per_class[i]
        deficit = max(target_n - crop_n, 0)

        # 如果 crop 對此 class 有害，不建議追加
        is_beneficial = cb > 0.01 or ab > 0.01
        suggested = deficit if is_beneficial else 0

        scarcity = 1.0 - (crop_n / max(max_crop, 1))
        room = 1.0 - b_f1
        score = max(cb, ab) * 0.4 + room * 0.3 + scarcity * 0.3
        recommendations.append({
            "idx": i,
            "name": CLASS_NAMES[i],
            "crop_n": crop_n,
            "target_n": target_n,
            "deficit": deficit,
            "suggested": suggested,
            "b_f1": b_f1,
            "cb": cb,
            "ab": ab,
            "score": score,
            "beneficial": is_beneficial,
        })
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("CROP RECOMMENDATION（建議裁切優先順序）")
    logger.info("=" * 80)
    logger.info(
        "  估算方式: 讓各 class 的 crop 數量比例"
        "對齊原圖出現比例 (total crops=%d)",
        total_crops,
    )
    logger.info("")
    logger.info(
        "  %-4s %-18s  %-6s %-6s %-6s  %-8s  %-9s %-9s",
        "Rank",
        "Class",
        "Crops",
        "Target",
        "+Need",
        "B F1",
        "C-B",
        "A-B",
    )
    logger.info("  " + "-" * 74)
    for rank, r in enumerate(recommendations, 1):
        if r["beneficial"]:
            if r["suggested"] > 0:
                advice = "← 建議再裁 {} 張".format(r["suggested"])
            else:
                advice = "← 已達標，可裁可不裁"
        elif r["cb"] < -0.03 and r["ab"] < -0.03:
            advice = "  (crop 有害，暫不裁切)"
        else:
            advice = "  (效果有限)"
        logger.info(
            "  #%-3d %-18s  %-6d %-6d %-6d  %-8.4f  %+-9.4f %+-9.4f %s",
            rank,
            r["name"],
            r["crop_n"],
            r["target_n"],
            r["deficit"],
            r["b_f1"],
            r["cb"],
            r["ab"],
            advice,
        )

    # 摘要
    logger.info("")
    to_crop = [
        r for r in recommendations if r["beneficial"] and r["suggested"] > 0
    ]
    if to_crop:
        logger.info("  具體行動:")
        total_suggested = 0
        for r in to_crop:
            logger.info(
                "    %s: 再裁切 ~%d 張 (目前 %d → 目標 %d)",
                r["name"],
                r["suggested"],
                r["crop_n"],
                r["target_n"],
            )
            total_suggested += r["suggested"]
        logger.info("    共計約 %d 張新 crop", total_suggested)
    else:
        logger.info("  各 class crop 數量已足夠或 crop 無正面效果。")

    saturated = [
        r for r in recommendations if r["beneficial"] and r["suggested"] == 0
    ]
    if saturated:
        names = ", ".join(r["name"] for r in saturated)
        logger.info("  已達標 (crop 有效但數量足夠): %s", names)

    harmful = [
        r for r in recommendations
        if r["cb"] < -0.03 and r["ab"] < -0.03
    ]
    if harmful:
        names = ", ".join(r["name"] for r in harmful)
        logger.info("  暫不裁切 (crop 反而有害): %s", names)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
