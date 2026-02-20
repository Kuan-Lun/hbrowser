"""
分析空間增強（水平翻轉、垂直翻轉、旋轉角度）對訓練效果的影響。

透過 ablation study 逐一啟用各空間增強，比較與 baseline 的 F1 差異，
判斷哪些增強有正面效果、最佳旋轉角度為何。

七組實驗（僅空間增強不同，其餘 augmentation 皆相同）：
  1. none    — 無翻轉、無旋轉（baseline）
  2. hflip   — + 水平翻轉
  3. vflip   — + 垂直翻轉
  4. rot15   — + 旋轉 15°
  5. rot45   — + 旋轉 45°
  6. rot90   — + 旋轉 90°
  7. current — 水平翻轉 + 垂直翻轉 + 旋轉 90°（目前 train.py 設定）

使用方式：
  python -m hvbrowser.hv_battle_ponychart_ml.analyze_augmentations
  # 或
  python hvbrowser/hv_battle_ponychart_ml/analyze_augmentations.py
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import subprocess
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
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 35
PATIENCE = 12


# ---------------------------------------------------------------------------
# Augmentation configs
# ---------------------------------------------------------------------------
class AugConfig:
    """描述一組空間增強設定。"""

    def __init__(
        self,
        name: str,
        hflip: bool = False,
        vflip: bool = False,
        degrees: float = 0,
    ) -> None:
        self.name = name
        self.hflip = hflip
        self.vflip = vflip
        self.degrees = degrees

    def __repr__(self) -> str:
        parts = []
        if self.hflip:
            parts.append("hflip")
        if self.vflip:
            parts.append("vflip")
        if self.degrees > 0:
            parts.append(f"rot{self.degrees:.0f}")
        return f"AugConfig({self.name}: {', '.join(parts) or 'none'})"


EXPERIMENTS: list[AugConfig] = [
    AugConfig("none"),
    AugConfig("hflip", hflip=True),
    AugConfig("vflip", vflip=True),
    AugConfig("rot15", degrees=15),
    AugConfig("rot45", degrees=45),
    AugConfig("rot90", degrees=90),
    AugConfig("current", hflip=True, vflip=True, degrees=90),
]


def build_train_transform(cfg: AugConfig) -> transforms.Compose:
    """根據 AugConfig 建立訓練用 transform pipeline。

    非空間增強（ColorJitter, GaussianBlur, RandomErasing）皆保持一致，
    僅變動翻轉與旋轉，以確保 ablation 公平比較。
    """
    spatial: list[Any] = []
    if cfg.hflip:
        spatial.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.vflip:
        spatial.append(transforms.RandomVerticalFlip(p=0.5))
    # 始終保留 translate 和 scale（非旋轉的空間微調），僅改變 degrees
    spatial.append(
        transforms.RandomAffine(
            degrees=cfg.degrees, translate=(0.05, 0.05), scale=(0.9, 1.1)
        )
    )
    return transforms.Compose(
        [
            *spatial,
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


VAL_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


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


def group_stratified_split(
    samples: list[tuple[str, list[int]]],
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """依時間戳分組的分層切分。"""
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
# Dataset
# ---------------------------------------------------------------------------
class PonyChartDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        samples: list[tuple[str, list[int]]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self._cache: list[Image.Image] = []
        for path, _ in samples:
            img = Image.open(path).convert("RGB")
            img = img.resize((256, 256), Image.Resampling.BOX)
            self._cache.append(img)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._cache[idx]
        if self.transform:
            image = self.transform(image)
        target = labels_to_binary(self.samples[idx][1])
        return image, target


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


@torch.no_grad()  # type: ignore[untyped-decorator]
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
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

    # 為每個 class 搜尋最佳 threshold
    best_thresholds: list[float] = []
    for i in range(NUM_CLASSES):
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.20, 0.80, 0.01):
            preds = (all_probs_arr[:, i] >= thr).astype(int)
            f1 = f1_score(all_targets_arr[:, i], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        best_thresholds.append(round(best_thr, 4))

    # 使用最佳 threshold 計算最終 metrics
    preds = np.zeros_like(all_probs_arr, dtype=int)
    for i in range(NUM_CLASSES):
        preds[:, i] = (all_probs_arr[:, i] >= best_thresholds[i]).astype(int)

    per_class_f1 = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(all_targets_arr[:, i], preds[:, i], zero_division=0.0)
        per_class_f1.append(float(f1))

    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": float(np.mean(per_class_f1)),
        "per_class_f1": per_class_f1,
        "thresholds": best_thresholds,
    }


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------
def train_model(
    train_samples: list[tuple[str, list[int]]],
    val_samples: list[tuple[str, list[int]]],
    train_transform: transforms.Compose,
    device: torch.device,
    num_workers: int,
    experiment_name: str,
) -> nn.Module:
    """訓練模型並回傳最佳模型。"""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info(
        "  Train: %d samples, Val: %d samples", len(train_samples), len(val_samples)
    )
    logger.info("=" * 60)

    train_ds = PonyChartDataset(train_samples, train_transform)
    val_ds = PonyChartDataset(val_samples, VAL_TRANSFORM)
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
    return model


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

    # Load data
    all_samples = load_all_samples()
    logger.info("Total samples: %d", len(all_samples))

    train_idx, test_idx = group_stratified_split(all_samples, test_size=0.15, seed=SEED)
    train_samples = [all_samples[i] for i in train_idx]
    test_samples = [all_samples[i] for i in test_idx]
    logger.info("Train: %d  Test: %d", len(train_samples), len(test_samples))

    # 從 train 再切出 val 用於訓練過程中的 early stopping
    sub_train_idx, val_idx = group_stratified_split(
        train_samples, test_size=0.15, seed=SEED
    )
    sub_train_samples = [train_samples[i] for i in sub_train_idx]
    val_samples = [train_samples[i] for i in val_idx]
    logger.info(
        "Sub-train: %d  Val: %d  Test: %d",
        len(sub_train_samples),
        len(val_samples),
        len(test_samples),
    )

    # Test set (共用)
    test_ds = PonyChartDataset(test_samples, VAL_TRANSFORM)
    use_persistent = num_workers > 0
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ── Run all experiments ──
    results: dict[str, dict[str, Any]] = {}
    for cfg in EXPERIMENTS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_tf = build_train_transform(cfg)
        model = train_model(
            sub_train_samples,
            val_samples,
            train_tf,
            device,
            num_workers,
            cfg.name,
        )
        result = evaluate(model, test_loader, criterion, device)
        results[cfg.name] = result
        logger.info(
            "  >> %s test F1=%.4f  thresholds=%s",
            cfg.name,
            result["macro_f1"],
            dict(zip(CLASS_NAMES, result["thresholds"])),
        )

    # ── Print comparison table ──
    baseline_f1 = results["none"]["macro_f1"]
    baseline_per_class = results["none"]["per_class_f1"]

    logger.info("")
    logger.info("=" * 90)
    logger.info(
        "AUGMENTATION ABLATION RESULTS (test set, %d images)",
        len(test_samples),
    )
    logger.info("=" * 90)

    # Macro F1 overview
    logger.info("")
    logger.info("%-12s  %-10s  %-10s  %s", "Experiment", "Macro F1", "Delta", "Config")
    logger.info("-" * 75)
    for cfg in EXPERIMENTS:
        r = results[cfg.name]
        delta = r["macro_f1"] - baseline_f1
        delta_str = f"{delta:+.4f}" if cfg.name != "none" else "baseline"
        desc_parts = []
        if cfg.hflip:
            desc_parts.append("HFlip")
        if cfg.vflip:
            desc_parts.append("VFlip")
        if cfg.degrees > 0:
            desc_parts.append(f"Rot({cfg.degrees:.0f}°)")
        desc = " + ".join(desc_parts) if desc_parts else "(no spatial aug)"
        logger.info(
            "%-12s  %-10.4f  %-10s  %s",
            cfg.name,
            r["macro_f1"],
            delta_str,
            desc,
        )

    # Per-class F1 table
    logger.info("")
    logger.info("Per-class F1 (delta vs baseline):")
    header = "  %-20s" + "  %-12s" * len(EXPERIMENTS)
    logger.info(header, "Class", *[cfg.name for cfg in EXPERIMENTS])
    logger.info("  " + "-" * (20 + 14 * len(EXPERIMENTS)))
    for i, name in enumerate(CLASS_NAMES):
        row_parts = [f"  {name:<20s}"]
        for cfg in EXPERIMENTS:
            f1 = results[cfg.name]["per_class_f1"][i]
            delta = f1 - baseline_per_class[i]
            if cfg.name == "none":
                row_parts.append(f"  {f1:<12.4f}")
            else:
                row_parts.append(f"  {f1:.4f}{delta:+.3f}")
        logger.info("".join(row_parts))

    # ── Flip analysis ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("FLIP ANALYSIS")
    logger.info("=" * 90)

    hflip_delta = results["hflip"]["macro_f1"] - baseline_f1
    vflip_delta = results["vflip"]["macro_f1"] - baseline_f1
    logger.info("  HFlip effect:  %+.4f", hflip_delta)
    logger.info("  VFlip effect:  %+.4f", vflip_delta)

    if hflip_delta > 0.005:
        logger.info("  >> 水平翻轉有正面效果，建議保留")
    elif hflip_delta < -0.005:
        logger.info("  >> 水平翻轉有負面效果，建議移除")
    else:
        logger.info(
            "  >> 水平翻轉效果不明顯 (%.4f)，可考慮移除以簡化 pipeline",
            hflip_delta,
        )

    if vflip_delta > 0.005:
        logger.info("  >> 垂直翻轉有正面效果，建議保留")
    elif vflip_delta < -0.005:
        logger.info("  >> 垂直翻轉有負面效果，建議移除")
    else:
        logger.info(
            "  >> 垂直翻轉效果不明顯 (%.4f)，可考慮移除以簡化 pipeline",
            vflip_delta,
        )

    # Per-class flip impact
    logger.info("")
    logger.info("  Per-class flip impact (F1 delta vs baseline):")
    logger.info("  %-20s  %-12s  %-12s", "Class", "HFlip", "VFlip")
    for i, name in enumerate(CLASS_NAMES):
        hd = results["hflip"]["per_class_f1"][i] - baseline_per_class[i]
        vd = results["vflip"]["per_class_f1"][i] - baseline_per_class[i]
        logger.info("  %-20s  %+.4f       %+.4f", name, hd, vd)

    # ── Rotation analysis ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("ROTATION ANALYSIS")
    logger.info("=" * 90)

    rot_configs = [("rot15", 15), ("rot45", 45), ("rot90", 90)]
    best_rot_name = "none"
    best_rot_f1 = baseline_f1
    for rname, deg in rot_configs:
        delta = results[rname]["macro_f1"] - baseline_f1
        rot_f1 = results[rname]["macro_f1"]
        logger.info(
            "  Rotation %3d°: %+.4f (F1=%.4f)", deg, delta, rot_f1
        )
        if results[rname]["macro_f1"] > best_rot_f1:
            best_rot_f1 = results[rname]["macro_f1"]
            best_rot_name = rname

    if best_rot_name == "none":
        logger.info("  >> 所有旋轉角度皆無正面效果，建議移除旋轉增強")
    else:
        best_deg = {"rot15": 15, "rot45": 45, "rot90": 90}[best_rot_name]
        delta = best_rot_f1 - baseline_f1
        logger.info("  >> 最佳旋轉角度: %d° (%+.4f F1)", best_deg, delta)
        if delta > 0.005:
            logger.info("  >> 建議使用 %d° 旋轉", best_deg)
        else:
            logger.info("  >> 效果有限 (%.4f)，旋轉非必要", delta)

    # Per-class rotation impact
    logger.info("")
    logger.info("  Per-class rotation impact (F1 delta vs baseline):")
    logger.info("  %-20s  %-10s  %-10s  %-10s", "Class", "15°", "45°", "90°")
    for i, name in enumerate(CLASS_NAMES):
        deltas = []
        for rname, _ in rot_configs:
            d = results[rname]["per_class_f1"][i] - baseline_per_class[i]
            deltas.append(d)
        logger.info(
            "  %-20s  %+.4f     %+.4f     %+.4f", name, *deltas
        )

    # ── Combined vs individual ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("COMBINED EFFECT ANALYSIS")
    logger.info("=" * 90)

    current_delta = results["current"]["macro_f1"] - baseline_f1
    sum_individual = hflip_delta + vflip_delta + (best_rot_f1 - baseline_f1)
    interaction = current_delta - sum_individual

    logger.info("  Current config (HFlip+VFlip+Rot90): %+.4f", current_delta)
    logger.info("  Sum of individual effects:           %+.4f", sum_individual)
    logger.info("  Interaction effect:                  %+.4f", interaction)
    if interaction > 0.005:
        logger.info("  >> 組合使用有正向交互作用，建議同時啟用")
    elif interaction < -0.005:
        logger.info("  >> 組合使用有負向交互作用，建議精簡增強組合")
    else:
        logger.info("  >> 交互作用微小，各增強可獨立決定是否啟用")

    # ── Final recommendation ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("RECOMMENDATION")
    logger.info("=" * 90)

    # 找出最佳單一組合
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_result = results[best_name]
    logger.info("  最佳實驗: %s (Macro F1=%.4f)", best_name, best_result["macro_f1"])
    logger.info("")

    # 建議組合
    recommended_parts = []
    if hflip_delta > 0.003:
        recommended_parts.append("RandomHorizontalFlip(p=0.5)")
    if vflip_delta > 0.003:
        recommended_parts.append("RandomVerticalFlip(p=0.5)")
    if best_rot_name != "none" and (best_rot_f1 - baseline_f1) > 0.003:
        best_deg = {"rot15": 15, "rot45": 45, "rot90": 90}[best_rot_name]
        recommended_parts.append(f"RandomAffine(degrees={best_deg})")

    if recommended_parts:
        logger.info("  建議在 train.py 中使用的空間增強:")
        for part in recommended_parts:
            logger.info("    - %s", part)
    else:
        logger.info("  建議移除所有空間增強（翻轉與旋轉），僅保留:")
        logger.info("    - RandomCrop")
        logger.info("    - ColorJitter")
        logger.info("    - GaussianBlur")
        logger.info("    - RandomErasing")

    # 與目前設定比較
    logger.info("")
    diff = best_result["macro_f1"] - results["current"]["macro_f1"]
    if abs(diff) < 0.003:
        logger.info("  目前設定 (current) 已接近最佳，無需調整")
    elif diff > 0:
        logger.info(
            "  切換至 '%s' 可提升 F1 約 %+.4f", best_name, diff
        )
    else:
        logger.info("  目前設定 (current) 即為最佳或接近最佳")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
