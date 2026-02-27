"""Data loading, dataset, transforms, and splitting utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    LABELS_FILE,
    NUM_CLASSES,
    PRE_RESIZE,
    RAWIMAGE_DIR,
)

logger = logging.getLogger(__name__)

ORIG_PATTERN = re.compile(r"^pony_chart_\d{8}_\d{6}\.png$")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def is_original(filename: str) -> bool:
    """Check if a filename matches the original image pattern."""
    return bool(ORIG_PATTERN.match(filename))


def separate_orig_crop(
    samples: list[tuple[str, list[int]]],
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Separate samples into originals and crops based on filename pattern."""
    orig = [s for s in samples if is_original(os.path.basename(s[0]))]
    crop = [s for s in samples if not is_original(os.path.basename(s[0]))]
    return orig, crop


def get_base_timestamp(filename: str) -> str:
    """Extract pony_chart_YYYYMMDD_HHMMSS from any variant."""
    parts = filename.replace(".png", "").replace(".jpg", "").split("_")
    return "_".join(parts[:4])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples() -> list[tuple[str, list[int]]]:
    """Load labeled samples from labels.json.

    Returns list of (image_path, [1-indexed labels]).
    """
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
    """Oversample crop 圖片使 per-class 出現比例接近 target_rates。"""
    if not crop_samples:
        return []

    current_rates = compute_class_rates(crop_samples)
    n = len(crop_samples)

    target_counts = [max(int(round(tr * n)), 0) for tr in target_rates]
    current_counts = [int(round(cr * n)) for cr in current_rates]

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
        available = [idx for idx in class_to_indices[cls] if idx not in extra_indices]
        n_to_sample = min(deficit, len(available))
        if n_to_sample <= 0:
            continue
        sampled = rng.choice(available, size=n_to_sample, replace=False)
        for idx in sampled:
            extra_indices.add(idx)
            extra_samples.append(crop_samples[idx])

    return list(crop_samples) + extra_samples


def labels_to_binary(label_list: list[int]) -> torch.Tensor:
    """Convert 1-indexed label list to binary vector."""
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in label_list:
        vec[lbl - 1] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def group_stratified_split(
    samples: list[tuple[str, list[int]]],
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Group-stratified split returning sample indices."""
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


def split_by_groups(
    samples: list[tuple[str, list[int]]],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split timestamp groups, returning group keys instead of indices."""
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
# Dataset
# ---------------------------------------------------------------------------
class PonyChartDataset(Dataset):  # type: ignore[misc]
    """Dataset that pre-loads and caches resized images in memory."""

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
            img = img.resize((PRE_RESIZE, PRE_RESIZE), Image.Resampling.BOX)
            self._cache.append(img)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image: Any = self._cache[idx]
        if self.transform:
            image = self.transform(image)
        target = labels_to_binary(self.samples[idx][1])
        return image, target


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(is_train: bool) -> transforms.Compose:
    """Return the default augmentation pipeline."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=90, translate=(0.05, 0.05), scale=(0.9, 1.1)
                ),
                transforms.RandomCrop((INPUT_SIZE, INPUT_SIZE)),
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
            transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    use_persistent = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        pin_memory=(device.type == "cuda"),
    )
