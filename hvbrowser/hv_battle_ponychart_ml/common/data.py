"""Data loading, dataset, transforms, and splitting utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS_FILE,
    NUM_CLASSES,
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
            img = img.resize((256, 256), Image.Resampling.BOX)
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
            transforms.CenterCrop((224, 224)),
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
