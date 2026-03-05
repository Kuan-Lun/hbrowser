"""Sample loading, balancing, and file-naming utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict

import numpy as np
import torch

from .constants import LABELS_FILE, NUM_CLASSES, RAWIMAGE_DIR

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


def compute_pos_weight(
    samples: list[tuple[str, list[int]]],
) -> torch.Tensor:
    """從訓練樣本計算 BCEWithLogitsLoss 的 pos_weight。

    公式: pos_weight[cls] = (N - pos_count) / pos_count
    """
    counts = [0] * NUM_CLASSES
    for _, labels in samples:
        for lbl in labels:
            counts[lbl - 1] += 1
    n = len(samples)
    return torch.tensor([(n - c) / max(c, 1) for c in counts], dtype=torch.float32)


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
