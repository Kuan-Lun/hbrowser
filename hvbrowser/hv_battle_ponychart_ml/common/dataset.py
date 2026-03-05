"""Dataset, cache budget, transforms, and DataLoader factory."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, PRE_RESIZE
from .sampling import labels_to_binary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache budget
# ---------------------------------------------------------------------------
def _estimate_image_bytes(pre_resize: int) -> int:
    """Estimate memory per cached PIL RGB image (pixels + object overhead)."""
    return pre_resize * pre_resize * 3 + 1024


# Reserve = min(total * 20%, available * 25%) to keep the system responsive.
_RESERVE_TOTAL_FRACTION = 0.20
_RESERVE_AVAIL_FRACTION = 0.25


def compute_cache_budget(
    pre_resize: int,
    n_datasets: int = 2,
    training_reserve: int = 0,
) -> int:
    """Return the total number of images that can be cached across all datasets.

    *training_reserve* is the estimated bytes needed for model training
    (use :func:`measure_training_memory` to compute it).  This is
    subtracted from available memory before computing the cache budget.

    The cache uses a shared-memory ``torch.uint8`` tensor so DataLoader
    workers (even with the ``spawn`` start method) access the same pages
    without per-worker duplication.
    """
    per_image = _estimate_image_bytes(pre_resize)
    mem = psutil.virtual_memory()
    reserve = min(
        int(mem.total * _RESERVE_TOTAL_FRACTION),
        int(mem.available * _RESERVE_AVAIL_FRACTION),
    )
    budget = max(mem.available - reserve - training_reserve, 0)
    total_images = budget // per_image
    logger.info(
        "Cache budget: %d images across %d datasets "
        "(available %.0f MB, reserve %.0f MB, training %.0f MB)",
        total_images,
        n_datasets,
        mem.available / 1024 / 1024,
        reserve / 1024 / 1024,
        training_reserve / 1024 / 1024,
    )
    return total_images


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PonyChartDataset(Dataset):  # type: ignore[misc]
    """Dataset with adaptive image caching based on available memory.

    *max_cached* controls how many images to keep in memory.  Use
    :func:`compute_cache_budget` to determine a sensible total, then
    split it among datasets proportionally.  Un-cached images are loaded
    from disk on each access.

    The cache is stored as a ``torch.uint8`` tensor in POSIX shared
    memory (``share_memory_()``), so DataLoader workers spawned via the
    ``spawn`` start method access the same physical pages without
    per-worker duplication.
    """

    def __init__(
        self,
        samples: list[tuple[str, list[int]]],
        transform: transforms.Compose | None = None,
        pre_resize: int = PRE_RESIZE,
        max_cached: int | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self._pre_resize = pre_resize

        if max_cached is not None:
            n_cache = min(max_cached, len(samples))
        else:
            n_cache = len(samples)

        self._n_cached = n_cache
        if n_cache > 0:
            self._cache = torch.empty(
                n_cache, pre_resize, pre_resize, 3, dtype=torch.uint8,
            )
            for i in range(n_cache):
                path = samples[i][0]
                img = Image.open(path).convert("RGB")
                img = img.resize((pre_resize, pre_resize), Image.Resampling.BOX)
                self._cache[i] = torch.from_numpy(np.array(img))
            self._cache.share_memory_()
        else:
            self._cache = None

        logger.info(
            "PonyChartDataset: cached %d/%d images",
            n_cache,
            len(samples),
        )

    def _load_image(self, idx: int) -> Image.Image:
        if self._cache is not None and idx < self._n_cached:
            return Image.fromarray(self._cache[idx].numpy())
        path = self.samples[idx][0]
        img = Image.open(path).convert("RGB")
        return img.resize((self._pre_resize, self._pre_resize), Image.Resampling.BOX)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image: Any = self._load_image(idx)
        if self.transform:
            image = self.transform(image)
        target = labels_to_binary(self.samples[idx][1])
        return image, target


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(is_train: bool, input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Return the default augmentation pipeline."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=90, translate=(0.05, 0.05), scale=(0.9, 1.1)
                ),
                transforms.RandomCrop((input_size, input_size)),
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
            transforms.CenterCrop((input_size, input_size)),
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
