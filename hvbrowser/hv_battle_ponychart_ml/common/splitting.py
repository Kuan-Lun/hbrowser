"""Train/val/test splitting strategies using hash-based assignment.

Each group's train/val assignment is determined solely by hashing its key,
making the split stable regardless of how many other samples are added or
removed.  This prevents data leakage when resuming training with new data.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict

from .sampling import get_base_timestamp

_HASH_MODULUS = 1000


def _is_val_group(group_key: str, val_ratio: float) -> bool:
    """Determine if a group belongs to the val set via hash."""
    h = hashlib.md5(group_key.encode()).hexdigest()
    return (int(h, 16) % _HASH_MODULUS) / _HASH_MODULUS < val_ratio


def _build_groups(
    samples: list[tuple[str, list[int]]],
) -> dict[str, list[int]]:
    """Build a mapping from base timestamp group key to sample indices."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path).replace(".png", "").replace(".jpg", "")
        parts = fname.split("_")
        base = "_".join(parts[:4])
        groups[base].append(idx)
    return groups


def group_hash_split(
    samples: list[tuple[str, list[int]]],
    test_size: float = 0.15,
) -> tuple[list[int], list[int]]:
    """Hash-based group split returning (train_idx, val_idx).

    Each group's assignment depends only on its own key, so adding or
    removing samples never changes existing assignments.
    """
    groups = _build_groups(samples)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for gk, indices in groups.items():
        if _is_val_group(gk, test_size):
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return train_idx, val_idx


def split_by_groups(
    samples: list[tuple[str, list[int]]],
    test_size: float,
) -> tuple[list[str], list[str]]:
    """Split timestamp groups, returning group keys instead of indices."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    train_gk: list[str] = []
    test_gk: list[str] = []
    for gk in groups:
        if _is_val_group(gk, test_size):
            test_gk.append(gk)
        else:
            train_gk.append(gk)
    return train_gk, test_gk
