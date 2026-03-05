"""Train/val/test splitting strategies."""

from __future__ import annotations

import os
from collections import defaultdict

from sklearn.model_selection import train_test_split

from .sampling import get_base_timestamp


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
