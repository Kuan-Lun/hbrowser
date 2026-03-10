"""Train/val/test splitting strategies using hash-based assignment.

Each group's assignment is determined solely by hashing its key,
making the split stable regardless of how many other samples are added or
removed.  This prevents data leakage when resuming training with new data.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass

from .sampling import get_base_timestamp

_HASH_MODULUS = 1000


@dataclass
class GroupSplit:
    """Result of a hash-based group split."""

    train: list[str]
    val: list[str]
    test: list[str]


def _hash_value(group_key: str) -> float:
    """Return a deterministic hash in [0, 1) for a group key."""
    h = hashlib.md5(group_key.encode()).hexdigest()
    return (int(h, 16) % _HASH_MODULUS) / _HASH_MODULUS


def _ratios_to_thresholds(ratios: list[float]) -> list[float]:
    """Convert sequential relative ratios to absolute hash thresholds.

    Each ratio cuts from the remaining pool.  For ``[0.20, 0.15]``::

        threshold[0] = 0.20
        threshold[1] = 0.20 + 0.80 * 0.15 = 0.32
    """
    thresholds: list[float] = []
    cumulative = 0.0
    for ratio in ratios:
        cumulative += (1.0 - cumulative) * ratio
        thresholds.append(cumulative)
    return thresholds


def _split_n_way(
    group_keys: list[str],
    ratios: list[float],
) -> list[list[str]]:
    """Assign *group_keys* into ``len(ratios) + 1`` buckets in one pass.

    *ratios* are sequential relative cuts (each is a fraction of the
    remaining pool).  The last bucket collects whatever is left over.
    """
    thresholds = _ratios_to_thresholds(ratios)
    buckets: list[list[str]] = [[] for _ in range(len(ratios) + 1)]
    for gk in group_keys:
        h = _hash_value(gk)
        assigned = len(thresholds)  # default: remainder bucket
        for i, t in enumerate(thresholds):
            if h < t:
                assigned = i
                break
        buckets[assigned].append(gk)
    return buckets


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
        if _hash_value(gk) < test_size:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return train_idx, val_idx


def split_by_groups(
    samples: list[tuple[str, list[int]]],
    test_size: float,
    val_size: float = 0.0,
) -> GroupSplit:
    """Split timestamp groups into train / val / test sets.

    Parameters
    ----------
    samples:
        Full sample list.
    test_size:
        Fraction of the total pool allocated to the test set.
    val_size:
        Fraction of the *remaining* pool (after test) allocated to
        the validation set.  Defaults to 0 (no validation split).

    Returns
    -------
    GroupSplit
        Group keys for each split.  When *val_size* is 0 the
        ``val`` field is an empty list.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        fname = os.path.basename(path)
        base = get_base_timestamp(fname)
        groups[base].append(idx)

    group_keys = list(groups.keys())

    if val_size > 0:
        buckets = _split_n_way(group_keys, [test_size, val_size])
        return GroupSplit(train=buckets[2], val=buckets[1], test=buckets[0])

    buckets = _split_n_way(group_keys, [test_size])
    return GroupSplit(train=buckets[1], val=[], test=buckets[0])
