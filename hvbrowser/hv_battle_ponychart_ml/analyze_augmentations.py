"""
分析空間增強（水平翻轉、垂直翻轉、旋轉角度）對訓練效果的影響。

透過 ablation study 逐一啟用各空間增強，比較與 baseline 的 F1 差異，
判斷哪些增強有正面效果、最佳旋轉角度為何。

七組實驗（僅空間增強不同，其餘 augmentation 皆相同）：
  1. none    — 無翻轉、無旋轉（baseline）
  2. hflip   — + 水平翻轉
  3. vflip   — + 垂直翻轉
  4. rot15   — + 旋轉 15 度
  5. rot45   — + 旋轉 45 度
  6. rot90   — + 旋轉 90 度
  7. current — 水平翻轉 + 垂直翻轉 + 旋轉 90 度（目前 train.py 設定）

使用方式：
  uv run python -m hvbrowser.hv_battle_ponychart_ml.analyze_augmentations
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .common import (
    BACKBONE,
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    SEED,
    evaluate,
    get_device,
    get_performance_cpu_count,
    get_transforms,
    group_stratified_split,
    load_samples,
    make_dataloader,
    train_model,
)
from .common.data import PonyChartDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation configs (unique to this script)
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
    spatial.append(
        transforms.RandomAffine(
            degrees=cfg.degrees,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        )
    )
    return transforms.Compose(
        [
            *spatial,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    num_workers = get_performance_cpu_count()
    logger.info("Device: %s  Workers: %d", device, num_workers)

    # Load data
    all_samples = load_samples()
    logger.info("Total samples: %d", len(all_samples))

    train_idx, test_idx = group_stratified_split(
        all_samples, test_size=0.15, seed=SEED
    )
    train_samples = [all_samples[i] for i in train_idx]
    test_samples = [all_samples[i] for i in test_idx]
    logger.info("Train: %d  Test: %d", len(train_samples), len(test_samples))

    # From train, split out val for early stopping
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

    # Test set (shared)
    test_ds = PonyChartDataset(test_samples, get_transforms(is_train=False))
    test_loader = make_dataloader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=num_workers, device=device,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ── Run all experiments ──
    results: dict[str, dict[str, Any]] = {}
    for cfg in EXPERIMENTS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_tf = build_train_transform(cfg)
        model, thresholds = train_model(
            sub_train_samples,
            val_samples,
            device,
            num_workers,
            cfg.name,
            backbone=BACKBONE,
            train_transform=train_tf,
        )
        result = evaluate(
            model, test_loader, criterion, device, thresholds
        )
        results[cfg.name] = result
        logger.info(
            "  >> %s test F1=%.4f  thresholds=%s",
            cfg.name,
            result["macro_f1"],
            dict(zip(CLASS_NAMES, thresholds)),
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
    logger.info(
        "%-12s  %-10s  %-10s  %s",
        "Experiment",
        "Macro F1",
        "Delta",
        "Config",
    )
    logger.info("-" * 75)
    for cfg in EXPERIMENTS:
        r = results[cfg.name]
        delta = r["macro_f1"] - baseline_f1
        delta_str = (
            f"{delta:+.4f}" if cfg.name != "none" else "baseline"
        )
        desc_parts = []
        if cfg.hflip:
            desc_parts.append("HFlip")
        if cfg.vflip:
            desc_parts.append("VFlip")
        if cfg.degrees > 0:
            desc_parts.append(f"Rot({cfg.degrees:.0f})")
        desc = (
            " + ".join(desc_parts) if desc_parts else "(no spatial aug)"
        )
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
    logger.info(
        "  " + "-" * (20 + 14 * len(EXPERIMENTS))
    )
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
    logger.info(
        "  %-20s  %-12s  %-12s", "Class", "HFlip", "VFlip"
    )
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
            "  Rotation %3d: %+.4f (F1=%.4f)", deg, delta, rot_f1
        )
        if results[rname]["macro_f1"] > best_rot_f1:
            best_rot_f1 = results[rname]["macro_f1"]
            best_rot_name = rname

    if best_rot_name == "none":
        logger.info(
            "  >> 所有旋轉角度皆無正面效果，建議移除旋轉增強"
        )
    else:
        best_deg = {"rot15": 15, "rot45": 45, "rot90": 90}[best_rot_name]
        delta = best_rot_f1 - baseline_f1
        logger.info(
            "  >> 最佳旋轉角度: %d (%+.4f F1)", best_deg, delta
        )
        if delta > 0.005:
            logger.info("  >> 建議使用 %d 旋轉", best_deg)
        else:
            logger.info("  >> 效果有限 (%.4f)，旋轉非必要", delta)

    # Per-class rotation impact
    logger.info("")
    logger.info("  Per-class rotation impact (F1 delta vs baseline):")
    logger.info(
        "  %-20s  %-10s  %-10s  %-10s", "Class", "15", "45", "90"
    )
    for i, name in enumerate(CLASS_NAMES):
        deltas = []
        for rname, _ in rot_configs:
            d = (
                results[rname]["per_class_f1"][i]
                - baseline_per_class[i]
            )
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
    sum_individual = (
        hflip_delta + vflip_delta + (best_rot_f1 - baseline_f1)
    )
    interaction = current_delta - sum_individual

    logger.info(
        "  Current config (HFlip+VFlip+Rot90): %+.4f", current_delta
    )
    logger.info(
        "  Sum of individual effects:           %+.4f", sum_individual
    )
    logger.info(
        "  Interaction effect:                  %+.4f", interaction
    )
    if interaction > 0.005:
        logger.info("  >> 組合使用有正向交互作用，建議同時啟用")
    elif interaction < -0.005:
        logger.info(
            "  >> 組合使用有負向交互作用，建議精簡增強組合"
        )
    else:
        logger.info(
            "  >> 交互作用微小，各增強可獨立決定是否啟用"
        )

    # ── Final recommendation ──
    logger.info("")
    logger.info("=" * 90)
    logger.info("RECOMMENDATION")
    logger.info("=" * 90)

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_result = results[best_name]
    logger.info(
        "  最佳實驗: %s (Macro F1=%.4f)",
        best_name,
        best_result["macro_f1"],
    )
    logger.info("")

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
        logger.info(
            "  建議移除所有空間增強（翻轉與旋轉），僅保留:"
        )
        logger.info("    - RandomCrop")
        logger.info("    - ColorJitter")
        logger.info("    - GaussianBlur")
        logger.info("    - RandomErasing")

    # 與目前設定比較
    logger.info("")
    diff = best_result["macro_f1"] - results["current"]["macro_f1"]
    if abs(diff) < 0.003:
        logger.info(
            "  目前設定 (current) 已接近最佳，無需調整"
        )
    elif diff > 0:
        logger.info(
            "  切換至 '%s' 可提升 F1 約 %+.4f", best_name, diff
        )
    else:
        logger.info(
            "  目前設定 (current) 即為最佳或接近最佳"
        )

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
