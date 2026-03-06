"""Model building with backbone registry pattern."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import psutil
import torch
import torch.nn as nn
from torchvision import models

from .constants import NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for a backbone architecture."""

    name: str
    build_fn: Callable[[bool], nn.Module]
    classifier_layer_index: int
    description: str


def _build_mobilenet_v3_small(pretrained: bool) -> nn.Module:
    weights = (
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return models.mobilenet_v3_small(weights=weights)


def _build_mobilenet_v3_large(pretrained: bool) -> nn.Module:
    weights = (
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return models.mobilenet_v3_large(weights=weights)


def _build_efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = (
        models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return models.efficientnet_b0(weights=weights)


def _build_efficientnet_b2(pretrained: bool) -> nn.Module:
    weights = (
        models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return models.efficientnet_b2(weights=weights)


BACKBONE_REGISTRY: dict[str, BackboneConfig] = {
    "mobilenet_v3_small": BackboneConfig(
        name="mobilenet_v3_small",
        build_fn=_build_mobilenet_v3_small,
        classifier_layer_index=3,
        description="MobileNetV3-Small (2.5M params, ~4MB ONNX)",
    ),
    "mobilenet_v3_large": BackboneConfig(
        name="mobilenet_v3_large",
        build_fn=_build_mobilenet_v3_large,
        classifier_layer_index=3,
        description="MobileNetV3-Large (5.4M params, ~9MB ONNX)",
    ),
    "efficientnet_b0": BackboneConfig(
        name="efficientnet_b0",
        build_fn=_build_efficientnet_b0,
        classifier_layer_index=1,
        description="EfficientNet-B0 (5.3M params, ~11MB ONNX)",
    ),
    "efficientnet_b2": BackboneConfig(
        name="efficientnet_b2",
        build_fn=_build_efficientnet_b2,
        classifier_layer_index=1,
        description="EfficientNet-B2 (9.1M params, ~18MB ONNX)",
    ),
}


def build_model(
    backbone: str = "mobilenet_v3_large",
    pretrained: bool = True,
) -> nn.Module:
    """Build a model with the specified backbone.

    Replaces the final classification layer for NUM_CLASSES output.
    """
    if backbone not in BACKBONE_REGISTRY:
        available = ", ".join(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {available}"
        )

    config = BACKBONE_REGISTRY[backbone]
    model = config.build_fn(pretrained)

    layer_idx = config.classifier_layer_index
    in_features: int = model.classifier[layer_idx].in_features
    model.classifier[layer_idx] = nn.Linear(in_features, NUM_CLASSES)

    return model


def _get_rss_bytes() -> int:
    """Return current process RSS in bytes."""
    return int(psutil.Process(os.getpid()).memory_info().rss)


def measure_training_memory(
    backbone: str,
    batch_size: int,
    input_size: int,
    device: torch.device,
) -> int:
    """Measure system RAM needed for training via a dry-run forward+backward.

    Returns 0 for CUDA (GPU VRAM is separate from system RAM).
    For MPS/CPU, performs a single training step and measures the RSS delta.
    """
    if device.type == "cuda":
        return 0

    gc.collect()
    rss_before = _get_rss_bytes()

    model = build_model(backbone=backbone, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    target = torch.zeros(batch_size, NUM_CLASSES, device=device)

    # Measure after forward (peak: weights + optimizer states + activations)
    logits = model(dummy)
    loss = criterion(logits, target)
    rss_peak = _get_rss_bytes()

    loss.backward()
    optimizer.step()

    gc.collect()
    rss_after = _get_rss_bytes()

    # Use the higher of peak (with activations) and post-step (with gradients)
    total = max(rss_peak - rss_before, rss_after - rss_before, 0)

    # Release MPS allocator cache back to OS so that subsequent
    # psutil.virtual_memory().available readings are accurate.
    del model, optimizer, criterion, dummy, target, logits, loss
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    logger.info(
        "Measured training memory: %.0f MB (device=%s)",
        total / 1024 / 1024,
        device.type,
    )
    return total
