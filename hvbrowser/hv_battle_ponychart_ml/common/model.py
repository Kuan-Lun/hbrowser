"""Model building with backbone registry pattern."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn
from torchvision import models

from .constants import NUM_CLASSES


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
