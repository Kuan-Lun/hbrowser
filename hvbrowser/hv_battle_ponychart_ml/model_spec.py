"""Model specification constants shared by inference and training."""

from __future__ import annotations

NUM_CLASSES = 6
CLASS_NAMES = [
    "Twilight Sparkle",
    "Rarity",
    "Fluttershy",
    "Rainbow Dash",
    "Pinkie Pie",
    "Applejack",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 320
PRE_RESIZE = 384
