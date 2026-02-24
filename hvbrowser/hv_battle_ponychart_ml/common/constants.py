"""Shared constants for PonyChart ML scripts."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
RAWIMAGE_DIR = SCRIPT_DIR / "rawimage"
LABELS_FILE = SCRIPT_DIR / "labels.json"
OUTPUT_ONNX = SCRIPT_DIR / "model.onnx"
OUTPUT_THRESHOLDS = SCRIPT_DIR / "thresholds.json"

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
