"""Shared constants for PonyChart ML scripts."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
RAWIMAGE_DIR = SCRIPT_DIR / "rawimage"
LABELS_FILE = SCRIPT_DIR / "labels.json"
OUTPUT_ONNX = SCRIPT_DIR / "model.onnx"
OUTPUT_CHECKPOINT = SCRIPT_DIR / "checkpoint.pt"
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

# Training hyperparameters (single source of truth)
BACKBONE = "efficientnet_b0"
BATCH_SIZE = 32
SEED = 42
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 100
PATIENCE = 12
MIN_DELTA = 0.005
LR_HEAD = 1e-3
LR_FEATURES = 3e-5
LR_CLASSIFIER = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-7

# Reduced settings for hyperparameter search (derived from main settings)
SEARCH_PHASE1_EPOCHS = PHASE1_EPOCHS
SEARCH_PHASE2_EPOCHS = PHASE2_EPOCHS
SEARCH_PATIENCE = PATIENCE
