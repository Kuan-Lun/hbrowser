"""Training-specific constants for PonyChart ML scripts."""

from __future__ import annotations

from pathlib import Path

from hvbrowser.hv_battle_ponychart_ml.model_spec import (  # noqa: F401
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    PRE_RESIZE,
)

__all__ = [
    "CLASS_NAMES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_SIZE",
    "NUM_CLASSES",
    "PRE_RESIZE",
]

SCRIPT_DIR = Path(__file__).resolve().parent.parent
RAWIMAGE_DIR = SCRIPT_DIR / "rawimage"
LABELS_FILE = SCRIPT_DIR / "labels.json"
OUTPUT_ONNX = SCRIPT_DIR / "model.onnx"
OUTPUT_CHECKPOINT = SCRIPT_DIR / "checkpoint.pt"
OUTPUT_THRESHOLDS = SCRIPT_DIR / "thresholds.json"

# Training hyperparameters (single source of truth)
BACKBONE = "efficientnet_b0"
BATCH_SIZE = 64
SEED = 42
PHASE1_EPOCHS = 30
PHASE1_PATIENCE = 5
PHASE2_EPOCHS = 100
PHASE2_PATIENCE = 12
MIN_DELTA = 0.005
LR_HEAD = 4e-3
LR_FEATURES = 1.2e-4
LR_CLASSIFIER = 1.2e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-7

# Data split ratios
HOLDOUT_TEST_SIZE = 0.20
VAL_SIZE = 0.15

# Resume vs from-scratch threshold (updated by compare_resume_scratch analysis)
# 以「上次 from-scratch 訓練時的樣本數」為基準計算累積 ratio：
#   cumulative_ratio = (目前樣本數 - n_samples_at_full_train) / n_samples_at_full_train
# 超過此值時自動切換為 from-scratch 訓練，避免多次增量訓練累積偏差。
# 例：full_train 用 1000 張，現在有 1400 張 → ratio=0.4 < 0.5 → resume
# 例：full_train 用 1000 張，現在有 1600 張 → ratio=0.6 > 0.5 → from-scratch
RETRAIN_NEW_DATA_RATIO = 0.5

# Reduced settings for hyperparameter search (derived from main settings)
SEARCH_PHASE1_EPOCHS = PHASE1_EPOCHS
SEARCH_PHASE2_EPOCHS = PHASE2_EPOCHS
SEARCH_PATIENCE = PHASE2_PATIENCE
