# PonyChart ML

PonyChart 角色辨識模型，用於自動辨識 HentaiVerse 戰鬥中出現的 PonyChart 圖片中的角色。

## 目錄結構

```
hv_battle_ponychart_ml/
├── common/                     # 共用模組 (SOLID 重構)
│   ├── __init__.py             # Re-export 所有 symbol
│   ├── constants.py            # 常數 (NUM_CLASSES, CLASS_NAMES, paths)
│   ├── device.py               # 裝置偵測
│   ├── data.py                 # 資料載入、Dataset、transforms、splitting
│   ├── model.py                # Backbone registry + build_model()
│   ├── training.py             # 訓練迴圈、evaluate、threshold 優化
│   └── export.py               # ONNX 匯出
├── rawimage/                   # 訓練用原始圖片 (PNG)
├── labels.json                 # 標註資料 {"rawimage/filename.png": [1,3]}
├── model.onnx                  # 訓練產出的 ONNX 模型 (推論用)
├── thresholds.json             # 各角色的分類閾值 (推論用)
├── label_images.py             # 圖片標註工具 (GUI)
├── train.py                    # 模型訓練腳本
├── compare_backbones.py        # Backbone 架構比較
├── compare_crops.py            # 裁切圖片效果分析
├── analyze_augmentations.py    # 資料增強 ablation study
├── learning_curve.py           # Learning curve 分析 + power-law 外推
├── search_batch_lr.py          # Batch size / LR 超參數搜尋
└── README.md
```

## 標籤對照

| 編號 | 角色 |
|------|------|
| 1 | Twilight Sparkle |
| 2 | Rarity |
| 3 | Fluttershy |
| 4 | Rainbow Dash |
| 5 | Pinkie Pie |
| 6 | Applejack |

## 工作流程

### 1. 收集圖片

將新的 PonyChart 截圖 (PNG) 放入 `rawimage/` 資料夾。

### 2. 標註圖片

```bash
# 需要 Pillow: pip install Pillow
python hvbrowser/hv_battle_ponychart_ml/label_images.py
```

操作方式：
- `1`~`6`: 加/取消對應角色標籤
- `A` / `D`: 上一張 / 下一張
- `S`: 儲存目前標籤

標註結果會即時更新到 `labels.json`。

### 3. 訓練模型

```bash
# 安裝訓練依賴 (只需一次)
uv pip install torch torchvision scikit-learn onnxscript

# 執行訓練 (預設使用 MobileNetV3-Small)
uv run python -m hvbrowser.hv_battle_ponychart_ml.train --epochs 50

# 使用其他 backbone
uv run python -m hvbrowser.hv_battle_ponychart_ml.train \
    --backbone mobilenet_v3_large
```

訓練完成後會直接覆寫 `model.onnx` 和 `thresholds.json`，下次推論自動使用新模型。

### 參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--epochs` | 35 | Phase 2 最大訓練輪數 |
| `--batch-size` | 32 | 批次大小 |
| `--seed` | 42 | 隨機種子 |
| `--device` | auto | `cpu` / `cuda` / `mps` / `auto` |
| `--backbone` | mobilenet_v3_large | 見下方支援的 backbone |

## 支援的 Backbone

| Backbone | 參數量 | ONNX 大小 | 說明 |
|----------|--------|-----------|------|
| `mobilenet_v3_small` | 2.5M | ~4MB | 輕量快速 |
| `mobilenet_v3_large` | 5.4M | ~9MB | 預設，精度最高 |
| `efficientnet_b0` | 5.3M | ~11MB | 精度接近 Large，但訓練較慢 |

所有 backbone 都使用 ImageNet 預訓練權重 + transfer learning。
推論端使用 ONNX Runtime，backbone 更換後只需重新匯出 `model.onnx`，推論程式碼不需改動。

## 分析腳本

所有分析腳本都支援 `--backbone` 參數：

```bash
# 比較三種 backbone 的效果
uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_backbones

# 分析裁切圖片的影響
uv run python -m hvbrowser.hv_battle_ponychart_ml.compare_crops

# 資料增強 ablation study
uv run python -m hvbrowser.hv_battle_ponychart_ml.analyze_augmentations

# Learning curve 分析 (估算增加資料的邊際效益)
uv run python -m hvbrowser.hv_battle_ponychart_ml.learning_curve

# Batch size / LR 超參數搜尋
uv run python -m hvbrowser.hv_battle_ponychart_ml.search_batch_lr
```

## 模型架構

- **Backbone**: 可選 MobileNetV3-Small/Large 或 EfficientNet-B0 (ImageNet 預訓練)
- **輸出**: 6 個 sigmoid 節點 (多標籤分類)
- **推論引擎**: ONNX Runtime (CPU)
- **推論速度**: 3-21ms / 張
