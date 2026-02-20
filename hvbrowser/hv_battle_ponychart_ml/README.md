# PonyChart ML

PonyChart 角色辨識模型，用於自動辨識 HentaiVerse 戰鬥中出現的 PonyChart 圖片中的角色。

## 目錄結構

```
hv_battle_ponychart_ml/
├── rawimage/           # 訓練用原始圖片 (PNG)
├── labels.json         # 標註資料 {"rawimage/filename.png": [1,3]}
├── model.onnx          # 訓練產出的 ONNX 模型 (推論用)
├── thresholds.json     # 各角色的分類閾值 (推論用)
├── label_images.py     # 圖片標註工具 (GUI)
├── train.py            # 模型訓練腳本
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

# 執行訓練
uv run python -m hvbrowser.hv_battle_ponychart_ml.train --epochs 50
```

訓練完成後會直接覆寫 `model.onnx` 和 `thresholds.json`，下次推論自動使用新模型。

### 參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--epochs` | 35 | Phase 2 最大訓練輪數 |
| `--batch-size` | 32 | 批次大小 |
| `--seed` | 42 | 隨機種子 |
| `--device` | auto | `cpu` / `cuda` / `mps` / `auto` |

## 模型架構

- **Backbone**: MobileNetV3-Small (ImageNet 預訓練)
- **輸出**: 6 個 sigmoid 節點 (多標籤分類)
- **推論引擎**: ONNX Runtime (CPU)
- **推論速度**: 3-21ms / 張
