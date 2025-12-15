# Gallery 模組重構說明

本目錄是重構後的 gallery 模組，遵循 SOLID 原則進行了模組化拆分。

## 目錄結構

```
gallery/
├── __init__.py              # 模組導出
├── models.py                # 數據模型（Tag, DriverPass）
├── driver_base.py           # Driver 抽象基類
├── eh_driver.py             # EHDriver 實現
├── exh_driver.py            # ExHDriver 實現
├── captcha/                 # 驗證碼處理模組
│   ├── models.py            # 驗證碼數據模型
│   ├── constants.py         # 驗證碼常量
│   ├── detector.py          # 驗證碼檢測器（核心邏輯）
│   ├── solver_interface.py  # 解決器抽象接口
│   ├── manager.py           # 驗證碼管理器
│   └── adapters/            # 第三方適配器
│       └── twocaptcha_adapter.py  # ⚠️ TwoCaptcha 適配器（可刪除）
├── browser/                 # 瀏覽器相關模組
│   ├── factory.py           # WebDriver 工廠
│   └── ban_handler.py       # IP ban 處理
└── utils/                   # 工具函數
    ├── log.py               # 日誌工具
    ├── url.py               # URL 工具
    └── window.py            # 視窗工具
```

## 設計原則

### 1. 單一職責原則 (SRP)
每個模組只負責一項功能：
- `captcha/`: 驗證碼處理
- `browser/`: 瀏覽器初始化和管理
- `utils/`: 工具函數

### 2. 開放封閉原則 (OCP)
- 通過接口擴展功能，無需修改核心代碼
- 例如：添加新的驗證碼解決器只需實現 `CaptchaSolver` 接口

### 3. 依賴反轉原則 (DIP)
- 核心邏輯依賴抽象接口，不依賴具體實現
- 第三方服務通過適配器隔離在 `adapters/` 目錄

## 移除 TwoCaptcha 依賴

當不再需要 TwoCaptcha 時，執行以下步驟：

### 1. 刪除適配器文件
```bash
rm src/hbrowser/gallery/captcha/adapters/twocaptcha_adapter.py
```

### 2. 修改 captcha/__init__.py
```python
# 移除這行
from .adapters import TwoCaptchaAdapter
```

### 3. 修改 driver_base.py
```python
# 移除導入
- from .captcha import CaptchaManager, TwoCaptchaAdapter
+ from .captcha import CaptchaManager

# 修改初始化（或改用其他解決器）
- solver = TwoCaptchaAdapter()
+ solver = YourNewAdapter()  # 使用新的解決器
```

### 4. 更新依賴
```toml
# pyproject.toml
dependencies = [
-   "2captcha-python",
]
```

## 向後兼容

原始的 `gallery.py` 文件已改為兼容層，從子模組重新導出所有類：

```python
# 舊代碼仍然可用
from hbrowser import EHDriver, ExHDriver, Tag

# 新代碼可以直接從子模組導入
from hbrowser.gallery.captcha import CaptchaManager
from hbrowser.gallery.utils import matchurl
```

## 擴展指南

### 添加新的驗證碼解決器

1. 創建新的適配器文件：
```python
# captcha/adapters/your_adapter.py
from ..solver_interface import CaptchaSolver
from ..models import ChallengeDetection, SolveResult

class YourAdapter(CaptchaSolver):
    def solve(self, challenge: ChallengeDetection, driver) -> SolveResult:
        # 實現你的解決邏輯
        pass
```

2. 在 `driver_base.py` 中使用：
```python
from .captcha.adapters.your_adapter import YourAdapter
solver = YourAdapter()
```

### 添加新的工具函數

在 `utils/` 目錄下創建新文件，然後在 `utils/__init__.py` 中導出。
