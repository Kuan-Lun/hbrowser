import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import cv2 as cv
import numpy as np
import onnxruntime as ort
from selenium.webdriver.common.by import By

from hbrowser.beep import beep_os_independent
from hbrowser.gallery.utils import setup_logger

from .hv import HVDriver
from .hv_battle_ponychart_ml.common.constants import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PRE_RESIZE,
)

logger = setup_logger(__name__)

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)


class _InlineModel:
    def __init__(self) -> None:
        self.loaded = False
        self.session: Any = None
        self.classes: list[str] = list(CLASS_NAMES)
        self.thresholds: dict[str, float] = {}

    def _dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "hv_battle_ponychart_ml")

    def load(self) -> None:
        if self.loaded:
            return

        d = self._dir()
        model_path = os.path.join(d, "model.onnx")
        th_path = os.path.join(d, "thresholds.json")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        with open(th_path, encoding="utf-8") as f:
            self.thresholds = json.load(f)
        self.loaded = True

    def _preprocess(self, bgr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """BGR 圖片 -> NCHW float32 tensor (matching training transforms)."""
        resized = cv.resize(bgr, (PRE_RESIZE, PRE_RESIZE), interpolation=cv.INTER_AREA)
        offset = (PRE_RESIZE - INPUT_SIZE) // 2
        cropped = resized[offset:offset + INPUT_SIZE, offset:offset + INPUT_SIZE]
        rgb = cv.cvtColor(cropped, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW -> NCHW
        return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(
        self, img_path: str, min_k: int = 1, max_k: int = 3
    ) -> tuple[list[str], dict[str, float]]:
        self.load()
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"無法讀取圖片: {img_path}")

        input_tensor = self._preprocess(img)
        input_name: str = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: input_tensor})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))

        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        picked = [c for c, p in scores.items() if p >= self.thresholds.get(c, 0.5)]
        if len(picked) < min_k:
            picked = [
                c
                for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
                    :max_k
                ]
            ]
        elif len(picked) > max_k:
            picked = [
                c
                for c, _ in sorted(
                    ((c, scores[c]) for c in picked),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:max_k]
            ]
        return picked, scores


_model = _InlineModel()


def preload_model() -> None:
    """預先載入 ONNX 模型，在 BattleDriver 初始化時呼叫以提早發現依賴問題。"""
    try:
        _model.load()
    except ImportError as e:
        msg = "onnxruntime 載入失敗。"
        if sys.platform == "win32" and "DLL load failed" in str(e):
            msg += (
                "\n可能原因：缺少 Microsoft Visual C++ Redistributable。"
                "\n請至 https://aka.ms/vs/17/release/vc_redist.x64.exe 下載安裝後重試。"
            )
        else:
            msg += "\n請執行: pip install onnxruntime"
        raise RuntimeError(msg) from e


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver = driver
        self._model = _model

    @property
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    def _save_pony_chart_image(self) -> str:
        """保存 PonyChart 圖片到 pony_chart 資料夾，回傳檔案路徑"""
        # 尋找 riddleimage 中的 img 元素
        riddleimage_div = self.driver.find_element(By.ID, "riddleimage")
        img_element = riddleimage_div.find_element(By.TAG_NAME, "img")
        img_src = img_element.get_attribute("src")

        if not img_src:
            raise ValueError("無法獲取圖片 src")

        # 創建 pony_chart 資料夾 - 使用主執行檔案的目錄
        if (
            hasattr(sys.modules["__main__"], "__file__")
            and sys.modules["__main__"].__file__
        ):
            main_script_dir = os.path.dirname(
                os.path.abspath(sys.modules["__main__"].__file__)
            )
        else:
            raise RuntimeError("無法獲取主執行檔案的目錄，請確保在正確的環境中運行。")

        pony_chart_dir = os.path.join(main_script_dir, "pony_chart")
        if not os.path.exists(pony_chart_dir):
            os.makedirs(pony_chart_dir)

        # 生成唯一的檔名 (使用時間戳)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pony_chart_{timestamp}.png"
        filepath = os.path.join(pony_chart_dir, filename)

        img_element.screenshot(filepath)
        return filepath

    # ---------------- ML & 自動作答邏輯 ----------------
    def _auto_answer(self, img_path: str) -> list[str] | None:
        """最簡化：模型推論 -> 依角色名稱精確比對 label 文字 -> 點擊。"""
        labels, _ = self._model.predict(img_path)
        drv = self.driver
        # 收集所有 label.lc 並建立標準化對照
        label_elements = drv.find_elements(By.CSS_SELECTOR, "label.lc")
        norm_map = {}
        for lab in label_elements:
            txt = lab.text.strip()
            if txt:
                norm_map[txt.lower()] = lab
        clicked = []
        for name in labels:
            _lab = norm_map.get(name.lower().strip())
            if _lab is None:
                continue
            try:
                _lab.click()
                clicked.append(name)
            except Exception:
                pass
        logger.info(f"[PonyChart][ML] Prediction: {labels} -> Clicked text: {clicked}")
        return labels

    def _check(self) -> bool:
        return bool(self.driver.find_elements(By.ID, "riddlesubmit") != [])

    def check(self) -> bool:
        isponychart: bool = self._check()
        if not isponychart:
            return isponychart

        img_path = self._save_pony_chart_image()

        beep_os_independent()

        # 新增：自動填入答案（若失敗不影響原流程）
        try:
            self._auto_answer(img_path)
        except Exception as e:  # pragma: no cover
            logger.error(f"[PonyChart] Auto-check failed: {e}")

        # 原始等待邏輯 (約 15 秒) 保留
        waitlimit = 15
        while waitlimit > 0 and self._check():
            time.sleep(1)
            waitlimit -= 1

        if waitlimit <= 1 and self._check():
            logger.warning(
                "PonyChart check timeout, please check your network connection"
            )
            # 改為依送出按鈕顯示文字 (value="Submit Answer") 來尋找並點擊，
            # 失敗時回退用 id
            try:
                self.hvdriver.driver.find_element(
                    By.XPATH, "//input[@type='submit' and @value='Submit Answer']"
                ).click()
            except Exception:
                try:
                    self.hvdriver.driver.find_element(By.ID, "riddlesubmit").click()
                except Exception:
                    pass

        time.sleep(1)

        return isponychart
