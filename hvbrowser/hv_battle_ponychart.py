import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import cv2 as cv
import numpy as np
from selenium.webdriver.common.by import By

from hbrowser.beep import beep_os_independent
from hbrowser.gallery.utils import setup_logger

from .hv import HVDriver

logger = setup_logger(__name__)

HOG_CFG: dict[str, tuple[int, int] | int] = {
    "win_size": (192, 192),
    "cell_size": (8, 8),
    "block_size": (16, 16),
    "block_stride": (8, 8),
    "nbins": 9,
}
IMG_SIZE = (192, 192)
PREPROCESS = dict(
    equalize=False,
    quant_step=1,
    denoise=False,
    denoise_ksize=1,
    remove_specks=True,
    despeckle_win=1,
    despeckle_diff=1,
)


def _despeckle(
    gray: np.ndarray[Any, Any], win: int, diff_thr: int
) -> np.ndarray[Any, Any]:
    if gray is None or getattr(gray, "ndim", 2) != 2:
        return gray
    if win < 3:
        return gray
    if win % 2 == 0:
        win += 1
    med = cv.medianBlur(gray, win)
    diff = cv.absdiff(gray, med)
    mask = diff > diff_thr
    if not np.any(mask):
        return gray
    out = gray.copy()
    out[mask] = med[mask]
    return out


class _InlineModel:
    def __init__(self) -> None:
        self.loaded = False
        self.W = None
        self.b = None
        self.mu = None
        self.sd = None
        self.classes: list[str] = []
        self.thresholds: dict[str, float] = {}

    def _dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "hv_battle_ponychart_ml")

    def load(self) -> None:
        if self.loaded:
            return
        d = self._dir()
        weights = os.path.join(d, "weights.npz")
        th = os.path.join(d, "thresholds.json")
        data = np.load(weights, allow_pickle=True)
        self.W = data["W"].astype(np.float32)
        self.b = data["b"].astype(np.float32)
        self.mu = data["scaler_mean"].astype(np.float32)
        self.sd = data["scaler_scale"].astype(np.float32)
        self.classes = [c for c in data["classes"]]
        with open(th, encoding="utf-8") as f:
            self.thresholds = json.load(f)
        self.loaded = True

    def _hog(self) -> cv.HOGDescriptor:
        win_size = HOG_CFG["win_size"]
        block_size = HOG_CFG["block_size"]
        block_stride = HOG_CFG["block_stride"]
        cell_size = HOG_CFG["cell_size"]
        nbins = HOG_CFG["nbins"]

        assert isinstance(win_size, tuple)
        assert isinstance(block_size, tuple)
        assert isinstance(block_stride, tuple)
        assert isinstance(cell_size, tuple)
        assert isinstance(nbins, int)

        return cv.HOGDescriptor(
            win_size,
            block_size,
            block_stride,
            cell_size,
            nbins,
        )

    def _pre(self, bgr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, IMG_SIZE, interpolation=cv.INTER_AREA)
        q = PREPROCESS["quant_step"]
        try:
            qv = int(q)
        except Exception:
            qv = 1
        if qv > 1:
            gray = (gray // qv) * qv
        if PREPROCESS["denoise"]:
            k = PREPROCESS["denoise_ksize"]
            if isinstance(k, int) and k >= 3 and k % 2 == 1:
                gray = cv.medianBlur(gray, k)
        if PREPROCESS["equalize"]:
            gray = cv.equalizeHist(gray)
        if PREPROCESS["remove_specks"]:
            gray = _despeckle(
                gray,
                PREPROCESS["despeckle_win"],
                PREPROCESS["despeckle_diff"],
            )
        return gray

    def predict(
        self, img_path: str, min_k: int = 1, max_k: int = 3
    ) -> tuple[list[str], dict[str, float]]:
        self.load()
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"無法讀取圖片: {img_path}")
        gray = self._pre(img)
        hog_result = self._hog().compute(gray)
        x = (
            np.array(hog_result).reshape(-1).astype(np.float32)
            if hog_result is not None
            else np.array([])
        )
        if self.mu is None:
            raise RuntimeError("Model not loaded properly: mu is None")
        x = (x - self.mu) / (self.sd + 1e-8) if self.sd is not None else x - self.mu
        logits = self.W @ x + self.b
        probs = 1.0 / (1.0 + np.exp(-logits))
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
                    ((c, scores[c]) for c in picked), key=lambda kv: kv[1], reverse=True
                )[:max_k]
            ]
        return picked, scores


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver = driver
        self._model = _InlineModel()

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
