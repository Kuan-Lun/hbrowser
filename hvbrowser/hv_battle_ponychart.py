import os
import sys
import time
from datetime import datetime
from typing import Any

from ponychart_classifier import predict
from selenium.webdriver.common.by import By

from hbrowser.gallery.utils import setup_logger
from hbrowser.notify import notify

from .hv import HVDriver

logger = setup_logger(__name__)


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver = driver

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
    def _auto_answer(self, img_path: str) -> frozenset[str] | None:
        """最簡化：模型推論 -> 依角色名稱精確比對 label 文字 -> 點擊。"""
        result = predict(img_path)
        labels: frozenset[str] = result.labels
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

        notify("PonyChart", "PonyChart detected")

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
