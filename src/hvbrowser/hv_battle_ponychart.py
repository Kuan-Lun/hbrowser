import time
import os
import sys
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from hbrowser.beep import beep_os_independent

from .hv import HVDriver


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def _save_pony_chart_image(self):
        """保存 PonyChart 圖片到 pony_chart 資料夾"""
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

        # 如果是相對路徑或本地檔案，使用 selenium 截圖
        if img_src.startswith("./") or not img_src.startswith("http"):
            # 對 img 元素進行截圖
            img_element.screenshot(filepath)
        else:
            raise ValueError("Unsupported image source format.")

    def _check(self) -> bool:
        return self.driver.find_elements(By.ID, "riddlesubmit") != []

    def check(self) -> bool:
        isponychart: bool = self._check()
        if not isponychart:
            return isponychart

        # 當檢測到 PonyChart 時，保存圖片
        self._save_pony_chart_image()

        beep_os_independent()

        waitlimit: float = 100
        while waitlimit > 0 and self._check():
            time.sleep(0.1)
            waitlimit -= 0.1

        if waitlimit <= 0:
            print("PonyChart check timeout, please check your network connection.")

        time.sleep(1)

        return isponychart
