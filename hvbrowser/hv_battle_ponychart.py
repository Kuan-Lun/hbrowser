import asyncio
import os
import sys
from datetime import datetime
from typing import Any

from ponychart_classifier import predict

from hbrowser.gallery.utils import setup_logger
from hbrowser.notify import notify

from .hv import HVDriver

logger = setup_logger(__name__)


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver = driver

    @property
    def page(self) -> Any:
        return self.hvdriver.page

    async def _wait_for_image_loaded(
        self, timeout: float = 10.0, min_size: int = 50, stable_checks: int = 3
    ) -> None:
        """等待 PonyChart 圖片完全下載並穩定。

        PonyChart 頁面會先載入小 placeholder 再換成真正的圖片，
        所以光檢查 naturalWidth > 0 不夠，必須：
        1. 尺寸超過 min_size（避開 4x4 之類的 placeholder）
        2. src 連續 stable_checks 次都沒變（避開換圖瞬間）
        """
        get_state_js = (
            "(() => {"
            " const div = document.getElementById('riddleimage');"
            " if (!div) return null;"
            " const img = div.querySelector('img');"
            " if (!img || !img.complete) return null;"
            " return {"
            "  src: img.currentSrc || img.src,"
            "  w: img.naturalWidth,"
            "  h: img.naturalHeight"
            " };"
            "})()"
        )
        deadline = asyncio.get_event_loop().time() + timeout
        last_src: str | None = None
        stable_count = 0
        while asyncio.get_event_loop().time() < deadline:
            state = await self.page.evaluate(get_state_js)
            if (
                state
                and state.get("w", 0) >= min_size
                and state.get("h", 0) >= min_size
            ):
                src = state.get("src")
                if src == last_src:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        return
                else:
                    last_src = src
                    stable_count = 1
            else:
                last_src = None
                stable_count = 0
            await asyncio.sleep(0.1)
        raise TimeoutError("PonyChart image did not finish loading in time")

    async def _save_pony_chart_image(self) -> str:
        """保存 PonyChart 圖片到 pony_chart 資料夾，回傳檔案路徑。"""
        await self._wait_for_image_loaded()

        riddleimage_div = await self.page.select("#riddleimage")
        img_element = await riddleimage_div.query_selector("img")
        img_src = await img_element.apply("(el) => el.src || ''")

        if not img_src:
            raise ValueError("無法獲取圖片 src")

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pony_chart_{timestamp}.png"
        filepath = os.path.join(pony_chart_dir, filename)

        await img_element.save_screenshot(filepath)
        return filepath

    async def _auto_answer(self, img_path: str) -> frozenset[str] | None:
        """模型推論後依角色名稱比對 label 文字並點擊。"""
        result = predict(img_path)
        labels: frozenset[str] = result.labels
        label_elements = await self.page.select_all("label.lc", timeout=2)
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
                await _lab.click()
                clicked.append(name)
            except Exception:
                pass
        logger.info(f"[PonyChart][ML] Prediction: {labels} -> Clicked text: {clicked}")
        return labels

    async def _check(self) -> bool:
        elements = await self.page.query_selector_all("#riddlesubmit")
        return bool(elements)

    async def check(self) -> bool:
        isponychart: bool = await self._check()
        if not isponychart:
            return isponychart

        img_path = await self._save_pony_chart_image()

        notify("PonyChart", "PonyChart detected")

        try:
            await self._auto_answer(img_path)
        except Exception as e:  # pragma: no cover
            logger.error(f"[PonyChart] Auto-check failed: {e}")

        waitlimit = 15
        while waitlimit > 0 and await self._check():
            await asyncio.sleep(1)
            waitlimit -= 1

        if waitlimit <= 1 and await self._check():
            logger.warning(
                "PonyChart check timeout, please check your network connection"
            )
            try:
                submit_elements = await self.hvdriver.page.xpath(
                    "//input[@type='submit' and @value='Submit Answer']", timeout=2
                )
                if submit_elements:
                    await submit_elements[0].click()
            except Exception:
                try:
                    riddle_submit = await self.hvdriver.page.select("#riddlesubmit")
                    await riddle_submit.click()
                except Exception:
                    pass

        await asyncio.sleep(1)

        return isponychart
