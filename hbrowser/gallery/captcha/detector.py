"""驗證碼檢測器 - 純粹的檢測邏輯，不依賴解決方案"""

from typing import Any

from .constants import RAY_RE, SITEKEY_RE, TURNSTILE_IFRAME_CSS
from .models import ChallengeDetection


class CaptchaDetector:
    """驗證碼檢測器 - 與解決方案無關"""

    async def detect(self, page: Any, timeout: float = 2.0) -> ChallengeDetection:
        """
        檢測當前頁面是否存在驗證碼挑戰

        Args:
            page: zendriver Tab 實例
            timeout: 檢測超時時間（秒）

        Returns:
            ChallengeDetection: 檢測結果
        """
        url = await page.evaluate("window.location.href")
        title = (await page.evaluate("document.title") or "").strip()
        html = await page.get_content() or ""

        # 檢測 Cloudflare managed challenge
        if self._is_managed_challenge(title, html):
            ray_id = self._extract_ray_id(html)
            return ChallengeDetection(
                url=url, kind="cf_managed_challenge", ray_id=ray_id
            )

        # 檢測 Turnstile widget
        iframe_data = await self._find_turnstile_iframe(page, timeout)
        if iframe_data:
            return ChallengeDetection(
                url=url,
                kind="turnstile_widget",
                sitekey=iframe_data["sitekey"],
                iframe_src=iframe_data["src"],
            )

        # 檢測 reCAPTCHA v2
        recaptcha_data = await self._find_recaptcha_div(page, timeout)
        if recaptcha_data:
            return ChallengeDetection(
                url=url,
                kind="recaptcha_v2",
                sitekey=recaptcha_data["sitekey"],
            )

        return ChallengeDetection(url=url, kind="none")

    def _is_managed_challenge(self, title: str, html: str) -> bool:
        """檢測是否為 Cloudflare managed challenge"""
        return (
            "請稍候" in title
            or "Just a moment" in title
            or "_cf_chl_opt" in html
            or "/cdn-cgi/challenge-platform/" in html
        )

    def _extract_ray_id(self, html: str) -> str | None:
        """從 HTML 中提取 Ray ID"""
        m = RAY_RE.search(html)
        return m.group(1) if m else None

    async def _find_turnstile_iframe(
        self, page: Any, timeout: float
    ) -> dict[str, Any] | None:
        """查找 Turnstile iframe 並提取 sitekey"""
        try:
            iframe = await page.select(TURNSTILE_IFRAME_CSS, timeout=timeout)
            iframe_src = iframe.attrs.get("src", "")
            m = SITEKEY_RE.search(iframe_src)
            sitekey = m.group(1) if m else None
            return {"src": iframe_src, "sitekey": sitekey}
        except (TimeoutError, Exception):
            return None

    async def _find_recaptcha_div(
        self, page: Any, timeout: float
    ) -> dict[str, Any] | None:
        """查找 reCAPTCHA div 並提取 sitekey"""
        try:
            recaptcha_div = await page.select(
                "div.g-recaptcha[data-sitekey]", timeout=timeout
            )
            sitekey = recaptcha_div.attrs.get("data-sitekey")
            return {"sitekey": sitekey}
        except (TimeoutError, Exception):
            return None
