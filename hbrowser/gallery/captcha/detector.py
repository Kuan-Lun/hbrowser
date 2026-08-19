"""驗證碼檢測器 - 純粹的檢測邏輯，不依賴解決方案"""

from typing import Any

from bs4 import BeautifulSoup

from ..utils import is_connection_error, wait_for_zendriver
from .constants import RAY_RE, SITEKEY_RE, TURNSTILE_WIDGET_CSS
from .models import ChallengeDetection

_DETECTION_READ_TIMEOUT_SECONDS = 2.0


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
        url = await wait_for_zendriver(
            page.evaluate("window.location.href"),
            timeout=_DETECTION_READ_TIMEOUT_SECONDS,
        )
        title = (
            await wait_for_zendriver(
                page.evaluate("document.title"),
                timeout=_DETECTION_READ_TIMEOUT_SECONDS,
            )
            or ""
        ).strip()
        html = (
            await wait_for_zendriver(
                page.get_content(), timeout=_DETECTION_READ_TIMEOUT_SECONDS
            )
            or ""
        )

        # 檢測 Cloudflare managed challenge
        if self._is_managed_challenge(title, html):
            ray_id = self._extract_ray_id(html)
            return ChallengeDetection(
                url=url, kind="cf_managed_challenge", ray_id=ray_id
            )

        # 檢測 Turnstile widget
        widget_data = await self._find_turnstile_widget(page, html, timeout)
        if widget_data:
            return ChallengeDetection(
                url=url,
                kind="turnstile_widget",
                sitekey=widget_data["sitekey"],
                iframe_src=widget_data["src"],
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
            or 'id="cf-challenge-running"' in html
            or 'id="challenge-running"' in html
            or 'id="challenge-stage"' in html
        )

    def _extract_ray_id(self, html: str) -> str | None:
        """從 HTML 中提取 Ray ID"""
        m = RAY_RE.search(html)
        return m.group(1) if m else None

    async def _find_turnstile_widget(
        self,
        page: Any,
        html: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Find either the static widget container, response field, or iframe."""
        widget = BeautifulSoup(html, "html.parser").select_one(TURNSTILE_WIDGET_CSS)
        if widget is not None:
            return self._turnstile_data(dict(widget.attrs))

        try:
            dynamic_widget = await wait_for_zendriver(
                page.select(TURNSTILE_WIDGET_CSS, timeout=timeout), timeout=timeout
            )
            return self._turnstile_data(dynamic_widget.attrs)
        except Exception as e:
            if is_connection_error(e):
                raise
            return None

    @staticmethod
    def _turnstile_data(attrs: dict[str, Any]) -> dict[str, str | None]:
        src_value = attrs.get("src")
        src = src_value if isinstance(src_value, str) and src_value else None
        sitekey_value = attrs.get("data-sitekey")
        sitekey = (
            sitekey_value if isinstance(sitekey_value, str) and sitekey_value else None
        )
        if sitekey is None and src is not None:
            match = SITEKEY_RE.search(src)
            sitekey = match.group(1) if match else None
        return {"src": src, "sitekey": sitekey}

    async def _find_recaptcha_div(
        self, page: Any, timeout: float
    ) -> dict[str, Any] | None:
        """查找 reCAPTCHA div 並提取 sitekey"""
        try:
            recaptcha_div = await wait_for_zendriver(
                page.select("div.g-recaptcha[data-sitekey]", timeout=timeout),
                timeout=timeout,
            )
            sitekey = recaptcha_div.attrs.get("data-sitekey")
            return {"sitekey": sitekey}
        except Exception as e:
            if is_connection_error(e):
                raise
            return None
