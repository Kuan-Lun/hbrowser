"""驗證碼檢測器 - 純粹的檢測邏輯，不依賴解決方案"""

from typing import Any

from bs4 import BeautifulSoup

from ..utils import (
    Deadline,
    ZendriverOperationTimeout,
    is_browser_generation_error,
    wait_for_selector,
    wait_for_zendriver,
)
from .constants import (
    MAX_CAPTCHA_DETECTION_TIMEOUT_SECONDS,
    RAY_RE,
    SITEKEY_RE,
    TURNSTILE_WIDGET_CSS,
)
from .models import ChallengeDetection

_DETECTION_READ_TIMEOUT_SECONDS = 2.0
_RECAPTCHA_WIDGET_CSS = "div.g-recaptcha[data-sitekey]"
_ANY_WIDGET_CSS = f"{TURNSTILE_WIDGET_CSS}, {_RECAPTCHA_WIDGET_CSS}"
_DETECTION_SNAPSHOT_SCRIPT = """(() => ({
    url: window.location.href,
    title: document.title,
    html: document.documentElement.outerHTML
}))()"""


def _read_timeout(deadline: Deadline) -> float:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise TimeoutError("CAPTCHA detection exceeded its semantic deadline")
    return min(_DETECTION_READ_TIMEOUT_SECONDS, remaining)


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
        if timeout > MAX_CAPTCHA_DETECTION_TIMEOUT_SECONDS:
            raise ValueError(
                "CAPTCHA detection timeout cannot exceed "
                f"{MAX_CAPTCHA_DETECTION_TIMEOUT_SECONDS:g} seconds"
            )
        detection_deadline = Deadline.after(timeout)
        snapshot = await wait_for_zendriver(
            page.evaluate(_DETECTION_SNAPSHOT_SCRIPT),
            timeout=_read_timeout(detection_deadline),
            owner=page,
        )
        if not isinstance(snapshot, dict):
            raise TypeError("CAPTCHA detection snapshot was not an object")
        url = snapshot.get("url")
        title = snapshot.get("title")
        html = snapshot.get("html")
        if not all(isinstance(value, str) for value in (url, title, html)):
            raise TypeError("CAPTCHA detection snapshot contained invalid fields")
        assert isinstance(url, str)
        assert isinstance(title, str)
        assert isinstance(html, str)
        title = title.strip()

        # 檢測 Cloudflare managed challenge
        if self._is_managed_challenge(title, html):
            ray_id = self._extract_ray_id(html)
            return ChallengeDetection(
                url=url, kind="cf_managed_challenge", ray_id=ray_id
            )

        # 檢測 Turnstile widget
        document = BeautifulSoup(html, "html.parser")
        widget = document.select_one(TURNSTILE_WIDGET_CSS)
        if widget is not None:
            widget_data = self._turnstile_data(dict(widget.attrs))
            return ChallengeDetection(
                url=url,
                kind="turnstile_widget",
                sitekey=widget_data["sitekey"],
                iframe_src=widget_data["src"],
            )

        recaptcha = document.select_one(_RECAPTCHA_WIDGET_CSS)
        if recaptcha is not None:
            sitekey_value = recaptcha.attrs.get("data-sitekey")
            return ChallengeDetection(
                url=url,
                kind="recaptcha_v2",
                sitekey=sitekey_value if isinstance(sitekey_value, str) else None,
            )

        try:
            dynamic_widget = await wait_for_selector(
                page,
                _ANY_WIDGET_CSS,
                deadline=detection_deadline,
            )
        except ZendriverOperationTimeout:
            raise
        except Exception as error:
            if is_browser_generation_error(error):
                raise
            return ChallengeDetection(url=url, kind="none")

        attrs = dict(dynamic_widget.attrs)
        if self._is_turnstile_attrs(attrs):
            widget_data = self._turnstile_data(attrs)
            return ChallengeDetection(
                url=url,
                kind="turnstile_widget",
                sitekey=widget_data["sitekey"],
                iframe_src=widget_data["src"],
            )
        return ChallengeDetection(
            url=url,
            kind="recaptcha_v2",
            sitekey=attrs.get("data-sitekey"),
        )

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

    @staticmethod
    def _is_turnstile_attrs(attrs: dict[str, Any]) -> bool:
        classes = attrs.get("class", ())
        if isinstance(classes, str):
            classes = classes.split()
        return (
            attrs.get("name") == "cf-turnstile-response"
            or str(attrs.get("id", "")).startswith("cf-chl-widget-")
            or "cf-turnstile" in classes
            or "turnstile" in str(attrs.get("src", ""))
        )

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
