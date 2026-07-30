import unittest
from dataclasses import dataclass
from typing import Any

from hbrowser.gallery.captcha import CaptchaDetector


@dataclass
class _Element:
    attrs: dict[str, str]


class _Page:
    def __init__(self, html: str, title: str = "") -> None:
        self.url = "https://forums.e-hentai.org/index.php?act=Login&CODE=00"
        self.title = title
        self.html = html
        self.selections: dict[str, _Element] = {}

    async def evaluate(self, expression: str) -> str:
        if expression == "window.location.href":
            return self.url
        if expression == "document.title":
            return self.title
        raise AssertionError(f"Unexpected evaluate expression: {expression}")

    async def get_content(self) -> str:
        return self.html

    async def select(self, selector: str, timeout: float) -> _Element:
        del timeout
        try:
            return self.selections[selector]
        except KeyError:
            raise LookupError(selector) from None


class CaptchaDetectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_detects_static_turnstile_without_iframe(self) -> None:
        page = _Page("""
            <div class="cf-turnstile"
                 data-sitekey="0x4AAAAAAC-TvH-cv03mjH96"></div>
            <input type="hidden"
                   id="cf-chl-widget-random_response"
                   name="cf-turnstile-response">
            """)

        detection = await CaptchaDetector().detect(page)

        self.assertEqual(detection.kind, "turnstile_widget")
        self.assertEqual(detection.sitekey, "0x4AAAAAAC-TvH-cv03mjH96")
        self.assertIsNone(detection.iframe_src)

    async def test_managed_challenge_takes_priority_over_widget(self) -> None:
        page = _Page(
            """
            <script>window._cf_chl_opt = {};</script>
            <input name="cf-turnstile-response">
            """,
            title="Just a moment...",
        )

        detection = await CaptchaDetector().detect(page)

        self.assertEqual(detection.kind, "cf_managed_challenge")

    async def test_embedded_iframe_is_not_managed_challenge(self) -> None:
        iframe_url = (
            "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/"
            "h/b/turnstile/f/av0/0x4AAAAAAC-TvH-cv03mjH96/light/normal"
        )
        page = _Page(f'<iframe src="{iframe_url}"></iframe>')

        detection = await CaptchaDetector().detect(page)

        self.assertEqual(detection.kind, "turnstile_widget")
        self.assertEqual(detection.sitekey, "0x4AAAAAAC-TvH-cv03mjH96")
        self.assertEqual(detection.iframe_src, iframe_url)

    async def test_recaptcha_detection_is_preserved(self) -> None:
        page = _Page("<form></form>")
        page.selections["div.g-recaptcha[data-sitekey]"] = _Element(
            {"data-sitekey": "recaptcha-sitekey"}
        )

        detection = await CaptchaDetector().detect(page)

        self.assertEqual(detection.kind, "recaptcha_v2")
        self.assertEqual(detection.sitekey, "recaptcha-sitekey")

    async def test_connection_errors_are_not_needed_for_static_detection(self) -> None:
        page: Any = _Page('<input type="hidden" name="cf-turnstile-response" value="">')

        detection = await CaptchaDetector().detect(page)

        self.assertEqual(detection.kind, "turnstile_widget")
