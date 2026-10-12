import asyncio
import logging
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from zendriver.core.connection import ProtocolException

from hbrowser.gallery.captcha import CaptchaDetector, LoginChallengeHandler
from hbrowser.gallery.captcha.page_challenge import PageChallengeHandler
from hbrowser.gallery.utils import (
    PageStateTimeout,
    ZendriverOperationTimeout,
    wait_for_zendriver,
)
from hbrowser.gallery.utils import deadline as deadline_module


@dataclass
class _Element:
    attrs: dict[str, str]


class _Page:
    def __init__(self, html: str, title: str = "") -> None:
        self.url = "https://forums.e-hentai.org/index.php?act=Login&CODE=00"
        self.title = title
        self.html = html
        self.selections: dict[str, _Element] = {}

    async def evaluate(self, expression: str) -> Any:
        if "document.documentElement.outerHTML" in expression:
            return {
                "url": self.url,
                "title": self.title,
                "html": self.html,
            }
        raise AssertionError(f"Unexpected evaluate expression: {expression}")

    async def get_content(self) -> str:
        return self.html

    async def select(self, selector: str, timeout: float) -> _Element:
        del timeout
        for configured_selector, element in self.selections.items():
            if configured_selector in selector:
                return element
        raise LookupError(selector)


class _AbsentDynamicWidgetPage(_Page):
    def __init__(self) -> None:
        super().__init__("<html></html>")
        self.query_count = 0
        self.now = 0.0

    async def query_selector(self, _selector: str) -> None:
        self.query_count += 1
        # Expire the semantic deadline only after a healthy query has started.
        # Host scheduling must not decide whether this query is issued.
        self.now += 3.0
        return None


class _FailedDynamicQueryPage(_Page):
    def __init__(self, failure: BaseException) -> None:
        super().__init__("<html></html>")
        self.failure = failure
        self.query_count = 0

    async def query_selector(self, _selector: str) -> None:
        self.query_count += 1
        raise self.failure


class CaptchaDetectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_detection_timeout_is_capped_at_five_seconds(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 5 seconds"):
            await CaptchaDetector().detect(_Page("<html></html>"), timeout=5.01)

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

    async def test_dynamic_widget_timeout_is_terminal(self) -> None:
        timeout = ZendriverOperationTimeout(timeout_seconds=2)
        page = _Page("<html></html>")
        page.select = AsyncMock(side_effect=timeout)  # type: ignore[method-assign]

        with self.assertRaises(ZendriverOperationTimeout) as raised:
            await CaptchaDetector().detect(page, timeout=1)

        self.assertIs(raised.exception, timeout)
        page.select.assert_awaited_once()

    async def test_absent_dynamic_widget_is_not_a_generation_failure(self) -> None:
        page = _AbsentDynamicWidgetPage()

        with patch.object(
            deadline_module,
            "time",
            SimpleNamespace(monotonic=lambda: page.now),
        ):
            detection = await CaptchaDetector().detect(page, timeout=2)

        self.assertEqual(detection.kind, "none")
        self.assertEqual(detection.url, page.url)
        self.assertEqual(page.query_count, 1)
        result = await wait_for_zendriver(
            asyncio.sleep(0, result="healthy"),
            timeout=2,
            owner=page,
        )
        self.assertEqual(result, "healthy")

    async def test_dynamic_dom_failures_never_mean_absent(self) -> None:
        for error in (
            ProtocolException({"code": -32000, "message": "Could not find node"}),
            LookupError("DOM document could not be resolved"),
            RuntimeError("DOM query failed"),
            TimeoutError("non-semantic transport timeout"),
        ):
            with self.subTest(error=type(error).__name__):
                page = _FailedDynamicQueryPage(error)
                with self.assertRaises(type(error)) as raised:
                    await CaptchaDetector().detect(page)
                self.assertIs(raised.exception, error)
                self.assertEqual(page.query_count, 1)

    async def test_explicit_missing_widget_timeout_is_absent(self) -> None:
        page = _FailedDynamicQueryPage(PageStateTimeout("widget was absent"))
        detection = await CaptchaDetector().detect(page)
        self.assertEqual(detection.kind, "none")
        self.assertEqual(page.query_count, 1)

    async def test_dynamic_query_cancellation_propagates(self) -> None:
        page = _FailedDynamicQueryPage(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await CaptchaDetector().detect(page)
        self.assertEqual(page.query_count, 1)

    async def test_dynamic_turnstile_is_detected(self) -> None:
        page = _Page("<html></html>")
        page.selections["div.cf-turnstile[data-sitekey]"] = _Element(
            {"class": "cf-turnstile", "data-sitekey": "dynamic-sitekey"}
        )
        detection = await CaptchaDetector().detect(page)
        self.assertEqual(detection.kind, "turnstile_widget")
        self.assertEqual(detection.sitekey, "dynamic-sitekey")

    async def test_static_recaptcha_is_detected(self) -> None:
        page = _Page('<div class="g-recaptcha" data-sitekey="static-sitekey"></div>')
        detection = await CaptchaDetector().detect(page)
        self.assertEqual(detection.kind, "recaptcha_v2")
        self.assertEqual(detection.sitekey, "static-sitekey")

    async def test_login_does_not_skip_challenge_after_selector_failure(self) -> None:
        error = ProtocolException({"code": -32000, "message": "Could not find node"})
        page = _FailedDynamicQueryPage(error)
        handler = LoginChallengeHandler(
            detector=CaptchaDetector(),
            logger=logging.getLogger("test-captcha-login-read-failure"),
            headless=False,
            manual_timeout=30,
            turnstile_tabs=15,
            automatic_solver=None,
        )
        with self.assertRaises(ProtocolException) as raised:
            await handler.resolve(page)
        self.assertIs(raised.exception, error)
        self.assertEqual(page.query_count, 1)

    async def test_manual_resolution_does_not_accept_failed_widget_query(self) -> None:
        for kind, html in (
            (
                "turnstile_widget",
                '<div class="cf-turnstile" data-sitekey="site"></div>',
            ),
            ("recaptcha_v2", '<div class="g-recaptcha" data-sitekey="site"></div>'),
        ):
            with self.subTest(kind=kind):
                error = ProtocolException(
                    {"code": -32000, "message": "Could not find node"}
                )
                page = _FailedDynamicQueryPage(error)
                page.html = html

                async def diagnostic(
                    _kind: str, bound_page: _FailedDynamicQueryPage = page
                ) -> None:
                    # A navigation replaces the initial widget's static HTML;
                    # the actual dynamic query now fails instead of proving absence.
                    bound_page.html = "<html></html>"

                handler = PageChallengeHandler(
                    detector=CaptchaDetector(),
                    logger=logging.getLogger("test-captcha-manual-read-failure"),
                    headless=False,
                    manual_timeout=30,
                    automatic_solver=None,
                    apply_identity=AsyncMock(),
                    navigate=AsyncMock(),
                    save_diagnostic=diagnostic,
                )
                with self.assertRaises(ProtocolException) as raised:
                    await handler.resolve(page, page.url)
                self.assertIs(raised.exception, error)
                self.assertEqual(page.query_count, 1)
