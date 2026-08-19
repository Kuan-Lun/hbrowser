"""Browser reads across the driver, captcha, and gallery layers must never
hang forever.

zendriver's own timeout= kwarg on Tab.select()/xpath()/find_all() only
checks a deadline between polling iterations; a single hung underlying CDP
call never returns control to that check, so it is not a real bound. Every
call exercised here previously relied on that fake protection (or had none
at all) and is now wrapped in wait_for_zendriver, which bounds the call
itself via asyncio.wait() regardless of what zendriver is doing internally.
"""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from hbrowser.gallery.browser.ban_handler import (
    _resolve_transient_blank_page,
    handle_ban_decorator,
)
from hbrowser.gallery.browser.proxy import verify_proxy_ip
from hbrowser.gallery.captcha.detector import CaptchaDetector
from hbrowser.gallery.captcha.login_challenge import LoginChallengeHandler
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.eh_driver import EHDriver
from hbrowser.gallery.forums_auth import detect_forums_auth_state


async def _hang(*_args: Any, **_kwargs: Any) -> Any:
    await asyncio.Event().wait()


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class _HangingElement:
    async def click(self) -> None:
        await asyncio.Event().wait()

    async def query_selector(self, selector: str) -> Any:
        await asyncio.Event().wait()

    async def query_selector_all(self, selector: str) -> Any:
        await asyncio.Event().wait()


class DriverHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gohomepage_times_out_instead_of_hanging_forever(self) -> None:
        driver = _TestDriver()
        driver.page = SimpleNamespace(evaluate=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(driver.gohomepage(), timeout=10)

    async def test_get_times_out_instead_of_hanging_forever(self) -> None:
        driver = _TestDriver()
        driver.page = SimpleNamespace(evaluate=_hang)
        driver.myget = AsyncMock()

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(driver.get("https://e-hentai.org/"), timeout=10)

    async def test_wait_old_url_read_times_out_instead_of_hanging_forever(
        self,
    ) -> None:
        driver = _TestDriver()
        driver.page = SimpleNamespace(evaluate=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(
                driver.wait(AsyncMock(), ischangeurl=True), timeout=10
            )

    async def test_find_element_chain_times_out_instead_of_hanging_forever(
        self,
    ) -> None:
        driver = _TestDriver()
        driver.page = _HangingElement()

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(driver.find_element_chain("#a", "#b"), timeout=10)


class CaptchaDetectorHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_detect_times_out_instead_of_hanging_forever(self) -> None:
        detector = CaptchaDetector()
        page = SimpleNamespace(evaluate=_hang, get_content=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(detector.detect(page), timeout=10)


class LoginChallengeHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_response_token_times_out_instead_of_hanging_forever(
        self,
    ) -> None:
        page = SimpleNamespace(evaluate=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(
                LoginChallengeHandler._read_response_token(page, "#token"),
                timeout=10,
            )

    async def test_write_response_token_returns_false_instead_of_hanging(
        self,
    ) -> None:
        """The write path already catches any exception as a soft failure;
        a hang must resolve to that same False outcome, not block forever."""

        page = SimpleNamespace(evaluate=_hang)

        result = await asyncio.wait_for(
            LoginChallengeHandler._write_response_token(page, "#token", "abc"),
            timeout=10,
        )

        self.assertFalse(result)


class ForumsAuthHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_detect_forums_auth_state_times_out_instead_of_hanging(
        self,
    ) -> None:
        page = SimpleNamespace(evaluate=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(detect_forums_auth_state(page), timeout=10)


class BanHandlerHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_myget_times_out_instead_of_hanging_forever(self) -> None:
        page = SimpleNamespace(get=_hang, get_content=_hang)
        myget = handle_ban_decorator(page)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(myget("https://e-hentai.org/"), timeout=15)

    async def test_resolve_transient_blank_page_times_out_instead_of_hanging(
        self,
    ) -> None:
        page = SimpleNamespace(reload=_hang, get_content=_hang)

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(
                _resolve_transient_blank_page(
                    page, "<html><head></head><body></body></html>"
                ),
                timeout=15,
            )


class ProxyHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_proxy_ip_swallows_a_hang_as_a_non_fatal_warning(
        self,
    ) -> None:
        """The whole check is already best-effort (except Exception: warn);
        a hung page.get()/select() must resolve to that same non-fatal
        warning path within a bounded time, not block startup forever."""

        page = SimpleNamespace(get=_hang, select=_hang)
        browser = Mock()

        with patch(
            "hbrowser.gallery.browser.proxy.asyncio.to_thread",
            new=AsyncMock(return_value="192.0.2.10"),
        ):
            await asyncio.wait_for(verify_proxy_ip(browser, page), timeout=20)


class EHDriverHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gallery2tag_times_out_instead_of_hanging_forever(self) -> None:
        driver = EHDriver()
        driver.page = SimpleNamespace(evaluate=_hang, xpath=_hang)
        driver.get = AsyncMock()  # type: ignore[method-assign]

        gallery = Mock()
        gallery.url = "https://e-hentai.org/g/1/deadbeef/"

        tags = await asyncio.wait_for(driver.gallery2tag(gallery, "artist"), timeout=10)

        self.assertEqual(tags, [])


if __name__ == "__main__":
    unittest.main()
