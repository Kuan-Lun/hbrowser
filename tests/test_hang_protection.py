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

from hbrowser.gallery.browser import ban_handler, proxy
from hbrowser.gallery.browser.ban_handler import (
    _resolve_transient_blank_page,
    handle_ban_decorator,
)
from hbrowser.gallery.browser.proxy import verify_proxy_ip
from hbrowser.gallery.captcha import login_challenge
from hbrowser.gallery.captcha.detector import CaptchaDetector
from hbrowser.gallery.captcha.login_challenge import LoginChallengeHandler
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.eh_driver import EHDriver
from hbrowser.gallery.forums_auth import detect_forums_auth_state
from hbrowser.gallery.utils import ZendriverOperationTimeout


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
                driver.wait(
                    AsyncMock(),
                    ischangeurl=True,
                    owner=driver.page,
                    operation_timeout=5.0,
                ),
                timeout=10,
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

    async def test_write_response_token_timeout_is_terminal(
        self,
    ) -> None:
        evaluate = AsyncMock(side_effect=_hang)
        page = SimpleNamespace(evaluate=evaluate)

        with (
            patch.object(
                login_challenge,
                "_TOKEN_MUTATION_TIMEOUT_SECONDS",
                0.05,
            ),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await LoginChallengeHandler._write_response_token(page, "#token", "abc")
        evaluate.assert_awaited_once()


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

        with (
            patch.object(ban_handler, "_PAGE_MUTATION_TIMEOUT_SECONDS", 0.05),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await myget("https://e-hentai.org/")

    async def test_resolve_transient_blank_page_times_out_instead_of_hanging(
        self,
    ) -> None:
        page = SimpleNamespace(reload=_hang, get_content=_hang)

        with (
            patch.object(ban_handler, "_PAGE_MUTATION_TIMEOUT_SECONDS", 0.05),
            patch(
                "hbrowser.gallery.browser.ban_handler.asyncio.sleep",
                new=AsyncMock(),
            ),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await _resolve_transient_blank_page(
                page, "<html><head></head><body></body></html>"
            )


class ProxyHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_proxy_ip_timeout_is_terminal(
        self,
    ) -> None:
        get = AsyncMock(side_effect=_hang)
        select = AsyncMock(side_effect=_hang)
        page = SimpleNamespace(get=get, select=select)
        browser = Mock()

        with (
            patch(
                "hbrowser.gallery.browser.proxy.asyncio.to_thread",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
            patch.object(proxy, "_PROXY_NAVIGATION_TIMEOUT_SECONDS", 0.05),
        ):
            with self.assertRaises(ZendriverOperationTimeout):
                await verify_proxy_ip(browser, page)
        get.assert_awaited_once_with("https://api.ipify.org")
        select.assert_not_awaited()


class EHDriverHangProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_gallery2tag_times_out_instead_of_hanging_forever(self) -> None:
        driver = EHDriver()
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(side_effect=_hang),
            xpath=AsyncMock(side_effect=_hang),
        )
        driver.get = AsyncMock()  # type: ignore[method-assign]

        gallery = Mock()
        gallery.url = "https://e-hentai.org/g/1/deadbeef/"

        with self.assertRaises(ZendriverOperationTimeout):
            await asyncio.wait_for(
                driver.gallery2tag(gallery, "artist"),
                timeout=10,
            )
        driver.get.assert_awaited_once_with(gallery.url)
        driver.page.xpath.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
