import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from hbrowser.gallery.browser.flaresolverr import (
    FlareSolverrProtocolError,
    FlareSolverrResult,
)
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.utils import wait_for_zendriver
from hbrowser.gallery.utils.protocol import ZendriverOwnerRetiredError


class _Driver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class IdentityApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_cookie_is_rejected_before_any_identity_mutation(
        self,
    ) -> None:
        invalid_fields: tuple[tuple[str, object], ...] = (
            ("expiry", "bad"),
            ("expiry", float("nan")),
            ("expiry", True),
            ("domain", 42),
            ("path", []),
            ("secure", "false"),
            ("httpOnly", 1),
            ("sameSite", []),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                driver = _Driver()
                page = SimpleNamespace(send=AsyncMock())
                driver.page = page
                result = FlareSolverrResult(
                    cookies=[{"name": "cf_clearance", "value": "opaque", field: value}],
                    user_agent="new-agent",
                )
                with self.assertRaises(FlareSolverrProtocolError):
                    await driver._apply_flaresolverr_result(result)
                page.send.assert_not_called()
                # Pure input rejection has not accepted a browser mutation.
                await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)

    async def test_failure_between_identity_commands_retires_generation(self) -> None:
        driver = _Driver()
        page = SimpleNamespace(send=AsyncMock())
        browser = SimpleNamespace(tabs=[page])
        page.browser = browser
        driver.page = page
        result = FlareSolverrResult(
            cookies=[{"name": "cf_clearance", "value": "opaque"}],
            user_agent="new-agent",
        )
        with (
            patch(
                "hbrowser.gallery.driver_base.cdp.network.set_cookies",
                side_effect=ValueError("cannot construct cookie command"),
            ),
            self.assertRaises(ValueError),
        ):
            await driver._apply_flaresolverr_result(result)
        page.send.assert_awaited_once()
        with self.assertRaises(ZendriverOwnerRetiredError):
            await wait_for_zendriver(asyncio.sleep(0), timeout=1, owner=page)
