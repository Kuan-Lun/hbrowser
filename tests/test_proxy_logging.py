from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import proxy


class ProxyLoggingTests(unittest.TestCase):
    def test_authenticated_proxy_log_does_not_include_credentials(self) -> None:
        logger = Mock()
        with (
            patch.dict(
                "os.environ",
                {
                    "RP_USERNAME": "secret-user",
                    "RP_PASSWORD": "secret-password",
                    "RP_DNS": "proxy.example:8443",
                },
                clear=False,
            ),
            patch.object(proxy, "logger", logger),
            patch.object(
                proxy,
                "_create_proxy_extension",
                return_value="proxy-extension.zip",
            ),
        ):
            result = proxy.configure_proxy()

        self.assertEqual(result, "proxy-extension.zip")
        logger.debug.assert_any_call("Using authenticated residential proxy")
        logger.debug.assert_any_call(
            "Residential proxy endpoint: host=%s port=%s",
            "proxy.example",
            "8443",
        )
        rendered_arguments = repr(logger.method_calls)
        self.assertNotIn("secret-user", rendered_arguments)
        self.assertNotIn("secret-password", rendered_arguments)


class ProxyVerificationLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonfatal_failure_logs_only_the_error_type(self) -> None:
        sentinel = "SENSITIVE-PROXY-FAILURE\nSECOND-LINE"
        logger = Mock()
        page = Mock()
        page.get = AsyncMock()
        page.select = AsyncMock(side_effect=ValueError(sentinel))

        with (
            patch.object(proxy, "logger", logger),
            patch(
                "hbrowser.gallery.browser.proxy.asyncio.to_thread",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        logger.warning.assert_called_once_with(
            "Could not verify proxy IP (non-fatal): error_type=%s",
            "ValueError",
        )
        self.assertNotIn(sentinel, repr(logger.method_calls))

    async def test_navigation_failure_is_terminal_and_skips_proxy_read(self) -> None:
        navigation_error = ValueError("navigation result unavailable")
        page = Mock()
        page.get = AsyncMock(side_effect=navigation_error)
        page.select = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.browser.proxy.asyncio.to_thread",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
            self.assertRaises(BrowserMutationOutcomeUnknownError) as raised,
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        page.get.assert_awaited_once_with("https://api.ipify.org")
        page.select.assert_not_awaited()
