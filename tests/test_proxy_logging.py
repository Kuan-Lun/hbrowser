from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from hbrowser.gallery.browser import proxy, proxy_rotator


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
        logger.info.assert_called_once_with("Using authenticated residential proxy")
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
        page.get = AsyncMock(side_effect=ValueError(sentinel))

        with (
            patch.object(proxy, "logger", logger),
            patch.object(
                proxy.asyncio,
                "to_thread",
                new=AsyncMock(return_value="192.0.2.10"),
            ),
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        logger.warning.assert_called_once_with(
            "Could not verify proxy IP (non-fatal): error_type=%s",
            "ValueError",
        )
        self.assertNotIn(sentinel, repr(logger.method_calls))


class ProxyRotatorLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_failure_warning_does_not_include_exception_text(self) -> None:
        sentinel = "SENSITIVE-STOP-FAILURE\nSECOND-LINE"
        logger = Mock()
        replacement_browser = Mock()
        replacement_page = Mock()

        with (
            patch.object(proxy_rotator, "logger", logger),
            patch.object(
                proxy_rotator,
                "stop_browser",
                new=AsyncMock(side_effect=RuntimeError(sentinel)),
            ),
            patch.object(
                proxy_rotator,
                "create_browser",
                new=AsyncMock(return_value=(replacement_browser, replacement_page)),
            ),
        ):
            result = await proxy_rotator.DriverRestartRotator().rotate(
                Mock(),
                headless=True,
            )

        self.assertEqual(result, (replacement_browser, replacement_page))
        logger.warning.assert_any_call(
            "Failed to stop current browser (non-fatal): error_type=%s",
            "RuntimeError",
        )
        self.assertNotIn(sentinel, repr(logger.method_calls))
