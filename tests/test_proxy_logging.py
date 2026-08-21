from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError, ProcessOwnershipError
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
        rendered_arguments = repr(logger.method_calls)
        self.assertNotIn("secret-user", rendered_arguments)
        self.assertNotIn("secret-password", rendered_arguments)
        self.assertNotIn("proxy.example", rendered_arguments)
        self.assertNotIn("8443", rendered_arguments)

    def test_proxy_extension_uses_json_literals_for_untrusted_configuration(
        self,
    ) -> None:
        proxy_host = 'proxy"host\\edge'
        proxy_user = 'user"name\nnext'
        proxy_password = 'pass\\word"\nnext'
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as directory:
            plugin_directory = Path(directory) / "plugin"
            plugin_directory.mkdir()
            with patch.object(
                tempfile,
                "mkdtemp",
                return_value=str(plugin_directory),
            ):
                result = proxy._create_proxy_extension(
                    proxy_host,
                    8443,
                    proxy_user,
                    proxy_password,
                )

            background = (Path(result) / "background.js").read_text(encoding="utf-8")
            self.assertIn(f"host: {json.dumps(proxy_host)}", background)
            self.assertIn(f"username: {json.dumps(proxy_user)}", background)
            self.assertIn(f"password: {json.dumps(proxy_password)}", background)

    def test_proxy_extension_cleanup_failure_is_an_ownership_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-proxy-test-") as directory:
            plugin_directory = Path(directory) / "plugin"
            plugin_directory.mkdir()
            creation_error = OSError("write failed")
            cleanup_error = OSError("remove failed")
            with (
                patch.object(
                    tempfile,
                    "mkdtemp",
                    return_value=str(plugin_directory),
                ),
                patch.object(Path, "write_text", side_effect=creation_error),
                patch.object(shutil, "rmtree", side_effect=cleanup_error),
                self.assertRaises(ProcessOwnershipError) as raised,
            ):
                proxy._create_proxy_extension("host", 8080, "user", "password")

            self.assertIs(raised.exception.__cause__, cleanup_error)
            self.assertIn(
                "Creation failure type: OSError",
                raised.exception.__notes__,
            )

    def test_invalid_proxy_port_does_not_echo_configuration(self) -> None:
        secret_port = "SECRET-INVALID-PORT"
        logger = Mock()
        with (
            patch.dict(
                "os.environ",
                {
                    "RP_USERNAME": "user",
                    "RP_PASSWORD": "password",
                    "RP_DNS": f"proxy.example:{secret_port}",
                },
                clear=False,
            ),
            patch.object(proxy, "logger", logger),
            patch.object(proxy, "_create_proxy_extension") as create_extension,
            self.assertRaisesRegex(
                ValueError, "Residential proxy port is invalid"
            ) as raised,
        ):
            proxy.configure_proxy()

        self.assertNotIn(secret_port, str(raised.exception))
        self.assertNotIn(secret_port, repr(logger.method_calls))
        create_extension.assert_not_called()


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

    async def test_equal_ip_failure_does_not_disclose_the_address(self) -> None:
        address = "192.0.2.123"
        page = Mock()
        page.get = AsyncMock()
        page.select = AsyncMock(return_value=Mock(text=address))

        with (
            patch(
                "hbrowser.gallery.browser.proxy.asyncio.to_thread",
                new=AsyncMock(return_value=address),
            ),
            self.assertRaisesRegex(RuntimeError, "local public address") as raised,
        ):
            await proxy.verify_proxy_ip(Mock(), page)

        self.assertNotIn(address, str(raised.exception))
