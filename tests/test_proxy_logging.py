from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

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
        logger.info.assert_called_once_with(
            "Using authenticated residential proxy: %s:%s",
            "proxy.example",
            "8443",
        )
        rendered_arguments = repr(logger.info.call_args)
        self.assertNotIn("secret-user", rendered_arguments)
        self.assertNotIn("secret-password", rendered_arguments)
