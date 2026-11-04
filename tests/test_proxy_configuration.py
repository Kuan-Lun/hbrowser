"""Partial authenticated-proxy configuration must never permit direct startup."""

from __future__ import annotations

import itertools
import unittest
from unittest.mock import AsyncMock, patch

from hbrowser.gallery.browser import factory, flaresolverr, proxy

_KEYS = ("RP_USERNAME", "RP_PASSWORD", "RP_DNS")
_VALUES = ("synthetic-user", "synthetic-password", "proxy.example:8443")


class ResidentialProxyConfigurationTests(unittest.TestCase):
    def test_configuration_is_either_disabled_complete_or_rejected(self) -> None:
        for states in itertools.product(("missing", "empty", "set"), repeat=3):
            environment = {
                key: value if state == "set" else ""
                for key, value, state in zip(_KEYS, _VALUES, states, strict=True)
                if state != "missing"
            }
            configured = sum(state == "set" for state in states)
            with (
                self.subTest(states=states),
                patch.dict("os.environ", environment, clear=True),
                patch.object(
                    proxy, "_create_proxy_extension", return_value="owned-extension"
                ) as create,
            ):
                if configured in (1, 2):
                    for operation in (
                        proxy.configure_proxy,
                        proxy.has_residential_proxy,
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "provided together"
                        ) as caught:
                            operation()
                        for value in _VALUES:
                            self.assertNotIn(value, str(caught.exception))
                    create.assert_not_called()
                elif configured == 0:
                    self.assertIsNone(proxy.configure_proxy())
                    self.assertFalse(proxy.has_residential_proxy())
                    create.assert_not_called()
                else:
                    self.assertEqual(proxy.configure_proxy(), "owned-extension")
                    self.assertTrue(proxy.has_residential_proxy())
                    create.assert_called_once_with(
                        proxy_host="proxy.example",
                        proxy_port=8443,
                        proxy_user="synthetic-user",
                        proxy_pass="synthetic-password",
                        cleanup_deadline=None,
                    )

    def test_solver_selection_rejects_partial_proxy_identity(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "USE_TOR": "0",
                    "FLARESOLVERR_URL": "http://127.0.0.1:8191/v1",
                    "RP_DNS": "proxy.example",
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "provided together"),
        ):
            flaresolverr.should_use_flaresolverr()


class ResidentialProxyStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_proxy_stops_before_browser_install_or_launch(self) -> None:
        for missing in _KEYS:
            for empty in (False, True):
                environment = dict(zip(_KEYS, _VALUES, strict=True))
                if empty:
                    environment[missing] = ""
                else:
                    del environment[missing]
                environment["USE_TOR"] = "0"
                with (
                    self.subTest(missing=missing, empty=empty),
                    patch.dict("os.environ", environment, clear=True),
                    patch.object(
                        factory,
                        "_install_chrome_in_owned_worker",
                        new=AsyncMock(
                            side_effect=AssertionError("unsafe browser install")
                        ),
                    ) as install,
                    patch.object(
                        factory, "_construct_owned_browser_async", new=AsyncMock()
                    ) as launch,
                    patch.object(proxy, "_create_proxy_extension") as extension,
                    self.assertRaisesRegex(ValueError, "provided together"),
                ):
                    await factory.create_browser()
                install.assert_not_awaited()
                launch.assert_not_awaited()
                extension.assert_not_called()
