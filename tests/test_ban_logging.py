from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from hbrowser.gallery.browser import ban_handler


class BanLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_ban_debug_log_does_not_include_page_content(self) -> None:
        secret = "secret-page-token"
        source = (
            "<html>Your IP address has been temporarily banned for 1 minute "
            f"{secret}</html>"
        )
        page = Mock()
        page.reload = AsyncMock()
        page.get_content = AsyncMock(return_value="<html>available</html>")
        logger = Mock()

        with (
            patch.object(ban_handler, "logger", logger),
            patch.object(
                ban_handler,
                "_wait_out_ban",
                new=AsyncMock(),
            ),
        ):
            await ban_handler._retry_until_unbanned(page, source)

        logger.debug.assert_called_once_with(
            "Ban page inspected: bytes=%d banned=%s blank=%s",
            len(source.encode("utf-8")),
            True,
            False,
        )
        self.assertNotIn(secret, repr(logger.method_calls))
