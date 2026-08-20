from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import ban_handler


class BanLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_navigation_failure_is_outcome_unknown_without_page_read(
        self,
    ) -> None:
        page = Mock()
        page.get = AsyncMock(side_effect=RuntimeError("sensitive URL"))
        page.get_content = AsyncMock()

        with self.assertRaises(BrowserMutationOutcomeUnknownError) as raised:
            await ban_handler.handle_ban_decorator(page)("https://example.test/private")

        page.get.assert_awaited_once_with("https://example.test/private")
        page.get_content.assert_not_awaited()
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    async def test_reload_failure_is_outcome_unknown_without_page_read(self) -> None:
        page = Mock()
        page.reload = AsyncMock(side_effect=RuntimeError("reload payload"))
        page.get_content = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.browser.ban_handler.asyncio.sleep",
                new=AsyncMock(),
            ),
            self.assertRaises(BrowserMutationOutcomeUnknownError) as raised,
        ):
            await ban_handler._resolve_transient_blank_page(
                page,
                "<html><head></head><body></body></html>",
            )

        page.reload.assert_awaited_once_with()
        page.get_content.assert_not_awaited()
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

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
