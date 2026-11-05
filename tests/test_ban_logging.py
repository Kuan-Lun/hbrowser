from __future__ import annotations

import unittest
from html import escape
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import ban_handler
from hbrowser.gallery.eh_driver import EHDriver, _RawPageSnapshot

_BAN_NOTICE = "Your IP address has been temporarily banned"


def _search_document(query: str, title: str) -> str:
    return (
        "<html><head><title>Search results</title></head><body>"
        f'<input id="f_search" value="{escape(query, quote=True)}">'
        f'<div class="searchtext">Results for {escape(query)}</div>'
        f'<div class="itg"><a href="/g/1/abcdef/">{escape(title)}</a></div>'
        "</body></html>"
    )


class BanClassificationTests(unittest.TestCase):
    def test_reflected_values_and_quoted_data_are_not_server_ban_notices(self) -> None:
        for text in (_BAN_NOTICE, _BAN_NOTICE + " for 30 days"):
            documents = (
                _search_document(text, text),
                f'<html><body><h1 id="gn">{text}</h1></body></html>',
                f'<html><body><h1 id="gj">{text}</h1></body></html>',
                f'<html><body><div class="glink">{text}</div></body></html>',
                f'<html><body><p data-message="{text}">Ready</p></body></html>',
                f"<html><head><title>{text}</title></head><body>Ready</body></html>",
                f"<html><body><!-- {text} --><p>Ready</p></body></html>",
                f"<html><body><script>const text = '{text}';</script></body></html>",
                f"<html><body><style>/* {text} */</style></body></html>",
                f"<html><body><template>{text}</template></body></html>",
                f"<html><body><textarea>{text}</textarea></body></html>",
                f"<html><body><blockquote>{text}</blockquote></body></html>",
                f"<html><body><q>{text}</q></body></html>",
                f"<html><body><pre><code>{text}</code></pre></body></html>",
                f"<html><body><p>The warning says: {text}</p></body></html>",
                f"<html><body><p>{text}ly</p></body></html>"
                if text == _BAN_NOTICE
                else "Ready",
            )
            for document in documents:
                with self.subTest(document=document):
                    status = ban_handler.check_ban_status(document)
                    self.assertFalse(status.is_banned)
                    self.assertFalse(status.should_wait)

    def test_structural_notice_is_banned_even_without_a_wait_duration(self) -> None:
        notices = (
            _BAN_NOTICE,
            f"<html>{_BAN_NOTICE}</html>",
            f"<html><body>{_BAN_NOTICE}.</body></html>",
            f"<nav>Home</nav><main><p>{_BAN_NOTICE}</p></main>",
            f'<div role="alert">{_BAN_NOTICE}: excessive pageloads.</div>',
            "<p>Your IP&nbsp;address has been <b>temporarily</b> banned.</p>",
            _search_document("30 days", "2 hours").replace(
                "</body>", f"<p>{_BAN_NOTICE}</p></body>"
            ),
        )
        for document in notices:
            with self.subTest(document=document):
                status = ban_handler.check_ban_status(document)
                self.assertTrue(status.is_banned)
                self.assertTrue(status.should_wait)

    def test_blank_page_contract_is_unchanged(self) -> None:
        status = ban_handler.check_ban_status("<html><head></head><body></body></html>")
        self.assertFalse(status.is_banned)
        self.assertTrue(status.is_blank_page)
        self.assertTrue(status.should_wait)

    def test_public_duration_parser_retains_its_generic_contract(self) -> None:
        self.assertEqual(
            ban_handler.parse_ban_time("1 day, 2 hours and 3 minutes"), 93780
        )


class BanLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reflected_ban_query_and_gallery_title_never_sleep_or_reload(
        self,
    ) -> None:
        for text in (_BAN_NOTICE, _BAN_NOTICE + " for 30 days"):
            with self.subTest(text=text):
                source = _search_document(text, text)
                with (
                    patch.object(
                        ban_handler, "navigate_and_wait", new=AsyncMock()
                    ) as navigate,
                    patch.object(
                        ban_handler,
                        "_read_content_before",
                        new=AsyncMock(return_value=source),
                    ) as read,
                    patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
                    patch.object(
                        ban_handler, "reload_and_wait", new=AsyncMock()
                    ) as reload,
                ):
                    await ban_handler.handle_ban_decorator(object())(
                        "https://example.test/search"
                    )
                navigate.assert_awaited_once()
                read.assert_awaited_once()
                wait.assert_not_awaited()
                reload.assert_not_awaited()

    async def test_real_notice_duration_ignores_reflected_query_and_gallery_data(
        self,
    ) -> None:
        reflected = _BAN_NOTICE + " for 30 days"
        source = _search_document(reflected, "2 hours 40 minutes").replace(
            "</body>",
            f"<p>{_BAN_NOTICE} for excessive pageloads.</p>"
            "<p>The ban expires in 1 minute.</p></body>",
        )
        with (
            patch.object(ban_handler, "navigate_and_wait", new=AsyncMock()),
            patch.object(
                ban_handler,
                "_read_content_before",
                new=AsyncMock(side_effect=[source, "Ready"]),
            ),
            patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
            patch.object(ban_handler, "reload_and_wait", new=AsyncMock()) as reload,
        ):
            await ban_handler.handle_ban_decorator(object())(
                "https://example.test/search"
            )
        wait.assert_awaited_once_with(60, deadline=None)
        reload.assert_awaited_once()

    async def test_missing_notice_duration_cannot_borrow_unrelated_time(self) -> None:
        source = _search_document(_BAN_NOTICE + " for 30 days", "1 hour").replace(
            "</body>",
            f"<p>{_BAN_NOTICE} for excessive pageloads.</p>"
            "<p>Member for 2 days.</p></body>",
        )
        with (
            patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
            patch.object(ban_handler, "reload_and_wait", new=AsyncMock()) as reload,
            self.assertRaisesRegex(RuntimeError, "positive wait duration"),
        ):
            await ban_handler._retry_until_unbanned(object(), source)
        wait.assert_not_awaited()
        reload.assert_not_awaited()

    async def test_wait_uses_only_the_adjacent_notice_duration_units(self) -> None:
        cases = (
            (f"{_BAN_NOTICE} for 1 minute. Member for 2 days.", 60),
            (f"{_BAN_NOTICE}. The ban expires in 1 day, 2 hours and 3 minutes.", 93780),
            (
                f"{_BAN_NOTICE} for excessive pageloads. "
                "The ban expires in 2 hours and 1 minute.",
                7260,
            ),
        )
        for source, seconds in cases:
            with (
                self.subTest(source=source),
                patch.object(
                    ban_handler,
                    "_read_content_before",
                    new=AsyncMock(return_value="Ready"),
                ),
                patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
                patch.object(ban_handler, "reload_and_wait", new=AsyncMock()),
            ):
                await ban_handler._retry_until_unbanned(object(), source)
            wait.assert_awaited_once_with(seconds, deadline=None)

    async def test_persistent_real_bans_continue_waiting_until_lifted(self) -> None:
        source = f"<html><body><p>{_BAN_NOTICE} for 1 minute.</p></body></html>"
        second = (
            f"<html><body><p>{_BAN_NOTICE}.</p>"
            "<p>The ban expires in 2 minutes.</p></body></html>"
        )
        ready = _search_document(_BAN_NOTICE, _BAN_NOTICE + " for 30 days")
        with (
            patch.object(
                ban_handler,
                "_read_content_before",
                new=AsyncMock(side_effect=[second, ready]),
            ) as read,
            patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
            patch.object(ban_handler, "reload_and_wait", new=AsyncMock()) as reload,
        ):
            await ban_handler._retry_until_unbanned(object(), source)
        self.assertEqual([call.args[0] for call in wait.await_args_list], [60, 120])
        self.assertEqual(reload.await_count, 2)
        self.assertEqual(read.await_count, 2)

    async def test_checkin_accepts_quoted_data_but_rejects_real_ban_notice(
        self,
    ) -> None:
        for banned in (False, True):
            body = f"<blockquote>{_BAN_NOTICE}</blockquote>"
            if banned:
                body += f"<p>{_BAN_NOTICE}.</p>"
            source = f"<html><body><main>{body}</main></body></html>"
            snapshot = _RawPageSnapshot(
                url="https://e-hentai.org/news.php",
                title="News",
                ready_state="complete",
                html=source,
                query_value=None,
            )
            driver = EHDriver(headless=True)
            with (
                self.subTest(banned=banned),
                patch.object(
                    driver, "_current_loader_id", new=AsyncMock(return_value="stable")
                ),
                patch.object(
                    driver,
                    "_read_raw_page_snapshot",
                    new=AsyncMock(return_value=snapshot),
                ),
            ):
                if banned:
                    with self.assertRaisesRegex(
                        RuntimeError, "blocked or unauthenticated"
                    ):
                        await driver._read_stable_punchin_document()
                else:
                    self.assertEqual(
                        await driver._read_stable_punchin_document(), source
                    )

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
