from __future__ import annotations

import unittest
from html import escape
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from h2h_galleryinfo_parser import GalleryURLParser

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import ban_handler
from hbrowser.gallery.eh_driver import EHDriver, _RawPageSnapshot
from hbrowser.gallery.utils import Deadline

_BAN_NOTICE = "Your IP address has been temporarily banned"


def _search_document(query: str, title: str) -> str:
    return (
        "<html><head><title>Search results</title></head><body>"
        f'<input id="f_search" value="{escape(query, quote=True)}">'
        f'<div class="searchtext">Results for {escape(query)}</div>'
        f'<div class="itg"><a href="/g/1/abcdef/">{escape(title)}</a></div>'
        "</body></html>"
    )


def _gallery_document(
    text: str,
    *,
    comment_markup: str = '<div class="c6">{text}</div>',
    data_first: bool = False,
) -> str:
    title = f'<div id="gd2"><h1 id="gn">{escape(text)}</h1></div>'
    comments = '<div id="cdiv">' + comment_markup.format(text=escape(text)) + "</div>"
    contents = comments + title if data_first else title + comments
    return (
        "<html><head><title>Gallery</title></head><body>"
        + contents
        + '<table id="taglist"><tr><td>Tags</td></tr></table></body></html>'
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
            f"<main><p>{_BAN_NOTICE}</p></main>",
            f'<div style="text-align:center">{_BAN_NOTICE}: excessive pageloads.</div>',
            '<html><body><pre style="white-space:pre-wrap">'
            f"{_BAN_NOTICE}</pre></body></html>",
            "<p>Your IP&nbsp;address has been <b>temporarily</b> banned.</p>",
        )
        for document in notices:
            with self.subTest(document=document):
                status = ban_handler.check_ban_status(document)
                self.assertTrue(status.is_banned)
                self.assertTrue(status.should_wait)

    def test_gallery_data_cannot_become_document_authority_in_any_position(
        self,
    ) -> None:
        for text in (_BAN_NOTICE, _BAN_NOTICE + " for 30 days"):
            for markup in (
                '<div class="c6">{text}</div>',
                '<div class="new-user-region">{text}</div>',
                '<div role="alert">{text}</div>',
                "<p>{text}</p>",
                "<main><section><div>{text}</div></section></main>",
                "<pre>{text}</pre>",
            ):
                for data_first in (False, True):
                    source = _gallery_document(
                        text, comment_markup=markup, data_first=data_first
                    )
                    with self.subTest(text=text, markup=markup, first=data_first):
                        self.assertFalse(
                            ban_handler.check_ban_status(source).should_wait
                        )

    def test_normal_document_cannot_be_stripped_into_a_standalone_notice(self) -> None:
        for content in (
            "<p>" + _BAN_NOTICE + "</p>" + _search_document("30 days", "1 hour"),
            _search_document("30 days", "1 hour").replace(
                "</body>", "<p>" + _BAN_NOTICE + "</p></body>"
            ),
            '<body><div class="c6">' + _BAN_NOTICE + "</div></body>",
            '<body><div role="alert">' + _BAN_NOTICE + "</div></body>",
            "<body><p>" + _BAN_NOTICE + "</p></body><p>Gallery data</p>",
            '<div class="c6"><body>' + _BAN_NOTICE + "</body></div>",
            "<body>" + _BAN_NOTICE + "</body><body>Gallery data</body>",
        ):
            with self.subTest(content=content):
                self.assertFalse(ban_handler.check_ban_status(content).should_wait)

    def test_blank_page_contract_is_unchanged(self) -> None:
        status = ban_handler.check_ban_status("<html><head></head><body></body></html>")
        self.assertFalse(status.is_banned)
        self.assertTrue(status.is_blank_page)
        self.assertTrue(status.should_wait)

    def test_public_duration_parser_retains_its_generic_contract(self) -> None:
        self.assertEqual(
            ban_handler.parse_ban_time("1 day, 2 hours and 3 minutes"), 93780
        )


class GalleryBanNavigationTests(unittest.IsolatedAsyncioTestCase):
    def _driver(self) -> EHDriver:
        browser = SimpleNamespace(tabs=[])
        element = SimpleNamespace(text="synthetic-tag", attrs={"href": "/tag/example"})
        page = SimpleNamespace(
            browser=browser,
            xpath=AsyncMock(return_value=[element]),
        )
        driver = EHDriver(headless=True)
        driver.bind_existing_browser(browser, page, owns_browser=False)
        return driver

    async def _invoke(self, driver: EHDriver, operation: str) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        if operation == "get":
            await driver.get(gallery.url)
            driver.page.xpath.assert_not_awaited()
        elif operation == "tags":
            tags = await driver.gallery2tag(gallery, "language")
            self.assertEqual([tag.name for tag in tags], ["synthetic-tag"])
            driver.page.xpath.assert_awaited_once()
        else:
            # The scripted availability response exercises the real bounded
            # download entry through its normal early return, without a popup.
            result = await driver._download(gallery, deadline=Deadline.after(10))
            self.assertFalse(result)
            driver.page.xpath.assert_awaited_once()

    async def test_gallery_user_content_never_starts_ban_wait_or_reload(self) -> None:
        for operation in ("get", "tags", "download"):
            for text in (_BAN_NOTICE, _BAN_NOTICE + " for 30 days"):
                for markup in (
                    '<div class="c6">{text}</div>',
                    '<section role="alert">{text}</section>',
                    "<p>{text}</p>",
                ):
                    source = _gallery_document(
                        text, comment_markup=markup, data_first=True
                    )
                    driver = self._driver()
                    with (
                        self.subTest(operation=operation, text=text, markup=markup),
                        patch.object(
                            ban_handler, "navigate_and_wait", new=AsyncMock()
                        ) as navigate,
                        patch.object(
                            ban_handler,
                            "_read_content_before",
                            new=AsyncMock(return_value=source),
                        ) as read,
                        patch.object(
                            ban_handler,
                            "_wait_out_ban",
                            new=AsyncMock(
                                side_effect=AssertionError(
                                    "user content cannot authorize waiting"
                                )
                            ),
                        ) as wait,
                        patch.object(
                            ban_handler,
                            "reload_and_wait",
                            new=AsyncMock(
                                side_effect=AssertionError(
                                    "user content cannot authorize reload"
                                )
                            ),
                        ) as reload,
                    ):
                        await self._invoke(driver, operation)
                    navigate.assert_awaited_once()
                    read.assert_awaited_once()
                    wait.assert_not_awaited()
                    reload.assert_not_awaited()

    async def test_gallery_entrypoints_respect_standalone_ban_and_missing_duration(
        self,
    ) -> None:
        for operation in ("get", "tags", "download"):
            for duration in (" for 1 minute", ""):
                source = f"<html><body><p>{_BAN_NOTICE}{duration}.</p></body></html>"
                driver = self._driver()
                with (
                    self.subTest(operation=operation, duration=duration),
                    patch.object(ban_handler, "navigate_and_wait", new=AsyncMock()),
                    patch.object(
                        ban_handler,
                        "_read_content_before",
                        new=AsyncMock(
                            side_effect=[source, _gallery_document("Normal title")]
                        ),
                    ) as read,
                    patch.object(ban_handler, "_wait_out_ban", new=AsyncMock()) as wait,
                    patch.object(
                        ban_handler, "reload_and_wait", new=AsyncMock()
                    ) as reload,
                ):
                    if duration:
                        await self._invoke(driver, operation)
                        wait.assert_awaited_once()
                        assert wait.await_args is not None
                        self.assertEqual(wait.await_args.args[0], 60)
                        reload.assert_awaited_once()
                        self.assertEqual(read.await_count, 2)
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "positive wait duration"
                        ):
                            await self._invoke(driver, operation)
                        driver.page.xpath.assert_not_awaited()
                        wait.assert_not_awaited()
                        reload.assert_not_awaited()


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

    async def test_embedded_notice_cannot_authorize_wait_on_a_normal_document(
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
        wait.assert_not_awaited()
        reload.assert_not_awaited()

    async def test_missing_notice_duration_cannot_borrow_unrelated_time(self) -> None:
        source = (
            f"<html><body><p>{_BAN_NOTICE} for excessive pageloads.</p>"
            "<p>Member for 2 days.</p></body></html>"
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
            body = (
                f"<p>{_BAN_NOTICE}.</p>"
                if banned
                else f"<blockquote>{_BAN_NOTICE}</blockquote>"
            )
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
