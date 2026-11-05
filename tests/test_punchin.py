from __future__ import annotations

import unittest
from html import escape
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import hbrowser
from hbrowser import PunchInComplete, RandomEncounterFound
from hbrowser.gallery.eh_driver import (
    EHDriver,
    _normalize_random_encounter_url,
    _parse_punch_in_result,
    _RawPageSnapshot,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ENCOUNTER = (
    "MzU5ODcxOS0xNzg1OTk0MDc4LTk5NjEyYzIzNWY2YTgwNmE0OTk3YjgzN2ZhOWQyNzM5NWJmYzkyZGY="
)
ENCOUNTER_URL = f"https://hentaiverse.org/?s=Battle&ss=ba&encounter={ENCOUNTER}"


def _news_page(content: str = "News content") -> str:
    return (
        "<html><head><title>E-Hentai Galleries</title></head><body>"
        '<div id="newsouter"><div id="newsinner">'
        '<nav id="nb"><a href="https://e-hentai.org/news.php">News</a></nav>'
        f'<main id="nt">{content}</main></div></div></body></html>'
    )


def _event_page(*hrefs: str) -> str:
    anchors = "".join(
        f'<a href="{escape(href, quote=True)}">Encounter</a>' for href in hrefs
    )
    return _news_page(f'<div id="eventpane">{anchors}</div>')


class RandomEncounterURLTests(unittest.TestCase):
    def test_accepts_and_canonicalizes_the_expected_url(self) -> None:
        result = _normalize_random_encounter_url(
            f"HTTPS://HENTAIVERSE.ORG/?encounter={ENCOUNTER}&ss=ba&s=Battle"
        )

        self.assertIsNotNone(result)
        assert result is not None
        parsed = urlsplit(result)
        self.assertEqual(
            (parsed.scheme, parsed.netloc, parsed.path),
            ("https", "hentaiverse.org", "/"),
        )
        self.assertEqual(
            parse_qs(parsed.query, keep_blank_values=True),
            {"s": ["Battle"], "ss": ["ba"], "encounter": [ENCOUNTER]},
        )

    def test_accepts_query_keys_after_one_url_decoding_pass(self) -> None:
        for key in (
            "%65ncounter",
            "encoun%74er",
            "%65%6E%63%6f%75%6e%74%65%72",
            "en\tcou\r\nnter",
        ):
            with self.subTest(key=key):
                result = _normalize_random_encounter_url(
                    f"https://hentaiverse.org/?s=Battle&ss=ba&{key}={ENCOUNTER}"
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(
                    parse_qs(urlsplit(result).query),
                    {"s": ["Battle"], "ss": ["ba"], "encounter": [ENCOUNTER]},
                )

    def test_rejects_keys_requiring_extra_decoding_or_space_removal(self) -> None:
        for key in ("%2565ncounter", "en+counter", "encounter+", "en%09counter"):
            with self.subTest(key=key):
                self.assertIsNone(
                    _normalize_random_encounter_url(
                        f"https://hentaiverse.org/?s=Battle&ss=ba&{key}={ENCOUNTER}"
                    )
                )

    def test_rejects_untrusted_or_malformed_urls(self) -> None:
        invalid_urls = {
            "relative": f"/?s=Battle&ss=ba&encounter={ENCOUNTER}",
            "insecure scheme": (
                f"http://hentaiverse.org/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "lookalike host": (
                f"https://hentaiverse.org.example/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "credentials": (
                f"https://user@hentaiverse.org/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "explicit port": (
                f"https://hentaiverse.org:443/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "invalid port": (
                "https://hentaiverse.org:not-a-port/?s=Battle&ss=ba&encounter="
                f"{ENCOUNTER}"
            ),
            "wrong path": (
                f"https://hentaiverse.org/battle?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "fragment": f"{ENCOUNTER_URL}#battle",
            "extra query field": f"{ENCOUNTER_URL}&next=unsafe",
            "missing s": f"https://hentaiverse.org/?ss=ba&encounter={ENCOUNTER}",
            "wrong s": (
                f"https://hentaiverse.org/?s=battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "duplicate s": f"{ENCOUNTER_URL}&s=Battle",
            "missing ss": f"https://hentaiverse.org/?s=Battle&encounter={ENCOUNTER}",
            "wrong ss": (
                f"https://hentaiverse.org/?s=Battle&ss=arena&encounter={ENCOUNTER}"
            ),
            "duplicate ss": f"{ENCOUNTER_URL}&ss=ba",
            "missing encounter": "https://hentaiverse.org/?s=Battle&ss=ba",
            "blank encounter": "https://hentaiverse.org/?s=Battle&ss=ba&encounter=",
            "whitespace encounter": (
                "https://hentaiverse.org/?s=Battle&ss=ba&encounter=%20%09"
            ),
            "duplicate encounter": f"{ENCOUNTER_URL}&encounter={ENCOUNTER}",
            "malformed query": ("https://hentaiverse.org/?s=Battle&ss=ba&encounter"),
        }

        for reason, url in invalid_urls.items():
            with self.subTest(reason=reason):
                self.assertIsNone(_normalize_random_encounter_url(url))


class PunchInDocumentTests(unittest.TestCase):
    def test_error_login_challenge_and_partial_documents_cannot_complete(self) -> None:
        documents = [
            "",
            "<html><title>Bad Gateway</title><body>502</body></html>",
            "<html><title>Just a moment...</title><body>Checking</body></html>",
            '<html><title>Login</title><input name="password"></html>',
            '<html><title>E-Hentai Galleries</title><div id="nt">News</div></html>',
        ]
        for document in documents:
            with (
                self.subTest(document=document),
                self.assertRaisesRegex(RuntimeError, "recognized news document"),
            ):
                _parse_punch_in_result(document)

    def test_realistic_news_fixture_returns_the_random_encounter(self) -> None:
        html_content = (FIXTURE_DIR / "ehentai_news_random_encounter.html").read_text(
            encoding="utf-8"
        )

        result = _parse_punch_in_result(html_content)

        self.assertIsInstance(result, RandomEncounterFound)
        assert isinstance(result, RandomEncounterFound)
        self.assertEqual(
            parse_qs(urlsplit(result.url).query),
            {"s": ["Battle"], "ss": ["ba"], "encounter": [ENCOUNTER]},
        )

    def test_ignores_hentaiverse_links_outside_the_event_pane(self) -> None:
        html_content = _news_page(
            f'<nav><a href="{escape(ENCOUNTER_URL, quote=True)}">HentaiVerse</a></nav>'
        )

        self.assertEqual(_parse_punch_in_result(html_content), PunchInComplete())

    def test_html_entity_encoded_keys_produce_the_expected_encounter(self) -> None:
        for key in ("&#101;ncounter", "en&#x63;ounter", "&#x25;65ncounter"):
            with self.subTest(key=key):
                html_content = _news_page(
                    '<div id="eventpane"><a href="https://hentaiverse.org/'
                    f'?s=Battle&amp;ss=ba&amp;{key}={ENCOUNTER}">'
                    "Encounter</a></div>"
                )

                result = _parse_punch_in_result(html_content)

                self.assertIsInstance(result, RandomEncounterFound)
                assert isinstance(result, RandomEncounterFound)
                self.assertEqual(
                    parse_qs(urlsplit(result.url).query)["encounter"], [ENCOUNTER]
                )

    def test_rejects_ambiguous_event_documents(self) -> None:
        documents = {
            "two encounters": _event_page(ENCOUNTER_URL, ENCOUNTER_URL),
            "two event panes": (
                _event_page(ENCOUNTER_URL).replace(
                    "</body>",
                    '<div id="eventpane"></div></body>',
                )
            ),
            "unsafe encounter": _event_page(
                f"https://example.com/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
        }

        for reason, html_content in documents.items():
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(
                    RuntimeError,
                    "random encounter markup",
                ),
            ):
                _parse_punch_in_result(html_content)

    def test_unrelated_event_markup_is_a_completed_check_in(self) -> None:
        html_content = _event_page("https://e-hentai.org/news.php")

        self.assertEqual(_parse_punch_in_result(html_content), PunchInComplete())


class EHDriverPunchInTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        capture_environment = patch.dict(
            environ,
            {"HBROWSER_CAPTURE_PUNCHIN_PAGES": ""},
        )
        capture_environment.start()
        self.addCleanup(capture_environment.stop)

    async def test_initial_encounter_is_returned_without_reload(self) -> None:
        html_content = (FIXTURE_DIR / "ehentai_news_random_encounter.html").read_text(
            encoding="utf-8"
        )
        events = list[str]()

        async def get(_url: str, **_kwargs: object) -> None:
            events.append("get")

        async def get_content(**_kwargs: object) -> str:
            events.append("content")
            return html_content

        driver = EHDriver(headless=True)
        driver.get = AsyncMock(side_effect=get)  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=get_content
        )
        driver.logger = Mock()

        result = await driver.punchin()

        self.assertIsInstance(result, RandomEncounterFound)
        self.assertEqual(events, ["get", "content"])
        driver.logger.info.assert_any_call("Daily check-in found a random encounter")
        self.assertNotIn(ENCOUNTER, str(driver.logger.mock_calls))

    async def test_reload_is_checked_when_initial_page_has_no_encounter(self) -> None:
        encounter_content = (
            FIXTURE_DIR / "ehentai_news_random_encounter.html"
        ).read_text(encoding="utf-8")
        events = list[str]()

        async def get(_url: str, **_kwargs: object) -> None:
            events.append("get")

        async def reload() -> None:
            events.append("reload")

        async def get_content(**_kwargs: object) -> str:
            events.append("content")
            return (
                _news_page("Initial news")
                if events.count("content") == 1
                else encounter_content
            )

        driver = EHDriver(headless=True)
        driver.get = AsyncMock(side_effect=get)  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(side_effect=reload),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=get_content
        )
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            return_value="initial-loader"
        )
        driver.logger = Mock()

        result = await driver.punchin()

        self.assertIsInstance(result, RandomEncounterFound)
        self.assertEqual(events, ["get", "content", "reload", "content"])
        self.assertEqual(
            driver._read_stable_punchin_document.await_args_list,
            [
                unittest.mock.call(deadline=unittest.mock.ANY),
                unittest.mock.call(
                    previous_loader_id="initial-loader",
                    deadline=unittest.mock.ANY,
                ),
            ],
        )
        driver.logger.info.assert_any_call(
            "Daily check-in found a random encounter after refresh"
        )
        self.assertNotIn(ENCOUNTER, str(driver.logger.mock_calls))

    async def test_opt_in_capture_saves_initial_and_reloaded_documents(self) -> None:
        initial_content = _news_page("Initial news")
        reloaded_content = _event_page(ENCOUNTER_URL)
        driver = EHDriver(headless=True)
        driver.get = AsyncMock()  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=[initial_content, reloaded_content],
        )
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            return_value="initial-loader"
        )
        driver.logger = Mock()
        driver.save_page_diagnostic = AsyncMock()  # type: ignore[method-assign]

        with patch.dict(
            environ,
            {"HBROWSER_CAPTURE_PUNCHIN_PAGES": "1"},
        ):
            result = await driver.punchin()

        self.assertIsInstance(result, RandomEncounterFound)
        self.assertEqual(driver._read_stable_punchin_document.await_count, 2)
        self.assertEqual(
            driver.save_page_diagnostic.await_args_list,
            [
                unittest.mock.call("punchin_initial", initial_content),
                unittest.mock.call("punchin_reloaded", reloaded_content),
            ],
        )

    async def test_opt_in_capture_redacts_encoded_key_but_returns_real_encounter(
        self,
    ) -> None:
        cases = (
            ("%65ncounter", ENCOUNTER, ENCOUNTER),
            ("&#x25;65ncounter", ENCOUNTER, ENCOUNTER),
            ("%65ncounter", "SPACE-PREFIX SPACE-TAIL", "SPACE-PREFIX SPACE-TAIL"),
            (
                "&#101;ncounter",
                "QUOTE-PREFIX&quot;QUOTE-TAIL",
                'QUOTE-PREFIX"QUOTE-TAIL',
            ),
        )
        for key, html_value, capability in cases:
            with self.subTest(key=key, html_value=html_value):
                content = _news_page(
                    '<div id="eventpane"><a href="https://hentaiverse.org/'
                    f'?s=Battle&amp;ss=ba&amp;{key}={html_value}">'
                    "Encounter</a></div>"
                )
                driver = EHDriver(headless=True)
                driver.page = SimpleNamespace(reload=AsyncMock())
                driver.logger = Mock()
                with (
                    TemporaryDirectory() as directory_name,
                    patch.dict(environ, {"HBROWSER_CAPTURE_PUNCHIN_PAGES": "1"}),
                    patch.object(driver, "get", new=AsyncMock()),
                    patch.object(
                        driver,
                        "_read_stable_punchin_document",
                        new=AsyncMock(return_value=content),
                    ),
                    patch(
                        "hbrowser.gallery.browser.page_diagnostic."
                        "_page_diagnostic_directory_hint",
                        return_value=Path(directory_name),
                    ),
                ):
                    result = await driver.punchin()
                    paths = list(Path(directory_name).glob("punchin_initial_*.html"))
                    self.assertEqual(len(paths), 1)
                    persisted = paths[0].read_text()

                self.assertIsInstance(result, RandomEncounterFound)
                assert isinstance(result, RandomEncounterFound)
                self.assertEqual(
                    parse_qs(urlsplit(result.url).query)["encounter"], [capability]
                )
                for fragment in capability.replace('"', " ").split():
                    self.assertNotIn(fragment, persisted)
                    self.assertNotIn(fragment, repr(driver.logger.mock_calls))
                saved_result = _parse_punch_in_result(persisted)
                self.assertIsInstance(saved_result, RandomEncounterFound)
                assert isinstance(saved_result, RandomEncounterFound)
                self.assertEqual(
                    parse_qs(urlsplit(saved_result.url).query)["encounter"],
                    ["REDACTED"],
                )
                driver.page.reload.assert_not_awaited()

    async def test_no_event_returns_a_typed_completed_outcome(self) -> None:
        driver = EHDriver(headless=True)
        driver.get = AsyncMock()  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                _news_page("Initial news"),
                _news_page("Reloaded news"),
            ]
        )
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            return_value="initial-loader"
        )
        driver.logger = Mock()

        result = await driver.punchin()

        self.assertIsInstance(result, PunchInComplete)
        self.assertIsInstance(result, hbrowser.PunchInComplete)

    async def test_non_string_page_content_is_rejected(self) -> None:
        driver = EHDriver(headless=True)
        driver.get = AsyncMock()  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            return_value=None,
        )
        driver.logger = Mock()

        with self.assertRaisesRegex(TypeError, "content was not a string"):
            await driver.punchin()

    async def test_reload_error_page_does_not_report_completed(self) -> None:
        driver = EHDriver(headless=True)
        logger = Mock()
        with (
            patch.object(driver, "get", new=AsyncMock()),
            patch.object(
                driver, "_current_loader_id", new=AsyncMock(return_value="old")
            ),
            patch.object(
                driver,
                "_read_stable_punchin_document",
                new=AsyncMock(
                    side_effect=[_news_page(), "<html>502 Bad Gateway</html>"]
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "recognized news document"),
        ):
            driver.page = SimpleNamespace(reload=AsyncMock())
            driver.logger = logger
            await driver.punchin()
        self.assertNotIn("Daily check-in completed", str(logger.mock_calls))


class PunchInDocumentReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_redirects_and_challenges_even_with_news_markup(self) -> None:
        snapshots = [
            _RawPageSnapshot(
                url=url,
                title="E-Hentai Galleries",
                ready_state="complete",
                html=_news_page(),
                query_value=None,
            )
            for url in (
                "https://forums.e-hentai.org/index.php?act=Login",
                "https://e-hentai.org.example/news.php",
                "http://e-hentai.org/news.php",
                "https://user@e-hentai.org/news.php",
                "https://e-hentai.org/news.php?redirect=1",
            )
        ]
        snapshots.extend(
            _RawPageSnapshot(
                url="https://e-hentai.org/news.php",
                title=title,
                ready_state="complete",
                html=_news_page(content),
                query_value=None,
            )
            for title, content in (
                ("Just a moment...", "News"),
                ("E-Hentai Galleries", '<div id="challenge-form"></div>'),
                ("E-Hentai Galleries", "You are not logged in"),
            )
        )
        for snapshot in snapshots:
            with self.subTest(snapshot=snapshot):
                driver = EHDriver(headless=True)
                with (
                    patch.object(
                        driver, "_current_loader_id", new=AsyncMock(return_value="new")
                    ),
                    patch.object(
                        driver,
                        "_read_raw_page_snapshot",
                        new=AsyncMock(return_value=snapshot),
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    await driver._read_stable_punchin_document(previous_loader_id="old")

    async def test_rejects_previous_loader_without_polling(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            return_value="old",
        )
        driver._read_raw_page_snapshot = AsyncMock()  # type: ignore[method-assign]

        with (
            patch(
                "hbrowser.gallery.eh_driver.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(RuntimeError, "lifecycle-confirmed"),
        ):
            await driver._read_stable_punchin_document(
                previous_loader_id="old",
            )

        sleep.assert_not_awaited()
        driver._read_raw_page_snapshot.assert_not_awaited()

    async def test_accepts_one_lifecycle_confirmed_stable_snapshot(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            side_effect=["stable", "stable"],
        )
        driver._read_raw_page_snapshot = AsyncMock(  # type: ignore[method-assign]
            return_value=_RawPageSnapshot(
                url="https://e-hentai.org/news.php",
                title="",
                ready_state="complete",
                html="<html><body>Stable</body></html>",
                query_value=None,
            ),
        )

        with patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await driver._read_stable_punchin_document()

        self.assertEqual(result, "<html><body>Stable</body></html>")
        sleep.assert_not_awaited()
        driver._read_raw_page_snapshot.assert_awaited_once_with(deadline=None)

    async def test_loader_change_during_snapshot_is_not_polled(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            side_effect=["first", "second"],
        )
        driver._read_raw_page_snapshot = AsyncMock(  # type: ignore[method-assign]
            return_value=_RawPageSnapshot(
                url="https://e-hentai.org/news.php",
                title="",
                ready_state="complete",
                html="<html><body>Unstable</body></html>",
                query_value=None,
            )
        )

        with (
            patch(
                "hbrowser.gallery.eh_driver.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(RuntimeError, "changed after lifecycle"),
        ):
            await driver._read_stable_punchin_document()

        sleep.assert_not_awaited()
        driver._read_raw_page_snapshot.assert_awaited_once()
