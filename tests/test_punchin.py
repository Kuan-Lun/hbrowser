from __future__ import annotations

import unittest
from html import escape
from os import environ
from pathlib import Path
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
ENCOUNTER_URL = "https://hentaiverse.org/?s=Battle&ss=ba&encounter=" f"{ENCOUNTER}"


def _event_page(*hrefs: str) -> str:
    anchors = "".join(
        f'<a href="{escape(href, quote=True)}">Encounter</a>' for href in hrefs
    )
    return f'<html><body><div id="eventpane">{anchors}</div></body></html>'


class RandomEncounterURLTests(unittest.TestCase):
    def test_accepts_and_canonicalizes_the_expected_url(self) -> None:
        result = _normalize_random_encounter_url(
            "HTTPS://HENTAIVERSE.ORG/?encounter=" f"{ENCOUNTER}&ss=ba&s=Battle"
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

    def test_rejects_untrusted_or_malformed_urls(self) -> None:
        invalid_urls = {
            "relative": f"/?s=Battle&ss=ba&encounter={ENCOUNTER}",
            "insecure scheme": (
                f"http://hentaiverse.org/?s=Battle&ss=ba&encounter={ENCOUNTER}"
            ),
            "lookalike host": (
                "https://hentaiverse.org.example/?s=Battle&ss=ba&encounter="
                f"{ENCOUNTER}"
            ),
            "credentials": (
                "https://user@hentaiverse.org/?s=Battle&ss=ba&encounter=" f"{ENCOUNTER}"
            ),
            "explicit port": (
                "https://hentaiverse.org:443/?s=Battle&ss=ba&encounter=" f"{ENCOUNTER}"
            ),
            "invalid port": (
                "https://hentaiverse.org:not-a-port/?s=Battle&ss=ba&encounter="
                f"{ENCOUNTER}"
            ),
            "wrong path": (
                "https://hentaiverse.org/battle?s=Battle&ss=ba&encounter="
                f"{ENCOUNTER}"
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
        html_content = (
            '<html><body><nav><a href="'
            f"{escape(ENCOUNTER_URL, quote=True)}"
            '">HentaiVerse</a></nav></body></html>'
        )

        self.assertEqual(_parse_punch_in_result(html_content), PunchInComplete())

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

        async def get(_url: str) -> None:
            events.append("get")

        async def get_content(**_kwargs: object) -> str:
            events.append("content")
            return html_content

        driver = EHDriver(headless=True)
        driver.get = AsyncMock(side_effect=get)  # type: ignore[method-assign]
        driver.wait = AsyncMock()  # type: ignore[method-assign]
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
        driver.wait.assert_not_awaited()
        self.assertNotIn(ENCOUNTER, str(driver.logger.mock_calls))

    async def test_reload_is_checked_when_initial_page_has_no_encounter(self) -> None:
        encounter_content = (
            FIXTURE_DIR / "ehentai_news_random_encounter.html"
        ).read_text(encoding="utf-8")
        events = list[str]()

        async def get(_url: str) -> None:
            events.append("get")

        async def wait(_function: object, *, ischangeurl: bool) -> None:
            self.assertFalse(ischangeurl)
            events.append("reload")

        async def get_content(**_kwargs: object) -> str:
            events.append("content")
            return (
                "<html><body>Initial news</body></html>"
                if events.count("content") == 1
                else encounter_content
            )

        driver = EHDriver(headless=True)
        driver.get = AsyncMock(side_effect=get)  # type: ignore[method-assign]
        driver.wait = AsyncMock(side_effect=wait)  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
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
                unittest.mock.call(),
                unittest.mock.call(previous_loader_id="initial-loader"),
            ],
        )
        self.assertNotIn(ENCOUNTER, str(driver.logger.mock_calls))

    async def test_opt_in_capture_saves_initial_and_reloaded_documents(self) -> None:
        initial_content = "<html><body>Initial news</body></html>"
        reloaded_content = _event_page(ENCOUNTER_URL)
        driver = EHDriver(headless=True)
        driver.get = AsyncMock()  # type: ignore[method-assign]
        driver.wait = AsyncMock()  # type: ignore[method-assign]
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
        driver._save_page_diagnostic = AsyncMock()  # type: ignore[method-assign]

        with patch.dict(
            environ,
            {"HBROWSER_CAPTURE_PUNCHIN_PAGES": "1"},
        ):
            result = await driver.punchin()

        self.assertIsInstance(result, RandomEncounterFound)
        self.assertEqual(driver._read_stable_punchin_document.await_count, 2)
        self.assertEqual(
            driver._save_page_diagnostic.await_args_list,
            [
                unittest.mock.call("punchin_initial", initial_content),
                unittest.mock.call("punchin_reloaded", reloaded_content),
            ],
        )

    async def test_no_event_returns_a_typed_completed_outcome(self) -> None:
        driver = EHDriver(headless=True)
        driver.get = AsyncMock()  # type: ignore[method-assign]
        driver.wait = AsyncMock()  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                "<html><body>Initial news</body></html>",
                "<html><body>Reloaded news</body></html>",
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
        driver.wait = AsyncMock()  # type: ignore[method-assign]
        driver.page = SimpleNamespace(
            reload=AsyncMock(),
        )
        driver._read_stable_punchin_document = AsyncMock(  # type: ignore[method-assign]
            return_value=None,
        )
        driver.logger = Mock()

        with self.assertRaisesRegex(TypeError, "content was not a string"):
            await driver.punchin()


class PunchInDocumentReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_a_new_dom_ready_loader_after_reload(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            side_effect=["old", "new", "new", "new", "new"],
        )
        driver._read_raw_page_snapshot = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                _RawPageSnapshot(
                    url="https://e-hentai.org/news.php",
                    title="",
                    ready_state="loading",
                    html="<html><body>Loading</body></html>",
                    query_value=None,
                ),
                _RawPageSnapshot(
                    url="https://e-hentai.org/news.php",
                    title="",
                    ready_state="complete",
                    html="<html><body>Ready</body></html>",
                    query_value=None,
                ),
            ]
        )

        with patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await driver._read_stable_punchin_document(
                previous_loader_id="old",
            )

        self.assertEqual(result, "<html><body>Ready</body></html>")
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(driver._read_raw_page_snapshot.await_count, 2)

    async def test_retries_when_loader_changes_during_snapshot(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            side_effect=["first", "second", "second", "second"],
        )
        driver._read_raw_page_snapshot = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                _RawPageSnapshot(
                    url="https://e-hentai.org/news.php",
                    title="",
                    ready_state="complete",
                    html="<html><body>Unstable</body></html>",
                    query_value=None,
                ),
                _RawPageSnapshot(
                    url="https://e-hentai.org/news.php",
                    title="",
                    ready_state="complete",
                    html="<html><body>Stable</body></html>",
                    query_value=None,
                ),
            ]
        )

        with patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await driver._read_stable_punchin_document()

        self.assertEqual(result, "<html><body>Stable</body></html>")
        sleep.assert_awaited_once()

    async def test_times_out_if_reload_keeps_the_previous_loader(self) -> None:
        driver = EHDriver(headless=True)
        driver._current_loader_id = AsyncMock(  # type: ignore[method-assign]
            return_value="old",
        )
        driver._read_raw_page_snapshot = AsyncMock()  # type: ignore[method-assign]
        loop = Mock()
        loop.time.side_effect = [0.0, 0.0, 1.0]

        with (
            patch(
                "hbrowser.gallery.eh_driver.asyncio.get_running_loop",
                return_value=loop,
            ),
            patch(
                "hbrowser.gallery.eh_driver.PUNCHIN_PAGE_TIMEOUT_SECONDS",
                0.5,
            ),
            patch(
                "hbrowser.gallery.eh_driver.asyncio.sleep",
                new=AsyncMock(),
            ),
            self.assertRaisesRegex(RuntimeError, "stable DOM-ready document"),
        ):
            await driver._read_stable_punchin_document(
                previous_loader_id="old",
            )

        driver._read_raw_page_snapshot.assert_not_awaited()
