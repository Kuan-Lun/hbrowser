from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import os
import unittest
from collections import defaultdict, deque
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlencode, urlsplit

from zendriver.core.connection import ProtocolException

import hbrowser
from hbrowser import (
    ConfirmedGalleryMissing,
    GalleryFound,
    GalleryLookupError,
    GallerySearchResult,
    InvalidSearchRequestError,
    MalformedSearchPageError,
    SearchAuthenticationError,
    SearchChallengeError,
    SearchLimitExceededError,
    SearchNavigationError,
    SearchPaginationError,
    SearchRateLimitError,
    SearchRequest,
)
from hbrowser.gallery.eh_driver import (
    _SEARCH_DIAGNOSTIC_FILE_LIMIT,
    _SEARCH_DIAGNOSTIC_FILENAME_PATTERN,
    _SEARCH_PAGE_SNAPSHOT_SCRIPT,
    EHDriver,
    _locked_search_diagnostic_directory,
    _NextPageState,
    _parse_search_page,
)
from hbrowser.gallery.exh_driver import ExHDriver

EXH_HOME = "https://exhentai.org/"
SCOPE_GALLERY = "https://exhentai.org/g/10/scope00000/"
GID_349189_GALLERY = "https://exhentai.org/g/349189/f1bcce529e/"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
_DEFAULT_QUERY = object()


class DriverDomainBoundaryTests(unittest.TestCase):
    def test_generic_driver_does_not_own_hentaiverse_urls(self) -> None:
        driver = EHDriver(headless=True)

        self.assertNotIn("HentaiVerse", driver.url)
        self.assertNotIn("HentaiVerse isekai", driver.url)


def _search_url(query: str, **extra: str) -> str:
    parameters = [("f_search", query), ("f_cats", "0"), *extra.items()]
    return f"{EXH_HOME}?{urlencode(parameters)}"


def _managed_diagnostic_paths(log_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in log_dir.iterdir()
        if _SEARCH_DIAGNOSTIC_FILENAME_PATTERN.fullmatch(path.name)
        and path.is_file()
        and not path.is_symlink()
    )


def _result_page(
    query: str,
    *gallery_urls: str,
    next_href: str | None = None,
) -> str:
    anchors = "".join(
        f'<a href="{escape(url, quote=True)}">Gallery</a>' for url in gallery_urls
    )
    next_control = (
        '<span id="unext">Next</span>'
        if next_href is None
        else f'<a id="unext" href="{escape(next_href, quote=True)}">Next</a>'
    )
    return (
        "<html><head><title>Search results</title></head><body>"
        f'<input id="f_search" value="{escape(query, quote=True)}">'
        f'<div class="searchtext">Found {len(gallery_urls)} result.</div>'
        f'<div class="itg">{anchors}</div>'
        f"{next_control}"
        "</body></html>"
    )


def _no_results_page(query: str, message: str = "No hits found") -> str:
    return (
        "<html><head><title>Search results</title></head><body>"
        f'<input id="f_search" value="{escape(query, quote=True)}">'
        f'<div class="searchtext">{message}</div>'
        '<div class="itg"></div>'
        '<span id="unext">Next</span>'
        "</body></html>"
    )


def _scope_document(
    scope_url: str = EXH_HOME,
    *,
    query: str = "",
) -> _Document:
    return _Document(
        url=scope_url,
        html_reads=(_result_page(query, SCOPE_GALLERY),),
    )


@dataclass
class _Document:
    url: str
    html_reads: tuple[str, ...]
    ready_state_reads: tuple[str, ...] = ("complete",)
    title: str = "Search results"
    live_query: str | None | object = _DEFAULT_QUERY
    loader_delay_polls: int = 0
    snapshot_errors: deque[BaseException] = field(default_factory=deque)
    snapshot_read_index: int = 0
    replacement_after_snapshot: _Document | None = None

    def snapshot(self) -> dict[str, object]:
        if self.snapshot_errors:
            raise self.snapshot_errors.popleft()
        index = min(self.snapshot_read_index, len(self.html_reads) - 1)
        ready_index = min(index, len(self.ready_state_reads) - 1)
        html = self.html_reads[index]
        if self.snapshot_read_index < len(self.html_reads) - 1:
            self.snapshot_read_index += 1

        if self.live_query is _DEFAULT_QUERY:
            _, _, query, _, _ = _parse_search_page(html, self.url)
        else:
            query = cast(str | None, self.live_query)
        return {
            "url": self.url,
            "title": self.title,
            "readyState": self.ready_state_reads[ready_index],
            "html": html,
            "query": query,
        }


class _FakePage:
    def __init__(self) -> None:
        self.current_document = _Document(
            url="about:blank",
            html_reads=("<html><head></head><body></body></html>",),
            title="",
        )
        self.pending_document: _Document | None = None
        self.pending_loader_polls = 0
        self.loader_generation = 0
        self.loader_observations = list[str]()
        self.snapshot_reads_while_navigation_pending = 0
        self.completed_navigation_loaders = list[str]()

    def schedule_navigation(self, document: _Document) -> None:
        if self.pending_document is not None:
            raise AssertionError("A navigation is already pending")
        self.pending_document = document
        self.pending_loader_polls = document.loader_delay_polls

    async def send(self, command: object) -> object:
        del command
        if self.pending_document is not None:
            if self.pending_loader_polls:
                self.pending_loader_polls -= 1
            else:
                self.current_document = self.pending_document
                self.pending_document = None
                self.loader_generation += 1
                self.completed_navigation_loaders.append(
                    f"loader-{self.loader_generation}"
                )
        loader_id = f"loader-{self.loader_generation}"
        self.loader_observations.append(loader_id)
        return SimpleNamespace(frame=SimpleNamespace(loader_id=loader_id))

    async def evaluate(self, expression: str) -> object:
        if expression == _SEARCH_PAGE_SNAPSHOT_SCRIPT:
            if self.pending_document is not None:
                self.snapshot_reads_while_navigation_pending += 1
            document = self.current_document
            snapshot = document.snapshot()
            if document.replacement_after_snapshot is not None:
                replacement = document.replacement_after_snapshot
                document.replacement_after_snapshot = None
                self.schedule_navigation(replacement)
            return snapshot
        if expression == "window.location.href":
            return self.current_document.url
        raise AssertionError(f"Unexpected evaluation: {expression}")

    async def wait(self, seconds: float) -> None:
        del seconds


class _HarnessExHDriver(ExHDriver):
    def __init__(self) -> None:
        super().__init__()
        self.page = _FakePage()
        self.routes: defaultdict[str, deque[_Document]] = defaultdict(deque)
        self.get_failures: defaultdict[str, deque[BaseException]] = defaultdict(deque)
        self.get_urls = list[str]()

    def add_route(self, url: str, *documents: _Document) -> None:
        self.routes[url].extend(documents)

    async def get(self, url: str) -> None:
        self.get_urls.append(url)
        if self.get_failures[url]:
            raise self.get_failures[url].popleft()
        if not self.routes[url]:
            raise AssertionError(f"No scripted GET response for {url!r}")
        self.page.schedule_navigation(self.routes[url].popleft())


class PublicSearchContractTests(unittest.TestCase):
    def test_public_models_and_errors_are_exported(self) -> None:
        expected_exports = (
            ConfirmedGalleryMissing,
            GalleryFound,
            GalleryLookupError,
            GallerySearchResult,
            InvalidSearchRequestError,
            MalformedSearchPageError,
            SearchAuthenticationError,
            SearchChallengeError,
            SearchLimitExceededError,
            SearchNavigationError,
            SearchPaginationError,
            SearchRateLimitError,
            SearchRequest,
        )

        for exported_type in expected_exports:
            with self.subTest(exported_type=exported_type.__name__):
                self.assertIs(
                    getattr(hbrowser, exported_type.__name__),
                    exported_type,
                )

    def test_old_search_api_is_completely_removed(self) -> None:
        self.assertFalse(hasattr(EHDriver, "search2gallery"))
        signature = inspect.signature(EHDriver.search)
        self.assertEqual(tuple(signature.parameters), ("self", "request"))

    def test_search_request_rejects_invalid_bounds(self) -> None:
        invalid_arguments = (
            {"scope_url": "", "query": "gid:1"},
            {"scope_url": EXH_HOME, "query": 1},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_pages": 0},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_pages": True},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_results": 0},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_results": False},
        )

        for arguments in invalid_arguments:
            with (
                self.subTest(arguments=arguments),
                self.assertRaises(InvalidSearchRequestError),
            ):
                SearchRequest(**arguments)

    def test_confirmed_missing_requires_two_confirmations(self) -> None:
        with self.assertRaises(ValueError):
            ConfirmedGalleryMissing(gid=1, confirmations=1)


class SearchPageParserTests(unittest.TestCase):
    def test_user_gid_349189_log_fixture_contains_the_gallery(self) -> None:
        html = (FIXTURE_DIR / "exhentai_gid_349189_error_snapshot.html").read_text()

        galleries, no_results, query, next_state, next_href = _parse_search_page(
            html,
            _search_url("gid:349189"),
        )

        self.assertEqual(
            [(gallery.gid, gallery.url_key) for gallery in galleries],
            [(349189, "f1bcce529e")],
        )
        self.assertFalse(no_results)
        self.assertEqual(query, "gid:349189")
        self.assertIs(next_state, _NextPageState.END)
        self.assertIsNone(next_href)

    def test_user_speechless_log_fixture_is_explicit_empty_result(self) -> None:
        fixture = FIXTURE_DIR / "exhentai_speechless_no_hits_snapshot.html"
        html = fixture.read_text()

        galleries, no_results, query, next_state, next_href = _parse_search_page(
            html,
            _search_url('artist:"naruko hanaharu$" language:speechless$'),
        )

        self.assertEqual(galleries, ())
        self.assertTrue(no_results)
        self.assertEqual(
            query,
            'artist:"naruko hanaharu$" language:speechless$',
        )
        self.assertIs(next_state, _NextPageState.END)
        self.assertIsNone(next_href)

    def test_parser_normalizes_safe_links_and_deduplicates_logically(self) -> None:
        html = _result_page(
            "test",
            "https://exhentai.org/g/1/deadbeef00/",
            "https://e-hentai.org/g/1/deadbeef00/",
            "/g/2/cafebabe01?ignored=yes#fragment",
            "http://exhentai.org/g/3/insecure00/",
            "https://evil.example/g/4/wronghost0/",
            "https://user@exhentai.org/g/5/userinfo00/",
            "https://exhentai.org:443/g/6/port000000/",
        )

        galleries, no_results, _, next_state, _ = _parse_search_page(
            html,
            EXH_HOME,
        )

        self.assertEqual([gallery.gid for gallery in galleries], [1, 2])
        self.assertEqual(
            [gallery.url for gallery in galleries],
            [
                "https://exhentai.org/g/1/deadbeef00/",
                "https://exhentai.org/g/2/cafebabe01/",
            ],
        )
        self.assertFalse(no_results)
        self.assertIs(next_state, _NextPageState.END)

    def test_parser_recognizes_explicit_empty_result_and_pagination_states(
        self,
    ) -> None:
        no_results = _no_results_page(
            "missing",
            "<span>No unfiltered</span> results found.",
        )
        galleries, empty, query, state, href = _parse_search_page(
            no_results,
            _search_url("missing"),
        )
        self.assertEqual(galleries, ())
        self.assertTrue(empty)
        self.assertEqual(query, "missing")
        self.assertIs(state, _NextPageState.END)
        self.assertIsNone(href)

        missing = no_results.replace('<span id="unext">Next</span>', "")
        self.assertIs(
            _parse_search_page(missing, EXH_HOME)[3],
            _NextPageState.END,
        )
        invalid = no_results.replace(
            '<span id="unext">Next</span>',
            '<a id="unext">Next</a>',
        )
        self.assertIs(
            _parse_search_page(invalid, EXH_HOME)[3],
            _NextPageState.INVALID,
        )
        duplicate = no_results.replace(
            '<span id="unext">Next</span>',
            '<span id="unext">Next</span>' '<a id="unext" href="?next=2">Next</a>',
        )
        self.assertIs(
            _parse_search_page(duplicate, EXH_HOME)[3],
            _NextPageState.INVALID,
        )
        nested_link = no_results.replace(
            '<span id="unext">Next</span>',
            '<span id="unext"><a href="?next=2">Next</a></span>',
        )
        self.assertIs(
            _parse_search_page(nested_link, EXH_HOME)[3],
            _NextPageState.INVALID,
        )

    def test_ordinary_table_text_cannot_manufacture_an_empty_result(self) -> None:
        html = _result_page("test", GID_349189_GALLERY).replace(
            "</body>",
            "<table><tr><td>No hits found</td></tr></table></body>",
        )

        galleries, no_results, _, _, _ = _parse_search_page(html, EXH_HOME)

        self.assertEqual([gallery.gid for gallery in galleries], [349189])
        self.assertFalse(no_results)

    def test_ordinary_paragraph_cannot_manufacture_an_empty_result(self) -> None:
        html = _result_page("test", GID_349189_GALLERY).replace(
            "</body>",
            "<div><p>No hits found</p></div></body>",
        )

        galleries, no_results, _, _, _ = _parse_search_page(html, EXH_HOME)

        self.assertEqual([gallery.gid for gallery in galleries], [349189])
        self.assertFalse(no_results)


class SearchNavigationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sleep_patch = patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        )
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()

    async def test_gid_349189_waits_for_new_loader_and_complete_snapshot(
        self,
    ) -> None:
        query = "gid:349189"
        search_url = _search_url(query)
        fixture = (FIXTURE_DIR / "exhentai_gid_349189_error_snapshot.html").read_text()
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())
        driver.add_route(
            search_url,
            _Document(
                url=search_url,
                html_reads=(
                    _result_page(query, GID_349189_GALLERY).replace(
                        '<div class="itg">'
                        f'<a href="{GID_349189_GALLERY}">Gallery</a></div>',
                        '<div class="itg"></div>',
                    ),
                    fixture,
                ),
                ready_state_reads=("loading", "complete"),
                loader_delay_polls=1,
            ),
        )

        result = await driver.search(SearchRequest(EXH_HOME, query))

        self.assertEqual(
            [gallery.gid for gallery in result.galleries],
            [349189],
        )
        self.assertEqual(driver.get_urls, [EXH_HOME, search_url])
        self.assertEqual(driver.page.snapshot_reads_while_navigation_pending, 0)
        self.assertGreaterEqual(
            driver.page.loader_observations.count("loader-1"),
            2,
        )
        self.assertEqual(driver.page.completed_navigation_loaders[-1], "loader-2")

    async def test_real_empty_result_without_unext_completes_search(self) -> None:
        query = 'artist:"naruko hanaharu$" language:speechless$'
        search_url = _search_url(query)
        fixture = (
            FIXTURE_DIR / "exhentai_speechless_no_hits_snapshot.html"
        ).read_text()
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())
        driver.add_route(
            search_url,
            _Document(url=search_url, html_reads=(fixture,)),
        )

        result = await driver.search(SearchRequest(EXH_HOME, query))

        self.assertEqual(result.galleries, ())
        self.assertEqual(result.pages_visited, 1)
        self.assertEqual(driver.get_urls, [EXH_HOME, search_url])

    async def test_stable_interactive_snapshot_is_dom_ready(self) -> None:
        query = "gid:349189"
        search_url = _search_url(query)
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())
        driver.add_route(
            search_url,
            _Document(
                url=search_url,
                html_reads=(_result_page(query, GID_349189_GALLERY),),
                ready_state_reads=("interactive",),
            ),
        )

        result = await driver.search(SearchRequest(EXH_HOME, query))

        self.assertEqual([gallery.gid for gallery in result.galleries], [349189])
        self.assertEqual(driver.get_urls, [EXH_HOME, search_url])

    async def test_loading_timeout_saves_the_last_snapshot(self) -> None:
        query = "gid:349189"
        search_url = _search_url(query)
        html = _result_page(query, GID_349189_GALLERY)
        driver = _HarnessExHDriver()
        driver.page.current_document = _Document(
            url=search_url,
            html_reads=(html,),
            ready_state_reads=("loading",),
        )
        fake_loop = Mock()
        fake_loop.time.side_effect = (0.0, 0.0, 11.0)
        diagnostic_path = Path("/tmp/search-error.html")

        with (
            patch(
                "hbrowser.gallery.eh_driver.asyncio.get_running_loop",
                return_value=fake_loop,
            ),
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=diagnostic_path),
            ) as save_diagnostic,
            self.assertRaises(SearchNavigationError) as raised,
        ):
            await driver._read_stable_search_page(search_url, query)

        save_diagnostic.assert_awaited_once_with(html)
        self.assertEqual(raised.exception.url, search_url)
        self.assertEqual(
            raised.exception.diagnostic_path,
            str(diagnostic_path),
        )
        self.assertIn("last readyState='loading'", raised.exception.reason)
        self.assertIn(
            "stable main-frame loader=True",
            raised.exception.reason,
        )
        self.assertIn(f"actual url={search_url!r}", raised.exception.reason)
        self.assertIn("title='Search results'", raised.exception.reason)
        self.assertIn(f"query={query!r}", raised.exception.reason)
        self.assertIn("loader before='loader-0'", raised.exception.reason)
        self.assertIn("loader after='loader-0'", raised.exception.reason)

    async def test_loader_timeout_saves_the_actual_gallery_page(self) -> None:
        target_url = _search_url("gid:349189")
        gallery_html = "<html><head><title>Gallery</title></head></html>"
        driver = _HarnessExHDriver()
        driver.page.current_document = _Document(
            url=GID_349189_GALLERY,
            html_reads=(gallery_html,),
            title="Gallery",
            live_query=None,
        )
        fake_loop = Mock()
        fake_loop.time.side_effect = (0.0, 0.0, 11.0)
        diagnostic_path = Path("/tmp/gallery-timeout.html")

        with (
            patch(
                "hbrowser.gallery.eh_driver.asyncio.get_running_loop",
                return_value=fake_loop,
            ),
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=diagnostic_path),
            ) as save_diagnostic,
            self.assertRaises(SearchNavigationError) as raised,
        ):
            await driver._wait_for_new_loader("loader-0", target_url)

        save_diagnostic.assert_awaited_once_with(gallery_html)
        self.assertEqual(raised.exception.diagnostic_path, str(diagnostic_path))
        self.assertIn(
            f"actual url={GID_349189_GALLERY!r}",
            raised.exception.reason,
        )
        self.assertIn("title='Gallery'", raised.exception.reason)
        self.assertIn("query=None", raised.exception.reason)
        self.assertIn("capture loader before='loader-0'", raised.exception.reason)
        self.assertIn("capture loader after='loader-0'", raised.exception.reason)

    async def test_snapshot_is_discarded_when_loader_changes_during_read(
        self,
    ) -> None:
        query = "gid:349189"
        url = _search_url(query)
        replacement = _Document(
            url=url,
            html_reads=(_result_page(query, GID_349189_GALLERY),),
        )
        racing_document = _Document(
            url=url,
            html_reads=(_no_results_page(query),),
            replacement_after_snapshot=replacement,
        )
        driver = _HarnessExHDriver()
        driver.add_route(url, racing_document)

        snapshot = await driver._navigate_search_url(query, url)

        self.assertEqual([gallery.gid for gallery in snapshot.galleries], [349189])
        self.assertFalse(snapshot.has_no_results)
        self.assertEqual(
            driver.page.completed_navigation_loaders,
            ["loader-1", "loader-2"],
        )

    async def test_tag_scope_builds_one_trusted_search_get(self) -> None:
        tag_url = "https://exhentai.org/tag/artist/test?foo=bar&next=old&f_cats=1"
        base_query = 'artist:"test"'
        full_query = f"{base_query} language:english"
        expected_url = (
            "https://exhentai.org/?foo=bar&"
            f"{urlencode({'f_search': full_query, 'f_cats': '0'})}"
        )
        driver = _HarnessExHDriver()
        driver.add_route(
            tag_url,
            _scope_document(tag_url, query=base_query),
        )
        driver.add_route(
            expected_url,
            _Document(
                url=expected_url,
                html_reads=(_result_page(full_query, GID_349189_GALLERY),),
            ),
        )

        result = await driver.search(SearchRequest(tag_url, "language:english"))

        self.assertEqual([gallery.gid for gallery in result.galleries], [349189])
        self.assertEqual(driver.get_urls, [tag_url, expected_url])
        parsed_query = parse_qs(urlsplit(driver.get_urls[-1]).query)
        self.assertEqual(parsed_query["f_search"], [full_query])
        self.assertEqual(parsed_query["f_cats"], ["0"])
        self.assertNotIn("next", parsed_query)

    async def test_tag_scope_allows_an_empty_additional_query(self) -> None:
        tag_url = "https://exhentai.org/tag/artist/test"
        base_query = "artist:test"
        expected_url = _search_url(base_query)
        driver = _HarnessExHDriver()
        driver.add_route(
            tag_url,
            _scope_document(tag_url, query=base_query),
        )
        driver.add_route(
            expected_url,
            _Document(
                url=expected_url,
                html_reads=(_result_page(base_query, GID_349189_GALLERY),),
            ),
        )

        result = await driver.search(SearchRequest(tag_url, ""))

        self.assertEqual([gallery.gid for gallery in result.galleries], [349189])

    async def test_pagination_uses_get_and_deduplicates_by_gid_and_url_key(
        self,
    ) -> None:
        query = "artist:test"
        first_url = _search_url(query)
        second_url = _search_url(query, next="2")
        first_gallery = "https://exhentai.org/g/1/deadbeef00/"
        duplicate_on_other_origin = "https://e-hentai.org/g/1/deadbeef00/"
        second_gallery = "https://e-hentai.org/g/2/cafebabe01/"
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())
        driver.add_route(
            first_url,
            _Document(
                url=first_url,
                html_reads=(
                    _result_page(
                        query,
                        first_gallery,
                        first_gallery,
                        next_href=second_url,
                    ),
                ),
            ),
        )
        driver.add_route(
            second_url,
            _Document(
                url=second_url,
                html_reads=(
                    _result_page(
                        query,
                        duplicate_on_other_origin,
                        second_gallery,
                    ),
                ),
            ),
        )

        result = await driver.search(SearchRequest(EXH_HOME, query))

        self.assertEqual(result.pages_visited, 2)
        self.assertEqual([gallery.gid for gallery in result.galleries], [1, 2])
        self.assertEqual(result.galleries[0].url, first_gallery)
        self.assertEqual(driver.get_urls, [EXH_HOME, first_url, second_url])

    async def test_pagination_accepts_omitted_default_f_cats(self) -> None:
        query = "artist:test"
        first_url = _search_url(query)
        omitted_url = f"{EXH_HOME}?{urlencode({'f_search': query, 'next': '2'})}"
        explicit_url = _search_url(query, next="2")

        for second_request_url, second_document_url in (
            (omitted_url, explicit_url),
            (explicit_url, omitted_url),
            (omitted_url, omitted_url),
        ):
            driver = _HarnessExHDriver()
            driver.add_route(EXH_HOME, _scope_document())
            driver.add_route(
                first_url,
                _Document(
                    url=first_url,
                    html_reads=(
                        _result_page(
                            query,
                            "https://exhentai.org/g/1/deadbeef00/",
                            next_href=second_request_url,
                        ),
                    ),
                ),
            )
            driver.add_route(
                second_request_url,
                _Document(
                    url=second_document_url,
                    html_reads=(
                        _result_page(
                            query,
                            "https://exhentai.org/g/2/cafebabe01/",
                        ),
                    ),
                ),
            )

            with self.subTest(
                request_url=second_request_url,
                document_url=second_document_url,
            ):
                result = await driver.search(SearchRequest(EXH_HOME, query))

                self.assertEqual(result.pages_visited, 2)
                self.assertEqual(
                    [gallery.gid for gallery in result.galleries],
                    [1, 2],
                )
                self.assertEqual(
                    driver.get_urls,
                    [EXH_HOME, first_url, second_request_url],
                )

    async def test_empty_query_after_scope_resolution_is_rejected(self) -> None:
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())

        with self.assertRaises(InvalidSearchRequestError):
            await driver.search(SearchRequest(EXH_HOME, ""))

        self.assertEqual(driver.get_urls, [EXH_HOME])

    async def test_cross_origin_and_unsafe_scopes_are_rejected_before_get(
        self,
    ) -> None:
        invalid_scopes = (
            "http://exhentai.org/",
            "https://exhentai.org.evil/",
            "https://user@exhentai.org/",
            "https://exhentai.org:443/",
            "https://[invalid/",
            "https://exhentai.org/#fragment",
            "https://e-hentai.org/",
        )

        for scope_url in invalid_scopes:
            driver = _HarnessExHDriver()
            with (
                self.subTest(scope_url=scope_url),
                self.assertRaises(InvalidSearchRequestError),
            ):
                await driver.search(SearchRequest(scope_url, "gid:1"))
            self.assertEqual(driver.get_urls, [])

    async def test_same_origin_scope_redirect_cannot_change_the_base_query(
        self,
    ) -> None:
        requested_scope = "https://exhentai.org/tag/artist/requested"
        redirected_scope = "https://exhentai.org/tag/artist/unrelated"
        driver = _HarnessExHDriver()
        driver.add_route(
            requested_scope,
            _scope_document(redirected_scope, query="artist:unrelated"),
        )

        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=None),
            ),
            self.assertRaises(MalformedSearchPageError),
        ):
            await driver.search(SearchRequest(requested_scope, "gid:1"))

        self.assertEqual(driver.get_urls, [requested_scope])

    async def test_scope_url_rejects_ambiguous_search_query_values(self) -> None:
        scope_url = (
            "https://exhentai.org/?f_search=artist%3Aone" "&f_search=artist%3Atwo"
        )
        driver = _HarnessExHDriver()
        driver.add_route(
            scope_url,
            _scope_document(scope_url, query="artist:two"),
        )

        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=None),
            ),
            self.assertRaises(MalformedSearchPageError),
        ):
            await driver.search(SearchRequest(scope_url, "gid:1"))

        self.assertEqual(driver.get_urls, [scope_url])

    async def test_only_explicit_context_loss_retries_a_get(self) -> None:
        url = _search_url("gid:1")
        transient_driver = _HarnessExHDriver()
        transient_driver.add_route(
            url,
            _Document(
                url=url,
                html_reads=(
                    _result_page(
                        "gid:1",
                        "https://exhentai.org/g/1/deadbeef00/",
                    ),
                ),
            ),
        )
        transient_driver.get_failures[url].append(
            ProtocolException({"message": "Execution context was destroyed."})
        )

        snapshot = await transient_driver._navigate_search_url("gid:1", url)

        self.assertEqual([gallery.gid for gallery in snapshot.galleries], [1])
        self.assertEqual(transient_driver.get_urls, [url, url])

        for error in (
            RuntimeError("unknown navigation failure"),
            ProtocolException({"message": "Permission denied"}),
        ):
            driver = _HarnessExHDriver()
            driver.get_failures[url].append(error)
            with (
                self.subTest(error=error),
                self.assertRaises(SearchNavigationError),
            ):
                await driver._navigate_search_url("gid:1", url)
            self.assertEqual(driver.get_urls, [url])

    async def test_cancellation_is_never_reclassified_or_retried(self) -> None:
        url = _search_url("gid:1")
        driver = _HarnessExHDriver()
        driver.get_failures[url].append(asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await driver._navigate_search_url("gid:1", url)

        self.assertEqual(driver.get_urls, [url])

    async def test_only_transient_snapshot_context_loss_is_polled(self) -> None:
        url = _search_url("gid:1")
        document = _Document(
            url=url,
            html_reads=(
                _result_page(
                    "gid:1",
                    "https://exhentai.org/g/1/deadbeef00/",
                ),
            ),
        )
        document.snapshot_errors.append(
            ProtocolException({"message": "Cannot find context with specified id"})
        )
        driver = _HarnessExHDriver()
        driver.add_route(url, document)

        snapshot = await driver._navigate_search_url("gid:1", url)

        self.assertEqual([gallery.gid for gallery in snapshot.galleries], [1])

        failing_document = _Document(
            url=url,
            html_reads=(_result_page("gid:1", GID_349189_GALLERY),),
        )
        failing_document.snapshot_errors.append(RuntimeError("unknown read error"))
        failing_driver = _HarnessExHDriver()
        failing_driver.add_route(url, failing_document)
        with self.assertRaises(SearchNavigationError):
            await failing_driver._navigate_search_url("gid:1", url)


class SearchDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_diagnostics_keep_only_the_newest_files(self) -> None:
        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            existing_paths = list[Path]()
            for index in range(_SEARCH_DIAGNOSTIC_FILE_LIMIT + 2):
                path = log_dir / f"search_error_{index:032x}.html"
                path.write_text(f"old-{index}")
                timestamp = index + 1
                os.utime(path, ns=(timestamp, timestamp))
                existing_paths.append(path)

            manual_path = log_dir / "search_error_manual_notes.html"
            manual_path.write_text("keep me")
            ignored_directory = log_dir / f"search_error_{'e' * 32}.html"
            ignored_directory.mkdir()
            ignored_symlink = log_dir / f"search_error_{'f' * 32}.html"
            ignored_symlink.symlink_to(manual_path)

            with patch(
                "hbrowser.gallery.eh_driver.get_log_dir",
                return_value=log_dir,
            ):
                created_path = await driver._save_search_diagnostic("new")

            self.assertIsNotNone(created_path)
            assert created_path is not None
            remaining_paths = _managed_diagnostic_paths(log_dir)
            self.assertEqual(
                len(remaining_paths),
                _SEARCH_DIAGNOSTIC_FILE_LIMIT,
            )
            self.assertIn(created_path, remaining_paths)
            self.assertEqual(created_path.read_text(), "new")
            for path in existing_paths[:3]:
                self.assertFalse(path.exists())
            self.assertTrue(manual_path.exists())
            self.assertTrue(ignored_directory.is_dir())
            self.assertTrue(ignored_symlink.is_symlink())
            self.assertEqual(created_path.stat().st_mode & 0o777, 0o600)
            lock_path = log_dir / ".hbrowser-search-diagnostics.lock"
            self.assertEqual(lock_path.stat().st_mode & 0o777, 0o600)

    async def test_concurrent_writers_keep_both_new_diagnostics(self) -> None:
        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            for index in range(_SEARCH_DIAGNOSTIC_FILE_LIMIT - 1):
                path = log_dir / f"search_error_{index:032x}.html"
                path.write_text("old")
                os.utime(path, ns=(10**18, 10**18))

            with patch(
                "hbrowser.gallery.eh_driver.get_log_dir",
                return_value=log_dir,
            ):
                first_path, second_path = await asyncio.gather(
                    driver._save_search_diagnostic("first"),
                    driver._save_search_diagnostic("second"),
                )

            self.assertIsNotNone(first_path)
            self.assertIsNotNone(second_path)
            assert first_path is not None
            assert second_path is not None
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertEqual(
                {first_path.read_text(), second_path.read_text()},
                {"first", "second"},
            )
            self.assertEqual(
                len(_managed_diagnostic_paths(log_dir)),
                _SEARCH_DIAGNOSTIC_FILE_LIMIT,
            )

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_child_does_not_inherit_a_stuck_thread_lock(self) -> None:
        context = multiprocessing.get_context("fork")
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            receive_connection, send_connection = context.Pipe(duplex=False)

            def write_in_child() -> None:
                receive_connection.close()
                try:
                    path = EHDriver._write_search_diagnostic(log_dir, "child")
                except BaseException as error:
                    send_connection.send((False, repr(error)))
                else:
                    send_connection.send((True, str(path)))
                finally:
                    send_connection.close()

            with _locked_search_diagnostic_directory(log_dir):
                process = context.Process(target=write_in_child)
                process.start()

            send_connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                self.fail("fork child remained blocked by the inherited lock")

            self.assertEqual(process.exitcode, 0)
            succeeded, detail = receive_connection.recv()
            receive_connection.close()
            process.close()
            self.assertTrue(succeeded, detail)
            self.assertTrue(Path(detail).exists())

    async def test_search_diagnostics_obey_total_byte_budget(self) -> None:
        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            existing_paths = list[Path]()
            for index in range(3):
                path = log_dir / f"search_error_{index:032x}.html"
                path.write_bytes(b"old!")
                os.utime(path, ns=(index + 1, index + 1))
                existing_paths.append(path)

            with (
                patch(
                    "hbrowser.gallery.eh_driver.get_log_dir",
                    return_value=log_dir,
                ),
                patch(
                    "hbrowser.gallery.eh_driver._SEARCH_DIAGNOSTIC_TOTAL_BYTES",
                    12,
                ),
                patch(
                    "hbrowser.gallery.eh_driver._SEARCH_DIAGNOSTIC_MAX_FILE_BYTES",
                    12,
                ),
            ):
                created_path = await driver._save_search_diagnostic("new!!")

            self.assertIsNotNone(created_path)
            assert created_path is not None
            remaining_paths = _managed_diagnostic_paths(log_dir)
            self.assertLessEqual(
                sum(path.stat().st_size for path in remaining_paths),
                12,
            )
            self.assertFalse(existing_paths[0].exists())
            self.assertFalse(existing_paths[1].exists())
            self.assertTrue(existing_paths[2].exists())

    async def test_oversized_search_diagnostic_is_truncated(self) -> None:
        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with (
                patch(
                    "hbrowser.gallery.eh_driver.get_log_dir",
                    return_value=log_dir,
                ),
                patch(
                    "hbrowser.gallery.eh_driver._SEARCH_DIAGNOSTIC_MAX_FILE_BYTES",
                    256,
                ),
                patch(
                    "hbrowser.gallery.eh_driver._SEARCH_DIAGNOSTIC_TOTAL_BYTES",
                    512,
                ),
            ):
                created_path = await driver._save_search_diagnostic("x" * 1000)

            self.assertIsNotNone(created_path)
            assert created_path is not None
            content = created_path.read_bytes()
            self.assertLessEqual(len(content), 256)
            self.assertIn(b"hbrowser search diagnostic truncated", content)

    async def test_partial_write_is_removed(self) -> None:
        class _FailingWriter:
            def __init__(self, descriptor: int) -> None:
                self.descriptor = descriptor

            def __enter__(self) -> _FailingWriter:
                return self

            def __exit__(self, *args: object) -> None:
                del args
                os.close(self.descriptor)

            def write(self, content: bytes) -> int:
                os.write(self.descriptor, content[:1])
                raise OSError("simulated partial write")

        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            with (
                patch(
                    "hbrowser.gallery.eh_driver.get_log_dir",
                    return_value=log_dir,
                ),
                patch(
                    "hbrowser.gallery.eh_driver.os.fdopen",
                    side_effect=lambda descriptor, mode: _FailingWriter(descriptor),
                ),
            ):
                created_path = await driver._save_search_diagnostic("content")

            self.assertIsNone(created_path)
            self.assertEqual(_managed_diagnostic_paths(log_dir), [])

    async def test_prune_failure_does_not_add_another_file(self) -> None:
        driver = _HarnessExHDriver()
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            existing_paths = list[Path]()
            for index in range(_SEARCH_DIAGNOSTIC_FILE_LIMIT):
                path = log_dir / f"search_error_{index:032x}.html"
                path.write_text("old")
                os.utime(path, ns=(index + 1, index + 1))
                existing_paths.append(path)

            original_unlink = Path.unlink

            def fail_oldest_unlink(path: Path, *args: object, **kwargs: object) -> None:
                if path == existing_paths[0]:
                    raise PermissionError("simulated retention failure")
                original_unlink(path, *args, **kwargs)

            with (
                patch(
                    "hbrowser.gallery.eh_driver.get_log_dir",
                    return_value=log_dir,
                ),
                patch.object(
                    Path, "unlink", autospec=True, side_effect=fail_oldest_unlink
                ),
            ):
                created_path = await driver._save_search_diagnostic("new")

            self.assertIsNone(created_path)
            self.assertEqual(
                len(_managed_diagnostic_paths(log_dir)),
                _SEARCH_DIAGNOSTIC_FILE_LIMIT,
            )


class SearchValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sleep_patch = patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        )
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()

    async def _assert_scope_error(
        self,
        document: _Document,
        error_type: type[Exception],
    ) -> None:
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, document)
        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=Path("/tmp/search-error.html")),
            ),
            self.assertRaises(error_type),
        ):
            await driver.search(SearchRequest(EXH_HOME, "gid:1"))
        self.assertEqual(driver.get_urls, [EXH_HOME])

    async def test_unknown_pages_are_classified_without_blind_reload(self) -> None:
        cases = (
            (
                _Document(
                    url=EXH_HOME,
                    title="Just a moment... | Security check",
                    html_reads=(
                        '<html><body><div id="challenge-stage"></div></body></html>',
                    ),
                ),
                SearchChallengeError,
            ),
            (
                _Document(
                    url="https://forums.e-hentai.org/index.php?act=Login",
                    title="Log In",
                    html_reads=(
                        '<html><body><input name="UserName">'
                        '<input name="PassWord"></body></html>',
                    ),
                ),
                SearchAuthenticationError,
            ),
            (
                _Document(
                    url=EXH_HOME,
                    title="ExHentai.org",
                    html_reads=(
                        '<html><body><img src="/img/kokomade.jpg"></body></html>',
                    ),
                ),
                SearchAuthenticationError,
            ),
            (
                _Document(
                    url=EXH_HOME,
                    html_reads=(
                        "<html><body>Your IP address has been temporarily "
                        "banned</body></html>",
                    ),
                ),
                SearchRateLimitError,
            ),
            (
                _Document(
                    url=EXH_HOME,
                    html_reads=(
                        "<html><body>"
                        '<input id="f_search" value="">'
                        '<div class="itg"></div>'
                        '<span id="unext">Next</span>'
                        "</body></html>",
                    ),
                ),
                MalformedSearchPageError,
            ),
        )

        for document, error_type in cases:
            with self.subTest(error_type=error_type.__name__):
                await self._assert_scope_error(document, error_type)

    async def test_missing_or_invalid_next_control_fails_closed(self) -> None:
        query = "gid:1"
        url = _search_url(query)
        valid = _result_page(
            query,
            "https://exhentai.org/g/1/deadbeef00/",
        )
        pages = (
            valid.replace('<span id="unext">Next</span>', ""),
            valid.replace(
                '<span id="unext">Next</span>',
                '<a id="unext">Next</a>',
            ),
        )

        for html in pages:
            driver = _HarnessExHDriver()
            driver.add_route(EXH_HOME, _scope_document())
            driver.add_route(url, _Document(url=url, html_reads=(html,)))
            with (
                self.subTest(html=html),
                patch.object(
                    driver,
                    "_save_search_diagnostic",
                    new=AsyncMock(return_value=None),
                ),
                self.assertRaises(SearchPaginationError),
            ):
                await driver.search(SearchRequest(EXH_HOME, query))

    async def test_changed_or_unsafe_next_url_fails_closed(self) -> None:
        query = "artist:test"
        first_url = _search_url(query)
        unsafe_urls = (
            "https://e-hentai.org/?f_search=artist%3Atest&f_cats=0&next=2",
            "https://evil.example/?f_search=artist%3Atest&next=2",
            "https://exhentai.org/gallery?f_search=artist%3Atest&next=2",
            "https://exhentai.org/?f_search=artist%3Aother&next=2",
            "https://exhentai.org/?f_search=artist%3Atest&f_cats=1&next=2",
            "https://exhentai.org/?f_search=artist%3Atest&f_cats=&next=2",
            "https://exhentai.org/?f_search=artist%3Atest&f_cats=0" "&f_cats=0&next=2",
            "https://exhentai.org/?f_search=artist%3Atest&f_cats=0" "&f_cats=1&next=2",
            "https://exhentai.org/?foo=changed&f_search=artist%3Atest"
            "&f_cats=0&next=2",
            f"{first_url}#unexpected",
        )

        for unsafe_url in unsafe_urls:
            driver = _HarnessExHDriver()
            driver.add_route(EXH_HOME, _scope_document())
            driver.add_route(
                first_url,
                _Document(
                    url=first_url,
                    html_reads=(
                        _result_page(
                            query,
                            GID_349189_GALLERY,
                            next_href=unsafe_url,
                        ),
                    ),
                ),
            )
            with (
                self.subTest(unsafe_url=unsafe_url),
                patch.object(
                    driver,
                    "_save_search_diagnostic",
                    new=AsyncMock(return_value=None),
                ),
                self.assertRaises(SearchPaginationError),
            ):
                await driver.search(SearchRequest(EXH_HOME, query))
            self.assertEqual(driver.get_urls, [EXH_HOME, first_url])

    async def test_pagination_cycle_fails_before_another_get(self) -> None:
        query = "artist:test"
        first_url = _search_url(query)
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document())
        driver.add_route(
            first_url,
            _Document(
                url=first_url,
                html_reads=(
                    _result_page(
                        query,
                        GID_349189_GALLERY,
                        next_href=first_url,
                    ),
                ),
            ),
        )

        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=None),
            ),
            self.assertRaises(SearchPaginationError),
        ):
            await driver.search(SearchRequest(EXH_HOME, query))

        self.assertEqual(driver.get_urls, [EXH_HOME, first_url])

    async def test_live_or_url_query_mismatch_is_malformed(self) -> None:
        query = "gid:1"
        requested_url = _search_url(query)
        pages = (
            _Document(
                url=requested_url,
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                    ),
                ),
                live_query="gid:2",
            ),
            _Document(
                url=_search_url("gid:2"),
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                    ),
                ),
            ),
            _Document(
                url="https://exhentai.org/?f_search=gid%3A1&f_cats=1",
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                    ),
                ),
            ),
        )

        for document in pages:
            driver = _HarnessExHDriver()
            driver.add_route(EXH_HOME, _scope_document())
            driver.add_route(requested_url, document)
            with (
                self.subTest(document=document),
                patch.object(
                    driver,
                    "_save_search_diagnostic",
                    new=AsyncMock(return_value=None),
                ),
                self.assertRaises(MalformedSearchPageError),
            ):
                await driver.search(SearchRequest(EXH_HOME, query))

    async def test_page_and_result_limits_are_enforced(self) -> None:
        query = "artist:test"
        first_url = _search_url(query)
        second_url = _search_url(query, next="2")

        page_limited = _HarnessExHDriver()
        page_limited.add_route(EXH_HOME, _scope_document())
        page_limited.add_route(
            first_url,
            _Document(
                url=first_url,
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                        next_href=second_url,
                    ),
                ),
            ),
        )
        with self.assertRaises(SearchLimitExceededError):
            await page_limited.search(SearchRequest(EXH_HOME, query, max_pages=1))
        self.assertEqual(page_limited.get_urls, [EXH_HOME, first_url])

        result_limited = _HarnessExHDriver()
        result_limited.add_route(EXH_HOME, _scope_document())
        result_limited.add_route(
            first_url,
            _Document(
                url=first_url,
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                        "https://exhentai.org/g/2/cafebabe01/",
                    ),
                ),
            ),
        )
        with self.assertRaises(SearchLimitExceededError):
            await result_limited.search(SearchRequest(EXH_HOME, query, max_results=1))


class GalleryLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sleep_patch = patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        )
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()

    async def test_two_empty_confirmations_use_two_independent_result_gets(
        self,
    ) -> None:
        gid = 349189
        query = f"gid:{gid}"
        search_url = _search_url(query)
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document(), _scope_document())
        driver.add_route(
            search_url,
            _Document(url=search_url, html_reads=(_no_results_page(query),)),
            _Document(url=search_url, html_reads=(_no_results_page(query),)),
        )

        result = await driver.lookup_gid(gid)

        self.assertEqual(
            result,
            ConfirmedGalleryMissing(gid=gid, confirmations=2),
        )
        self.assertEqual(
            driver.get_urls,
            [EXH_HOME, search_url, EXH_HOME, search_url],
        )
        result_loader_ids = driver.page.completed_navigation_loaders[1::2]
        self.assertEqual(len(result_loader_ids), 2)
        self.assertEqual(len(set(result_loader_ids)), 2)

    async def test_empty_then_found_is_not_misclassified_as_missing(self) -> None:
        gid = 349189
        query = f"gid:{gid}"
        search_url = _search_url(query)
        driver = _HarnessExHDriver()
        driver.add_route(EXH_HOME, _scope_document(), _scope_document())
        driver.add_route(
            search_url,
            _Document(url=search_url, html_reads=(_no_results_page(query),)),
            _Document(
                url=search_url,
                html_reads=(_result_page(query, GID_349189_GALLERY),),
            ),
        )

        result = await driver.lookup_gid(gid)

        self.assertIsInstance(result, GalleryFound)
        assert isinstance(result, GalleryFound)
        self.assertEqual(result.requested_gid, gid)
        self.assertEqual(result.gallery.gid, gid)
        self.assertEqual(driver.get_urls.count(search_url), 2)

    async def test_lookup_errors_and_contradictions_never_become_missing(
        self,
    ) -> None:
        gid = 349189
        query = f"gid:{gid}"
        search_url = _search_url(query)

        malformed_driver = _HarnessExHDriver()
        malformed_driver.add_route(
            EXH_HOME,
            _scope_document(),
            _scope_document(),
        )
        malformed_driver.add_route(
            search_url,
            _Document(url=search_url, html_reads=(_no_results_page(query),)),
            _Document(
                url=search_url,
                html_reads=(
                    "<html><body>"
                    f'<input id="f_search" value="{query}">'
                    '<span id="unext">Next</span>'
                    "</body></html>",
                ),
            ),
        )
        with (
            patch.object(
                malformed_driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=None),
            ),
            self.assertRaises(MalformedSearchPageError),
        ):
            await malformed_driver.lookup_gid(gid)

        contradictory_driver = _HarnessExHDriver()
        contradictory_driver.add_route(EXH_HOME, _scope_document())
        contradictory_driver.add_route(
            search_url,
            _Document(
                url=search_url,
                html_reads=(
                    _result_page(
                        query,
                        GID_349189_GALLERY,
                        "https://exhentai.org/g/2/cafebabe01/",
                    ),
                ),
            ),
        )
        with self.assertRaises(GalleryLookupError):
            await contradictory_driver.lookup_gid(gid)

    async def test_lookup_validates_gid_before_any_get(self) -> None:
        for gid in (0, -1, True, "1"):
            driver = _HarnessExHDriver()
            with (
                self.subTest(gid=gid),
                self.assertRaises(InvalidSearchRequestError),
            ):
                await driver.lookup_gid(gid)  # type: ignore[arg-type]
            self.assertEqual(driver.get_urls, [])
