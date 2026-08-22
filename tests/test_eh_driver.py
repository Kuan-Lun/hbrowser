from __future__ import annotations

import asyncio
import inspect
import unittest
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, AsyncMock, Mock, call, patch
from urllib.parse import parse_qs, urlencode, urlsplit

from h2h_galleryinfo_parser import GalleryURLParser
from websockets.exceptions import ConnectionClosedError
from zendriver.core.connection import ProtocolException

import hbrowser
from hbrowser import (
    ArchiveDownloadOutcomeUnknownError,
    BrowserMutationOutcomeUnknownError,
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
    _SEARCH_PAGE_SNAPSHOT_SCRIPT,
    EHDriver,
    _NextPageState,
    _parse_search_page,
)
from hbrowser.gallery.exh_driver import ExHDriver
from hbrowser.gallery.utils import (
    Deadline,
    ZendriverOperationTimeout,
    is_browser_generation_error,
)
from hbrowser.gallery.utils.protocol import (
    _LIFECYCLE_ATTRIBUTE,
    ZendriverOwnerRetiredError,
    wait_for_zendriver,
)

EXH_HOME = "https://exhentai.org/"
SCOPE_GALLERY = "https://exhentai.org/g/10/scope00000/"
GID_349189_GALLERY = "https://exhentai.org/g/349189/f1bcce529e/"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
_DEFAULT_QUERY = object()


class _ScriptedDeadline:
    def __init__(self, *remaining: float) -> None:
        self._remaining = deque(remaining)

    def remaining(self) -> float:
        return self._remaining.popleft() if self._remaining else 0.0

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0


def _popup_after_mutation(
    new_tab: object | None,
) -> Callable[..., Awaitable[tuple[object, object | None]]]:
    async def invoke(
        _browser: object,
        _existing_tabs: object,
        mutation: Callable[[], Awaitable[object]],
        **_kwargs: object,
    ) -> tuple[object, object | None]:
        result = await mutation()
        return result, new_tab

    return invoke


async def _assert_generation_rejects_new_work(
    test_case: unittest.TestCase,
    *,
    browser: object,
    owner: object,
    exact_connection: object,
) -> None:
    lifecycle = vars(browser)[_LIFECYCLE_ATTRIBUTE]
    test_case.assertTrue(lifecycle.retired)
    test_case.assertIn(exact_connection, lifecycle.shutdown_connections)
    started = False

    async def unexpected_operation() -> None:
        nonlocal started
        started = True

    coroutine = unexpected_operation()
    with test_case.assertRaises(ZendriverOwnerRetiredError):
        await wait_for_zendriver(coroutine, timeout=1, owner=owner)
    test_case.assertFalse(started)
    test_case.assertEqual(
        inspect.getcoroutinestate(coroutine),
        inspect.CORO_CLOSED,
    )


class DriverDomainBoundaryTests(unittest.TestCase):
    def test_generic_driver_does_not_own_hentaiverse_urls(self) -> None:
        driver = EHDriver(headless=True)

        self.assertNotIn("HentaiVerse", driver.url)
        self.assertNotIn("HentaiVerse isekai", driver.url)


def _search_url(query: str, **extra: str) -> str:
    parameters = [("f_search", query), ("f_cats", "0"), *extra.items()]
    return f"{EXH_HOME}?{urlencode(parameters)}"


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
        self.get_deadlines = list[Deadline | None]()

    def add_route(self, url: str, *documents: _Document) -> None:
        self.routes[url].extend(documents)

    async def get(self, url: str, *, deadline: Deadline | None = None) -> None:
        self.get_urls.append(url)
        self.get_deadlines.append(deadline)
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
            {"scope_url": EXH_HOME, "query": "gid:1", "max_pages": 101},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_results": 0},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_results": False},
            {"scope_url": EXH_HOME, "query": "gid:1", "max_results": 5_001},
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

    async def test_snapshot_completed_after_deadline_is_rejected(self) -> None:
        driver = _HarnessExHDriver()
        driver.page.evaluate = AsyncMock(
            return_value={
                "url": EXH_HOME,
                "title": "Home",
                "readyState": "complete",
                "html": "<html></html>",
                "query": None,
            }
        )
        deadline = Mock()
        deadline.remaining.side_effect = (1.0, 0.0)

        with self.assertRaisesRegex(TimeoutError, "completed after"):
            await driver._read_raw_page_snapshot(deadline=deadline)

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
        # The harness overrides Driver.get and therefore does not emit the
        # production lifecycle receipt. The atomic loader guard discards this
        # one pre-navigation snapshot before returning the final document.
        self.assertEqual(driver.page.snapshot_reads_while_navigation_pending, 1)
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
        browser = SimpleNamespace()
        driver.browser = browser
        driver.page.browser = browser
        driver.page.websocket = object()
        driver.page.mapper = {}
        driver.page.current_document = _Document(
            url=search_url,
            html_reads=(html,),
            ready_state_reads=("loading",),
        )
        diagnostic_path = Path("/tmp/search-error.html")

        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(return_value=diagnostic_path),
            ) as save_diagnostic,
            self.assertRaises(SearchNavigationError) as raised,
        ):
            await driver._read_stable_search_page(
                search_url,
                query,
                deadline=_ScriptedDeadline(*([1.0] * 7), 0.0),  # type: ignore[arg-type]
            )

        save_diagnostic.assert_awaited_once_with(html)
        self.assertEqual(raised.exception.url, search_url)
        self.assertEqual(
            raised.exception.diagnostic_path,
            str(diagnostic_path),
        )
        self.assertIn("did not become stable", raised.exception.reason)
        self.assertIn("readyState='loading'", raised.exception.reason)
        self.assertIn(f"actual url={search_url!r}", raised.exception.reason)
        self.assertIn("title='Search results'", raised.exception.reason)
        self.assertIn(f"query={query!r}", raised.exception.reason)
        self.assertIn("loader before='loader-0'", raised.exception.reason)
        self.assertFalse(is_browser_generation_error(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        lifecycle = getattr(browser, _LIFECYCLE_ATTRIBUTE, None)
        if lifecycle is not None:
            self.assertFalse(lifecycle.retired)

    async def test_loading_timeout_diagnostic_failure_keeps_page_healthy(
        self,
    ) -> None:
        query = "gid:349189"
        search_url = _search_url(query)
        driver = _HarnessExHDriver()
        browser = SimpleNamespace()
        driver.browser = browser
        driver.page.browser = browser
        driver.page.websocket = object()
        driver.page.mapper = {}
        driver.page.current_document = _Document(
            url=search_url,
            html_reads=(_result_page(query, GID_349189_GALLERY),),
            ready_state_reads=("loading",),
        )
        with (
            patch.object(
                driver,
                "_save_search_diagnostic",
                new=AsyncMock(side_effect=OSError("diagnostic unavailable")),
            ),
            self.assertRaises(SearchNavigationError) as raised,
        ):
            await driver._read_stable_search_page(
                search_url,
                query,
                deadline=_ScriptedDeadline(*([1.0] * 7), 0.0),  # type: ignore[arg-type]
            )

        self.assertFalse(is_browser_generation_error(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIn(
            "could not save diagnostic: OSError",
            raised.exception.reason,
        )
        lifecycle = getattr(browser, _LIFECYCLE_ATTRIBUTE, None)
        if lifecycle is not None:
            self.assertFalse(lifecycle.retired)

    async def test_stable_snapshot_preserves_the_actual_gallery_page(self) -> None:
        target_url = _search_url("gid:349189")
        gallery_html = "<html><head><title>Gallery</title></head></html>"
        driver = _HarnessExHDriver()
        browser = SimpleNamespace()
        driver.browser = browser
        driver.page.browser = browser
        driver.page.websocket = object()
        driver.page.mapper = {}
        driver.page.current_document = _Document(
            url=GID_349189_GALLERY,
            html_reads=(gallery_html,),
            title="Gallery",
            live_query=None,
        )
        snapshot = await driver._read_stable_search_page(
            target_url,
            "gid:349189",
            deadline=Deadline.after(1),
        )

        self.assertEqual(snapshot.url, GID_349189_GALLERY)
        self.assertEqual(snapshot.title, "Gallery")
        self.assertIsNone(snapshot.query_value)
        self.assertEqual(snapshot.html, gallery_html)

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

    async def test_navigation_get_is_never_replayed_after_invocation(self) -> None:
        sentinel = "SENSITIVESEARCHTOKEN987654"
        query = f"gid:1 {sentinel}"
        url = _search_url(query)
        transient_driver = _HarnessExHDriver()
        transient_driver.logger = Mock()
        transient_driver.add_route(
            url,
            _Document(
                url=url,
                html_reads=(
                    _result_page(
                        query,
                        "https://exhentai.org/g/1/deadbeef00/",
                    ),
                ),
            ),
        )
        transient_driver.get_failures[url].append(
            ProtocolException({"message": "Execution context was destroyed."})
        )

        with self.assertRaises(SearchNavigationError) as raised:
            await transient_driver._navigate_search_url(query, url)

        self.assertEqual(transient_driver.get_urls, [url])
        self.assertTrue(is_browser_generation_error(raised.exception))
        self.assertIsInstance(
            raised.exception.__cause__,
            BrowserMutationOutcomeUnknownError,
        )
        transient_driver.logger.warning.assert_not_called()
        self.assertNotIn(
            sentinel,
            repr(transient_driver.logger.warning.call_args_list),
        )

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
                await driver._navigate_search_url(query, url)
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
    async def test_search_diagnostic_uses_shared_bounded_pipeline(self) -> None:
        driver = _HarnessExHDriver()
        expected_path = Path("search-error.html")
        with patch.object(
            driver,
            "save_page_diagnostic",
            new=AsyncMock(return_value=expected_path),
        ) as save_page_diagnostic:
            result = await driver._save_search_diagnostic("<html>failure</html>")

        self.assertEqual(result, expected_path)
        save_page_diagnostic.assert_awaited_once_with(
            "search_error",
            "<html>failure</html>",
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
        driver.logger = Mock()
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
        lookup_deadlines = [
            deadline for deadline in driver.get_deadlines if deadline is not None
        ]
        self.assertEqual(len(lookup_deadlines), 4)
        self.assertEqual(
            {deadline.expires_at for deadline in lookup_deadlines},
            {lookup_deadlines[0].expires_at},
        )
        result_loader_ids = driver.page.completed_navigation_loaders[1::2]
        self.assertEqual(len(result_loader_ids), 2)
        self.assertEqual(len(set(result_loader_ids)), 2)
        exact_lookup_calls = [
            logged_call
            for logged_call in driver.logger.debug.call_args_list
            if logged_call.args
            and logged_call.args[0].startswith("Exact gallery lookup")
        ]
        self.assertEqual(
            exact_lookup_calls,
            [
                call(
                    "Exact gallery lookup was empty: gid=%d confirmation=%d/%d",
                    gid,
                    1,
                    2,
                ),
                call(
                    "Exact gallery lookup was empty: gid=%d confirmation=%d/%d",
                    gid,
                    2,
                    2,
                ),
            ],
        )
        driver.logger.info.assert_called_once_with(
            "Gallery absence confirmed: gid=%d confirmations=%d",
            gid,
            2,
        )
        info_text = repr(driver.logger.info.call_args_list)
        self.assertNotIn("Gallery search started", info_text)
        self.assertNotIn("Gallery search completed", info_text)
        driver.logger.warning.assert_not_called()

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


class GalleryDownloadRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_archive_control_reuses_live_gallery_tab(self) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0
        gallery_closed = False

        async def navigation_url(_: str) -> str:
            if gallery_closed:
                raise ConnectionClosedError(None, None)
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        async def close_gallery() -> None:
            nonlocal gallery_closed
            gallery_closed = True

        archive_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [], [], [archive_link]]),
            close=AsyncMock(side_effect=close_gallery),
            activate=AsyncMock(),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            xpath=AsyncMock(return_value=[]),
            get_content=AsyncMock(
                return_value=(
                    "Downloads should start processing within a couple of minutes."
                )
            ),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        browser = SimpleNamespace(
            connection=SimpleNamespace(),
            tabs=[gallery_page],
            update_targets=AsyncMock(),
        )

        async def open_archive_tab() -> None:
            browser.tabs.append(archive_page)

        archive_link.click.side_effect = open_archive_tab
        driver.page = gallery_page
        driver.browser = browser
        driver.myget = AsyncMock()

        with patch(
            "hbrowser.gallery.eh_driver.asyncio.sleep",
            new=AsyncMock(),
        ):
            downloaded = await driver.download(gallery)

        self.assertTrue(downloaded)
        self.assertEqual(driver.myget.await_count, 1)
        self.assertEqual(gallery_page.xpath.await_count, 4)
        gallery_page.close.assert_not_awaited()
        archive_link.click.assert_awaited_once()
        archive_page.close.assert_awaited_once()
        gallery_page.activate.assert_awaited_once()
        self.assertIs(driver.page, gallery_page)

    async def test_archive_timeout_runs_no_cleanup_or_replayed_click(
        self,
    ) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        first_link = SimpleNamespace(click=AsyncMock())
        second_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [first_link], [], [second_link]]),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        first_archive_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="archive-1"),
            xpath=AsyncMock(return_value=[]),
            get_content=AsyncMock(side_effect=TimeoutError),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        second_archive_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="archive-2"),
            xpath=AsyncMock(return_value=[]),
            get_content=AsyncMock(
                return_value=(
                    "Downloads should start processing within a couple of minutes."
                )
            ),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        browser = SimpleNamespace(tabs=[gallery_page])
        gallery_page.browser = browser
        first_archive_page.browser = browser
        second_archive_page.browser = browser
        first_link._tab = gallery_page
        second_link._tab = gallery_page

        async def open_archive_page(page: object) -> None:
            browser.tabs.append(page)

        async def close_archive_page(page: object) -> None:
            browser.tabs.remove(page)

        async def open_first_archive_page() -> None:
            await open_archive_page(first_archive_page)

        async def open_second_archive_page() -> None:
            await open_archive_page(second_archive_page)

        async def close_first_archive_page() -> None:
            await close_archive_page(first_archive_page)

        async def close_second_archive_page() -> None:
            await close_archive_page(second_archive_page)

        first_link.click.side_effect = open_first_archive_page
        second_link.click.side_effect = open_second_archive_page
        first_archive_page.close.side_effect = close_first_archive_page
        second_archive_page.close.side_effect = close_second_archive_page
        driver.page = gallery_page
        driver.browser = browser
        driver.myget = AsyncMock()

        async def save_retired_diagnostic(kind: str, html: str) -> None:
            self.assertEqual((kind, html), ("download_timeout", ""))
            lifecycle = getattr(browser, _LIFECYCLE_ATTRIBUTE)
            self.assertTrue(lifecycle.retired)

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(first_archive_page)),
            ),
            patch(
                "hbrowser.gallery.eh_driver.asyncio.sleep",
                new=AsyncMock(),
            ),
            patch.object(
                driver,
                "save_page_diagnostic",
                new=AsyncMock(side_effect=save_retired_diagnostic),
            ) as save_diagnostic,
        ):
            with self.assertRaises(ArchiveDownloadOutcomeUnknownError):
                await driver.download(gallery)

        gallery_page.close.assert_not_awaited()
        first_link.click.assert_awaited_once()
        second_link.click.assert_not_awaited()
        first_archive_page.close.assert_not_awaited()
        second_archive_page.close.assert_not_awaited()
        gallery_page.activate.assert_not_awaited()
        self.assertIs(driver.page, first_archive_page)
        first_archive_page.get_content.assert_awaited_once_with()
        save_diagnostic.assert_awaited_once_with("download_timeout", "")
        await _assert_generation_rejects_new_work(
            self,
            browser=browser,
            owner=gallery_page,
            exact_connection=first_archive_page,
        )

    async def test_protocol_timeout_after_click_runs_no_browser_cleanup(
        self,
    ) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        archive_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            activate=AsyncMock(),
            xpath=AsyncMock(return_value=[]),
            get_content=AsyncMock(side_effect=timeout),
            close=AsyncMock(),
        )
        driver.page = gallery_page
        driver.browser = SimpleNamespace(tabs=[gallery_page, archive_page])
        driver.get = AsyncMock()  # type: ignore[method-assign]

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(archive_page)),
            ),
            patch("hbrowser.gallery.eh_driver.asyncio.sleep", new=AsyncMock()),
            patch.object(
                driver,
                "save_page_diagnostic",
                new=AsyncMock(),
            ) as save_diagnostic,
            patch.object(
                driver,
                "_close_archive_tab_and_restore",
                new=AsyncMock(),
            ) as close_and_restore,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver.download(gallery)

        self.assertIs(raised.exception, timeout)
        driver.get.assert_awaited_once_with(gallery.url, deadline=ANY)
        archive_link.click.assert_awaited_once_with()
        archive_page.get_content.assert_awaited_once_with()
        save_diagnostic.assert_not_awaited()
        close_and_restore.assert_not_awaited()
        archive_page.close.assert_not_awaited()

    async def test_archive_tab_activation_failure_runs_no_followup_mutation(
        self,
    ) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()
        activation_error = RuntimeError("activation dispatch failed")
        archive_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            activate=AsyncMock(),
            close=AsyncMock(),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            activate=AsyncMock(side_effect=activation_error),
            xpath=AsyncMock(),
            get_content=AsyncMock(),
            close=AsyncMock(),
        )
        driver.page = gallery_page
        driver.browser = SimpleNamespace(tabs=[gallery_page, archive_page])
        driver.get = AsyncMock()  # type: ignore[method-assign]

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(archive_page)),
            ),
            patch.object(
                driver,
                "_close_archive_tab_and_restore",
                new=AsyncMock(),
            ) as close_and_restore,
            self.assertRaises(ArchiveDownloadOutcomeUnknownError) as raised,
        ):
            await driver.download(gallery)

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        archive_link.click.assert_awaited_once_with()
        archive_page.activate.assert_awaited_once_with()
        archive_page.xpath.assert_not_awaited()
        archive_page.close.assert_not_awaited()
        gallery_page.activate.assert_not_awaited()
        close_and_restore.assert_not_awaited()

    async def test_original_link_click_failure_runs_no_followup_mutation(
        self,
    ) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()
        click_error = RuntimeError("Original click dispatch failed")
        archive_link = SimpleNamespace(click=AsyncMock())
        original_link = SimpleNamespace(click=AsyncMock(side_effect=click_error))
        gallery_page = SimpleNamespace(
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            activate=AsyncMock(),
            close=AsyncMock(),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            activate=AsyncMock(),
            xpath=AsyncMock(return_value=[original_link]),
            get_content=AsyncMock(),
            close=AsyncMock(),
        )
        driver.page = gallery_page
        driver.browser = SimpleNamespace(tabs=[gallery_page, archive_page])
        driver.get = AsyncMock()  # type: ignore[method-assign]

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(archive_page)),
            ),
            patch.object(
                driver,
                "_close_archive_tab_and_restore",
                new=AsyncMock(),
            ) as close_and_restore,
            self.assertRaises(ArchiveDownloadOutcomeUnknownError) as raised,
        ):
            await driver.download(gallery)

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        archive_link.click.assert_awaited_once_with()
        archive_page.activate.assert_awaited_once_with()
        original_link.click.assert_awaited_once_with()
        archive_page.get_content.assert_not_awaited()
        archive_page.close.assert_not_awaited()
        gallery_page.activate.assert_not_awaited()
        close_and_restore.assert_not_awaited()

    async def test_archive_click_timeout_is_not_replayed(self) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        archive_link = SimpleNamespace(
            click=AsyncMock(side_effect=ZendriverOperationTimeout(timeout_seconds=5.0))
        )
        gallery_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        browser = SimpleNamespace(tabs=[gallery_page])
        gallery_page.browser = browser
        archive_link._tab = gallery_page
        driver.page = gallery_page
        driver.browser = browser
        driver.myget = AsyncMock()

        with (
            patch("hbrowser.gallery.eh_driver.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(ZendriverOperationTimeout),
        ):
            await driver.download(gallery)

        archive_link.click.assert_awaited_once_with()
        driver.myget.assert_awaited_once_with(gallery.url, deadline=ANY)

    async def test_archive_generic_click_failure_is_not_replayed(self) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        click_error = RuntimeError("click dispatch failed")
        archive_link = SimpleNamespace(
            click=AsyncMock(side_effect=click_error),
        )
        gallery_page = SimpleNamespace(
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        driver.page = gallery_page
        driver.browser = SimpleNamespace(tabs=[gallery_page])
        driver.myget = AsyncMock()

        with (
            patch("hbrowser.gallery.eh_driver.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(ArchiveDownloadOutcomeUnknownError) as raised,
        ):
            await driver.download(gallery)

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        archive_link.click.assert_awaited_once_with()
        driver.myget.assert_awaited_once_with(gallery.url, deadline=ANY)

    async def test_missing_archive_tab_after_click_is_not_replayed(self) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        archive_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            websocket=object(),
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        browser = SimpleNamespace(tabs=[gallery_page])
        gallery_page.browser = browser
        archive_link._tab = gallery_page
        driver.page = gallery_page
        driver.browser = browser
        driver.myget = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(None)),
            ),
            patch("hbrowser.gallery.eh_driver.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(ArchiveDownloadOutcomeUnknownError),
        ):
            await driver.download(gallery)

        archive_link.click.assert_awaited_once_with()
        driver.myget.assert_awaited_once_with(gallery.url, deadline=ANY)
        lifecycle = getattr(browser, _LIFECYCLE_ATTRIBUTE, None)
        if lifecycle is not None:
            self.assertFalse(lifecycle.retired)

    async def test_client_offline_cleanup_activation_failure_is_terminal(self) -> None:
        gallery = GalleryURLParser("https://exhentai.org/g/7654321/deadbeef00/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        archive_link = SimpleNamespace(click=AsyncMock())
        gallery_page = SimpleNamespace(
            mapper={},
            target=SimpleNamespace(target_id="gallery"),
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(side_effect=[[], [archive_link]]),
            close=AsyncMock(),
            activate=AsyncMock(side_effect=RuntimeError("activation failed")),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            xpath=AsyncMock(return_value=[]),
            get_content=AsyncMock(
                return_value="Your H@H client appears to be offline."
            ),
            close=AsyncMock(),
            activate=AsyncMock(),
        )
        browser = SimpleNamespace(tabs=[gallery_page, archive_page])
        driver.page = gallery_page
        driver.browser = browser
        driver.myget = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.eh_driver.mutate_and_wait_for_new_tab",
                new=AsyncMock(side_effect=_popup_after_mutation(archive_page)),
            ),
            patch(
                "hbrowser.gallery.eh_driver.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            with self.assertRaises(ArchiveDownloadOutcomeUnknownError) as raised:
                await driver.download(gallery)

        gallery_page.close.assert_not_awaited()
        archive_page.close.assert_awaited_once()
        gallery_page.activate.assert_awaited_once()
        self.assertIs(driver.page, gallery_page)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertIn(
            "Archive operation also failed before cleanup: ClientOfflineException",
            raised.exception.__notes__,
        )

    async def test_archive_close_failure_skips_gallery_activation(self) -> None:
        driver = EHDriver(headless=True)
        close_error = RuntimeError("close dispatch failed")
        gallery_page = SimpleNamespace(
            target=SimpleNamespace(target_id="gallery"),
            activate=AsyncMock(),
        )
        archive_page = SimpleNamespace(
            target=SimpleNamespace(target_id="archive"),
            close=AsyncMock(side_effect=close_error),
        )
        driver.browser = SimpleNamespace(tabs=[gallery_page, archive_page])
        driver.page = archive_page

        with self.assertRaises(ArchiveDownloadOutcomeUnknownError) as raised:
            await driver._close_archive_tab_and_restore(
                gallery_page,
                archive_page,
            )

        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        archive_page.close.assert_awaited_once_with()
        gallery_page.activate.assert_not_awaited()
        self.assertIs(driver.page, archive_page)


class GalleryDownloadLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_gallery_token_never_reaches_logs_or_terminal_retry_error(
        self,
    ) -> None:
        sentinel = "SENSITIVEGALLERYTOKEN987654"
        gallery = GalleryURLParser(f"https://exhentai.org/g/7654321/{sentinel}/")
        driver = EHDriver(headless=True)
        driver.logger = Mock()

        evaluation_count = 0

        async def navigation_url(_: str) -> str:
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count % 2:
                return "about:blank"
            return gallery.url

        page = SimpleNamespace(
            mapper={},
            evaluate=AsyncMock(side_effect=navigation_url),
            xpath=AsyncMock(return_value=[]),
            close=AsyncMock(),
        )
        driver.page = page
        driver.browser = SimpleNamespace(tabs=[])
        driver.myget = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.eh_driver.wait_for_xpath",
                new=AsyncMock(side_effect=TimeoutError),
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await driver.download(gallery)

        self.assertIn("did not become ready", str(raised.exception))
        self.assertIn("gid=7654321", str(raised.exception))
        driver.logger.warning.assert_any_call(
            "Archive Download control did not become ready before its "
            "shared deadline: gid=%d error_type=%s",
            gallery.gid,
            "TimeoutError",
        )
        self.assertNotIn(sentinel, repr(driver.logger.method_calls))
        self.assertNotIn(sentinel, str(raised.exception))
        page.close.assert_not_awaited()
