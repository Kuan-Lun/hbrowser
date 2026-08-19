"""E-Hentai Driver 實現"""

import asyncio
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import random
from threading import Lock
from typing import Any, Never
from urllib.parse import (
    parse_qs,
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)
from uuid import uuid4

from bs4 import BeautifulSoup
from h2h_galleryinfo_parser import GalleryURLParser
from zendriver import cdp
from zendriver.core.connection import ProtocolException

from ..exceptions import (
    ClientOfflineException,
    GalleryLookupError,
    InsufficientFundsException,
    InvalidSearchRequestError,
    MalformedSearchPageError,
    SearchAuthenticationError,
    SearchChallengeError,
    SearchLimitExceededError,
    SearchNavigationError,
    SearchPageError,
    SearchPaginationError,
    SearchRateLimitError,
)
from .browser.ban_handler import check_ban_status
from .browser.mapper import prune_zendriver_connection_mapper
from .driver_base import Driver
from .models import Tag
from .punchin_models import PunchInComplete, PunchInResult, RandomEncounterFound
from .search_models import (
    MINIMUM_MISSING_CONFIRMATIONS,
    ConfirmedGalleryMissing,
    GalleryFound,
    GalleryLookupResult,
    GallerySearchResult,
    SearchRequest,
)
from .utils import (
    get_log_dir,
    is_connection_error,
    log_context,
    wait_for_new_tab,
    wait_for_zendriver,
)

_PAGE_READ_TIMEOUT_SECONDS = 5.0

MAX_DOWNLOAD_RETRIES = 5
SEARCH_PAGE_TIMEOUT_SECONDS = 10.0
SEARCH_PAGE_POLL_SECONDS = 0.1
SEARCH_NAVIGATION_RETRIES = 1
PUNCHIN_PAGE_TIMEOUT_SECONDS = 10.0
PUNCHIN_PAGE_POLL_SECONDS = 0.1
_SEARCH_DIAGNOSTIC_FILE_LIMIT = 20
_SEARCH_DIAGNOSTIC_MAX_FILE_BYTES = 2 * 1024 * 1024
_SEARCH_DIAGNOSTIC_TOTAL_BYTES = 20 * 1024 * 1024
_SEARCH_DIAGNOSTIC_FILENAME_PATTERN = re.compile(
    r"search_error_(?:(?P<sequence>[0-9a-f]{16})_)?[0-9a-f]{32}\.html\Z"
)
_SEARCH_DIAGNOSTIC_LOCK_FILENAME = ".hbrowser-search-diagnostics.lock"
_SEARCH_DIAGNOSTIC_THREAD_LOCK = Lock()
_SEARCH_DIAGNOSTIC_MAX_SEQUENCE = (1 << 64) - 1
_SEARCH_DOCUMENT_READY_STATES = frozenset({"interactive", "complete"})

_GALLERY_PATH_PATTERN = re.compile(r"/g/\d+/[A-Za-z0-9]+/?")
_GALLERY_HOSTS = frozenset({"e-hentai.org", "exhentai.org"})
_HENTAIVERSE_HOST = "hentaiverse.org"
_RANDOM_ENCOUNTER_QUERY_KEYS = frozenset({"s", "ss", "encounter"})
_PUNCHIN_PAGE_CAPTURE_ENVIRONMENT_VARIABLE = "HBROWSER_CAPTURE_PUNCHIN_PAGES"
_NO_RESULTS_MARKERS = (
    "no hits found",
    "no unfiltered results found",
)
_NO_RESULTS_SELECTORS = (
    ".searchtext",
    "table.itg > tbody > tr > td:only-child",
    "#toppane + div > p:only-child",
)
_PAGINATION_QUERY_KEYS = frozenset(
    {
        "next",
        "prev",
        "seek",
        "page",
        "range",
    }
)
_TRANSIENT_CONTEXT_ERRORS = (
    "cannot find context with specified id",
    "cannot find default execution context",
    "execution context was destroyed",
    "inspected target navigated",
)
_SEARCH_PAGE_SNAPSHOT_SCRIPT = """(() => {
    const root = document.documentElement;
    return {
        url: window.location.href,
        title: document.title || "",
        readyState: document.readyState || "",
        html: root ? root.outerHTML : "",
        query: document.querySelector("#f_search")?.value ?? null,
    };
})()"""


class _NextPageState(Enum):
    NEXT = "next"
    END = "end"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class _RawPageSnapshot:
    url: str
    title: str
    ready_state: str
    html: str
    query_value: str | None


@dataclass(frozen=True, slots=True)
class _SearchPageSnapshot:
    url: str
    title: str
    ready_state: str
    html: str
    galleries: tuple[GalleryURLParser, ...]
    has_no_results: bool
    query_value: str | None
    next_state: _NextPageState
    next_href: str | None


@dataclass(frozen=True, slots=True)
class _SearchDiagnosticCandidate:
    path: Path
    size: int
    order: tuple[int, int, str]


def _reset_search_diagnostic_thread_lock_after_fork() -> None:
    global _SEARCH_DIAGNOSTIC_THREAD_LOCK

    _SEARCH_DIAGNOSTIC_THREAD_LOCK = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_search_diagnostic_thread_lock_after_fork)


def _lock_search_diagnostic_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.lockf(descriptor, fcntl.LOCK_EX, 0, 0, os.SEEK_SET)
    elif os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(descriptor, getattr(msvcrt, "LK_LOCK"), 1)


def _unlock_search_diagnostic_descriptor(descriptor: int) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.lockf(descriptor, fcntl.LOCK_UN, 0, 0, os.SEEK_SET)
    elif os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(
            descriptor,
            getattr(msvcrt, "LK_UNLCK"),
            1,
        )


@contextmanager
def _locked_search_diagnostic_directory(directory: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    lock_path = directory / _SEARCH_DIAGNOSTIC_LOCK_FILENAME

    with _SEARCH_DIAGNOSTIC_THREAD_LOCK:
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            _lock_search_diagnostic_descriptor(descriptor)
            try:
                yield
            finally:
                _unlock_search_diagnostic_descriptor(descriptor)
        finally:
            os.close(descriptor)


def _bounded_search_diagnostic_content(content: str) -> bytes:
    maximum_bytes = min(
        _SEARCH_DIAGNOSTIC_MAX_FILE_BYTES,
        _SEARCH_DIAGNOSTIC_TOTAL_BYTES,
    )
    if maximum_bytes <= 0:
        raise OSError("Search diagnostic byte limits must be positive")

    content_bytes = content.encode("utf-8", errors="ignore")
    if len(content_bytes) <= maximum_bytes:
        return content_bytes

    marker = (
        "\n<!-- hbrowser search diagnostic truncated at " f"{maximum_bytes} bytes -->\n"
    ).encode()
    if len(marker) >= maximum_bytes:
        return marker[:maximum_bytes]
    prefix = content_bytes[: maximum_bytes - len(marker)]
    return prefix.decode("utf-8", errors="ignore").encode("utf-8") + marker


def _write_private_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        file = os.fdopen(descriptor, "wb")
        descriptor = None
        with file:
            written = file.write(content)
            if written != len(content):
                raise OSError(
                    f"Short search diagnostic write: {written}/{len(content)} bytes"
                )
    except BaseException as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as close_error:
                error.add_note(f"Could not close diagnostic file: {close_error!r}")
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                error.add_note(
                    f"Could not remove partial diagnostic file: {cleanup_error!r}"
                )
        raise


def _normalize_gallery_url(href: str, page_url: str) -> GalleryURLParser | None:
    try:
        absolute_url = urljoin(page_url, href)
        parsed_url = urlsplit(absolute_url)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
    except ValueError:
        return None
    if (
        parsed_url.scheme != "https"
        or hostname not in _GALLERY_HOSTS
        or username is not None
        or password is not None
        or port is not None
        or _GALLERY_PATH_PATTERN.fullmatch(parsed_url.path) is None
    ):
        return None

    gallery_path = f"{parsed_url.path.rstrip('/')}/"
    gallery_url = urlunsplit(("https", hostname, gallery_path, "", ""))
    try:
        return GalleryURLParser(gallery_url)
    except ValueError:
        return None


def _normalize_random_encounter_url(href: str) -> str | None:
    """Return a canonical URL for one trusted random encounter anchor."""
    try:
        parsed_url = urlsplit(href)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
        query_pairs = parse_qsl(
            parsed_url.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=len(_RANDOM_ENCOUNTER_QUERY_KEYS),
        )
    except ValueError:
        return None

    if (
        parsed_url.scheme != "https"
        or hostname != _HENTAIVERSE_HOST
        or username is not None
        or password is not None
        or port is not None
        or parsed_url.path != "/"
        or parsed_url.fragment
        or len(query_pairs) != len(_RANDOM_ENCOUNTER_QUERY_KEYS)
    ):
        return None

    query = dict(query_pairs)
    if len(query) != len(query_pairs) or query.keys() != _RANDOM_ENCOUNTER_QUERY_KEYS:
        return None
    if query["s"] != "Battle" or query["ss"] != "ba":
        return None

    encounter = query["encounter"]
    if not encounter.strip():
        return None

    return urlunsplit(
        (
            "https",
            _HENTAIVERSE_HOST,
            "/",
            urlencode(
                (
                    ("s", "Battle"),
                    ("ss", "ba"),
                    ("encounter", encounter),
                )
            ),
            "",
        )
    )


def _looks_like_random_encounter_url(href: str) -> bool:
    """Identify encounter-like markup that must not be silently ignored."""
    try:
        parsed_url = urlsplit(href)
        query_pairs = parse_qsl(
            parsed_url.query,
            keep_blank_values=True,
            max_num_fields=16,
        )
    except ValueError:
        lowered = href.casefold()
        return "encounter" in lowered and ("battle" in lowered or "ss=ba" in lowered)

    query: dict[str, list[str]] = {}
    for key, value in query_pairs:
        query.setdefault(key, []).append(value)
    return "encounter" in query or (
        "Battle" in query.get("s", []) and "ba" in query.get("ss", [])
    )


def _parse_punch_in_result(html_content: str) -> PunchInResult:
    """Classify the random-encounter state of one check-in document."""
    soup = BeautifulSoup(html_content, "html.parser")
    event_panes = soup.select("#eventpane")
    if not event_panes:
        return PunchInComplete()
    if len(event_panes) != 1:
        raise RuntimeError(
            "Daily check-in page contained ambiguous random encounter markup"
        )

    encounter_urls = list[str]()
    for anchor in event_panes[0].select("a[href]"):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        encounter_url = _normalize_random_encounter_url(href)
        if encounter_url is not None:
            encounter_urls.append(encounter_url)
        elif _looks_like_random_encounter_url(href):
            raise RuntimeError(
                "Daily check-in page contained untrusted random encounter markup"
            )

    if not encounter_urls:
        return PunchInComplete()
    if len(encounter_urls) != 1:
        raise RuntimeError(
            "Daily check-in page contained ambiguous random encounter markup"
        )
    return RandomEncounterFound(url=encounter_urls[0])


def _punchin_page_capture_enabled() -> bool:
    return os.getenv(
        _PUNCHIN_PAGE_CAPTURE_ENVIRONMENT_VARIABLE,
        "",
    ).strip().casefold() in {"1", "true", "yes", "on"}


def _parse_search_page(
    html_content: str,
    page_url: str,
) -> tuple[
    tuple[GalleryURLParser, ...],
    bool,
    str | None,
    _NextPageState,
    str | None,
]:
    soup = BeautifulSoup(html_content, "html.parser")
    galleries = list[GalleryURLParser]()
    seen_galleries = set[tuple[int, str]]()

    for anchor in soup.select(".itg a[href]"):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        gallery = _normalize_gallery_url(href, page_url)
        if gallery is None:
            continue
        identity = (gallery.gid, gallery.url_key)
        if identity not in seen_galleries:
            galleries.append(gallery)
            seen_galleries.add(identity)

    empty_state_texts = {
        " ".join(node.get_text(" ", strip=True).split()).casefold().rstrip(".")
        for selector in _NO_RESULTS_SELECTORS
        for node in soup.select(selector)
    }
    has_no_results = bool(empty_state_texts.intersection(_NO_RESULTS_MARKERS))

    input_element = soup.select_one("#f_search")
    input_value = input_element.get("value") if input_element is not None else None
    query_value = input_value if isinstance(input_value, str) else None

    next_elements = soup.select("#unext")
    if not next_elements:
        next_state = _NextPageState.END if has_no_results else _NextPageState.MISSING
        next_href = None
    elif len(next_elements) != 1:
        next_state = _NextPageState.INVALID
        next_href = None
    else:
        next_element = next_elements[0]
        if (
            next_element.name == "span"
            and next_element.get("href") is None
            and next_element.select_one("a[href]") is None
        ):
            next_state = _NextPageState.END
            next_href = None
        elif next_element.name == "a":
            href = next_element.get("href")
            if isinstance(href, str) and href.strip():
                next_state = _NextPageState.NEXT
                next_href = href
            else:
                next_state = _NextPageState.INVALID
                next_href = None
        else:
            next_state = _NextPageState.INVALID
            next_href = None

    return (
        tuple(galleries),
        has_no_results,
        query_value,
        next_state,
        next_href,
    )


def _is_transient_context_error(error: BaseException) -> bool:
    if not isinstance(error, ProtocolException):
        return False
    message = str(error).casefold()
    return any(marker in message for marker in _TRANSIENT_CONTEXT_ERRORS)


def _trusted_origin(url: str) -> tuple[str, str]:
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        username = parsed_url.username
        password = parsed_url.password
        port = parsed_url.port
    except ValueError as error:
        raise InvalidSearchRequestError(f"Invalid search scope URL: {url!r}") from error
    if (
        parsed_url.scheme != "https"
        or hostname not in _GALLERY_HOSTS
        or username is not None
        or password is not None
        or port is not None
    ):
        raise InvalidSearchRequestError(
            "Search scope must be an HTTPS e-hentai.org or exhentai.org URL "
            "without credentials or an explicit port"
        )
    assert hostname is not None
    return parsed_url.scheme, hostname


def _query_values(url: str, key: str) -> list[str]:
    return parse_qs(
        urlsplit(url).query,
        keep_blank_values=True,
    ).get(key, [])


def _canonical_query_pairs(url: str) -> list[tuple[str, str]]:
    pairs = parse_qsl(
        urlsplit(url).query,
        keep_blank_values=True,
    )
    if not any(key == "f_cats" for key, _ in pairs):
        pairs.append(("f_cats", "0"))
    return pairs


def _search_context(url: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key, value)
            for key, value in _canonical_query_pairs(url)
            if key not in _PAGINATION_QUERY_KEYS
        )
    )


def _combine_search_query(base_query: str, additional_query: str) -> str:
    values = [value.strip() for value in (base_query, additional_query)]
    return " ".join(value for value in values if value)


def _build_search_url(
    scope_url: str,
    *,
    origin: tuple[str, str],
    query: str,
) -> str:
    parsed_scope = urlsplit(scope_url)
    preserved_params = [
        (key, value)
        for key, value in parse_qsl(parsed_scope.query, keep_blank_values=True)
        if key not in _PAGINATION_QUERY_KEYS and key not in {"f_search", "f_cats"}
    ]
    preserved_params.extend((("f_search", query), ("f_cats", "0")))
    return urlunsplit(
        (
            origin[0],
            origin[1],
            "/",
            urlencode(preserved_params),
            "",
        )
    )


class EHDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"

    @staticmethod
    def _write_search_diagnostic(directory: Path, content: str) -> Path:
        content_bytes = _bounded_search_diagnostic_content(content)
        if _SEARCH_DIAGNOSTIC_FILE_LIMIT <= 0:
            raise OSError("Search diagnostic file limit must be positive")

        with _locked_search_diagnostic_directory(directory):
            candidates = list[_SearchDiagnosticCandidate]()
            maximum_sequence = -1
            for directory_entry in directory.iterdir():
                match = _SEARCH_DIAGNOSTIC_FILENAME_PATTERN.fullmatch(
                    directory_entry.name
                )
                if match is None:
                    continue
                try:
                    candidate_stat = directory_entry.lstat()
                except OSError as error:
                    raise OSError(
                        "Could not inspect search diagnostic "
                        f"{directory_entry}: {error!r}"
                    ) from error
                if not stat.S_ISREG(candidate_stat.st_mode):
                    continue

                sequence_text = match.group("sequence")
                if sequence_text is None:
                    order = (
                        0,
                        candidate_stat.st_mtime_ns,
                        directory_entry.name,
                    )
                else:
                    sequence = int(sequence_text, 16)
                    maximum_sequence = max(maximum_sequence, sequence)
                    order = (1, sequence, directory_entry.name)
                candidates.append(
                    _SearchDiagnosticCandidate(
                        path=directory_entry,
                        size=candidate_stat.st_size,
                        order=order,
                    )
                )

            candidates.sort(key=lambda candidate: candidate.order)
            allowed_count = _SEARCH_DIAGNOSTIC_FILE_LIMIT - 1
            allowed_bytes = _SEARCH_DIAGNOSTIC_TOTAL_BYTES - len(content_bytes)
            remaining_count = len(candidates)
            remaining_bytes = sum(candidate.size for candidate in candidates)
            for diagnostic_candidate in candidates:
                if (
                    remaining_count <= allowed_count
                    and remaining_bytes <= allowed_bytes
                ):
                    break
                try:
                    diagnostic_candidate.path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise OSError(
                        "Could not enforce search diagnostic retention while "
                        f"removing {diagnostic_candidate.path}: {error!r}"
                    ) from error
                remaining_count -= 1
                remaining_bytes -= diagnostic_candidate.size

            if remaining_count > allowed_count or remaining_bytes > allowed_bytes:
                raise OSError(
                    "Could not make room within search diagnostic retention "
                    f"bounds: files={remaining_count}, bytes={remaining_bytes}"
                )
            if maximum_sequence >= _SEARCH_DIAGNOSTIC_MAX_SEQUENCE:
                raise OSError("Search diagnostic sequence is exhausted")

            sequence = maximum_sequence + 1
            path = directory / (f"search_error_{sequence:016x}_{uuid4().hex}.html")
            _write_private_file(path, content_bytes)
            return path

    async def _close_page_safely(self, page: object) -> None:
        try:
            await page.close()  # type: ignore[attr-defined]
        except Exception as error:
            if is_connection_error(error):
                raise
            self.logger.debug(
                "Failed to close page (non-fatal): error_type=%s",
                type(error).__name__,
            )

    def _browser_has_tab(self, page: object) -> bool:
        page_target = getattr(page, "target", None)
        page_target_id = getattr(page_target, "target_id", None)
        for tab in self.browser.tabs:
            if tab is page:
                return True
            tab_target = getattr(tab, "target", None)
            tab_target_id = getattr(tab_target, "target_id", None)
            if page_target_id is not None and tab_target_id == page_target_id:
                return True
        return False

    async def _close_archive_tab_and_restore(
        self,
        gallery_tab: Any,
        archive_tab: Any,
    ) -> None:
        cleanup_error: BaseException | None = None
        if self._browser_has_tab(archive_tab):
            try:
                await self._close_page_safely(archive_tab)
            except BaseException as error:
                cleanup_error = error

        self.page = gallery_tab
        try:
            await gallery_tab.activate()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(
                    "Gallery activation also failed during archive cleanup: "
                    f"{type(error).__name__}"
                )

        if cleanup_error is not None:
            raise cleanup_error

    async def _current_loader_id(self) -> str:
        frame_tree = await wait_for_zendriver(
            self.page.send(cdp.page.get_frame_tree()),
            timeout=_PAGE_READ_TIMEOUT_SECONDS,
        )
        return str(frame_tree.frame.loader_id)

    async def _read_stable_punchin_document(
        self,
        *,
        previous_loader_id: str | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + PUNCHIN_PAGE_TIMEOUT_SECONDS

        while loop.time() < deadline:
            try:
                loader_before = await self._current_loader_id()
                if (
                    previous_loader_id is not None
                    and loader_before == previous_loader_id
                ):
                    await asyncio.sleep(PUNCHIN_PAGE_POLL_SECONDS)
                    continue
                snapshot = await self._read_raw_page_snapshot()
                loader_after = await self._current_loader_id()
            except Exception as error:
                if is_connection_error(error):
                    raise
                if _is_transient_context_error(error):
                    await asyncio.sleep(PUNCHIN_PAGE_POLL_SECONDS)
                    continue
                raise RuntimeError(
                    "Unable to read a stable daily check-in document"
                ) from error

            if (
                loader_before == loader_after
                and snapshot.ready_state in _SEARCH_DOCUMENT_READY_STATES
            ):
                return snapshot.html
            await asyncio.sleep(PUNCHIN_PAGE_POLL_SECONDS)

        raise RuntimeError(
            "Daily check-in page did not expose a stable DOM-ready document"
        )

    async def _read_raw_page_snapshot(self) -> _RawPageSnapshot:
        page_data = await wait_for_zendriver(
            self.page.evaluate(_SEARCH_PAGE_SNAPSHOT_SCRIPT),
            timeout=_PAGE_READ_TIMEOUT_SECONDS,
        )
        if not isinstance(page_data, dict):
            raise TypeError("Search-page snapshot was not an object")

        url = page_data.get("url")
        title = page_data.get("title")
        ready_state = page_data.get("readyState")
        html_content = page_data.get("html")
        query_value = page_data.get("query")
        if (
            not isinstance(url, str)
            or not isinstance(title, str)
            or not isinstance(ready_state, str)
            or not isinstance(html_content, str)
            or (query_value is not None and not isinstance(query_value, str))
        ):
            raise TypeError("Search-page snapshot contained non-string fields")

        return _RawPageSnapshot(
            url=url,
            title=title.strip(),
            ready_state=ready_state,
            html=html_content,
            query_value=query_value,
        )

    async def _read_search_page(self) -> _SearchPageSnapshot:
        raw_snapshot = await self._read_raw_page_snapshot()

        (
            galleries,
            has_no_results,
            serialized_query,
            next_state,
            next_href,
        ) = _parse_search_page(raw_snapshot.html, raw_snapshot.url)
        return _SearchPageSnapshot(
            url=raw_snapshot.url,
            title=raw_snapshot.title,
            ready_state=raw_snapshot.ready_state,
            html=raw_snapshot.html,
            galleries=galleries,
            has_no_results=has_no_results,
            query_value=(
                raw_snapshot.query_value
                if raw_snapshot.query_value is not None
                else serialized_query
            ),
            next_state=next_state,
            next_href=next_href,
        )

    async def _save_search_diagnostic(self, html_content: str) -> Path | None:
        try:
            path = await asyncio.to_thread(
                self._write_search_diagnostic,
                get_log_dir(),
                html_content,
            )
        except OSError as error:
            self.logger.warning(
                "Failed to save search diagnostic: error_type=%s",
                type(error).__name__,
            )
            return None
        self.logger.warning("Search diagnostic saved to: %s", path)
        return path

    async def _raise_search_page_error(
        self,
        error_type: type[SearchPageError],
        query: str,
        reason: str,
        snapshot: _SearchPageSnapshot,
    ) -> Never:
        diagnostic_path = await self._save_search_diagnostic(snapshot.html)
        raise error_type(
            query=query,
            url=snapshot.url,
            title=snapshot.title,
            reason=reason,
            diagnostic_path=(
                str(diagnostic_path) if diagnostic_path is not None else None
            ),
        )

    async def _wait_for_new_loader(
        self,
        old_loader_id: str,
        target_url: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SEARCH_PAGE_TIMEOUT_SECONDS
        last_error: Exception | None = None
        last_loader_id: str | None = None

        while loop.time() < deadline:
            try:
                current_loader_id = await self._current_loader_id()
            except Exception as error:
                if is_connection_error(error):
                    raise
                if not _is_transient_context_error(error):
                    raise SearchNavigationError(
                        url=target_url,
                        reason=f"could not read the main-frame loader: {error!r}",
                    ) from error
                last_error = error
                await asyncio.sleep(SEARCH_PAGE_POLL_SECONDS)
                continue
            last_loader_id = current_loader_id
            if current_loader_id != old_loader_id:
                return
            await asyncio.sleep(SEARCH_PAGE_POLL_SECONDS)

        reason = (
            "the trusted GET did not replace the main-frame loader"
            f"; old loader={old_loader_id!r}"
            f"; last observed loader={last_loader_id!r}"
        )
        diagnostic_path: Path | None = None
        try:
            loader_before = await self._current_loader_id()
            if loader_before != old_loader_id:
                return
            snapshot = await self._read_raw_page_snapshot()
            loader_after = await self._current_loader_id()
            if loader_after != old_loader_id:
                return
            reason += (
                f"; actual url={snapshot.url!r}"
                f"; title={snapshot.title!r}"
                f"; query={snapshot.query_value!r}"
                f"; readyState={snapshot.ready_state!r}"
                f"; capture loader before={loader_before!r}"
                f"; capture loader after={loader_after!r}"
            )
            diagnostic_path = await self._save_search_diagnostic(snapshot.html)
        except Exception as error:
            if is_connection_error(error):
                raise
            reason += f"; could not capture timeout diagnostic: {error!r}"
        if last_error is not None:
            reason += f"; last transient context error: {last_error!r}"
        raise SearchNavigationError(
            url=target_url,
            reason=reason,
            diagnostic_path=(
                str(diagnostic_path) if diagnostic_path is not None else None
            ),
        )

    async def _read_stable_search_page(
        self,
        target_url: str,
        diagnostic_query: str,
    ) -> _SearchPageSnapshot:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SEARCH_PAGE_TIMEOUT_SECONDS
        last_transient_error: Exception | None = None
        last_snapshot: _SearchPageSnapshot | None = None
        last_loader_stable: bool | None = None
        last_loader_before: str | None = None
        last_loader_after: str | None = None

        while loop.time() < deadline:
            try:
                loader_before = await self._current_loader_id()
                snapshot = await self._read_search_page()
                loader_after = await self._current_loader_id()
            except Exception as error:
                if is_connection_error(error):
                    raise
                if _is_transient_context_error(error):
                    last_transient_error = error
                    await asyncio.sleep(SEARCH_PAGE_POLL_SECONDS)
                    continue
                if isinstance(error, (TypeError, ValueError)):
                    raise MalformedSearchPageError(
                        query=diagnostic_query,
                        url=target_url,
                        title="",
                        reason=f"the atomic page snapshot was malformed: {error}",
                    ) from error
                raise SearchNavigationError(
                    url=target_url,
                    reason=f"could not read the navigated document: {error!r}",
                ) from error

            last_snapshot = snapshot
            last_loader_before = loader_before
            last_loader_after = loader_after
            last_loader_stable = loader_before == loader_after
            if (
                last_loader_stable
                and snapshot.ready_state in _SEARCH_DOCUMENT_READY_STATES
            ):
                return snapshot
            await asyncio.sleep(SEARCH_PAGE_POLL_SECONDS)

        reason = (
            "the page did not expose a stable DOM-ready snapshot from one "
            "main-frame loader"
        )
        diagnostic_path: Path | None = None
        if last_snapshot is None:
            reason += "; no readable snapshot was captured"
        else:
            reason += (
                f"; actual url={last_snapshot.url!r}"
                f"; title={last_snapshot.title!r}"
                f"; query={last_snapshot.query_value!r}"
                f"; last readyState={last_snapshot.ready_state!r}"
                f"; stable main-frame loader={last_loader_stable}"
                f"; loader before={last_loader_before!r}"
                f"; loader after={last_loader_after!r}"
            )
            diagnostic_path = await self._save_search_diagnostic(last_snapshot.html)
        if last_transient_error is not None:
            reason += f"; last transient context error: {last_transient_error!r}"
        raise SearchNavigationError(
            url=target_url,
            reason=reason,
            diagnostic_path=(
                str(diagnostic_path) if diagnostic_path is not None else None
            ),
        )

    async def _navigate_search_url(
        self,
        diagnostic_query: str,
        url: str,
    ) -> _SearchPageSnapshot:
        old_loader_id: str | None = None
        for attempt in range(SEARCH_NAVIGATION_RETRIES + 1):
            try:
                old_loader_id = await self._current_loader_id()
                break
            except Exception as error:
                if is_connection_error(error):
                    raise
                if (
                    not _is_transient_context_error(error)
                    or attempt == SEARCH_NAVIGATION_RETRIES
                ):
                    raise SearchNavigationError(
                        url=url,
                        reason=(
                            "could not capture the pre-navigation main-frame "
                            f"loader: {error!r}"
                        ),
                    ) from error
                await asyncio.sleep(SEARCH_PAGE_POLL_SECONDS)
        assert old_loader_id is not None

        for attempt in range(SEARCH_NAVIGATION_RETRIES + 1):
            try:
                await self.get(url)
                break
            except Exception as error:
                if is_connection_error(error):
                    raise
                if (
                    not _is_transient_context_error(error)
                    or attempt == SEARCH_NAVIGATION_RETRIES
                ):
                    raise SearchNavigationError(
                        url=url,
                        reason=f"trusted GET failed: {error!r}",
                    ) from error
                self.logger.warning(
                    "Transient browser context loss during search navigation; "
                    "retrying the trusted GET once: query_length=%d error_type=%s",
                    len(diagnostic_query),
                    type(error).__name__,
                )
        await self._wait_for_new_loader(old_loader_id, url)
        return await self._read_stable_search_page(
            url,
            diagnostic_query,
        )

    @staticmethod
    def _is_challenge_page(snapshot: _SearchPageSnapshot) -> bool:
        title = snapshot.title.casefold()
        html = snapshot.html.casefold()
        return (
            "just a moment" in title
            or "請稍候" in title
            or "_cf_chl_opt" in html
            or "cf-chl-widget" in html
            or "cf-turnstile" in html
            or "g-recaptcha" in html
            or 'id="challenge-form"' in html
            or 'id="cf-challenge-running"' in html
            or 'id="challenge-running"' in html
            or 'id="challenge-stage"' in html
        )

    @staticmethod
    def _is_authentication_page(snapshot: _SearchPageSnapshot) -> bool:
        try:
            hostname = urlsplit(snapshot.url).hostname
        except ValueError:
            hostname = None
        title = snapshot.title.casefold()
        html = snapshot.html.casefold()
        return (
            hostname == "forums.e-hentai.org"
            or title in {"log in", "login"}
            or "you are not logged in" in html
            or "kokomade.jpg" in html
            or (
                ('name="username"' in html or "name='username'" in html)
                and ('name="password"' in html or "name='password'" in html)
            )
        )

    async def _validate_search_page(
        self,
        snapshot: _SearchPageSnapshot,
        *,
        expected_origin: tuple[str, str],
        expected_query: str | None,
        expected_context: tuple[tuple[str, str], ...] | None = None,
        diagnostic_query: str,
    ) -> None:
        if check_ban_status(snapshot.html).should_wait:
            await self._raise_search_page_error(
                SearchRateLimitError,
                diagnostic_query,
                "the shared ban handler returned a still-banned or blank page",
                snapshot,
            )
        if self._is_challenge_page(snapshot):
            await self._raise_search_page_error(
                SearchChallengeError,
                diagnostic_query,
                "the trusted navigation reached a managed challenge",
                snapshot,
            )
        if self._is_authentication_page(snapshot):
            await self._raise_search_page_error(
                SearchAuthenticationError,
                diagnostic_query,
                "the trusted navigation lost the authenticated gallery session",
                snapshot,
            )

        try:
            actual_origin = _trusted_origin(snapshot.url)
        except InvalidSearchRequestError:
            await self._raise_search_page_error(
                MalformedSearchPageError,
                diagnostic_query,
                "the result page URL was not a trusted gallery origin",
                snapshot,
            )
        if actual_origin != expected_origin:
            await self._raise_search_page_error(
                MalformedSearchPageError,
                diagnostic_query,
                "the result page changed gallery origin",
                snapshot,
            )

        if snapshot.query_value is None:
            await self._raise_search_page_error(
                MalformedSearchPageError,
                diagnostic_query,
                "the result page did not contain #f_search",
                snapshot,
            )
        if expected_query is not None:
            if snapshot.query_value != expected_query:
                await self._raise_search_page_error(
                    MalformedSearchPageError,
                    diagnostic_query,
                    "the live search query did not match the requested query "
                    f"({snapshot.query_value!r} != {expected_query!r})",
                    snapshot,
                )
            url_queries = _query_values(snapshot.url, "f_search")
            if url_queries != [expected_query]:
                await self._raise_search_page_error(
                    MalformedSearchPageError,
                    diagnostic_query,
                    "the result URL did not contain exactly one matching "
                    "f_search value",
                    snapshot,
                )
        if expected_context is not None:
            try:
                parsed_result_url = urlsplit(snapshot.url)
                actual_context = _search_context(snapshot.url)
            except ValueError:
                await self._raise_search_page_error(
                    MalformedSearchPageError,
                    diagnostic_query,
                    "the result URL query could not be parsed",
                    snapshot,
                )
            if (
                parsed_result_url.path != "/"
                or parsed_result_url.fragment
                or actual_context != expected_context
            ):
                await self._raise_search_page_error(
                    MalformedSearchPageError,
                    diagnostic_query,
                    "the result URL changed path, fragment, or non-pagination "
                    "search parameters",
                    snapshot,
                )

        if bool(snapshot.galleries) == snapshot.has_no_results:
            await self._raise_search_page_error(
                MalformedSearchPageError,
                diagnostic_query,
                "the page contained either contradictory result markers or "
                "neither galleries nor an explicit no-results marker",
                snapshot,
            )
        if snapshot.next_state is _NextPageState.MISSING:
            await self._raise_search_page_error(
                SearchPaginationError,
                diagnostic_query,
                "the result page did not contain an explicit #unext control",
                snapshot,
            )
        if snapshot.next_state is _NextPageState.INVALID:
            await self._raise_search_page_error(
                SearchPaginationError,
                diagnostic_query,
                "the #unext control was malformed",
                snapshot,
            )
        if snapshot.has_no_results and snapshot.next_state is not _NextPageState.END:
            await self._raise_search_page_error(
                SearchPaginationError,
                diagnostic_query,
                "an explicit no-results page unexpectedly advertised a next page",
                snapshot,
            )

    async def _resolve_search_request(
        self,
        request: SearchRequest,
    ) -> tuple[tuple[str, str], str, str]:
        request_origin = _trusted_origin(request.scope_url)
        driver_origin = _trusted_origin(self.url[self.name])
        if request_origin != driver_origin:
            raise InvalidSearchRequestError(
                f"{self.name} searches cannot cross to {request_origin[1]!r}"
            )
        if urlsplit(request.scope_url).fragment:
            raise InvalidSearchRequestError(
                "Search scope URLs must not contain a fragment"
            )

        scope_snapshot = await self._navigate_search_url(
            f"<scope {request.scope_url}>",
            request.scope_url,
        )
        await self._validate_search_page(
            scope_snapshot,
            expected_origin=request_origin,
            expected_query=None,
            diagnostic_query=f"<scope {request.scope_url}>",
        )
        if self._page_identity(scope_snapshot.url) != self._page_identity(
            request.scope_url
        ):
            await self._raise_search_page_error(
                MalformedSearchPageError,
                f"<scope {request.scope_url}>",
                "the navigated scope URL did not match the trusted GET target",
                scope_snapshot,
            )
        assert scope_snapshot.query_value is not None
        scope_url_queries = _query_values(scope_snapshot.url, "f_search")
        if scope_url_queries and scope_url_queries != [scope_snapshot.query_value]:
            await self._raise_search_page_error(
                MalformedSearchPageError,
                f"<scope {request.scope_url}>",
                "the scope URL contained ambiguous or mismatched f_search values",
                scope_snapshot,
            )
        query = _combine_search_query(scope_snapshot.query_value, request.query)
        if not query:
            raise InvalidSearchRequestError(
                "The resolved search query must not be empty"
            )
        search_url = _build_search_url(
            scope_snapshot.url,
            origin=request_origin,
            query=query,
        )
        return request_origin, query, search_url

    async def _validated_next_url(
        self,
        snapshot: _SearchPageSnapshot,
        *,
        expected_origin: tuple[str, str],
        expected_query: str,
        expected_context: tuple[tuple[str, str], ...],
    ) -> str:
        if snapshot.next_state is not _NextPageState.NEXT or snapshot.next_href is None:
            await self._raise_search_page_error(
                SearchPaginationError,
                expected_query,
                "the result page did not provide a usable next-page URL",
                snapshot,
            )

        next_url = urljoin(snapshot.url, snapshot.next_href)
        try:
            next_origin = _trusted_origin(next_url)
            parsed_next = urlsplit(next_url)
        except InvalidSearchRequestError, ValueError:
            await self._raise_search_page_error(
                SearchPaginationError,
                expected_query,
                "the next-page URL was malformed or untrusted",
                snapshot,
            )
        if (
            next_origin != expected_origin
            or parsed_next.path != "/"
            or parsed_next.fragment
            or _query_values(next_url, "f_search") != [expected_query]
            or _search_context(next_url) != expected_context
        ):
            await self._raise_search_page_error(
                SearchPaginationError,
                expected_query,
                "the next-page URL changed origin, path, fragment, or query",
                snapshot,
            )
        return urlunsplit(
            (
                parsed_next.scheme,
                parsed_next.netloc,
                parsed_next.path,
                parsed_next.query,
                "",
            )
        )

    @staticmethod
    def _page_identity(url: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        parsed_url = urlsplit(url)
        return (
            parsed_url.hostname or "",
            parsed_url.path,
            tuple(sorted(_canonical_query_pairs(url))),
        )

    async def checkh2h(self) -> bool:
        """檢查 H@H 客戶端是否在線"""
        self.logger.info("Checking H@H client status")
        await self.get("https://e-hentai.org/hentaiathome.php")
        table = await wait_for_zendriver(
            self.page.select("#hct", timeout=10),
            timeout=_PAGE_READ_TIMEOUT_SECONDS,
        )
        header_row = await wait_for_zendriver(
            table.query_selector("tr"), timeout=_PAGE_READ_TIMEOUT_SECONDS
        )
        headers = await wait_for_zendriver(
            header_row.query_selector_all("th"),
            timeout=_PAGE_READ_TIMEOUT_SECONDS,
        )
        status_index = [
            index for index, th in enumerate(headers) if th.text == "Status"
        ][0]
        rows = await wait_for_zendriver(
            table.query_selector_all("tr"), timeout=_PAGE_READ_TIMEOUT_SECONDS
        )
        for row in rows[1:]:
            cells = await wait_for_zendriver(
                row.query_selector_all("td"), timeout=_PAGE_READ_TIMEOUT_SECONDS
            )
            status = cells[status_index].text
            if status.lower() == "online":
                self.logger.info("H@H client is online")
                return True
        self.logger.warning("H@H client is offline")
        return False

    async def punchin(self) -> PunchInResult:
        """Check in and return any trusted HentaiVerse random encounter."""
        with log_context(activity="Check-in"):
            return await self._punchin()

    async def _punchin(self) -> PunchInResult:
        self.logger.info("Starting daily check-in")
        await self.get("https://e-hentai.org/news.php")

        capture_pages = _punchin_page_capture_enabled()
        initial_html_content = await self._read_stable_punchin_document()
        if not isinstance(initial_html_content, str):
            raise TypeError("Initial daily check-in page content was not a string")
        if capture_pages:
            await self._save_page_diagnostic(
                "punchin_initial",
                initial_html_content,
            )
        initial_result = _parse_punch_in_result(initial_html_content)
        if isinstance(initial_result, RandomEncounterFound):
            self.logger.info("Daily check-in found a random encounter")
            return initial_result

        # 刷新以免沒簽到成功
        initial_loader_id = await self._current_loader_id()
        self.logger.debug("Refreshing the daily check-in page")
        await self.wait(self.page.reload, ischangeurl=False)
        self.logger.debug("Daily check-in page refresh completed")

        html_content = await self._read_stable_punchin_document(
            previous_loader_id=initial_loader_id,
        )
        if not isinstance(html_content, str):
            raise TypeError("Daily check-in page content was not a string")
        if capture_pages:
            await self._save_page_diagnostic(
                "punchin_reloaded",
                html_content,
            )
        result = _parse_punch_in_result(html_content)
        if isinstance(result, RandomEncounterFound):
            self.logger.info("Daily check-in found a random encounter after refresh")
        else:
            self.logger.info("Daily check-in completed; no random encounter found")
        return result

    async def search(self, request: SearchRequest) -> GallerySearchResult:
        """Execute one bounded gallery search using trusted URL navigations."""
        if not isinstance(request, SearchRequest):
            raise InvalidSearchRequestError(
                "search() requires a SearchRequest instance"
            )

        origin, query, initial_url = await self._resolve_search_request(request)
        self.logger.debug(
            "Gallery search started: query=%r max_pages=%d max_results=%d",
            query,
            request.max_pages,
            request.max_results,
        )
        expected_context = _search_context(initial_url)

        galleries = list[GalleryURLParser]()
        seen_galleries = set[tuple[int, str]]()
        seen_pages = set[tuple[str, str, tuple[tuple[str, str], ...]]]()
        current_url = initial_url
        pages_visited = 0

        while True:
            snapshot = await self._navigate_search_url(query, current_url)
            await self._validate_search_page(
                snapshot,
                expected_origin=origin,
                expected_query=query,
                expected_context=expected_context,
                diagnostic_query=query,
            )
            page_identity = self._page_identity(snapshot.url)
            if page_identity != self._page_identity(current_url):
                await self._raise_search_page_error(
                    (
                        MalformedSearchPageError
                        if pages_visited == 0
                        else SearchPaginationError
                    ),
                    query,
                    "the navigated result URL did not match the trusted GET target",
                    snapshot,
                )
            if page_identity in seen_pages:
                await self._raise_search_page_error(
                    SearchPaginationError,
                    query,
                    "pagination returned a previously processed result URL",
                    snapshot,
                )
            seen_pages.add(page_identity)
            pages_visited += 1

            for gallery in snapshot.galleries:
                identity = (gallery.gid, gallery.url_key)
                if identity in seen_galleries:
                    continue
                if len(galleries) == request.max_results:
                    raise SearchLimitExceededError(
                        query=query,
                        max_pages=request.max_pages,
                        max_results=request.max_results,
                        pages_visited=pages_visited,
                        results_found=len(galleries) + 1,
                    )
                seen_galleries.add(identity)
                galleries.append(gallery)

            if snapshot.next_state is _NextPageState.END:
                result = GallerySearchResult(
                    request=request,
                    galleries=tuple(galleries),
                    pages_visited=pages_visited,
                )
                self.logger.debug(
                    "Gallery search completed: query=%r galleries=%d pages=%d",
                    query,
                    len(result.galleries),
                    result.pages_visited,
                )
                return result

            if pages_visited == request.max_pages:
                raise SearchLimitExceededError(
                    query=query,
                    max_pages=request.max_pages,
                    max_results=request.max_results,
                    pages_visited=pages_visited,
                    results_found=len(galleries),
                )
            next_url = await self._validated_next_url(
                snapshot,
                expected_origin=origin,
                expected_query=query,
                expected_context=expected_context,
            )
            if self._page_identity(next_url) in seen_pages:
                await self._raise_search_page_error(
                    SearchPaginationError,
                    query,
                    "the next-page URL creates a pagination cycle",
                    snapshot,
                )
            current_url = next_url

    async def lookup_gid(self, gid: int) -> GalleryLookupResult:
        """Resolve an exact GID or confirm its absence with two fresh searches."""
        if not isinstance(gid, int) or isinstance(gid, bool) or gid < 1:
            raise InvalidSearchRequestError("gid must be a positive integer")

        request = SearchRequest(
            scope_url=self.url[self.name],
            query=f"gid:{gid}",
            max_pages=1,
        )
        for confirmation in range(1, MINIMUM_MISSING_CONFIRMATIONS + 1):
            result = await self.search(request)
            if len(result.galleries) > 1:
                raise GalleryLookupError(
                    f"Exact lookup for GID {gid} returned "
                    f"{len(result.galleries)} logical galleries"
                )
            if result.galleries:
                return GalleryFound(
                    requested_gid=gid,
                    gallery=result.galleries[0],
                )
            self.logger.debug(
                "Exact gallery lookup was empty: gid=%d confirmation=%d/%d",
                gid,
                confirmation,
                MINIMUM_MISSING_CONFIRMATIONS,
            )

        self.logger.info(
            "Gallery absence confirmed: gid=%d confirmations=%d",
            gid,
            MINIMUM_MISSING_CONFIRMATIONS,
        )
        return ConfirmedGalleryMissing(
            gid=gid,
            confirmations=MINIMUM_MISSING_CONFIRMATIONS,
        )

    async def download(self, gallery: GalleryURLParser) -> bool:
        self.logger.info(
            "Gallery archive download started: gid=%d max_attempts=%d",
            gallery.gid,
            MAX_DOWNLOAD_RETRIES,
        )
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            if attempt > 1:
                self.logger.debug(
                    "Gallery archive download retry started: gid=%d attempt=%d/%d",
                    gallery.gid,
                    attempt,
                    MAX_DOWNLOAD_RETRIES,
                )
            downloaded = await self._download(gallery, attempt)
            if downloaded is not None:
                return downloaded

        raise RuntimeError(
            f"Failed to download gallery after {MAX_DOWNLOAD_RETRIES} "
            f"attempts: gid={gallery.gid}"
        )

    async def _download(self, gallery: GalleryURLParser, attempt: int) -> bool | None:
        prune_zendriver_connection_mapper(self.page)

        await self.get(gallery.url)
        try:
            xpath_query_list = [
                "//p[contains(text(), "
                "'This gallery is unavailable due to a copyright claim "
                "by Irodori Comics.')]",
                "//input[@id='f_search']",
            ]
            xpath_query = " | ".join(xpath_query_list)
            results = await wait_for_zendriver(
                self.page.xpath(xpath_query, timeout=2),
                timeout=_PAGE_READ_TIMEOUT_SECONDS,
            )
            if results:
                self.logger.warning(
                    "Gallery unavailable or deleted: gid=%d",
                    gallery.gid,
                )
                return False
        except TimeoutError:
            pass

        existing_tabs = {t.target.target_id for t in self.browser.tabs}
        gallery_tab = self.page

        key_xpath = "//a[contains(text(), 'Archive Download')]"
        try:
            archive_links = await wait_for_zendriver(
                self.page.xpath(key_xpath, timeout=2),
                timeout=_PAGE_READ_TIMEOUT_SECONDS,
            )
            if archive_links:
                await wait_for_zendriver(
                    archive_links[0].click(), timeout=_PAGE_READ_TIMEOUT_SECONDS
                )
            else:
                raise RuntimeError("Archive Download not found")
        except Exception as error:
            if is_connection_error(error):
                raise
            self.logger.warning(
                "Archive Download control unavailable; retrying: "
                "gid=%d attempt=%d/%d error_type=%s",
                gallery.gid,
                attempt,
                MAX_DOWNLOAD_RETRIES,
                type(error).__name__,
            )
            return None

        new_tab = await wait_for_new_tab(self.browser, existing_tabs)
        if not new_tab:
            self.logger.warning(
                "Archive download tab did not open; retrying: " "gid=%d attempt=%d/%d",
                gallery.gid,
                attempt,
                MAX_DOWNLOAD_RETRIES,
            )
            return None

        self.page = new_tab
        retrytime: int | None = None
        try:
            await new_tab.activate()
            original_links = await wait_for_zendriver(
                self.page.xpath("//a[contains(text(), 'Original')]", timeout=10),
                timeout=_PAGE_READ_TIMEOUT_SECONDS,
            )
            if original_links:
                await wait_for_zendriver(
                    original_links[0].click(), timeout=_PAGE_READ_TIMEOUT_SECONDS
                )

            try:
                deadline = asyncio.get_event_loop().time() + 10
                while asyncio.get_event_loop().time() < deadline:
                    html = await wait_for_zendriver(
                        self.page.get_content(),
                        timeout=_PAGE_READ_TIMEOUT_SECONDS,
                    )
                    if (
                        "Downloads should start processing within a couple of minutes."
                        in html
                    ):
                        break
                    if "Your H@H client appears to be offline." in html:
                        raise ClientOfflineException()
                    if "Cannot start download: Insufficient funds" in html:
                        raise InsufficientFundsException()
                    await asyncio.sleep(0.5)
                else:
                    html = await wait_for_zendriver(
                        self.page.get_content(),
                        timeout=_PAGE_READ_TIMEOUT_SECONDS,
                    )
                    if "Cannot start download: Insufficient funds" in html:
                        raise InsufficientFundsException()
                    raise TimeoutError()
            except TimeoutError:
                error_file = await self._save_page_diagnostic("download_timeout")
                retrytime = 60
                if error_file is None:
                    self.logger.warning(
                        "Archive download timed out; diagnostic unavailable; "
                        "retrying: gid=%d attempt=%d/%d delay=%ds",
                        gallery.gid,
                        attempt,
                        MAX_DOWNLOAD_RETRIES,
                        retrytime,
                    )
                else:
                    self.logger.warning(
                        "Archive download timed out; diagnostic=%s; retrying: "
                        "gid=%d attempt=%d/%d delay=%ds",
                        error_file,
                        gallery.gid,
                        attempt,
                        MAX_DOWNLOAD_RETRIES,
                        retrytime,
                    )
        except BaseException as operation_error:
            try:
                await self._close_archive_tab_and_restore(gallery_tab, new_tab)
            except BaseException as cleanup_error:
                operation_error.add_note(
                    "Archive cleanup also failed: " f"{type(cleanup_error).__name__}"
                )
            raise
        else:
            await self._close_archive_tab_and_restore(gallery_tab, new_tab)

        if retrytime is not None:
            await asyncio.sleep(retrytime)
            return None

        await asyncio.sleep(random())
        await asyncio.sleep(random())
        self.logger.info("Gallery archive download queued: gid=%d", gallery.gid)
        return True

    async def gallery2tag(self, gallery: GalleryURLParser, filter: str) -> list[Tag]:
        await self.get(gallery.url)
        try:
            elements = await wait_for_zendriver(
                self.page.xpath(f"//a[contains(@id, 'ta_{filter}')]", timeout=2),
                timeout=_PAGE_READ_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return list()

        tag = list()
        for element in elements:
            tag.append(
                Tag(
                    filter=filter,
                    name=element.text,
                    href=element.attrs.get("href", ""),
                )
            )
        return tag
