"""Public models for deterministic gallery searches and exact GID lookups."""

from dataclasses import dataclass

from h2h_galleryinfo_parser import GalleryURLParser

from ..exceptions import InvalidSearchRequestError

DEFAULT_SEARCH_MAX_PAGES = 100
DEFAULT_SEARCH_MAX_RESULTS = 5_000
MINIMUM_MISSING_CONFIRMATIONS = 2


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """A self-contained gallery search.

    ``scope_url`` identifies the trusted E-Hentai/ExHentai search scope. Its
    existing ``f_search`` value is treated as the base scope (for example, an
    artist tag URL); ``query`` is appended as an additional condition.
    """

    scope_url: str
    query: str
    max_pages: int = DEFAULT_SEARCH_MAX_PAGES
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS

    def __post_init__(self) -> None:
        if not isinstance(self.scope_url, str) or not self.scope_url.strip():
            raise InvalidSearchRequestError("scope_url must be a non-empty string")
        if not isinstance(self.query, str):
            raise InvalidSearchRequestError("query must be a string")
        if not isinstance(self.max_pages, int) or isinstance(self.max_pages, bool):
            raise InvalidSearchRequestError("max_pages must be an integer")
        if self.max_pages < 1:
            raise InvalidSearchRequestError("max_pages must be at least 1")
        if not isinstance(self.max_results, int) or isinstance(
            self.max_results,
            bool,
        ):
            raise InvalidSearchRequestError("max_results must be an integer")
        if self.max_results < 1:
            raise InvalidSearchRequestError("max_results must be at least 1")


@dataclass(frozen=True, slots=True)
class GallerySearchResult:
    """The converged, logically de-duplicated result of one search request."""

    request: SearchRequest
    galleries: tuple[GalleryURLParser, ...]
    pages_visited: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, SearchRequest):
            raise TypeError("request must be a SearchRequest")
        if not isinstance(self.galleries, tuple) or not all(
            isinstance(gallery, GalleryURLParser) for gallery in self.galleries
        ):
            raise TypeError("galleries must be a tuple of GalleryURLParser values")
        if (
            not isinstance(self.pages_visited, int)
            or isinstance(self.pages_visited, bool)
            or not 1 <= self.pages_visited <= self.request.max_pages
        ):
            raise ValueError("pages_visited must be within the request's page bounds")
        if len(self.galleries) > self.request.max_results:
            raise ValueError("galleries must not exceed the request's result bound")


@dataclass(frozen=True, slots=True)
class GalleryFound:
    """An exact-GID lookup resolved to a live gallery."""

    requested_gid: int
    gallery: GalleryURLParser

    def __post_init__(self) -> None:
        if (
            not isinstance(self.requested_gid, int)
            or isinstance(self.requested_gid, bool)
            or self.requested_gid < 1
        ):
            raise ValueError("requested_gid must be a positive integer")
        if not isinstance(self.gallery, GalleryURLParser):
            raise TypeError("gallery must be a GalleryURLParser")


@dataclass(frozen=True, slots=True)
class ConfirmedGalleryMissing:
    """An exact-GID lookup was empty in multiple independent searches."""

    gid: int
    confirmations: int

    def __post_init__(self) -> None:
        if not isinstance(self.gid, int) or isinstance(self.gid, bool) or self.gid < 1:
            raise ValueError("gid must be a positive integer")
        if (
            not isinstance(self.confirmations, int)
            or isinstance(self.confirmations, bool)
            or self.confirmations < MINIMUM_MISSING_CONFIRMATIONS
        ):
            raise ValueError(
                "ConfirmedGalleryMissing requires at least "
                f"{MINIMUM_MISSING_CONFIRMATIONS} confirmations"
            )


type GalleryLookupResult = GalleryFound | ConfirmedGalleryMissing
