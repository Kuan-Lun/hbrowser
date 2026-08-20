class ClientOfflineException(Exception):
    def __init__(self, message: str = "H@H client appears to be offline.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InsufficientFundsException(Exception):
    def __init__(
        self, message: str = "Insufficient funds to start the download."
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class LoginFailedException(Exception):
    def __init__(self, message: str = "Login did not succeed.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class BrowserMutationOutcomeUnknownError(RuntimeError):
    """A browser mutation was invoked but its final outcome is unknowable.

    This marker makes the current browser generation terminal. It is reserved
    for failures after a state-changing operation has started; read-only
    failures and mutations rejected before invocation must not use it.
    """


class BrowserIdentityApplyException(BrowserMutationOutcomeUnknownError):
    """Applying an external solver identity failed and cannot be rolled back."""


class DriverBrowserBindingError(RuntimeError):
    """A driver was asked to replace its immutable browser/page binding."""


class ArchiveDownloadOutcomeUnknownError(BrowserMutationOutcomeUnknownError):
    """An archive mutation was sent but its final outcome could not be observed."""


class LoginTokenInjectionOutcomeUnknownError(BrowserMutationOutcomeUnknownError):
    """A login challenge token mutation had an unobservable final outcome."""


class GallerySearchError(RuntimeError):
    """Base class for gallery-search and exact-lookup failures."""


class InvalidSearchRequestError(GallerySearchError, ValueError):
    """A search request cannot be represented as a safe, bounded search."""


class SearchNavigationError(GallerySearchError):
    """A trusted search URL could not be navigated to or read."""

    def __init__(
        self,
        *,
        url: str,
        reason: str,
        diagnostic_path: str | None = None,
    ) -> None:
        self.url = url
        self.reason = reason
        self.diagnostic_path = diagnostic_path

        message = f"Could not navigate gallery search URL {url!r}: {reason}"
        if diagnostic_path is not None:
            message += f"; diagnostic saved to {diagnostic_path}"
        super().__init__(message)


class SearchPageError(GallerySearchError):
    """Base class for classified, non-result search pages."""

    def __init__(
        self,
        *,
        query: str,
        url: str,
        title: str,
        reason: str,
        diagnostic_path: str | None = None,
    ) -> None:
        self.query = query
        self.url = url
        self.title = title
        self.reason = reason
        self.diagnostic_path = diagnostic_path

        message = (
            f"Search page for {query!r} is not a valid result page: {reason} "
            f"(url={url!r}, title={title!r})"
        )
        if diagnostic_path is not None:
            message += f"; diagnostic saved to {diagnostic_path}"
        super().__init__(message)


class SearchChallengeError(SearchPageError):
    """A search navigation reached a CAPTCHA or managed challenge."""


class SearchAuthenticationError(SearchPageError):
    """A search navigation lost the expected authenticated site state."""


class SearchRateLimitError(SearchPageError):
    """A search navigation remained rate-limited after the shared handler."""


class MalformedSearchPageError(SearchPageError):
    """A page on the expected origin violated the search-page contract."""


class SearchPaginationError(MalformedSearchPageError):
    """A pagination control or target URL was missing, unsafe, or cyclic."""


class SearchLimitExceededError(GallerySearchError):
    """A bounded search would exceed its explicit page or result limit."""

    def __init__(
        self,
        *,
        query: str,
        max_pages: int,
        max_results: int,
        pages_visited: int,
        results_found: int,
    ) -> None:
        self.query = query
        self.max_pages = max_pages
        self.max_results = max_results
        self.pages_visited = pages_visited
        self.results_found = results_found
        super().__init__(
            f"Search for {query!r} exceeded its bounds "
            f"(pages={pages_visited}/{max_pages}, "
            f"results={results_found}/{max_results})"
        )


class GalleryLookupError(GallerySearchError):
    """An exact-GID lookup returned a contradictory result set."""
