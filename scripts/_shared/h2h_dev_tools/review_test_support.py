"""Export verified synthetic responses without importing tooling in fake CLIs."""

from importlib.abc import InspectLoader

from . import review_test_response


def synthetic_review_response_source() -> str:
    """Return the single response implementation for a disposable fake CLI.

    Call only after the normal source loader has verified this tree's selected
    component. Its get_source rechecks the registered bytes, including changes
    after initial import. Ordinary installed-package imports are not a fallback.
    The returned module is self-contained and may be appended to a local fixture
    script; callers retain their CLI, fault injection and output-file handling.
    """
    loader = review_test_response.__loader__
    if (
        not isinstance(loader, InspectLoader)
        or getattr(loader, "_h2h_source_registration", None) is None
    ):
        raise RuntimeError("Review test support requires verified source loading")
    source = loader.get_source(review_test_response.__name__)
    if source is None:
        raise RuntimeError("Verified review response source is unavailable")
    return source
