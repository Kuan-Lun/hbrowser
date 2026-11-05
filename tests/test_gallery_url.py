import pytest

from hbrowser.gallery.utils import matchurl


@pytest.mark.parametrize(
    ("urls", "expected"),
    [
        (("/", "/"), True),
        (("/", "///", "//"), True),
        (("/", "/gallery/"), False),
        (("/gallery/", "/"), False),
        (("/gallery", "/gallery///"), True),
        (("https://e-hentai.org", "https://e-hentai.org///"), True),
        (("https://e-hentai.org", "https://exhentai.org/"), False),
        (("", "/"), False),
        (("/", ""), False),
        (("", ""), False),
        (("/",), True),
        ((), True),
    ],
)
def test_matchurl_ignores_trailing_slashes_without_accepting_empty_input(
    urls: tuple[str, ...], expected: bool
) -> None:
    assert matchurl(*urls) is expected
