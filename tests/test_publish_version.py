from __future__ import annotations

import io
import json
import runpy
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import cast

import pytest

_CHECK = cast(
    Callable[[Path], None],
    runpy.run_path(str(Path(__file__).parents[1] / "scripts/check-publish-version.py"))[
        "check_version"
    ],
)


@pytest.mark.parametrize(
    ("current", "published", "expected"),
    [("1.2.3", "1.2.2", "true"), ("1.2.3", "1.2.3", "false")],
)
def test_registry_retry_only_uploads_unpublished_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    current: str,
    published: str,
    expected: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{current}"\n')

    def response(url: str, *, timeout: int) -> io.BytesIO:
        assert url == "https://pypi.org/pypi/hbrowser/json"
        assert timeout == 30
        return io.BytesIO(json.dumps({"info": {"version": published}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", response)
    _CHECK(tmp_path)
    assert capsys.readouterr().out == (
        f"current={current}\npublished={published}\npublish={expected}\n"
    )


def test_registry_newer_version_fails_without_passing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b'{"info":{"version":"1.3.0"}}'),
    )
    with pytest.raises(SystemExit, match="older than published"):
        _CHECK(tmp_path)
    assert capsys.readouterr().out == ""


def test_registry_error_does_not_turn_into_first_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')

    def unavailable(url: str, *, timeout: int) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 404, "absent", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(urllib.error.HTTPError):
        _CHECK(tmp_path)
    assert capsys.readouterr().out == ""
