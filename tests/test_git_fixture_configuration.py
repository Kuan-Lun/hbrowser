from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN = "conftest"
_PROBE = r"""
import os
import subprocess
from pathlib import Path


def test_git_fixture_uses_only_explicit_configuration(tmp_path, monkeypatch):
    assert os.environ["GIT_DIR"] == os.environ["POISON_AUTHORITY"]
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name)
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments):
        return subprocess.run(
            ("git", *arguments), cwd=repository, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Isolated Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (repository / "payload.txt").write_text("fixture data\n")
    git("add", ".")
    git("commit", "-m", "test: use an isolated Git fixture")
    assert git("log", "-1", "--format=%s") == "test: use an isolated Git fixture"
    assert not Path(os.environ["POISON_MARKER"]).exists()

    # Tests of configuration policy may opt in after the autouse boundary.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.environ["POISON_CONFIG"])
    assert git("config", "--get", "commit.gpgsign") == "true"
"""


@pytest.mark.parametrize("source", ("global", "system", "parameters", "count", "file"))
def test_pytest_git_fixtures_ignore_host_configuration(
    tmp_path: Path, source: str
) -> None:
    marker = tmp_path / "ambient-git-program-ran"
    hooks = tmp_path / "host-hooks"
    hooks.mkdir()
    program = (
        "#!/bin/sh\n"
        'printf invoked > "$POISON_MARKER"\n'
        "echo ambient-git-program-must-not-run >&2\n"
        "exit 93\n"
    )
    for executable in (hooks / "pre-commit", tmp_path / "host-signer"):
        executable.write_text(program, encoding="utf-8")
        executable.chmod(0o755)
    config = tmp_path / "host.gitconfig"
    config.write_text(
        "[commit]\n\tgpgsign = true\n"
        f"[gpg]\n\tprogram = {(tmp_path / 'host-signer').as_posix()}\n"
        f"[core]\n\thooksPath = {hooks.as_posix()}\n",
        encoding="utf-8",
    )
    child = tmp_path / "child"
    child.mkdir()
    (child / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (child / "test_probe.py").write_text(_PROBE, encoding="utf-8")
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    authority = str(tmp_path / "preserved-authority.git")
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_DIR=authority,
        GIT_WORK_TREE=str(tmp_path / "preserved-worktree"),
        GIT_INDEX_FILE=str(tmp_path / "preserved-index"),
        POISON_AUTHORITY=authority,
        POISON_CONFIG=str(config),
        POISON_MARKER=str(marker),
        PYTHONPATH=os.pathsep.join((str(_ROOT), str(_ROOT / "tests"))),
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    if source in ("global", "system", "file"):
        key = "GIT_CONFIG" if source == "file" else f"GIT_CONFIG_{source.upper()}"
        environment[key] = str(config)
    elif source == "parameters":
        environment["GIT_CONFIG_PARAMETERS"] = f"'include.path={config.as_posix()}'"
    else:
        environment.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="include.path",
            GIT_CONFIG_VALUE_0=str(config),
        )
    result = subprocess.run(
        (
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            _PLUGIN,
            "-p",
            "no:cacheprovider",
            "-c",
            str(child / "pytest.ini"),
            "-q",
            str(child / "test_probe.py"),
        ),
        cwd=child,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert not marker.exists()
