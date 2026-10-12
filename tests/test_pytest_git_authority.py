from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CHILD_SESSION = r"""import json, os, subprocess
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption('--check-backend-pairs', action='store_true')

def pytest_sessionstart(session):
    state = Path(os.environ['H2H_TEST_CHILD_STATE'])
    child = state / str(os.getpid())
    child.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=child, text=True).strip()
    git('init', '-b', 'child')
    git('config', 'user.name', 'Child Test')
    git('config', 'user.email', 'child@example.invalid')
    (child / 'child.txt').write_text('child data\n')
    git('add', 'child.txt')
    git('commit', '-m', 'test: create isolated child')
    assert Path(git('rev-parse', '--show-toplevel')).resolve() == child.resolve()
    (state / (str(os.getpid()) + '.json')).write_text(json.dumps({
        'head': git('rev-parse', 'HEAD'), 'tree': git('ls-files'),
    }))
"""


@pytest.mark.skipif(os.name != "posix", reason="Repository shell entry requires POSIX")
@pytest.mark.parametrize("profile", ("full", "process-ownership"))
def test_pytest_entries_do_not_inherit_candidate_git_authority(
    tmp_path: Path, profile: str
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "check-full.sh",
        "check-process-ownership.sh",
        "pytest-source.py",
        "runtime_test_owner.py",
    ):
        shutil.copy2(_ROOT / "scripts" / name, scripts / name)
    shutil.copytree(
        _ROOT / "hbrowser",
        repository / "hbrowser",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        _ROOT / "scripts/_shared",
        scripts / "_shared",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    components = tomllib.loads((_ROOT / "pyproject.toml").read_text())["tool"][
        "h2h-dev-tools"
    ]["components"]
    (repository / "pyproject.toml").write_text(
        "[tool.h2h-dev-tools]\ncomponents = "
        + json.dumps(components)
        + '\n[tool.pytest.ini_options]\nmarkers = ["deep: fixture selection"]\n'
    )
    (repository / "conftest.py").write_text(_CHILD_SESSION)
    (repository / "test_probe.py").write_text(
        "import pytest\n" + "def test_selected_profile():\n    pass\n"
    )
    (repository / "tests").mkdir()
    for name in (
        "test_owned_process.py",
        "test_browser_factory.py",
        "test_runtime_test_owner.py",
        "test_tor_cleanup.py",
        "test_diagnostics.py",
        "test_page_diagnostic_transport.py",
        "test_driver_shutdown_priority.py",
    ):
        (repository / "tests" / name).write_text("def test_process_fixture(): pass\n")
    state = tmp_path / "children"
    state.mkdir()
    gate_authority = tmp_path / "gate-authority"
    executables = {
        "scripts/check-fast.sh": 'printf "%s" "$GIT_DIR" > '
        + shlex.quote(str(gate_authority))
        + "\n",
        ".venv/bin/python": (
            'case "$1" in scripts/_shared/h2h_dev_tools/offline_wheel.py) '
            "exit 0;; esac\n" + "exec " + shlex.quote(sys.executable) + ' "$@"\n'
        ),
    }
    for name, body in executables.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="Parent Test",
        GIT_AUTHOR_EMAIL="parent@example.invalid",
        GIT_COMMITTER_NAME="Parent Test",
        GIT_COMMITTER_EMAIL="parent@example.invalid",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_XDIST_AUTO_NUM_WORKERS="1",
        H2H_TEST_CHILD_STATE=str(state),
    )
    environment.pop("PYTEST_ADDOPTS", None)

    def git(*args: str) -> str:
        return subprocess.check_output(
            ("git", *args), cwd=repository, env=environment, text=True
        ).strip()

    git("init", "-b", "main")
    git("add", ".")
    git("commit", "-m", "test: initialize gate")
    git("switch", "-c", "task")
    (repository / "candidate").write_text("candidate data\n")
    git("add", "candidate")
    git("commit", "-m", "test: stage candidate")
    task = git("rev-parse", "HEAD")
    git("switch", "main")
    git("merge", "--no-ff", "--no-commit", "task")
    primary, tree = git("rev-parse", "HEAD"), git("write-tree")
    metadata = repository / ".git"
    original_index = (metadata / "index").read_bytes()
    original_config = (metadata / "config").read_bytes()
    result = subprocess.run(
        ("bash", str(scripts / f"check-{profile}.sh")),
        cwd=repository,
        env=environment
        | {
            "GIT_DIR": str(metadata),
            "GIT_WORK_TREE": str(repository),
            "GIT_INDEX_FILE": str(metadata / "index"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (metadata / "index").read_bytes() == original_index
    assert (metadata / "config").read_bytes() == original_config
    assert git("rev-parse", "HEAD") == primary
    assert git("write-tree") == tree
    assert git("rev-parse", "MERGE_HEAD") == task
    reports = list(state.glob("*.json"))
    assert reports, "The real pytest child session never ran"
    assert all(json.loads(path.read_text())["tree"] == "child.txt" for path in reports)
    if profile == "full":
        assert gate_authority.read_text() == str(metadata)
