"""Local gate and hook entry points preserve Git-authoritative path bytes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _shared.h2h_dev_tools.executable_fixture import write_python_executable

_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shell entry points")


@pytest.fixture(params=["checkout\ninside", "checkout\n\n"])
def repository(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    assert isinstance(request.param, str)
    root = tmp_path / request.param
    root.mkdir()
    subprocess.run(
        ("git", "init", "-b", "master", str(root)), check=True, capture_output=True
    )
    return root


def _copy(root: Path, relative: str) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_ROOT / relative, destination)


@pytest.mark.parametrize(
    "entry",
    ["check-fast.sh", "check-full.sh", "format.sh", "check-process-ownership.sh"],
)
def test_quality_gate_keeps_git_root_with_line_breaks(
    repository: Path, tmp_path: Path, entry: str
) -> None:
    for name in ("check-fast.sh", "check-full.sh", "format.sh"):
        _copy(repository, f"scripts/{name}")
    _copy(repository, "scripts/check-process-ownership.sh")
    log = tmp_path / "calls.jsonl"
    for relative in (
        ".venv/bin/ruff",
        ".venv/bin/mypy",
        ".venv/bin/python",
        "node_modules/.bin/markdownlint-cli2",
    ):
        executable = repository / relative
        executable.parent.mkdir(parents=True, exist_ok=True)
        write_python_executable(
            executable,
            "import json, os, sys\n"
            f"with open({str(log)!r}, 'a') as output:\n"
            "    record = {'cwd': os.getcwd(), 'argv': sys.argv}\n"
            "    output.write(json.dumps(record) + '\\n')\n",
        )
    nested = repository / "nested"
    nested.mkdir()

    result = subprocess.run(
        ("bash", str(repository / "scripts" / entry)),
        cwd=nested,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert (
        len(calls)
        == {
            "check-fast.sh": 5,
            "check-full.sh": 7,
            "format.sh": 3,
            "check-process-ownership.sh": 1,
        }[entry]
    )
    assert all(call["cwd"] == str(repository) for call in calls)


def test_commit_hook_preserves_relative_message_and_git_authority(
    repository: Path, tmp_path: Path
) -> None:
    for relative in (
        ".githooks/commit-msg",
        "scripts/_shared/h2h_dev_tools/commit-message.sh",
    ):
        _copy(repository, relative)
    message = tmp_path / "message"
    message.write_text("fix: preserve source path\n")
    environment = dict(os.environ)
    environment.update(GIT_DIR=str(repository / ".git"), GIT_WORK_TREE=str(repository))

    result = subprocess.run(
        ("bash", str(repository / ".githooks/commit-msg"), "message"),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_rebase_hook_keeps_primary_protection_in_root_with_line_breaks(
    repository: Path,
) -> None:
    for relative in (
        ".githooks/pre-rebase",
        "scripts/detect-primary-branch.sh",
        "scripts/_shared/h2h_dev_tools/primary-branch.sh",
    ):
        _copy(repository, relative)
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Path Test",
            "-c",
            "user.email=path@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "test: initialize",
        ),
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        ("bash", str(repository / ".githooks/pre-rebase"), "upstream", "master"),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert "Rebasing primary is not allowed" in result.stderr


def _two_checkouts(tmp_path: Path) -> tuple[Path, Path]:
    sibling = tmp_path / "checkout"
    intended = tmp_path / "checkout\n\n"
    for root in (sibling, intended):
        root.mkdir()
        subprocess.run(
            ("git", "init", "-b", "master", str(root)),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Path Test",
                "-c",
                "user.email=path@example.invalid",
                "commit",
                "--allow-empty",
                "-m",
                "test: initialize independent checkout",
            ),
            check=True,
            capture_output=True,
        )
        for name in ("rebuild-env.sh", "install-git-hooks.sh", "format.sh"):
            _copy(root, f"scripts/{name}")
        for name in (
            "rebuild-env.sh",
            "resolve-sources.sh",
            "environment.py",
            "install-git-hooks.sh",
            "primary-branch.sh",
        ):
            _copy(root, f"scripts/_shared/h2h_dev_tools/{name}")
        (root / ".venv").mkdir()
        (root / ".venv/preserve-me").write_text("original environment\n")
        (root / "format-target.txt").write_text("original formatting\n")
        (root / "candidate.whl").write_bytes(b"fixture source location")
        for relative in (".venv/bin/ruff", "node_modules/.bin/markdownlint-cli2"):
            executable = root / relative
            executable.parent.mkdir(parents=True, exist_ok=True)
            write_python_executable(
                executable,
                "from pathlib import Path\n"
                "Path('format-target.txt').write_text('formatted by local tool\\n')\n",
            )
    return intended, sibling


@pytest.mark.parametrize(
    "entry", ["format.sh", "install-git-hooks.sh", "rebuild-env.sh"]
)
@pytest.mark.parametrize("directory_search", ["unset", "current", "foreign"])
def test_mutating_entries_never_target_a_truncated_sibling_checkout(
    tmp_path: Path, entry: str, directory_search: str
) -> None:
    intended, sibling = _two_checkouts(tmp_path)
    sibling_config = (sibling / ".git/config").read_bytes()
    source_plan = tmp_path / "source-plan.jsonl"
    binary = tmp_path / "controlled-tools"
    binary.mkdir()
    uv = binary / "uv"
    write_python_executable(
        uv,
        "import json, sys\nfrom pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        f"assert Path.cwd() in (Path({str(intended)!r}), Path({str(sibling)!r}))\n"
        "if arguments == ['--version']:\n"
        "    print('uv 0.13.0')\n"
        "elif arguments[0] == 'venv':\n"
        "    destination = Path(arguments[-1])\n"
        "    if '--clear' in arguments:\n"
        "        assert destination == Path('.venv')\n"
        "        (destination / 'preserve-me').unlink()\n"
        "        (destination / 'rebuilt').touch()\n"
        "    python = destination / 'bin/python'\n"
        "    python.parent.mkdir(parents=True, exist_ok=True)\n"
        f"    python.symlink_to({sys.executable!r})\n"
        "elif arguments[:2] == ['pip', 'compile']:\n"
        "    requirements = Path(arguments[arguments.index('--python') + 2])\n"
        f"    with open({str(source_plan)!r}, 'a') as output:\n"
        "        record = {'cwd': str(Path.cwd()), 'plan': requirements.read_text()}\n"
        "        output.write(json.dumps(record) + '\\n')\n"
        "    Path(arguments[arguments.index('--output-file') + 1]).write_text('')\n"
        "else:\n"
        "    assert arguments[:2] == ['pip', 'install']\n",
    )
    npm = binary / "npm"
    write_python_executable(
        npm,
        "from pathlib import Path\nPath('.npm-ran').touch()\n",
    )
    nested = (sibling if entry == "rebuild-env.sh" else intended) / "nested"
    if directory_search != "unset":
        nested = intended
    nested.mkdir(exist_ok=True)
    if entry == "rebuild-env.sh":
        (nested / "candidate.whl").write_bytes(b"caller-selected fixture source")
    arguments = (
        ("--source", "sample=./candidate.whl") if entry == "rebuild-env.sh" else ()
    )

    environment = dict(os.environ)
    environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
    environment.pop("CDPATH", None)
    entry_path = str(intended / "scripts" / entry)
    if directory_search != "unset":
        environment["CDPATH"] = "." if directory_search == "current" else str(sibling)
        entry_path = f"scripts/{entry}"
    result = subprocess.run(
        ("bash", entry_path, *arguments),
        cwd=nested,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The other checkout is a valid, tempting target: check its real state,
    # rather than merely accepting a failure to find a nonexistent directory.
    assert (sibling / ".git/config").read_bytes() == sibling_config
    assert (sibling / ".venv/preserve-me").read_text() == "original environment\n"
    assert (sibling / "format-target.txt").read_text() == "original formatting\n"
    assert not (sibling / ".venv/rebuilt").exists()
    assert not (sibling / ".npm-ran").exists()
    if entry == "format.sh":
        assert (
            intended / "format-target.txt"
        ).read_text() == "formatted by local tool\n"
    elif entry == "install-git-hooks.sh":
        configured = subprocess.run(
            (
                "git",
                "-C",
                str(intended),
                "config",
                "--local",
                "--get",
                "core.hooksPath",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        assert configured.stdout == ".githooks\n"
    else:
        assert not (intended / ".venv/preserve-me").exists()
        assert (intended / ".venv/rebuilt").exists()
        assert (intended / ".npm-ran").exists()
        plans = [json.loads(line) for line in source_plan.read_text().splitlines()]
        assert plans == [
            {
                "cwd": str(intended),
                "plan": (
                    f"sample @ {(nested / 'candidate.whl').as_uri()}\n-e .[dev]\n"
                ),
            }
        ]


@pytest.mark.parametrize("directory_search", ["current", "foreign"])
def test_primary_query_ignores_cdpath_and_emits_only_branch_name(
    tmp_path: Path, directory_search: str
) -> None:
    intended, sibling = _two_checkouts(tmp_path)
    for root in (intended, sibling):
        _copy(root, "scripts/detect-primary-branch.sh")
    subprocess.run(
        ("git", "-C", str(sibling), "branch", "-m", "main"),
        check=True,
        capture_output=True,
    )
    environment = dict(os.environ)
    environment["CDPATH"] = "." if directory_search == "current" else str(sibling)

    result = subprocess.run(
        ("bash", "scripts/detect-primary-branch.sh"),
        cwd=intended,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "master\n"
