from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_PATH = "hbrowser/version_probe.py"


def _run(
    repository: Path, environment: dict[str, str], *command: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _git(repository: Path, environment: dict[str, str], *arguments: str) -> str:
    result = _run(repository, environment, "git", *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("stale_bytecode", [False, True])
@pytest.mark.parametrize(
    "runtime_change", [False, True], ids=["development", "runtime"]
)
def test_local_version_entry_checks_the_exact_merge_candidate(
    tmp_path: Path, runtime_change: bool, stale_bytecode: bool
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(_ROOT / "scripts/check-version.py", scripts / "check-version.py")
    shutil.copytree(
        _ROOT / "scripts/_shared",
        scripts / "_shared",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    configuration = tomllib.loads(
        (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    policy = configuration["tool"]["h2h"]["version"]
    components = configuration["tool"]["h2h-dev-tools"]["components"]
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "version-entry-fixture"\nversion = "0.1.0"\n'
        "[tool.h2h.version]\n"
        + "".join(f"{name} = {json.dumps(value)}\n" for name, value in policy.items())
        + "[tool.h2h-dev-tools]\ncomponents = "
        + json.dumps(components)
        + "\n",
        encoding="utf-8",
    )
    (repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    runtime = repository / _RUNTIME_PATH
    runtime.parent.mkdir(parents=True)
    runtime.write_text("value = 1\n", encoding="utf-8")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_") and name != "WORKFLOW_MERGE_TASK_REF"
    }
    # This child must exercise its normal adjacent cache, independent of the
    # outer canonical pytest runner's private cache prefix.
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    _git(repository, environment, "init", "-b", "main")
    _git(repository, environment, "config", "user.name", "Version Entry Test")
    _git(repository, environment, "config", "user.email", "version@example.invalid")
    _git(repository, environment, "add", ".")
    _git(repository, environment, "commit", "-m", "chore: initialize fixture")
    _git(repository, environment, "switch", "-c", "task/version-entry")
    tests = repository / "tests"
    tests.mkdir()
    (tests / "test_probe.py").write_text("def test_probe(): pass\n", encoding="utf-8")
    if runtime_change:
        runtime.write_text("value = 2\n", encoding="utf-8")
    _git(repository, environment, "add", ".")
    _git(repository, environment, "commit", "-m", "fix: exercise local release paths")
    task_oid = _git(repository, environment, "rev-parse", "HEAD")
    _git(repository, environment, "switch", "main")
    _git(repository, environment, "merge", "--no-ff", "--no-commit", task_oid)
    candidate_tree = _git(repository, environment, "write-tree")
    # An unstaged runtime edit must not replace the exact candidate under review.
    runtime.write_text("value = 3\n", encoding="utf-8")
    environment["WORKFLOW_MERGE_TASK_REF"] = task_oid

    if stale_bytecode:
        module = scripts / "_shared/h2h_dev_tools/version_policy.py"
        stale = tmp_path / "stale-version.py"
        prefix = b"raise SystemExit('STALE SNAPSHOT EXECUTED')\n"
        stale.write_bytes(prefix + b" " * (module.stat().st_size - len(prefix)))
        stamp = module.stat().st_mtime
        os.utime(stale, (stamp, stamp))
        cache = (
            module.parent
            / "__pycache__"
            / f"{module.stem}.{sys.implementation.cache_tag}.pyc"
        )
        cache.parent.mkdir(exist_ok=True)
        py_compile.compile(
            str(stale),
            cfile=str(cache),
            dfile=str(module),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        ordinary = _run(
            repository,
            environment,
            sys.executable,
            "-B",
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "from _shared.h2h_dev_tools import version_policy",
        )
        assert ordinary.returncode != 0
        assert "STALE SNAPSHOT EXECUTED" in ordinary.stdout + ordinary.stderr

    result = _run(
        repository,
        environment,
        sys.executable,
        "-B",
        "scripts/check-version.py",
        "--index",
    )

    assert "STALE SNAPSHOT EXECUTED" not in result.stdout + result.stderr
    if runtime_change:
        assert result.returncode != 0
        assert "release surface changed without a project version bump" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repository, environment, "write-tree") == candidate_tree
    assert _git(repository, environment, "rev-parse", "MERGE_HEAD") == task_oid
    assert runtime.read_text(encoding="utf-8") == "value = 3\n"
