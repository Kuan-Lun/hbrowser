"""Regression tests for task-branch merge and no-op cleanup policy."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MERGE_SCRIPT = _PROJECT_ROOT / "scripts" / "git-flow-merge.sh"
_PRIMARY_SCRIPT = _PROJECT_ROOT / "scripts" / "detect-primary-branch.sh"
_TASK_BRANCH = "task/example"
_GIT_LOCAL_ENVIRONMENT_VARIABLES = frozenset(
    subprocess.run(
        ("git", "rev-parse", "--local-env-vars"),
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
)


def _run(
    cwd: Path,
    *command: str,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    for variable in _GIT_LOCAL_ENVIRONMENT_VARIABLES:
        process_env.pop(variable, None)
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=process_env,
    )


def _git(
    cwd: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(cwd, "git", *arguments, check=check)


def _git_stdout(cwd: Path, *arguments: str) -> str:
    return _git(cwd, *arguments).stdout.strip()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _initialize_repository(tmp_path: Path) -> tuple[Path, str, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Git Flow Test")
    _git(repository, "config", "user.email", "git-flow@example.invalid")
    _git(repository, "config", "commit.gpgSign", "false")
    _git(repository, "config", "core.hooksPath", ".githooks")

    scripts = repository / "scripts"
    scripts.mkdir()
    shutil.copy2(_MERGE_SCRIPT, scripts / _MERGE_SCRIPT.name)
    shutil.copy2(_PRIMARY_SCRIPT, scripts / _PRIMARY_SCRIPT.name)

    hooks = repository / ".githooks"
    hooks.mkdir()
    _write_executable(
        hooks / "pre-merge-commit",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${GIT_FLOW_TEST_GATE_LOG:-}" ]]; then
    printf 'gate-called\\n' >> "$GIT_FLOW_TEST_GATE_LOG"
fi
""",
    )
    _write_executable(
        hooks / "post-checkout",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${GIT_FLOW_TEST_UPDATE_REF:-}" ]]; then
    git update-ref \
        "$GIT_FLOW_TEST_UPDATE_REF" \
        "${GIT_FLOW_TEST_UPDATE_OID:?}"
fi
""",
    )

    (repository / "payload.txt").write_text("base\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "test: initial state")
    initial_oid = _git_stdout(repository, "rev-parse", "HEAD")
    return repository, initial_oid, tmp_path / "gate.log"


def _run_merge_script(
    worktree: Path,
    gate_log: Path,
    *arguments: str,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = {"GIT_FLOW_TEST_GATE_LOG": str(gate_log)}
    if env is not None:
        process_env.update(env)
    return _run(
        worktree,
        str(worktree / "scripts" / "git-flow-merge.sh"),
        *arguments,
        check=False,
        env=process_env,
    )


def _assert_branch_missing(repository: Path, branch: str) -> None:
    result = _git(
        repository,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{branch}",
        check=False,
    )
    assert result.returncode == 1


def test_equal_tip_noop_switches_to_primary_without_merge(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)

    result = _run_merge_script(repository, gate_log)

    assert result.returncode == 0, result.stderr
    assert "Cleaned up no-op task" in result.stdout
    assert _git_stdout(repository, "branch", "--show-current") == "main"
    assert _git_stdout(repository, "rev-parse", "HEAD") == initial_oid
    head_line = _git_stdout(
        repository,
        "rev-list",
        "--parents",
        "-n",
        "1",
        "HEAD",
    ).split()
    assert len(head_line) == 1
    _assert_branch_missing(repository, _TASK_BRANCH)
    assert not gate_log.exists()


def test_ancestor_noop_removes_linked_task_worktree(tmp_path: Path) -> None:
    repository, _, gate_log = _initialize_repository(tmp_path)
    task_worktree = tmp_path / "task-worktree"
    _git(
        repository,
        "worktree",
        "add",
        "-b",
        _TASK_BRANCH,
        str(task_worktree),
        "main",
    )
    (repository / "payload.txt").write_text("primary advanced\n")
    _git(repository, "add", "payload.txt")
    _git(repository, "commit", "-m", "test: advance primary")
    primary_oid = _git_stdout(repository, "rev-parse", "main")

    result = _run_merge_script(task_worktree, gate_log)

    assert result.returncode == 0, result.stderr
    assert not task_worktree.exists()
    assert _git_stdout(repository, "rev-parse", "main") == primary_oid
    _assert_branch_missing(repository, _TASK_BRANCH)
    assert not gate_log.exists()


def test_git_subprocesses_discard_hook_local_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_path = str(tmp_path / "outer-repository-state")
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "outer-index"))
    for variable in _GIT_LOCAL_ENVIRONMENT_VARIABLES - {"GIT_INDEX_FILE"}:
        monkeypatch.setenv(variable, leaked_path)

    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    task_worktree = tmp_path / "task-worktree"
    _git(
        repository,
        "worktree",
        "add",
        "-b",
        _TASK_BRANCH,
        str(task_worktree),
        "main",
    )

    result = _run_merge_script(task_worktree, gate_log)

    assert result.returncode == 0, result.stderr
    assert not task_worktree.exists()
    assert _git_stdout(repository, "rev-parse", "main") == initial_oid
    _assert_branch_missing(repository, _TASK_BRANCH)
    assert not gate_log.exists()


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_noop_rejects_dirty_task_worktree(
    tmp_path: Path,
    dirty_kind: str,
) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)
    if dirty_kind == "tracked":
        (repository / "payload.txt").write_text("dirty\n")
    else:
        (repository / "untracked.txt").write_text("dirty\n")

    result = _run_merge_script(repository, gate_log, "--cleanup-only")

    assert result.returncode != 0
    assert "worktree is not clean" in result.stderr
    assert _git_stdout(repository, "branch", "--show-current") == _TASK_BRANCH
    assert _git_stdout(repository, "rev-parse", _TASK_BRANCH) == initial_oid
    assert not gate_log.exists()


def test_noop_rejects_active_git_operation(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)
    bisect_start = Path(
        _git_stdout(
            repository,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "BISECT_START",
        )
    )
    bisect_start.write_text("main\n")

    result = _run_merge_script(repository, gate_log, "--cleanup-only")

    assert result.returncode != 0
    assert "Git operation is active: BISECT_START" in result.stderr
    assert _git_stdout(repository, "branch", "--show-current") == _TASK_BRANCH
    assert _git_stdout(repository, "rev-parse", _TASK_BRANCH) == initial_oid
    assert not gate_log.exists()


def test_cleanup_only_rejects_unique_commits(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)
    (repository / "payload.txt").write_text("task change\n")
    _git(repository, "add", "payload.txt")
    _git(repository, "commit", "-m", "test: task change")
    task_oid = _git_stdout(repository, "rev-parse", "HEAD")

    result = _run_merge_script(repository, gate_log, "--cleanup-only")

    assert result.returncode != 0
    assert "task contains commits not present in primary" in result.stderr
    assert _git_stdout(repository, "branch", "--show-current") == _TASK_BRANCH
    assert _git_stdout(repository, "rev-parse", _TASK_BRANCH) == task_oid
    assert _git_stdout(repository, "rev-parse", "main") == initial_oid
    assert not gate_log.exists()


def test_cleanup_only_does_not_treat_equal_trees_as_noop(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)
    (repository / "payload.txt").write_text("temporary change\n")
    _git(repository, "add", "payload.txt")
    _git(repository, "commit", "-m", "test: temporary change")
    (repository / "payload.txt").write_text("base\n")
    _git(repository, "add", "payload.txt")
    _git(repository, "commit", "-m", "test: restore original tree")
    task_oid = _git_stdout(repository, "rev-parse", "HEAD")
    tree_comparison = _git(
        repository,
        "diff",
        "--quiet",
        "main",
        _TASK_BRANCH,
        check=False,
    )
    assert tree_comparison.returncode == 0

    result = _run_merge_script(repository, gate_log, "--cleanup-only")

    assert result.returncode != 0
    assert "task contains commits not present in primary" in result.stderr
    assert _git_stdout(repository, "rev-parse", _TASK_BRANCH) == task_oid
    assert _git_stdout(repository, "rev-parse", "main") == initial_oid
    assert not gate_log.exists()


def test_noop_revalidates_task_oid_after_switch(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    _git(repository, "switch", "-c", _TASK_BRANCH)
    tree_oid = _git_stdout(repository, "rev-parse", "HEAD^{tree}")
    moved_oid = _git_stdout(
        repository,
        "commit-tree",
        tree_oid,
        "-p",
        initial_oid,
        "-m",
        "test: concurrent task update",
    )

    result = _run_merge_script(
        repository,
        gate_log,
        env={
            "GIT_FLOW_TEST_UPDATE_REF": f"refs/heads/{_TASK_BRANCH}",
            "GIT_FLOW_TEST_UPDATE_OID": moved_oid,
        },
    )

    assert result.returncode != 0
    assert "Git ref changed during cleanup" in result.stderr
    assert _git_stdout(repository, "branch", "--show-current") == "main"
    assert _git_stdout(repository, "rev-parse", _TASK_BRANCH) == moved_oid
    assert not gate_log.exists()


def test_unique_commit_preserves_full_merge_and_cleanup_path(tmp_path: Path) -> None:
    repository, initial_oid, gate_log = _initialize_repository(tmp_path)
    task_worktree = tmp_path / "task-worktree"
    _git(
        repository,
        "worktree",
        "add",
        "-b",
        _TASK_BRANCH,
        str(task_worktree),
        "main",
    )
    (task_worktree / "payload.txt").write_text("task change\n")
    _git(task_worktree, "add", "payload.txt")
    _git(task_worktree, "commit", "-m", "test: task change")
    task_oid = _git_stdout(task_worktree, "rev-parse", "HEAD")

    result = _run_merge_script(task_worktree, gate_log)

    assert result.returncode == 0, result.stderr
    assert "Merged task/example into main" in result.stdout
    assert not task_worktree.exists()
    merge_line = _git_stdout(
        repository,
        "rev-list",
        "--parents",
        "-n",
        "1",
        "main",
    ).split()
    assert len(merge_line) == 3
    assert initial_oid in merge_line[1:]
    assert task_oid in merge_line[1:]
    assert (repository / "payload.txt").read_text() == "task change\n"
    _assert_branch_missing(repository, _TASK_BRANCH)
    assert gate_log.read_text() == "gate-called\n"
