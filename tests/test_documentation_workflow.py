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

pytestmark = pytest.mark.usefixtures("offline_code_review_cli")

_ROOT = Path(__file__).resolve().parents[1]
_COMPONENTS = tomllib.loads((_ROOT / "pyproject.toml").read_text())["tool"][
    "h2h-dev-tools"
]["components"]


_REVIEW_ENTRY = r"""from pathlib import Path
import os
import runpy
import sys

shared = Path(__file__).resolve().parent / "_shared/h2h_dev_tools"
runpy.run_path(str(shared / "source_loader.py"))["load_package"](
    shared, "_shared.h2h_dev_tools"
)
sys.dont_write_bytecode = True
from _shared.h2h_dev_tools.code_review import main

trace = Path(os.environ["DOCUMENTATION_WORKFLOW_TRACE"])
with trace.open("a", encoding="utf-8") as stream:
    stream.write(sys.argv[1] + "\n")

# Preserve the real evidence verifier; inject only after its successful read.
mutation = os.environ.get("DOCUMENTATION_MUTATE_FINAL_VERIFY")
if mutation and "full" in trace.read_text(encoding="utf-8").splitlines():
    import subprocess
    from _shared.h2h_dev_tools.code_review import Reviewer

    original_verify = Reviewer._verify

    def verify_then_mutate(self, target):
        receipt = original_verify(self, target)
        root = Path(__file__).resolve().parents[1]
        def git(*arguments):
            return subprocess.check_output(
                ("git", *arguments), cwd=root, text=True).strip()
        if mutation in {"tree", "unstaged"}:
            (root / "tests/example.py").write_text("value = 2\n", encoding="utf-8")
            if mutation == "tree":
                git("add", "tests/example.py")
        elif mutation == "head":
            git("update-ref", "HEAD", git("rev-parse", "MERGE_HEAD"))
        elif mutation == "merge_head":
            merge_head = Path(git("rev-parse", "--path-format=absolute",
                                  "--git-path", "MERGE_HEAD"))
            merge_head.write_text(git("rev-parse", "HEAD") + "\n", encoding="ascii")
        elif mutation == "untracked":
            (root / "untracked.txt").write_text("not reviewed\n", encoding="utf-8")
        else:
            raise AssertionError(f"unknown fixture mutation: {mutation}")
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(f"mutated:{mutation}\n")
        return receipt

    Reviewer._verify = verify_then_mutate

main(Path(__file__).resolve().parents[1])
"""


def _run(repository: Path, *command: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_") and key != "WORKFLOW_MERGE_TASK_REF"
    }
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        DOCUMENTATION_WORKFLOW_TRACE=str(repository.parent / "checks.log"),
    )
    return subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _git(repository: Path, *arguments: str) -> str:
    result = _run(repository, "git", *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _write(repository: Path, name: str, content: str) -> None:
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repository: Path, message: str) -> None:
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", message)


def _repository(tmp_path: Path, *, install_hooks: bool = True) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    for name, value in (
        ("user.name", "Documentation Workflow"),
        ("user.email", "documentation@example.invalid"),
        ("commit.gpgsign", "false"),
        ("workflow.primaryBranch", "main"),
    ):
        _git(repository, "config", name, value)
    for name in (
        "AGENTS.md",
        "scripts/detect-primary-branch.sh",
        "scripts/git-flow-merge.sh",
        "scripts/review-merge-candidate.sh",
        "scripts/check_change_scope.py",
        "scripts/check-docs.py",
        "scripts/check-version.py",
        ".githooks/pre-commit",
        ".githooks/pre-merge-commit",
        ".githooks/commit-msg",
        ".markdownlint-cli2.jsonc",
    ):
        destination = repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / name, destination)
    shutil.copytree(_ROOT / "scripts/_shared", repository / "scripts/_shared")
    _write(repository, "scripts/review-code.py", _REVIEW_ENTRY)
    _write(repository, ".gitignore", ".venv/\nnode_modules/\n__pycache__/\n")
    _write(repository, "README.md", "# Fixture\n")
    _write(
        repository,
        "pyproject.toml",
        '[project]\nname = "workflow-fixture"\nversion = "0.1.0"\n'
        '[tool.h2h.version]\nrelease-paths = ["src/**"]\n'
        'development-paths = ["README.md", "docs/**", "tests/**", '
        '"scripts/**", "AGENTS.md"]\n'
        + "[tool.h2h-dev-tools]\ncomponents = "
        + json.dumps(_COMPONENTS)
        + "\n",
    )
    _write(
        repository,
        ".venv/bin/python",
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n',
    )
    (repository / ".venv/bin/python").chmod(0o755)
    (repository / "node_modules").symlink_to(
        _ROOT / "node_modules", target_is_directory=True
    )
    for profile in ("fast", "full"):
        path = repository / "scripts" / f"check-{profile}.sh"
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f'printf "{profile}\\n" >> "$DOCUMENTATION_WORKFLOW_TRACE"\n'
            f'if [[ "${{DOCUMENTATION_FAIL_PROFILE:-}}" == "{profile}" ]]; '
            "then exit 42; fi\n"
            'if [[ "${DOCUMENTATION_INVALIDATE_REVIEW:-}" == 1 ]]; then\n'
            '    rm -f "$(git rev-parse --git-path h2h-review/receipts)"/*.json\n'
            "fi\n"
            'if [[ "${DOCUMENTATION_CHANGE_CANDIDATE:-}" == 1 ]]; then\n'
            "    printf 'changed = True\\n' > tests/gate-change.py\n"
            "    git add tests/gate-change.py\n"
            "fi\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
    _commit(repository, "chore: seed isolated workflow")
    if install_hooks:
        _git(repository, "config", "core.hooksPath", ".githooks")
    _git(repository, "switch", "-c", "docs/task")
    return repository


def _checks(repository: Path) -> list[str]:
    path = repository.parent / "checks.log"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.mark.skipif(os.name != "posix", reason="POSIX newline paths and shell hooks")
@pytest.mark.parametrize("profile", ["documentation", "full"])
def test_real_local_workflow_preserves_root_trailing_newlines(
    tmp_path: Path, profile: str
) -> None:
    repository = _repository(tmp_path)
    renamed = repository.with_name("repository\n\n")
    repository.rename(renamed)
    repository = renamed
    base = _git(repository, "rev-parse", "HEAD")
    if profile == "documentation":
        _write(repository, "README.md", "# Updated guide\n")
    else:
        _write(repository, "tests/local-path.py", "value = 1\n")
    _commit(repository, "test: exercise real local entry paths")
    task = _git(repository, "rev-parse", "HEAD")

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repository, "branch", "--show-current") == "main"
    assert _git(repository, "branch", "--list", "docs/task") == ""
    assert _git(repository, "show", "-s", "--format=%P", "HEAD").split() == [base, task]
    assert _git(repository, "status", "--porcelain") == ""
    assert not (repository / ".git/MERGE_HEAD").exists()
    assert _checks(repository) == (
        ["fast", "run", "verify", "full", "verify"] if profile == "full" else []
    )


def test_merge_message_exemption_requires_a_real_merge(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    base = _git(repository, "rev-parse", "HEAD")
    _write(repository, "README.md", "# Updated guide\n")
    _git(repository, "add", "README.md")
    candidate_tree = _git(repository, "write-tree")

    rejected = _run(repository, "git", "commit", "-m", "Merge a normal task change")

    assert rejected.returncode != 0, rejected.stdout + rejected.stderr
    assert "Commit message must follow Conventional Commits" in rejected.stderr
    assert _git(repository, "rev-parse", "HEAD") == base
    assert _git(repository, "write-tree") == candidate_tree
    assert (
        _run(repository, "git", "rev-parse", "--verify", "MERGE_HEAD").returncode != 0
    )
    _commit(repository, "docs: update the guide with a conventional message")
    task = _git(repository, "rev-parse", "HEAD")

    merged = _run(repository, "scripts/git-flow-merge.sh")

    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert _git(repository, "show", "-s", "--format=%P", "HEAD").split() == [base, task]
    assert _git(repository, "show", "-s", "--format=%s", "HEAD").startswith("Merge ")
    assert _git(repository, "status", "--porcelain") == ""


def test_documentation_commit_checks_staged_content_without_program_tools(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "README.md", "# Staged documentation\n")
    _git(repository, "add", "README.md")
    _write(repository, "README.md", "# Unstaged\n\n### Invalid heading jump\n")
    _write(repository, "tests/unstaged.py", "not valid Python !\n")

    _git(repository, "commit", "-m", "docs: update guide")

    assert _checks(repository) == []
    assert _git(repository, "show", "HEAD:README.md") == "# Staged documentation"


@pytest.mark.parametrize(
    "path", ("tests/example.py", "scripts/example.py", "AGENTS.md")
)
def test_mixed_commit_dispatches_to_fast_profile(tmp_path: Path, path: str) -> None:
    repository = _repository(tmp_path)
    _write(repository, "README.md", "# Updated guide\n")
    _write(repository, path, "changed\n")

    _commit(repository, "chore: update implementation and guide")

    assert _checks(repository) == ["fast"]


def test_invalid_documentation_blocks_commit_without_program_checks(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    before = _git(repository, "rev-parse", "HEAD")
    _write(repository, "README.md", "# Guide\n\n### Invalid heading jump\n")
    _git(repository, "add", "README.md")

    result = _run(repository, "git", "commit", "-m", "docs: invalid guide")

    assert result.returncode != 0
    assert "MD001" in result.stdout + result.stderr
    assert _git(repository, "rev-parse", "HEAD") == before
    assert _checks(repository) == []


@pytest.mark.parametrize("classifier", ("raise SystemExit(23)\n", "print('unknown')\n"))
def test_classifier_failure_or_unknown_profile_blocks_commit(
    tmp_path: Path, classifier: str
) -> None:
    repository = _repository(tmp_path)
    before = _git(repository, "rev-parse", "HEAD")
    _write(repository, "README.md", "# Updated guide\n")
    _git(repository, "add", "README.md")
    _write(repository, "scripts/check_change_scope.py", classifier)

    result = _run(repository, "git", "commit", "-m", "docs: update guide")

    assert result.returncode != 0
    assert _git(repository, "rev-parse", "HEAD") == before
    assert _checks(repository) == []


@pytest.mark.parametrize("code_first", (False, True))
@pytest.mark.parametrize("advance_primary", (False, True))
def test_merge_dispatch_uses_complete_candidate_not_last_commit(
    tmp_path: Path, code_first: bool, advance_primary: bool
) -> None:
    repository = _repository(tmp_path)
    if code_first:
        _write(repository, "tests/example.py", "value = 1\n")
        _commit(repository, "test: add example")
    _write(repository, "README.md", "# Updated guide\n")
    _commit(repository, "docs: update guide")
    if advance_primary:
        _git(repository, "switch", "main")
        _git(repository, "switch", "-c", "docs/parallel")
        _write(repository, "docs/parallel.md", "# Parallel guide\n")
        _commit(repository, "docs: add parallel guide")
        result = _run(repository, "scripts/git-flow-merge.sh")
        assert result.returncode == 0, result.stdout + result.stderr
        _git(repository, "switch", "docs/task")
    (repository.parent / "checks.log").write_text("", encoding="utf-8")

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _checks(repository) == (
        ["run", "verify", "full", "verify"] if code_first else []
    )
    assert _git(repository, "branch", "--show-current") == "main"
    assert (
        len(_git(repository, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
    )
    assert not _git(repository, "branch", "--list", "docs/task")
    assert _git(repository, "status", "--porcelain") == ""


@pytest.mark.parametrize("failure", ("documentation", "full"))
def test_failed_selected_merge_profile_aborts_and_retains_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repository = _repository(tmp_path, install_hooks=False)
    primary = _git(repository, "rev-parse", "main")
    if failure == "documentation":
        _write(repository, "README.md", "# Guide\n\n### Invalid heading jump\n")
    else:
        _write(repository, "tests/example.py", "value = 1\n")
        monkeypatch.setenv("DOCUMENTATION_FAIL_PROFILE", "full")
    _commit(repository, "test: construct failing candidate")
    task = _git(repository, "rev-parse", "HEAD")
    _git(repository, "config", "core.hooksPath", ".githooks")

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode != 0
    assert (
        "merge commit or release gate failed; task branch was retained" in result.stderr
    )
    assert _git(repository, "rev-parse", "main") == primary
    assert _git(repository, "rev-parse", "docs/task") == task
    assert _git(repository, "branch", "--show-current") == "docs/task"
    assert _git(repository, "status", "--porcelain") == ""
    assert not (repository / ".git/MERGE_HEAD").exists()
    assert _checks(repository) == (
        ["run", "verify", "full"] if failure == "full" else []
    )
    if failure == "documentation":
        assert "MD001" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "entrypoint",
    (
        (".githooks/pre-merge-commit",),
        (".githooks/pre-commit",),
        ("git", "commit", "--no-edit"),
    ),
)
def test_full_merge_gate_requires_review_before_program_checks(
    tmp_path: Path, entrypoint: tuple[str, ...]
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/example.py", "value = 1\n")
    _commit(repository, "test: add example")
    _git(repository, "switch", "main")
    _git(repository, "merge", "--no-ff", "--no-commit", "docs/task")
    (repository.parent / "checks.log").write_text("", encoding="utf-8")

    result = _run(repository, *entrypoint)

    assert result.returncode != 0
    assert "h2h code review:" in result.stderr
    assert _checks(repository) == ["verify"]


def test_failed_review_aborts_candidate_without_running_full_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/example.py", "value = 1\n")
    _commit(repository, "test: add example")
    primary = _git(repository, "rev-parse", "main")
    task = _git(repository, "rev-parse", "HEAD")
    (repository.parent / "checks.log").write_text("", encoding="utf-8")
    monkeypatch.setenv("DOCUMENTATION_FAIL_REVIEW", "1")

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode != 0
    assert "code review failed" in result.stderr
    assert _checks(repository) == ["run"]
    assert _git(repository, "rev-parse", "main") == primary
    assert _git(repository, "rev-parse", "docs/task") == task
    assert _git(repository, "branch", "--show-current") == "docs/task"
    assert not (repository / ".git/MERGE_HEAD").exists()


@pytest.mark.parametrize("invalidation", ("receipt", "tree"))
def test_full_merge_gate_rechecks_review_after_program_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalidation: str
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/example.py", "value = 1\n")
    _commit(repository, "test: add example")
    primary = _git(repository, "rev-parse", "main")
    task = _git(repository, "rev-parse", "HEAD")
    (repository.parent / "checks.log").write_text("", encoding="utf-8")
    monkeypatch.setenv(
        "DOCUMENTATION_INVALIDATE_REVIEW"
        if invalidation == "receipt"
        else "DOCUMENTATION_CHANGE_CANDIDATE",
        "1",
    )

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode != 0
    assert "h2h code review:" in result.stderr
    assert _checks(repository) == ["run", "verify", "full", "verify"]
    assert _git(repository, "rev-parse", "main") == primary
    assert _git(repository, "rev-parse", "docs/task") == task
    merge_head = repository / ".git/MERGE_HEAD"
    admission = repository / ".git/h2h-merge-workflow.lock"
    if invalidation == "receipt":
        assert _git(repository, "branch", "--show-current") == "docs/task"
        assert not merge_head.exists()
        assert not admission.exists()
        assert _git(repository, "status", "--porcelain") == ""
        return

    # A gate's staged change is external to the captured candidate. Preserve
    # those bytes and the original workflow's admission instead of aborting it.
    assert "Candidate changed outside this workflow" in result.stderr
    assert _git(repository, "branch", "--show-current") == "main"
    assert merge_head.read_text(encoding="ascii") == task + "\n"
    staged = _run(repository, "git", "show", ":tests/gate-change.py")
    assert staged.returncode == 0
    assert staged.stdout == "changed = True\n"
    assert (repository / "tests/gate-change.py").read_bytes() == b"changed = True\n"
    index = _git(repository, "write-tree")
    owner_bytes = (admission / "owner.json").read_bytes()
    owner = json.loads(owner_bytes)
    assert owner["repository"] == str(repository)
    assert isinstance(owner["owner_pid"], int) and owner["owner_pid"] > 0
    assert len(owner["token"]) == 64 and set(owner["token"]) <= set("0123456789abcdef")
    identity = admission.stat()

    blocked = _run(repository, "scripts/git-flow-merge.sh", "--cleanup-only")

    assert blocked.returncode != 0
    assert "admission is already held" in blocked.stderr
    assert (admission.stat().st_dev, admission.stat().st_ino) == (
        identity.st_dev,
        identity.st_ino,
    )
    assert (admission / "owner.json").read_bytes() == owner_bytes
    assert merge_head.read_text(encoding="ascii") == task + "\n"
    assert _git(repository, "write-tree") == index
    assert (repository / "tests/gate-change.py").read_bytes() == b"changed = True\n"
    assert _git(repository, "branch", "--show-current") == "main"
    assert _git(repository, "rev-parse", "main") == primary
    assert _git(repository, "rev-parse", "docs/task") == task
    assert _checks(repository) == ["run", "verify", "full", "verify"]


def test_merge_version_gate_uses_repository_python_without_system_shebang(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    script = repository / "scripts/check-version.py"
    script.write_text(
        "#!/nonexistent/system/python\n" + script.read_text().split("\n", 1)[1],
        encoding="utf-8",
    )
    _commit(repository, "test: remove the system interpreter prerequisite")

    result = _run(repository, "scripts/git-flow-merge.sh")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(repository, "branch", "--show-current") == "main"


def test_repository_policy_rejects_combined_runtime_and_version_commit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, install_hooks=False)
    configuration = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    policy = configuration["tool"]["h2h"]["version"]
    policy_text = "\n".join(
        f"{key} = {json.dumps(value)}" for key, value in policy.items()
    )
    metadata = (
        '[project]\nname = "release-fixture"\nversion = "0.1.0"\n'
        "[tool.h2h.version]\n"
        + policy_text
        + "\n"
        + "[tool.h2h-dev-tools]\ncomponents = "
        + json.dumps(_COMPONENTS)
        + "\n"
    )
    release_directory = next(
        pattern.removesuffix("**")
        for pattern in policy["release-paths"]
        if pattern.endswith("/**")
    )
    runtime = release_directory + "version_probe.py"
    _write(repository, "pyproject.toml", metadata)
    _write(repository, runtime, "value = 1\n")
    _commit(repository, "chore: seed actual repository version policy")
    base = _git(repository, "rev-parse", "HEAD")
    _write(repository, runtime, "value = 2\n")
    _write(repository, "pyproject.toml", metadata.replace('"0.1.0"', '"0.1.1"'))
    _commit(repository, "fix: combine runtime change and version bump")

    result = _run(
        repository, ".venv/bin/python", "scripts/check-version.py", "--base", base
    )

    assert result.returncode != 0
    assert "version transition requires its dedicated release commit" in result.stderr


@pytest.mark.parametrize(
    "mutation", ("stable", "tree", "head", "merge_head", "unstaged", "untracked")
)
def test_full_merge_gate_rejects_drift_during_final_offline_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/example.py", "value = 1\n")
    _commit(repository, "test: add reviewed example")
    _git(repository, "switch", "main")
    _git(repository, "merge", "--no-ff", "--no-commit", "docs/task")
    (repository.parent / "checks.log").write_text("", encoding="utf-8")
    reviewed = _run(
        repository, ".venv/bin/python", "scripts/review-code.py", "run", "--index"
    )
    assert reviewed.returncode == 0, reviewed.stdout + reviewed.stderr
    if mutation != "stable":
        monkeypatch.setenv("DOCUMENTATION_MUTATE_FINAL_VERIFY", mutation)

    result = _run(repository, ".githooks/pre-merge-commit")

    expected_checks = ["run", "verify", "full", "verify"]
    if mutation == "stable":
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("Verified exact-candidate code review:") == 2
    else:
        expected_checks.append(f"mutated:{mutation}")
        assert _checks(repository) == expected_checks
        assert result.returncode != 0, result.stdout + result.stderr
        assert "h2h code review:" in result.stderr
        # Only the first verification may announce success.
        assert result.stdout.count("Verified exact-candidate code review:") == 1
    assert _checks(repository) == expected_checks
