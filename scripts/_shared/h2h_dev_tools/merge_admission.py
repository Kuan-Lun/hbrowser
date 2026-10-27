"""Exclusive cooperating merge workflows and exact failure-recovery authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NoReturn, TypedDict

from . import git_index


class Snapshot(TypedDict):
    branch: str
    head: str
    primary_oid: str | None
    task_oid: str | None
    merge_head: str | None
    metadata: dict[str, str]
    index: str
    tree: str | None
    worktree: str
    status: str


class AdmissionError(RuntimeError):
    """Admission or recovery cannot be proved safe; retain the existing state."""


@dataclass(frozen=True)
class Workflow:
    token: str
    lock_directory: str
    lock_device: int
    lock_inode: int
    owner_pid: int
    repository: str
    worktree: str = ""
    primary_ref: str = ""
    primary_oid: str = ""
    task_ref: str = ""
    task_oid: str = ""
    entry_ref: str = ""
    phase: str = "admitted"
    candidate: Snapshot | None = None

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(asdict(self), sort_keys=True) + "\n")
        temporary.replace(path)

    def verify_lock(self) -> None:
        directory = Path(self.lock_directory)
        identity = directory.lstat()
        if not stat.S_ISDIR(identity.st_mode) or (
            identity.st_dev,
            identity.st_ino,
        ) != (self.lock_device, self.lock_inode):
            raise AdmissionError("Workflow admission directory identity changed")
        owner_path = directory / "owner.json"
        if not stat.S_ISREG(owner_path.lstat().st_mode) or json.loads(
            owner_path.read_text()
        ) != {
            "token": self.token,
            "owner_pid": self.owner_pid,
            "repository": self.repository,
        }:
            raise AdmissionError("Workflow admission token no longer matches")


def _capture_git(
    repository: Path, *arguments: str
) -> subprocess.CompletedProcess[bytes]:
    # These calls stay inside the existing canonical Git process owner. Recovery
    # retains its ten-second deadline; normal phases retain their outer lifetime.
    # Regular files let this leader return after Git exits even when a callback's
    # descendants retain stdout/stderr, so the owner can detect and clean them.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        result = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            stdout=output,
            stderr=errors,
            check=False,
        )
        # Snapshot both extents before reading either stream. A surviving
        # callback may keep appending; reads must not chase that moving EOF.
        output_size = os.fstat(output.fileno()).st_size
        error_size = os.fstat(errors.fileno()).st_size
        output.seek(0)
        errors.seek(0)
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            output.read(output_size),
            errors.read(error_size),
        )


def _git(repository: Path, *arguments: str, missing_ref: bool = False) -> bytes:
    result = _capture_git(repository, *arguments)
    if missing_ref and result.returncode == 1 and not result.stdout:
        return b""
    if result.returncode:
        raise AdmissionError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    output = _git(repository, *arguments)
    if not output.endswith(b"\n"):
        raise AdmissionError("Git did not terminate its requested single record")
    return os.fsdecode(output[:-1])


def _reference(repository: Path, name: str) -> str | None:
    value = _git(repository, "rev-parse", "--verify", "--quiet", name, missing_ref=True)
    if not value:
        return None
    if not value.endswith(b"\n"):
        raise AdmissionError("Git did not terminate its reference record")
    return value[:-1].decode("ascii")


def _load(path: Path) -> Workflow:
    document = json.loads(path.read_text())
    state = Workflow(**document)
    if (
        not isinstance(state.token, str)
        or len(state.token) != 64
        or any(c not in "0123456789abcdef" for c in state.token)
        or state.phase
        not in {
            "admitted",
            "prepared",
            "merging",
            "candidate",
            "committed",
            "aborted",
            "restored",
            "cleanup",
            "retained",
        }
    ):
        raise AdmissionError("Invalid workflow admission state")
    state.verify_lock()
    return state


def acquire(repository: Path, state_path: Path, owner_pid: int) -> None:
    common = Path(
        _git_text(repository, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    directory = common / "h2h-merge-workflow.lock"
    token = secrets.token_hex(32)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as error:
        raise AdmissionError(
            f"Merge workflow admission is already held: {directory}; "
            "do not reclaim it until the existing workflow and owned "
            "cleanup are verified"
        ) from error
    # Only this successful mkdir authorizes rollback of partial initialization.
    # A failed rollback leaves the directory as a fail-closed admission barrier.
    try:
        identity = directory.lstat()
        state = Workflow(
            token,
            str(directory),
            identity.st_dev,
            identity.st_ino,
            owner_pid,
            str(repository),
        )
        (directory / "owner.json").write_text(
            json.dumps(
                {
                    "token": token,
                    "owner_pid": owner_pid,
                    "repository": str(repository),
                }
            )
            + "\n"
        )
        state.save(state_path)
    except BaseException as original:
        try:
            (directory / "owner.json").unlink(missing_ok=True)
            directory.rmdir()
        except OSError as cleanup_error:
            raise AdmissionError(
                f"Admission initialization and cleanup failed; retain "
                f"{directory}: {cleanup_error}"
            ) from original
        raise


def check_clean(state_path: Path, worktree: Path) -> None:
    _load(state_path)
    if _git(worktree, "status", "--porcelain=v1", "--untracked-files=all", "-z"):
        raise AdmissionError(f"worktree is not clean: {worktree}")


def configure(
    state_path: Path,
    *,
    worktree: Path,
    primary_ref: str,
    primary_oid: str,
    task_ref: str,
    task_oid: str,
) -> None:
    state = _load(state_path)
    if state.phase != "admitted":
        raise AdmissionError("Workflow identity was already configured")
    _assert_index_visible(Path(state.repository))
    _assert_index_visible(worktree)
    entry_ref = _git_text(Path(state.repository), "symbolic-ref", "--quiet", "HEAD")
    if entry_ref not in {primary_ref, task_ref}:
        raise AdmissionError("Caller checkout is neither the primary nor task branch")
    replace(
        state,
        entry_ref=entry_ref,
        worktree=str(worktree),
        primary_ref=primary_ref,
        primary_oid=primary_oid,
        task_ref=task_ref,
        task_oid=task_oid,
        phase="prepared",
    ).save(state_path)


def _assert_index_visible(root: Path) -> None:
    try:
        git_index.assert_index_visible(lambda *args: _git(root, *args))
    except git_index.GitIndexError as error:
        raise AdmissionError(str(error)) from error


def _snapshot(state: Workflow) -> Snapshot:
    root = Path(state.worktree)
    _assert_index_visible(root)
    metadata: dict[str, str] = {}
    merge_head: str | None = None
    for name in (
        "MERGE_HEAD",
        "MERGE_MODE",
        "MERGE_MSG",
        "AUTO_MERGE",
        "MERGE_AUTOSTASH",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "REBASE_HEAD",
        "BISECT_START",
        "rebase-apply",
        "rebase-merge",
        "sequencer",
    ):
        path = Path(
            _git_text(root, "rev-parse", "--path-format=absolute", "--git-path", name)
        )
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if name not in {"MERGE_HEAD", "MERGE_MODE", "MERGE_MSG", "AUTO_MERGE"}:
            raise AdmissionError(f"Another Git operation is active: {name}")
        if not stat.S_ISREG(info.st_mode):
            raise AdmissionError(f"Git merge metadata is not a regular file: {name}")
        content = path.read_bytes()
        metadata[name] = hashlib.sha256(content).hexdigest()
        if name == "MERGE_HEAD":
            merge_head = content.decode("ascii")
    unmerged = _git(root, "ls-files", "--unmerged", "-z")
    return {
        "branch": _git_text(root, "symbolic-ref", "--quiet", "HEAD"),
        "head": _git_text(root, "rev-parse", "HEAD"),
        "primary_oid": _reference(root, state.primary_ref),
        "task_oid": _reference(root, state.task_ref),
        "merge_head": merge_head,
        "metadata": metadata,
        "index": hashlib.sha256(_git(root, "ls-files", "--stage", "-z")).hexdigest(),
        "tree": None if unmerged else _git_text(root, "write-tree"),
        "worktree": hashlib.sha256(
            _git(root, "diff", "--binary", "--no-ext-diff", "--no-textconv")
        ).hexdigest(),
        "status": _git(
            root, "status", "--porcelain=v1", "--untracked-files=all", "-z"
        ).hex(),
    }


def _clean_primary(
    state: Workflow, snapshot: Snapshot, *, allow_removed_task: bool = False
) -> bool:
    return (
        snapshot["branch"] == state.primary_ref
        and snapshot["head"] == state.primary_oid
        and snapshot["primary_oid"] == state.primary_oid
        and snapshot["task_oid"]
        in ({state.task_oid, None} if allow_removed_task else {state.task_oid})
        and not snapshot["metadata"]
        and not snapshot["status"]
    )


def _clean_task(state: Workflow, snapshot: Snapshot) -> bool:
    return (
        state.repository == state.worktree
        and state.entry_ref == state.task_ref
        and snapshot["branch"] == state.task_ref
        and snapshot["head"] == state.task_oid
        and snapshot["primary_oid"] == state.primary_oid
        and snapshot["task_oid"] == state.task_oid
        and not snapshot["metadata"]
        and not snapshot["status"]
    )


def finish(state_path: Path) -> str:
    state = _load(state_path)
    snapshot = _snapshot(state)
    if state.phase not in {"candidate", "committed"} or not _completed_commit(
        state, snapshot
    ):
        raise AdmissionError(
            "The owned candidate has not become its exact merge commit"
        )
    _record_completed(state, snapshot, state_path)
    return snapshot["head"]


def _completed_commit(state: Workflow, snapshot: Snapshot) -> bool:
    if (
        state.candidate is None
        or snapshot["branch"] != state.primary_ref
        or snapshot["status"]
        or snapshot["head"] == state.primary_oid
        or snapshot["primary_oid"] != snapshot["head"]
        or snapshot["task_oid"]
        not in (
            {state.task_oid, None} if state.phase == "committed" else {state.task_oid}
        )
        or snapshot["metadata"]
    ):
        return False
    if snapshot["tree"] != state.candidate["tree"] or (
        state.phase == "committed" and snapshot["head"] != state.candidate["head"]
    ):
        return False
    root = Path(state.worktree)
    parents = _git_text(root, "show", "-s", "--format=%P", "HEAD")
    return parents.split() == [state.primary_oid, state.task_oid] and snapshot[
        "tree"
    ] == _git_text(root, "rev-parse", "HEAD^{tree}")


def _record_completed(state: Workflow, snapshot: Snapshot, state_path: Path) -> None:
    replace(state, phase="committed", candidate=snapshot).save(state_path)


def _walk_error(error: OSError) -> NoReturn:
    raise error


def _tracked_path(root: Path, relative: Path, tracked: set[bytes]) -> bool:
    # Git inventories always use slash separators. as_posix() preserves literal
    # backslashes and surrogateescaped filename bytes on POSIX.
    if os.fsencode(relative.as_posix()) in tracked:
        return True
    # A case-only rename can resolve to an existing tracked spelling on native
    # case-insensitive filesystems. Prefer exact directory entries so a distinct
    # untracked hardlink is never mistaken for its tracked inode. Walk without
    # following symlink ancestors; ambiguous aliases remain protected.
    actual = root
    for index, component in enumerate(relative.parts):
        with os.scandir(actual) as scan:
            entries = list(scan)
        exact = next((entry for entry in entries if entry.name == component), None)
        if exact is None:
            identity = (actual / component).lstat()
            matches = []
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
                    matches.append(entry)
            if len(matches) != 1:
                return False
            exact = matches[0]
        actual = actual / exact.name
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(actual.lstat().st_mode):
            return False
    return os.fsencode(actual.relative_to(root).as_posix()) in tracked


def _protect_checkout_paths(state: Workflow, *, target_tree: str) -> None:
    root = Path(state.worktree)
    # Include all conflict stages: recovery can lack a complete index tree.
    # Check explicit incoming destinations, including D/F changes where the index names
    # the destination but its physical directory contains ignored user files.
    tracked: set[bytes] = set()
    gitlinks: set[bytes] = set()
    current_links: dict[bytes, tuple[bytes, bytes]] = {}
    for record in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        entry_mode, oid, stage = metadata.split(b" ")
        tracked.add(encoded_path)
        if entry_mode == b"160000":
            gitlinks.add(encoded_path)
            if stage == b"0":
                current_links[encoded_path] = (entry_mode, oid)
    destinations: dict[bytes, tuple[bytes, bytes]] = {}
    for record in _git(root, "ls-tree", "-r", "-z", target_tree).split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        entry_mode, _kind, oid = metadata.split(b" ")
        destinations[encoded_path] = (entry_mode, oid)
    # A removed gitlink is absent from the target inventory but can still own
    # initialized source or private files. It needs the same protection as a
    # gitlink replaced by a regular file/directory or another commit.
    for encoded in sorted(destinations.keys() | gitlinks):
        path = os.fsdecode(encoded)
        tracked_ancestor = False
        for ancestor in reversed(Path(path).parents):
            if ancestor == Path("."):
                continue
            try:
                mode = (root / ancestor).lstat().st_mode
            except FileNotFoundError:
                break
            if not stat.S_ISDIR(mode):
                if not _tracked_path(root, ancestor, tracked):
                    raise AdmissionError(
                        "Untracked or ignored checkout obstruction; admission retained"
                    )
                # This exact tracked file/symlink becomes a directory. Never
                # follow its current target while inspecting future child paths.
                tracked_ancestor = True
                break
        if tracked_ancestor:
            continue
        target = root / path
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(mode):
            if not _tracked_path(root, Path(path), tracked):
                raise AdmissionError(
                    "Untracked or ignored checkout obstruction; admission retained"
                )
            continue
        if encoded in current_links and current_links[encoded] == destinations.get(
            encoded
        ):
            # The stage-0 gitlink's mode and commit are unchanged. This checkout
            # does not replace its directory; its contents belong to the nested
            # repository, not the superproject's tracked-file inventory.
            continue
        # Native path lookup also catches case-folded spellings which literal
        # Git pathspecs can miss. Inspect only affected destination directories.
        for directory, directories, files in os.walk(
            target, followlinks=False, onerror=_walk_error
        ):
            if not directories and not files:
                raise AdmissionError(
                    "Untracked or ignored empty-directory obstruction; "
                    "admission retained"
                )
            for name in [*directories, *files]:
                child = Path(directory) / name
                if not stat.S_ISDIR(child.lstat().st_mode) and not _tracked_path(
                    root, child.relative_to(root), tracked
                ):
                    raise AdmissionError(
                        "Untracked or ignored checkout obstruction; admission retained"
                    )


def _preview_merge_tree(state: Workflow) -> str:
    root = Path(state.worktree)
    options = _git(
        root,
        "config",
        "--null",
        "--get-all",
        f"branch.{state.primary_ref.removeprefix('refs/heads/')}.mergeOptions",
        missing_ref=True,
    )
    if options and not options.endswith(b"\0"):
        raise AdmissionError("Branch mergeOptions inventory is not NUL terminated")
    configured_options = options[:-1].split(b"\0") if options else []
    # The installer sets this sole commit-shape option. It cannot change ort's
    # predicted tree, and the actual merge already pins --no-ff explicitly.
    if configured_options not in ([], [b""], [b"--no-ff"]):
        raise AdmissionError(
            "Canonical merge requires ort without custom branch mergeOptions; "
            "configured options were not changed"
        )
    result = _capture_git(
        root,
        "merge-tree",
        "--write-tree",
        "--no-messages",
        "-z",
        state.primary_oid,
        state.task_oid,
    )
    if result.returncode == 1:
        raise AdmissionError("Merge preview reports conflicts; no merge was started")
    if result.returncode:
        raise AdmissionError(
            "Could not preview merge (Git 2.38 or newer is required): "
            + result.stderr.decode(errors="replace").strip()
        )
    tree, separator, trailing = result.stdout.partition(b"\0")
    if (
        not separator
        or trailing
        or len(tree) not in {40, 64}
        or any(value not in b"0123456789abcdef" for value in tree)
    ):
        raise AdmissionError("Merge preview did not return one complete tree object")
    oid = tree.decode("ascii")
    if _git_text(root, "cat-file", "-t", oid) != "tree":
        raise AdmissionError("Merge preview result is not a tree object")
    return oid


def switch_primary(state_path: Path) -> None:
    state = _load(state_path)
    if state.phase not in {"prepared", "cleanup"}:
        raise AdmissionError("Workflow is not ready to switch to primary")
    snapshot = _snapshot(state)
    if not _clean_task(state, snapshot):
        raise AdmissionError("Task changed before primary checkout")
    # Git's ignored-file guard does not preserve empty directories. Inspect the
    # destination while the task index still distinguishes private material,
    # then run the actual switch inside this same canonical process owner.
    _protect_checkout_paths(state, target_tree=state.primary_oid)
    if _snapshot(state) != snapshot:
        raise AdmissionError(
            "Checkout changed during path protection; admission retained"
        )
    # Inherit the owner's streams. Captured pipes could remain open in hook
    # descendants after Git exits, preventing this leader from returning so
    # the canonical owner can detect and clean those surviving processes.
    result = subprocess.run(
        (
            "git",
            "-C",
            state.worktree,
            "switch",
            "--no-overwrite-ignore",
            state.primary_ref.removeprefix("refs/heads/"),
        ),
        check=False,
    )
    if result.returncode:
        raise AdmissionError("Primary checkout command failed")
    if not _clean_primary(state, _snapshot(state)):
        raise AdmissionError(
            "Primary checkout left unexpected state; admission retained"
        )


def begin(state_path: Path, *, cleanup: bool) -> None:
    state = _load(state_path)
    if state.phase != "prepared":
        raise AdmissionError("Workflow is not ready to begin its Git operation")
    if not cleanup:
        if not _clean_primary(state, _snapshot(state)):
            raise AdmissionError(
                "Primary changed before this workflow's merge invocation"
            )
        # The ort result may create destinations absent from both parent trees
        # (directory renames), and --no-overwrite-ignore does not protect those
        # writes. Preview without touching the index/worktree, then inspect all
        # output and task destinations under the same cooperating admission.
        predicted = _preview_merge_tree(state)
        _protect_checkout_paths(state, target_tree=predicted)
        _protect_checkout_paths(state, target_tree=state.task_oid)
        if not _clean_primary(state, _snapshot(state)):
            raise AdmissionError(
                "Primary changed during merge preview; admission retained"
            )
    replace(state, phase="cleanup" if cleanup else "merging").save(state_path)


def capture(state_path: Path) -> str | None:
    state = _load(state_path)
    if state.phase != "merging":
        raise AdmissionError("No owned merge invocation awaits candidate capture")
    # Once observation starts, any incomplete read is conservative retained state.
    replace(state, phase="retained").save(state_path)
    snapshot = _snapshot(state)
    if (
        snapshot["branch"] == state.primary_ref
        and snapshot["head"] == state.primary_oid
        and snapshot["primary_oid"] == state.primary_oid
        and snapshot["task_oid"] == state.task_oid
        and snapshot["merge_head"] == state.task_oid + "\n"
    ):
        replace(state, phase="candidate", candidate=snapshot).save(state_path)
        return snapshot["tree"]
    elif _clean_primary(state, snapshot):
        replace(state, phase="prepared").save(state_path)
    else:
        raise AdmissionError(
            "Merge outcome has unknown ownership; candidate and admission retained"
        )
    return None


def recover_abort(state_path: Path) -> None:
    state = _load(state_path)
    if state.phase == "merging":
        capture(state_path)
        state = _load(state_path)
    if state.phase == "candidate":
        snapshot = _snapshot(state)
        if _completed_commit(state, snapshot):
            _record_completed(state, snapshot, state_path)
            return
        if snapshot != state.candidate:
            raise AdmissionError(
                "Candidate changed outside this workflow; candidate and "
                "admission retained"
            )
        _protect_checkout_paths(state, target_tree=state.primary_oid)
        _git(Path(state.worktree), "merge", "--abort")
        if not _clean_primary(state, _snapshot(state)):
            raise AdmissionError(
                "Merge abort left unexpected Git state; admission retained"
            )
        replace(state, phase="aborted").save(state_path)
    elif state.phase == "admitted":
        return
    elif state.phase in {"prepared", "cleanup", "committed", "aborted", "restored"}:
        # These phases never confer authority to abort an unrelated MERGE_HEAD.
        snapshot = _snapshot(state)
        if snapshot["metadata"]:
            raise AdmissionError(
                "Unowned Git operation detected; candidate and admission retained"
            )
        if state.phase in {"prepared", "aborted", "restored"} and not (
            _clean_primary(state, snapshot) or _clean_task(state, snapshot)
        ):
            raise AdmissionError(
                "Checkout changed outside this workflow; admission retained"
            )
        if state.phase == "cleanup" and not (
            _clean_primary(state, snapshot, allow_removed_task=True)
            or _clean_task(state, snapshot)
        ):
            raise AdmissionError("Cleanup checkout changed; admission retained")
        if state.phase == "committed":
            if not _completed_commit(state, snapshot):
                raise AdmissionError(
                    "Completed merge state changed; admission retained"
                )
            _record_completed(state, snapshot, state_path)
    else:
        raise AdmissionError(
            "Recovery ownership is unproven; candidate and admission retained"
        )


def recover_restore(state_path: Path) -> None:
    state = _load(state_path)
    if state.phase in {"admitted", "committed", "cleanup", "restored"}:
        return
    if state.phase not in {"prepared", "aborted"}:
        raise AdmissionError("Abort did not establish safe checkout authority")
    snapshot = _snapshot(state)
    if _clean_task(state, snapshot):
        replace(state, phase="restored").save(state_path)
        return
    if not _clean_primary(state, snapshot):
        raise AdmissionError(
            "Primary changed before recovery checkout; admission retained"
        )
    if state.repository == state.worktree and state.entry_ref == state.task_ref:
        if (
            _git_text(Path(state.worktree), "rev-parse", state.task_ref)
            != state.task_oid
        ):
            raise AdmissionError(
                "Task ref changed before recovery checkout; admission retained"
            )
        _protect_checkout_paths(state, target_tree=state.task_oid)
        _git(
            Path(state.worktree),
            "switch",
            "--no-overwrite-ignore",
            state.task_ref.removeprefix("refs/heads/"),
        )
        if not _clean_task(state, _snapshot(state)):
            raise AdmissionError(
                "Recovery checkout left unexpected state; admission retained"
            )
    replace(state, phase="restored").save(state_path)


def remove_task_worktree(state_path: Path, task_worktree: Path) -> None:
    state = _load(state_path)
    if state.phase not in {"cleanup", "committed"}:
        raise AdmissionError("Workflow has no completed task cleanup authority")
    primary = Path(state.worktree)
    target = task_worktree.resolve(strict=True)
    if target == primary.resolve() or state.task_ref == state.primary_ref:
        raise AdmissionError("Primary worktree cannot be removed")
    common = Path(state.lock_directory).parent
    for root in (primary, target):
        actual = Path(
            _git_text(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        if not actual.samefile(common):
            raise AdmissionError("Task worktree belongs to another Git repository")
    snapshot = _snapshot(state)
    valid_primary = (
        _completed_commit(state, snapshot)
        if state.phase == "committed"
        else _clean_primary(state, snapshot)
    )
    if not valid_primary or snapshot["task_oid"] != state.task_oid:
        raise AdmissionError("Primary or task changed before worktree removal")
    try:
        _git(primary, "merge-base", "--is-ancestor", state.task_oid, snapshot["head"])
    except AdmissionError as error:
        raise AdmissionError(
            "Task is not integrated into the primary history"
        ) from error
    registrations: list[dict[bytes, bytes]] = []
    for record in _git(primary, "worktree", "list", "--porcelain", "-z").split(b"\0\0"):
        if not record:
            continue
        fields: dict[bytes, bytes] = {}
        for field in record.split(b"\0"):
            key, _, value = field.partition(b" ")
            if not key or key in fields:
                raise AdmissionError("Invalid Git worktree registration")
            fields[key] = value
        if fields.get(b"branch") == os.fsencode(state.task_ref):
            registrations.append(fields)
    if (
        len(registrations) != 1
        or registrations[0].get(b"HEAD") != state.task_oid.encode("ascii")
        or b"worktree" not in registrations[0]
        or Path(os.fsdecode(registrations[0][b"worktree"])).resolve() != target
    ):
        raise AdmissionError("Task worktree registration no longer matches")
    task_snapshot = _snapshot(replace(state, worktree=str(target)))
    if (
        task_snapshot["branch"] != state.task_ref
        or task_snapshot["head"] != state.task_oid
        or task_snapshot["primary_oid"] != snapshot["primary_oid"]
        or task_snapshot["task_oid"] != state.task_oid
        or task_snapshot["metadata"]
        or task_snapshot["status"]
    ):
        raise AdmissionError("Task worktree changed before removal")
    # Deliberately omit --exclude-standard: ignored environments, private files,
    # symlinks and even empty directories remain user material. Git's own remove
    # checks still apply; cooperating admission is not a filesystem writer lock.
    if _git(target, "ls-files", "--others", "--directory", "-z"):
        raise AdmissionError("Task worktree contains untracked or ignored material")
    _git(primary, "worktree", "remove", str(target))


def validate_release(state_path: Path) -> Workflow:
    state = _load(state_path)
    if state.phase not in {"admitted", "restored", "cleanup", "committed"}:
        raise AdmissionError("Workflow outcome is unproven; admission retained")
    return state


def release(state_path: Path) -> None:
    # Only the site-isolated owner calls this after its final command, diagnostics
    # and cleanup receipt are complete. Revalidate before the first unlink; the
    # terminal filesystem operations must never start another child or writer.
    state = validate_release(state_path)
    directory = Path(state.lock_directory)
    (directory / "owner.json").unlink()
    directory.rmdir()
    state_path.unlink()


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "acquire",
            "check-clean",
            "configure",
            "switch-primary",
            "begin",
            "cleanup",
            "capture",
            "finish",
            "recover-abort",
            "recover-restore",
            "remove-task-worktree",
            "validate-release",
        ),
    )
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--task-worktree", type=Path)
    for name in ("primary-ref", "primary-oid", "task-ref", "task-oid"):
        parser.add_argument("--" + name)
    args = parser.parse_args(arguments)
    try:
        if args.operation == "acquire":
            if args.repository is None or args.owner_pid is None:
                raise AdmissionError("Acquisition requires repository and workflow PID")
            acquire(args.repository, args.state, args.owner_pid)
        elif args.operation == "check-clean":
            if args.worktree is None:
                raise AdmissionError("Cleanliness check requires its worktree")
            check_clean(args.state, args.worktree)
        elif args.operation == "configure":
            if any(
                value is None
                for value in (
                    args.worktree,
                    args.primary_ref,
                    args.primary_oid,
                    args.task_ref,
                    args.task_oid,
                )
            ):
                raise AdmissionError(
                    "Configuration requires exact workflow refs and worktree"
                )
            configure(
                args.state,
                worktree=args.worktree,
                primary_ref=args.primary_ref,
                primary_oid=args.primary_oid,
                task_ref=args.task_ref,
                task_oid=args.task_oid,
            )
        elif args.operation == "remove-task-worktree":
            if args.task_worktree is None:
                raise AdmissionError(
                    "Task worktree removal requires its registered path"
                )
            remove_task_worktree(args.state, args.task_worktree)
        elif args.operation == "capture":
            tree = capture(args.state)
            if tree is not None:
                # Export the exact owned observation without another Git read
                # in the shell. Recovery capture does not publish this output.
                args.state.with_suffix(".tree").write_text(
                    tree + "\n", encoding="ascii"
                )
        elif args.operation == "finish":
            # Return the already-verified commit to the shell. Re-reading HEAD
            # there could adopt a later external commit as cleanup authority.
            commit = finish(args.state)
            args.state.with_suffix(".commit").write_text(
                commit + "\n", encoding="ascii"
            )
        elif args.operation in {"begin", "cleanup"}:
            begin(args.state, cleanup=args.operation == "cleanup")
        else:
            {
                "switch-primary": switch_primary,
                "recover-abort": recover_abort,
                "recover-restore": recover_restore,
                "validate-release": validate_release,
            }[args.operation](args.state)
    except (AdmissionError, OSError, ValueError, TypeError) as error:
        # The shell runs this entry inside its canonical owner. Recovery
        # captures both streams under its original deadline; never launch
        # another diagnostic owner or grant it a new deadline here.
        import sys

        print(f"git-flow-merge admission: {error}", file=sys.stderr)
        return 1
    return 0
