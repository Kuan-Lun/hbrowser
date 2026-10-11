"""Own a merge/review command and its process tree until cleanup completes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .process import ProcessCommand

REVIEW_OWNER_ENVIRONMENT = (
    "H2H_REVIEW_PROCESS_OWNER",
    "H2H_REVIEW_OWNER_SOURCE",
    "H2H_REVIEW_LOCK_TARGET",
    "H2H_REVIEW_PENDING_RECEIPT",
)


def without_review_owner(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in environment.items()
        if name not in REVIEW_OWNER_ENVIRONMENT
    }


def owner_source_files() -> tuple[str, ...]:
    from . import bundle_verify

    # The review owner now validates and publishes review evidence itself. Use
    # the authoritative review closure rather than a second dependency list.
    return tuple(sorted(bundle_verify.component_files(["code-review"])))


def owner_source_digest() -> str:
    """Bind the complete copied owner implementation used by managed review."""
    digest = hashlib.sha256()
    for name in owner_source_files():
        source = Path(__file__).with_name(name).read_bytes()
        digest.update(name.encode("ascii") + b"\0" + hashlib.sha256(source).digest())
    return digest.hexdigest()


def _run_owned(phase: ProcessCommand) -> int:
    from . import process

    try:
        return process.run_command(phase, hard_deadline=None)
    except KeyboardInterrupt:
        return process.INTERRUPTED_EXIT_CODE
    except process.RunnerSignalInterrupt as error:
        return 128 + error.signal_number
    except process.ProcessOwnershipError as error:
        if not error.cleanup_proven:
            raise
        print(
            f"merge-command: surviving descendants required cleanup: {error}",
            file=sys.stderr,
        )
        return 1


def _review(
    repository: Path, command: tuple[str, ...], environment: dict[str, str]
) -> int:
    from . import process

    # Classify under ownership before requiring any online-review prerequisites.
    # Only an empty, successful classifier tree admits the next phase.
    with tempfile.TemporaryFile() as output:
        status = _run_owned(
            process.ProcessCommand(
                "Classify review candidate",
                (
                    sys.executable,
                    "scripts/check_change_scope.py",
                    "--index",
                    "--base",
                    "HEAD",
                ),
                repository,
                environment=environment,
                stdout=output,
            )
        )
        if status != 0:
            return status
        output.seek(0)
        scope = output.read().decode("utf-8").strip()
    if scope == "documentation":
        print("Documentation-only candidate: online code review is not required.")
        return 0
    if scope != "full":
        print(f"merge-command: invalid check scope: {scope}", file=sys.stderr)
        return 1
    if os.name != "posix":
        print(
            "merge-command: online code review currently requires POSIX",
            file=sys.stderr,
        )
        return 1

    from . import code_review

    reviewer = code_review.Reviewer(repository)
    with ExitStack() as locks:
        try:
            target = reviewer._target(index=True)
            reviewer._assert_workspace(target)
            locks.enter_context(reviewer._review_lock(target, exclusive=True))
            reviewer._receipt_path(target).unlink(missing_ok=True)
            # Keep both atomic renames on Git metadata's filesystem, including
            # repositories on a different volume from TMPDIR.
            pending_directory = locks.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="pending-", dir=reviewer._directory()
                )
            )
            pending = Path(pending_directory) / "receipt.json"
        except (code_review.ReviewError, OSError, ValueError) as error:
            print(f"merge-command: {error}", file=sys.stderr)
            return 1
        environment.update(
            H2H_REVIEW_PROCESS_OWNER=str(os.getpid()),
            H2H_REVIEW_OWNER_SOURCE=owner_source_digest(),
            H2H_REVIEW_LOCK_TARGET=target.key,
            H2H_REVIEW_PENDING_RECEIPT=str(pending),
        )
        status = _run_owned(
            process.ProcessCommand(
                "Code review", command, repository, environment=environment
            )
        )
        if status == 0:
            try:
                published = reviewer._publish_managed_receipt(target, pending)
                print(f"Code review passed after owned cleanup; wrote {published}")
            except (code_review.ReviewError, OSError, ValueError) as error:
                print(f"merge-command: {error}", file=sys.stderr)
                status = 1
        return status


def main() -> int:
    from . import process

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-receipt", required=True, type=Path)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("repository", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a Git merge command is required")
    environment = without_review_owner(os.environ)
    try:
        if arguments.review:
            status = _review(arguments.repository, tuple(command), environment)
        else:
            status = _run_owned(
                process.ProcessCommand(
                    "Git merge and hooks",
                    tuple(command),
                    arguments.repository,
                    environment=environment,
                )
            )
    except process.ProcessOwnershipError as error:
        print(f"merge-command: cleanup could not be proven: {error}", file=sys.stderr)
        return process.TERMINATION_FAILED_EXIT_CODE
    # An exit status alone cannot distinguish a cleaned interrupt from an owner
    # killed before/during teardown. Every started phase must complete cleanup
    # before publishing proof, including classification-only documentation merges.
    receipt: Path = arguments.cleanup_receipt
    temporary_receipt = receipt.with_suffix(".pending")
    temporary_receipt.write_text("owned_tree_empty\n", encoding="ascii")
    temporary_receipt.replace(receipt)
    # Reserve 125 for an unproven owner cleanup, never for a hook's own exit code.
    return 1 if status == process.TERMINATION_FAILED_EXIT_CODE else status


if __name__ == "__main__":
    # The shell copies the source closure before switching to primary. Give its
    # directly executed entry the same relative imports as packaged/snapshot use.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "h2h_dev_tools"
    if sys.argv[1:] == ["--print-source-files"]:
        print("\n".join(owner_source_files()))
        raise SystemExit(0)
    raise SystemExit(main())
