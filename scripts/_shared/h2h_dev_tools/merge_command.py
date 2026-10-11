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
    receipt: Path = arguments.cleanup_receipt
    # Windows still owns classification through a Job Object. Documentation can
    # finish without online review; full review is rejected by the reviewer until
    # that platform's online-review lifecycle has been implemented and verified.
    managed_review = arguments.review and os.name == "posix"
    with ExitStack() as locks:
        if managed_review:
            from . import code_review

            reviewer = code_review.Reviewer(arguments.repository)
            try:
                target = reviewer._target(index=True)
                reviewer._assert_workspace(target)
                locks.enter_context(reviewer._review_lock(target, exclusive=True))
                reviewer._receipt_path(target).unlink(missing_ok=True)
                # Both pending and final evidence must share Git metadata's
                # filesystem, even when TMPDIR is on a different volume.
                pending_directory = locks.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix="pending-", dir=reviewer._directory()
                    )
                )
                pending = Path(pending_directory) / "receipt.json"
            except (code_review.ReviewError, OSError, ValueError) as error:
                # No command was started: this stage has no owned descendants.
                print(f"merge-command: {error}", file=sys.stderr)
                receipt.write_text("owned_tree_empty\n", encoding="ascii")
                return 1
            environment.update(
                H2H_REVIEW_PROCESS_OWNER=str(os.getpid()),
                H2H_REVIEW_OWNER_SOURCE=owner_source_digest(),
                H2H_REVIEW_LOCK_TARGET=target.key,
                H2H_REVIEW_PENDING_RECEIPT=str(pending),
            )
        phase = process.ProcessCommand(
            "Code review" if arguments.review else "Git merge and hooks",
            tuple(command),
            arguments.repository,
            environment=environment,
        )
        try:
            status = process.run_command(phase, hard_deadline=None)
        except KeyboardInterrupt:
            status = process.INTERRUPTED_EXIT_CODE
        except process.RunnerSignalInterrupt as error:
            status = 128 + error.signal_number
        except process.ProcessOwnershipError as error:
            if not error.cleanup_proven:
                print(
                    f"merge-command: cleanup could not be proven: {error}",
                    file=sys.stderr,
                )
                return process.TERMINATION_FAILED_EXIT_CODE
            print(
                f"merge-command: surviving descendants required cleanup: {error}",
                file=sys.stderr,
            )
            status = 1
        if managed_review and status == 0 and pending.exists():
            try:
                published = reviewer._publish_managed_receipt(target, pending)
                print(f"Code review passed after owned cleanup; wrote {published}")
            except (code_review.ReviewError, OSError, ValueError) as error:
                print(f"merge-command: {error}", file=sys.stderr)
                status = 1
    # An exit status alone cannot distinguish a cleaned interrupt from an owner
    # killed before/during teardown. Publish evidence only after the runner has
    # returned or completed its controlled interruption and final cleanup.
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
