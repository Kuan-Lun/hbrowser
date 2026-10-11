"""Own a merge/review command and its process tree until cleanup completes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

OWNER_SOURCE_FILES = (
    "__init__.py",
    "git-flow-merge.sh",
    "merge_command.py",
    "process.py",
)


def owner_source_digest() -> str:
    """Bind the complete copied owner implementation used by managed review."""
    digest = hashlib.sha256()
    for name in OWNER_SOURCE_FILES:
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
    environment = dict(os.environ)
    environment.pop("H2H_REVIEW_PROCESS_OWNER", None)
    environment.pop("H2H_REVIEW_OWNER_SOURCE", None)
    if arguments.review:
        environment["H2H_REVIEW_PROCESS_OWNER"] = str(os.getpid())
        environment["H2H_REVIEW_OWNER_SOURCE"] = owner_source_digest()
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
                f"merge-command: cleanup could not be proven: {error}", file=sys.stderr
            )
            return process.TERMINATION_FAILED_EXIT_CODE
        print(
            f"merge-command: surviving descendants required cleanup: {error}",
            file=sys.stderr,
        )
        status = 1
    # An exit status alone cannot distinguish a cleaned interrupt from an owner
    # killed before/during teardown. Publish evidence only after the runner has
    # returned or completed its controlled interruption and final cleanup.
    receipt: Path = arguments.cleanup_receipt
    temporary_receipt = receipt.with_suffix(".pending")
    temporary_receipt.write_text("owned_tree_empty\n", encoding="ascii")
    temporary_receipt.replace(receipt)
    # Reserve 125 for an unproven owner cleanup, never for a hook's own exit code.
    return 1 if status == process.TERMINATION_FAILED_EXIT_CODE else status


if __name__ == "__main__":
    if sys.argv[1:] == ["--print-source-files"]:
        print("\n".join(OWNER_SOURCE_FILES))
        raise SystemExit(0)
    # The shell copies the source closure before switching to primary. Give its
    # directly executed entry the same relative imports as packaged/snapshot use.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "h2h_dev_tools"
    raise SystemExit(main())
