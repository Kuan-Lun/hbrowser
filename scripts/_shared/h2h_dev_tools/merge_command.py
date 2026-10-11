"""Own a Git merge command and its hook process tree until cleanup completes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    from . import process

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-receipt", required=True, type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a Git merge command is required")
    phase = process.ProcessCommand(
        "Git merge and hooks", tuple(command), arguments.repository
    )
    try:
        status = process.run_command(phase, hard_deadline=None)
    except KeyboardInterrupt:
        status = process.INTERRUPTED_EXIT_CODE
    except process.RunnerSignalInterrupt as error:
        status = 128 + error.signal_number
    except process.ProcessOwnershipError as error:
        print(f"merge-command: cleanup could not be proven: {error}", file=sys.stderr)
        return process.TERMINATION_FAILED_EXIT_CODE
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
    # The shell copies the source closure before switching to primary. Give its
    # directly executed entry the same relative imports as packaged/snapshot use.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "h2h_dev_tools"
    raise SystemExit(main())
