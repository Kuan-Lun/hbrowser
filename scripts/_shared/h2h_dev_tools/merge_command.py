"""Own a merge/review command and its process tree until cleanup completes."""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from .process import ProcessCommand

REVIEW_OWNER_ENVIRONMENT = (
    "H2H_REVIEW_PROCESS_OWNER",
    "H2H_REVIEW_OWNER_SOURCE",
    "H2H_REVIEW_LOCK_TARGET",
    "H2H_REVIEW_PENDING_RECEIPT",
)
RECOVERY_TIMEOUT_SECONDS = 10.0
RECOVERY_DIAGNOSTIC_LIMIT_BYTES = 64 * 1024


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


def _run_owned(phase: ProcessCommand, *, hard_deadline: float | None = None) -> int:
    from . import process

    try:
        return process.run_command(phase, hard_deadline=hard_deadline)
    except KeyboardInterrupt:
        return process.INTERRUPTED_EXIT_CODE
    except process.RunnerSignalInterrupt as error:
        return 128 + error.signal_number
    except process.ProcessOwnershipError as error:
        if not error.cleanup_proven:
            raise
        process.report_diagnostic(
            f"merge-command: surviving descendants required cleanup: {error}"
        )
        return 1


def _replay_recovery_output(output: BinaryIO) -> None:
    from . import process

    try:
        output.seek(0)
        remaining = RECOVERY_DIAGNOSTIC_LIMIT_BYTES
        while remaining:
            chunk = output.read(min(8192, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            process.report_diagnostic(
                chunk.decode("utf-8", errors="replace").rstrip("\n")
            )
        if remaining == 0 and output.read(1):
            process.report_diagnostic(
                "merge-command: recovery diagnostics truncated after 64 KiB"
            )
    except OSError as error:
        process.report_diagnostic(
            f"merge-command: could not read recovery diagnostics: {error}"
        )


def _run_git(phase: ProcessCommand, *, recovery: bool) -> int:
    from . import process

    if not recovery:
        return _run_owned(phase)
    # Git resets SIGPIPE itself. A closed caller pipe must not stop a recovery
    # checkout after updating HEAD but before running its post-checkout hook.
    # Only this human diagnostic channel is captured; review/data output keeps
    # its normal semantics. Replay is bounded and follows proven tree cleanup.
    output = tempfile.TemporaryFile()
    try:
        status = _run_owned(
            replace(phase, stdout=output, stderr=output),
            hard_deadline=time.monotonic() + RECOVERY_TIMEOUT_SECONDS,
        )
        _replay_recovery_output(output)
        return status
    finally:
        try:
            output.close()
        except OSError as error:
            process.report_diagnostic(
                f"merge-command: could not close recovery diagnostics: {error}"
            )


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
        process.report_diagnostic(
            "Documentation-only candidate: online code review is not required.",
            standard_output=True,
        )
        return 0
    if scope != "full":
        process.report_diagnostic(f"merge-command: invalid check scope: {scope}")
        return 1
    if os.name != "posix":
        process.report_diagnostic(
            "merge-command: online code review currently requires POSIX"
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
            process.report_diagnostic(f"merge-command: {error}")
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
                process.report_diagnostic(
                    f"Code review passed after owned cleanup; wrote {published}",
                    standard_output=True,
                )
            except (code_review.ReviewError, OSError, ValueError) as error:
                process.report_diagnostic(f"merge-command: {error}")
                status = 1
        return status


def _execute(
    repository: Path,
    command: tuple[str, ...],
    *,
    review: bool,
    recovery: bool,
    receipt: Path,
) -> int:
    from . import process

    environment = without_review_owner(os.environ)
    try:
        if review:
            status = _review(repository, command, environment)
        else:
            status = _run_git(
                process.ProcessCommand(
                    "Git workflow and hooks",
                    command,
                    repository,
                    environment=environment,
                ),
                recovery=recovery,
            )
    except process.ProcessOwnershipError as error:
        process.report_diagnostic(
            f"merge-command: cleanup could not be proven: {error}"
        )
        return process.TERMINATION_FAILED_EXIT_CODE
    # An exit status alone cannot distinguish a cleaned interrupt from an owner
    # killed before/during teardown. Every started phase must complete cleanup
    # before publishing proof, including classification-only documentation merges.
    temporary_receipt = receipt.with_suffix(".pending")
    temporary_receipt.write_text("owned_tree_empty\n", encoding="ascii")
    temporary_receipt.replace(receipt)
    # Reserve 125 for an unproven owner cleanup, never for a hook's own exit code.
    return 1 if status == process.TERMINATION_FAILED_EXIT_CODE else status


def main() -> int:
    from . import process

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-receipt", required=True, type=Path)
    parser.add_argument("--review", action="store_true")
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="bound this failure-recovery command and its cleanup to 10 seconds",
    )
    parser.add_argument(
        "--recovery-output-fds", nargs=2, type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a Git workflow command is required")
    if arguments.review and arguments.recovery:
        parser.error("review cannot run as failure recovery")
    if arguments.recovery_output_fds and not arguments.recovery:
        parser.error("recovery output descriptors require --recovery")
    with ExitStack() as diagnostics:
        if arguments.recovery_output_fds:
            # Startup errors go to /dev/null. Only the common diagnostic writer
            # receives the original destinations, through ordinary text streams.
            for descriptor, redirect in zip(
                arguments.recovery_output_fds,
                (redirect_stdout, redirect_stderr),
                strict=True,
            ):
                stream = diagnostics.enter_context(
                    os.fdopen(descriptor, "w", encoding="utf-8", closefd=False)
                )
                diagnostics.enter_context(redirect(stream))
        try:
            return _execute(
                arguments.repository,
                tuple(command),
                review=arguments.review,
                recovery=arguments.recovery,
                receipt=arguments.cleanup_receipt,
            )
        except OSError as error:
            if not arguments.recovery:
                raise
            # A failed receipt write is still a failure and publishes no proof;
            # reporting it must not block recovery on the caller's pipe.
            process.report_diagnostic(f"merge-command: {error}")
            return 1


if __name__ == "__main__":
    # The shell copies the source closure before switching to primary. Give its
    # directly executed entry the same relative imports as packaged/snapshot use.
    package = Path(__file__).resolve().parent
    runpy.run_path(str(package / "source_loader.py"))["load_package"](
        package, "h2h_dev_tools"
    )
    __package__ = "h2h_dev_tools"
    if sys.argv[1:] == ["--print-source-files"]:
        print("\n".join(owner_source_files()))
        raise SystemExit(0)
    if len(sys.argv) == 3 and sys.argv[1] == "--workflow-diagnostic":
        # The shell gives only this diagnostic FD to us; startup errors use
        # /dev/null and must never fall back to its potentially blocked stderr.
        from .process import report_diagnostic

        with (
            os.fdopen(3, "w", encoding="utf-8", closefd=False) as diagnostic,
            redirect_stderr(diagnostic),
        ):
            report_diagnostic(f"git-flow-merge: {sys.argv[2]}")
        raise SystemExit(0)
    raise SystemExit(main())
