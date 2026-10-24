"""Bounded concurrent review invocations with registered nested process owners."""

from __future__ import annotations

import errno
import io
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from . import process

CONCURRENCY = 4
CLEANUP_RESERVE = 6.0
_ACTIVE_DEADLINE: ContextVar[float | None] = ContextVar("review_deadline", default=None)


class ExecutionError(RuntimeError):
    """An invocation failed; no aggregate evidence may be published."""


class GitCommandError(ExecutionError):
    """A query was rejected before admission or after proven process cleanup."""


def remaining() -> float | None:
    """Bound blocking Git and filesystem/result validation by the current phase."""
    deadline = _ACTIVE_DEADLINE.get()
    if deadline is None:
        return None
    value = deadline - time.monotonic()
    if value <= 0:
        raise ExecutionError("Aggregate review deadline expired")
    return value


def current_deadline() -> float | None:
    """Pass the existing absolute bound to deadline-aware generic helpers."""
    remaining()
    return _ACTIVE_DEADLINE.get()


def git_output(
    repository: Path,
    *arguments: str,
    deadline: float | None = None,
    input_bytes: bytes | None = None,
    max_output_bytes: int | None = None,
) -> bytes:
    """Read Git only after its canonical owner proves the whole query empty.

    Git configuration and wrappers may execute descendants even for read queries.
    Regular files separate command output from ownership diagnostics and cannot
    wait for a surviving descendant to close a pipe. The existing stdin bridge
    admits batch input without adding a second process-launch implementation.
    """
    active = current_deadline()
    if active is not None:
        deadline = active if deadline is None else min(active, deadline)
    if deadline is not None and (
        not math.isfinite(deadline) or deadline <= time.monotonic()
    ):
        raise ExecutionError("Git query deadline expired or is invalid")
    stack = ExitStack()
    failure: BaseException | None = None
    try:
        output = stack.enter_context(tempfile.TemporaryFile())
        errors = stack.enter_context(tempfile.TemporaryFile())
        command = ("git", *arguments)
        if input_bytes is not None:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            source = Path(directory) / "stdin"
            source.write_bytes(input_bytes)
            command = (
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).with_name("review_worker.py")),
                str(source),
                *command,
            )
        # The owner reports only to this in-memory sink. In particular, query
        # diagnostics cannot corrupt a classifier's stdout or start a writer
        # after uncertain cleanup. Exceptions retain their canonical evidence.
        diagnostics = stack.enter_context(io.StringIO())
        budget = None if deadline is None else deadline - time.monotonic()
        if budget is not None and budget <= 0:
            raise ExecutionError("Git query deadline expired before admission")
        # The caller may already be inside a phase's execution cutoff. Reserve
        # part of that SAME remaining slice for this nested query, rather than
        # requiring another full six seconds or extending the parent deadline.
        reserve = (
            CLEANUP_RESERVE if budget is None else min(CLEANUP_RESERVE, budget / 2)
        )
        if os.name == "posix":
            # Capture files are already open. Nested POSIX closure additionally
            # holds its own lock, the parent's lock, and a token/obligation read:
            # three descriptors at once, exceeding Popen's two-FD error pipe.
            # Reject pressure BEFORE registering an owner. This is a capacity
            # probe, not a reservation; later ownership failures stay unproven.
            try:
                with (
                    Path(os.devnull).open("rb"),
                    Path(os.devnull).open("rb"),
                    Path(os.devnull).open("rb"),
                ):
                    pass
            except OSError as error:
                if error.errno not in {errno.EMFILE, errno.ENFILE}:
                    raise
                raise GitCommandError(
                    "Git query lacks file descriptors for owned cleanup"
                ) from error
        try:
            with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
                status = process.run_command(
                    process.ProcessCommand(
                        "Review Git query",
                        command,
                        repository,
                        stdout=output,
                        stderr=errors,
                    ),
                    hard_deadline=deadline,
                    termination_reserve_seconds=reserve,
                )
        except process.ProcessOwnershipError as error:
            if not error.cleanup_proven:
                raise
            raise ExecutionError(
                "Git query left surviving descendants; cleanup completed"
            ) from error
        if status:
            if status == process.TIMEOUT_EXIT_CODE:
                raise ExecutionError("Git query deadline expired")
            errors.seek(0)
            detail = errors.read(64 * 1024).decode("utf-8", errors="replace")
            raise GitCommandError(
                f"Git query failed with exit {status}: {detail.strip()}"
            )
        remaining()
        if deadline is not None and time.monotonic() >= deadline:
            raise ExecutionError("Git query exceeded its deadline")
        if max_output_bytes is not None:
            if output.seek(0, io.SEEK_END) > max_output_bytes:
                raise ExecutionError("Git output exceeds review memory budget")
        output.seek(0)
        data = (
            output.read()
            if max_output_bytes is None
            else output.read(max_output_bytes + 1)
        )
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            stack.close()
        except BaseException as close_error:
            if failure is None:
                raise
            # A spool or temporary-directory failure must never downgrade an
            # unproven canonical owner into an ordinary recoverable I/O error.
            failure.add_note(f"Git query capture closure also failed: {close_error}")
    remaining()
    if deadline is not None and time.monotonic() >= deadline:
        raise ExecutionError("Git query capture closure exceeded its deadline")
    return data


@contextmanager
def within_deadline(deadline: float) -> Iterator[None]:
    previous = _ACTIVE_DEADLINE.get()
    token = _ACTIVE_DEADLINE.set(
        deadline if previous is None else min(previous, deadline)
    )
    try:
        remaining()
        yield
        remaining()
    finally:
        _ACTIVE_DEADLINE.reset(token)


def shard_deadline(started: float, deadline: float, count: int) -> float:
    """Reserve 20% (up to ten minutes) for integration within the SAME budget."""
    if count <= 1:
        return deadline
    reserve = min(600.0, (deadline - started) * 0.2)
    if reserve <= CLEANUP_RESERVE:
        raise ExecutionError(
            "Aggregate budget cannot admit shards and integration cleanup"
        )
    return deadline - reserve


@dataclass(frozen=True)
class Invocation:
    id: str
    command: tuple[str, ...]
    prompt: str
    directory: Path


@dataclass
class _Active:
    invocation: Invocation
    owned: process.OwnedProcess
    streams: ExitStack


def _cleanup_invocation(item: _Active, *, deadline: float) -> tuple[bool, list[str]]:
    """Clean one independent owner without writing shared diagnostic streams."""
    messages: list[str] = []
    empty = False
    try:
        item.owned.process.poll()
        empty = process.terminate_owned_tree(
            item.owned, deadline=deadline, diagnostic=messages.append
        )
    except Exception as error:
        messages.append(f"Review cleanup failed: {error}")
    try:
        if not process.close_owner(
            item.owned, deadline=deadline, diagnostic=messages.append
        ):
            empty = False
    except Exception as error:
        messages.append(f"Review owner closure failed: {error}")
        empty = False
    try:
        item.streams.close()
    except Exception as error:
        messages.append(f"Review stream closure failed: {error}")
        empty = False
    return empty, messages


def run_invocations(
    repository: Path,
    invocations: Sequence[Invocation],
    *,
    deadline: float,
    validate: Callable[[Invocation], None],
) -> list[dict[str, object]]:
    """Share one deadline; inspect results only after that invocation is empty.

    A child group is registered through start_owned_process before it exists.
    Each cancelled owner closes its admission before signalling its own group.
    Independent owners clean concurrently within the same absolute deadline.
    No invocation result, including a late passing file, rescues a failed run.
    Streams close with each completed owner, so concurrency also bounds FDs.
    """
    if os.name != "posix":
        raise ExecutionError("Online review ownership currently requires POSIX")
    active: dict[int, _Active] = {}
    completed: list[dict[str, object]] = []
    next_index = 0
    execution_deadline = deadline - CLEANUP_RESERVE
    collection_ownership_error: process.ProcessOwnershipError | None = None
    cleanup_ready = Event()
    cleanup_items: tuple[_Active, ...] = ()
    cleanup_deadline = deadline
    cleanup_workers: list[Future[tuple[bool, list[str]]]] = []

    def clean_slot(index: int) -> tuple[bool, list[str]]:
        cleanup_ready.wait()
        if index >= len(cleanup_items):
            return True, []
        return _cleanup_invocation(cleanup_items[index], deadline=cleanup_deadline)

    with (
        process.controlled_termination_signals() as controller,
        ThreadPoolExecutor(
            max_workers=CONCURRENCY, thread_name_prefix="h2h-review-cleanup"
        ) as cleanup,
    ):

        def collect_completed() -> None:
            """Validate every exited owner before admitting further work."""
            nonlocal collection_ownership_error
            for pid, item in list(active.items()):
                invocation, owned = item.invocation, item.owned
                code = owned.process.poll()
                if code is None:
                    continue
                remaining()
                with controller.defer():
                    try:
                        empty = process.wait_for_owned_tree_exit(
                            owned,
                            deadline=execution_deadline,
                            poll_deadline=min(
                                execution_deadline, time.monotonic() + 0.25
                            ),
                        )
                    except process.ProcessOwnershipError as error:
                        # This owner is already registered in active. Only the
                        # complete final cleanup can upgrade its failed probe.
                        collection_ownership_error = error
                        raise
                    if not empty:
                        raise ExecutionError(
                            f"Review leader left surviving descendants: {invocation.id}"
                        )
                    # Normal collection still precedes validation/admission.
                    # Retain the cleanup reserve for all remaining owners; a
                    # failed close stays active for the concurrent final cleanup.
                    messages: list[str] = []
                    if not process.close_owner(
                        owned, deadline=execution_deadline, diagnostic=messages.append
                    ):
                        failure = ExecutionError(
                            "Review owner cleanup could not be proven"
                        )
                        for message in messages:
                            failure.add_note(message)
                        raise failure
                    del active[pid]
                    item.streams.close()
                if code != 0:
                    raise ExecutionError(
                        f"Review invocation {invocation.id} exited with {code}"
                    )
                remaining()
                validate(invocation)
                remaining()  # Late validation is never admitted as evidence.
                completed.append(
                    {
                        "id": invocation.id,
                        "exit_code": code,
                        "owned_tree_empty": True,
                    }
                )

        try:
            with within_deadline(execution_deadline):
                # These jobs remain blocked until final cleanup. Consequently
                # each submission must start a worker, and all four workers
                # exist before any child is admitted. No later submit can fail
                # after enqueuing cleanup for an owner whose handoff is unknown.
                for index in range(CONCURRENCY):
                    remaining()
                    try:
                        with controller.defer():
                            cleanup_workers.append(cleanup.submit(clean_slot, index))
                    except RuntimeError as error:
                        raise ExecutionError(
                            "Cannot prepare review cleanup workers before admission"
                        ) from error
                while next_index < len(invocations) or active:
                    remaining()
                    collect_completed()
                    if next_index < len(invocations) and len(active) < CONCURRENCY:
                        remaining()  # Check EACH admission, including after prompt IO.
                        invocation = invocations[next_index]
                        invocation.directory.mkdir(parents=True, exist_ok=False)
                        prompt_path = invocation.directory / "prompt.txt"
                        prompt_path.write_text(invocation.prompt, encoding="utf-8")
                        prompt_path.chmod(0o400)
                        with ExitStack() as streams:
                            output = streams.enter_context(
                                (invocation.directory / "events.jsonl").open("wb")
                            )
                            errors = streams.enter_context(
                                (invocation.directory / "stderr.log").open("wb")
                            )
                            from .merge_command import without_review_owner

                            phase = process.ProcessCommand(
                                invocation.id,
                                (
                                    sys.executable,
                                    "-I",
                                    "-B",
                                    str(Path(__file__).with_name("review_worker.py")),
                                    str(prompt_path),
                                    *invocation.command,
                                ),
                                repository,
                                environment=without_review_owner(os.environ),
                                stdout=output,
                                stderr=errors,
                            )
                            remaining()
                            # Prompt/file IO can outlast another owner's failure.
                            # Drain again immediately before registering a child.
                            collect_completed()
                            remaining()
                            with controller.defer():
                                owned = process.start_owned_process(
                                    phase,
                                    deadline=deadline,
                                    signal_controller=controller,
                                )
                                active[owned.process.pid] = _Active(
                                    invocation, owned, streams.pop_all()
                                )
                        next_index += 1
                        continue
                    if active:
                        time.sleep(
                            min(0.05, max(0.0, execution_deadline - time.monotonic()))
                        )
        finally:
            with controller.defer():
                cleanup_deadline = min(deadline, time.monotonic() + CLEANUP_RESERVE)
                # A busy scope must not spend the other owners' signal/proof
                # budget. Each worker owns one process/scope/stream set; parent
                # ledger changes still use the canonical filesystem locks. The
                # main thread retains signal handling and joins every worker.
                cleanup_items = tuple(active.values())
                cleanup_ready.set()
                # On partial startup failure an enqueued job may have no
                # returned Future. There are no children yet, so it is a no-op;
                # the executor still joins every thread after this trigger.
                results = []
                for worker in cleanup_workers:
                    try:
                        results.append(worker.result())
                    except BaseException as error:
                        # Keep collecting the other independent owners' proofs.
                        results.append(
                            (False, [f"Review cleanup worker failed: {error}"])
                        )
                failed_cleanup = any(not empty for empty, _messages in results)
                if failed_cleanup:
                    details = "; ".join(
                        message for _empty, messages in results for message in messages
                    )
                    raise process.ProcessOwnershipError(
                        "Partitioned review cleanup could not be proven"
                        + (f": {details}" if details else "")
                    )
                # Only after all command owners are empty may the controlling
                # thread spend the remainder on optional human diagnostics.
                for _empty, messages in results:
                    for message in messages:
                        process.report_diagnostic(
                            message,
                            deadline=cleanup_deadline,
                            signal_controller=controller,
                        )
                if collection_ownership_error is not None:
                    collection_ownership_error.cleanup_proven = True
    return completed
