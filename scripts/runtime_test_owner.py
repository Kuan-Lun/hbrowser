"""PYTEST_DONT_REWRITE

Compose runtime process receipts with this repository's pytest owner.

This development-only adapter records an outer obligation before pytest can
import runtime code. It never discovers processes or manufactures runtime
receipts. A killed interpreter leaves its durable POSIX obligation unresolved.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from unittest.mock import patch

from _shared.h2h_dev_tools._process_scope import ENVIRONMENT_KEY, PosixScope

_CLEANUP_SECONDS = 5.0
_WORKER_FILES = (
    "process.py",
    "_process_control.py",
    "_process_observation.py",
    "_process_supervisor.py",
    "_directory_cleanup_worker.py",
)


def _worker_contents(directory: Path) -> dict[str, tuple[bytes, int]]:
    result = {}
    for name in _WORKER_FILES:
        path = directory / name
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Runtime worker source is not a regular file: {path}")
        result[name] = (path.read_bytes(), stat.S_IMODE(metadata.st_mode))
    return result


class _WorkerSources:
    """Project exact worker sources into an initially cache-free directory.

    Production keeps its -E -S argv and normal sibling imports. Only this
    development adapter redirects its two script paths; no wrapper executes in
    a runtime worker, and no runtime receipt is replaced or manufactured.
    """

    def __init__(self) -> None:
        from hbrowser.gallery.browser import process

        self.source = Path(process.__file__).resolve().parent
        self.contents = _worker_contents(self.source)
        self.directory = Path(tempfile.mkdtemp(prefix="hbrowser-test-workers-"))
        self.paths = contextlib.ExitStack()
        try:
            for name, (content, mode) in self.contents.items():
                path = self.directory / name
                path.write_bytes(content)
                path.chmod(mode)
            self.verify()
            self.paths.enter_context(
                patch.object(
                    process,
                    "_SUPERVISOR_PATH",
                    self.directory / "_process_supervisor.py",
                )
            )
            self.paths.enter_context(
                patch.object(
                    process,
                    "_PRIVATE_CLEANUP_WORKER_PATH",
                    self.directory / "_directory_cleanup_worker.py",
                )
            )
        except BaseException:
            self.paths.close()
            shutil.rmtree(self.directory)
            raise

    def verify(self) -> None:
        if (
            _worker_contents(self.source) != self.contents
            or _worker_contents(self.directory) != self.contents
            or {p.name for p in self.directory.iterdir() if p.name != "__pycache__"}
            != set(_WORKER_FILES)
        ):
            raise RuntimeError("Runtime worker source projection changed")
        cache = self.directory / "__pycache__"
        if cache.exists() or cache.is_symlink():
            if cache.is_symlink() or not cache.is_dir():
                raise RuntimeError("Runtime worker cache is not a private directory")
            if any(
                not stat.S_ISREG(p.lstat().st_mode) or p.suffix != ".pyc"
                for p in cache.iterdir()
            ):
                raise RuntimeError("Unexpected runtime worker cache entry")

    def close(self) -> None:
        # Called only after every real runtime owner and nested scope is empty.
        # An unproven cleanup retains both the projection and the path binding.
        try:
            self.verify()
        finally:
            self.paths.close()
            shutil.rmtree(self.directory)


class _Observation:
    def __init__(self, registry: _Registry) -> None:
        self.registry = registry
        self.thread = threading.get_ident()
        self.starting = True
        self.completed = False
        self.callback: Callable[[float], None] | None = None

    def bind(self, cleanup: Callable[[float], None], *, deadline: float) -> None:
        with self.registry.locked(deadline):
            if not self.starting or self.completed:
                raise RuntimeError("Runtime observation cannot change stable ownership")
            self.callback = cleanup

    def finish_start(self, *, deadline: float) -> None:
        with self.registry.locked(deadline):
            self.starting = False
            self.registry.changed.notify_all()
        self.registry.deliver_cancel()

    def complete(self, *, deadline: float) -> None:
        with self.registry.locked(deadline):
            self.completed = True
            self.registry.observations.discard(self)
            self.registry.changed.notify_all()


class _Registry:
    def __init__(self) -> None:
        self.changed = threading.Condition(threading.RLock())
        self.observations: set[_Observation] = set()
        self.closing = False
        self.sealed = False
        self.cleaning = False
        self.cancelled: int | None = None
        self.deadline: float | None = None
        self.scope: PosixScope | None = None
        self.previous_scope = os.environ.get(ENVIRONMENT_KEY)
        self.handlers = {
            number: signal.getsignal(number)
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        self.handler_attempts: list[signal.Signals] = []
        self.deferring_cancel = True
        self.entry_cleanup_started = False
        self.previous_observer: object = None
        self.worker_sources: _WorkerSources | None = None
        self.installed = False
        self.finished = False

    def setup(self) -> None:
        # The complete state object is registered before any external effect.
        # Install cancellation handlers before creating a durable obligation;
        # pending signals cannot escape between scope creation and assignment.
        self.deadline = time.monotonic() + _CLEANUP_SECONDS
        for number in self.handlers:
            self.handler_attempts.append(number)
            signal.signal(number, self.on_signal)
        if os.name == "posix":
            self.scope = PosixScope.create(deadline=self.deadline)
            os.environ[ENVIRONMENT_KEY] = self.scope.environment(None)[ENVIRONMENT_KEY]
        if self.cancelled is None:
            self.deadline = None
        self.deferring_cancel = False
        self.deliver_cancel()

    @contextlib.contextmanager
    def locked(self, deadline: float) -> Iterator[None]:
        deadline = self.cleanup_deadline(deadline)
        if not self.changed.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Runtime test ownership state deadline expired")
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("Runtime test ownership state deadline expired")
            yield
        finally:
            self.changed.release()

    def admit(self, *, recovery: bool, deadline: float) -> _Observation:
        with self.locked(deadline):
            if self.sealed or (self.closing and not recovery):
                raise RuntimeError("Runtime test owner has closed process admission")
            observation = _Observation(self)
            self.observations.add(observation)
            return observation

    def cleanup_deadline(self, deadline: float) -> float:
        return deadline if self.deadline is None else min(deadline, self.deadline)

    def on_signal(self, number: int, frame: FrameType | None) -> None:
        del frame
        # The first cancellation transfers responsibility to cleanup. A later
        # signal must not interrupt that handoff or replace its fixed deadline.
        if self.cleaning or self.cancelled is not None:
            return
        self.cancelled = number
        if self.deadline is None:
            self.deadline = time.monotonic() + _CLEANUP_SECONDS
        self.closing = True
        self.deliver_cancel()

    def deliver_cancel(self) -> None:
        if (
            self.cancelled is None
            or self.cleaning
            or self.deferring_cancel
            or threading.current_thread() is not threading.main_thread()
        ):
            return
        # Signal delivery can interrupt admission/binding on this same thread.
        # The RLock is reentrant, and a STARTING token defers the exception until
        # the runtime's finally block has established the final callback.
        with self.locked(time.monotonic() + _CLEANUP_SECONDS):
            if any(
                item.starting and item.thread == threading.get_ident()
                for item in self.observations
            ):
                return
        raise KeyboardInterrupt("Runtime test invocation cancelled")

    def install(self) -> None:
        if self.installed:
            return
        self.deferring_cancel = True
        from hbrowser.gallery.browser import _process_observation

        self.previous_observer = _process_observation.observer
        if self.previous_observer is not None:
            raise RuntimeError("Runtime process observer is already installed")
        self.worker_sources = _WorkerSources()
        _process_observation.observer = self
        self.installed = True
        # Failed installation keeps cancellation deferred until its entry
        # transfers responsibility to close, including the first signal there.
        self.deferring_cancel = False
        self.deliver_cancel()

    def close(self) -> None:
        if self.finished:
            return
        self.cleaning = True
        if self.deadline is None:
            self.deadline = time.monotonic() + _CLEANUP_SECONDS
        deadline = self.deadline
        failure: BaseException | None = None
        try:
            with self.locked(deadline):
                self.closing = True
            if self.scope is not None:
                self.scope.close_admission(deadline=deadline)
            attempted: set[_Observation] = set()
            while True:
                with self.locked(deadline):
                    if not self.observations:
                        self.sealed = True
                        break
                    available = tuple(
                        item
                        for item in self.observations
                        if not item.starting and item not in attempted
                    )
                    if not available:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Runtime test cleanup proof is incomplete"
                            )
                        self.changed.wait(timeout=max(0.0, deadline - time.monotonic()))
                        continue
                for item in available:
                    attempted.add(item)
                    if item.callback is None:
                        failure = RuntimeError("Runtime startup has no cleanup owner")
                        continue
                    try:
                        item.callback(deadline)
                    except BaseException as error:
                        if failure is None:
                            failure = error
            if failure is not None:
                raise RuntimeError("Runtime test cleanup failed") from failure
            if self.scope is not None:
                while self.scope.has_pending(deadline=deadline):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Nested test cleanup obligations remain")
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                self.scope.proven_empty = True
                self.scope.close(deadline=deadline)
            if self.worker_sources is not None:
                self.worker_sources.close()
            self.finished = True
        finally:
            # The observer is an invocation-lifetime fence, including atexit
            # and background threads after the command has returned. Never
            # restore an unobserved admission path after publishing emptiness.
            if self.previous_scope is None:
                os.environ.pop(ENVIRONMENT_KEY, None)
            else:
                os.environ[ENVIRONMENT_KEY] = self.previous_scope
            for number in reversed(self.handler_attempts):
                # A failed signal.signal call may have installed its handler
                # before raising. Restore attempted changes that remain ours.
                if signal.getsignal(number) == self.on_signal:
                    signal.signal(number, self.handlers[number])


_registry: _Registry | None = None
_managed_by_run = False


def _current() -> _Registry:
    global _registry
    if _registry is None:
        _registry = _Registry()
        try:
            _registry.setup()
        except BaseException:
            # This layer owns setup failure, including uncertain scope cleanup.
            # Entry finally blocks must not retry and erase that uncertainty.
            _registry.deferring_cancel = True
            _registry.entry_cleanup_started = True
            _registry.close()
            raise
    if _registry.entry_cleanup_started and not _registry.finished:
        raise RuntimeError("Runtime test owner cleanup is unproven")
    return _registry


def pytest_addhooks() -> None:
    # Explicit -p plugins are registered before entry-point plugins/conftests.
    # This occurs inside the shared runner's fresh source-cache environment.
    try:
        _current().install()
    except BaseException:
        if (
            not _managed_by_run
            and _registry is not None
            and not _registry.entry_cleanup_started
        ):
            # This entry owns install failure. Pytest may unconfigure next;
            # that hook must not retry a close whose proof is uncertain.
            _registry.deferring_cancel = True
            _registry.entry_cleanup_started = True
            _registry.close()
        raise


def pytest_unconfigure() -> None:
    if (
        not _managed_by_run
        and _registry is not None
        and not _registry.entry_cleanup_started
    ):
        _registry.deferring_cancel = True
        _registry.entry_cleanup_started = True
        _registry.close()
        if _registry.cancelled is not None:
            raise KeyboardInterrupt("Runtime test invocation cancelled")


def run(main: Callable[[], object], *, pytest_entry: bool = False) -> object:
    """Own an invocation; runtime import happens after pytest source setup."""
    global _managed_by_run
    if _managed_by_run:
        raise RuntimeError("Runtime test owner may only run once per interpreter")
    _managed_by_run = True
    try:
        registry = _current()
        if pytest_entry:
            sys.argv[1:1] = ["-p", "runtime_test_owner"]
        else:
            registry.install()
        return main()
    finally:
        if _registry is not None:
            if not _registry.entry_cleanup_started:
                _registry.deferring_cancel = True
                _registry.entry_cleanup_started = True
                _registry.close()
            if _registry.finished and _registry.cancelled is not None:
                raise SystemExit(128 + _registry.cancelled)
