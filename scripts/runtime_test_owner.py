"""PYTEST_DONT_REWRITE

Compose runtime process receipts with this repository's pytest owner.

This development-only adapter records an outer obligation before pytest can
import runtime code. It never discovers processes or manufactures runtime
receipts. A killed interpreter leaves its durable POSIX obligation unresolved.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType

from _shared.h2h_dev_tools._process_scope import ENVIRONMENT_KEY, PosixScope

_CLEANUP_SECONDS = 5.0


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
        self.scope = (
            PosixScope.create(deadline=time.monotonic() + _CLEANUP_SECONDS)
            if os.name == "posix"
            else None
        )
        self.previous_scope = os.environ.get(ENVIRONMENT_KEY)
        if self.scope is not None:
            os.environ[ENVIRONMENT_KEY] = self.scope.environment(None)[ENVIRONMENT_KEY]
        self.handlers = {
            number: signal.signal(number, self.on_signal)
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        self.previous_observer: object = None
        self.installed = False
        self.finished = False

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
        if self.cleaning:
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
        from hbrowser.gallery.browser import _process_observation

        self.previous_observer = _process_observation.observer
        if self.previous_observer is not None:
            raise RuntimeError("Runtime process observer is already installed")
        _process_observation.observer = self
        self.installed = True

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
            self.finished = True
        finally:
            # The observer is an invocation-lifetime fence, including atexit
            # and background threads after the command has returned. Never
            # restore an unobserved admission path after publishing emptiness.
            if self.previous_scope is None:
                os.environ.pop(ENVIRONMENT_KEY, None)
            else:
                os.environ[ENVIRONMENT_KEY] = self.previous_scope
            for number, handler in self.handlers.items():
                signal.signal(number, handler)


_registry: _Registry | None = None
_managed_by_run = False


def _current() -> _Registry:
    global _registry
    if _registry is None:
        _registry = _Registry()
    return _registry


def pytest_addhooks() -> None:
    # Explicit -p plugins are registered before entry-point plugins/conftests.
    # This occurs inside the shared runner's fresh source-cache environment.
    _current().install()


def pytest_unconfigure() -> None:
    if not _managed_by_run and _registry is not None:
        _registry.close()


def run(main: Callable[[], object], *, pytest_entry: bool = False) -> object:
    """Own an invocation; runtime import happens after pytest source setup."""
    global _managed_by_run
    if _managed_by_run:
        raise RuntimeError("Runtime test owner may only run once per interpreter")
    _managed_by_run = True
    registry = _current()
    try:
        if pytest_entry:
            sys.argv[1:1] = ["-p", "runtime_test_owner"]
        else:
            registry.install()
        return main()
    finally:
        registry.close()
        if registry.cancelled is not None:
            raise SystemExit(128 + registry.cancelled)
