import array
import atexit
import ctypes
import gc
import io
import math
import os
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import ANY, Mock, call, patch

import runtime_test_owner
import zendriver as zd
from _shared.h2h_dev_tools import process as command_process

from hbrowser.gallery.browser import _directory_cleanup_worker as cleanup_worker_module
from hbrowser.gallery.browser import _directory_identity as identity_module
from hbrowser.gallery.browser import _process_control as control_module
from hbrowser.gallery.browser import _process_observation
from hbrowser.gallery.browser import _process_supervisor as supervisor_module
from hbrowser.gallery.browser import process as process_module


@contextmanager
def _mock_popen(**options: Any) -> Iterator[Mock]:
    # Only the fully mocked spawn boundary is exempt. In particular, a mock
    # OwnedProcess may still launch a REAL private-cleanup worker outside this
    # context; that worker must remain observed.
    with (
        patch.object(subprocess, "Popen", **options) as mocked,
        patch.object(_process_observation, "observer", None),
    ):
        yield mocked


def _runtime_harness_source(source: str) -> str:
    entry = Path(__file__).resolve().parents[1] / "scripts/pytest-source.py"
    return (
        "import runpy\n"
        f"boundary = runpy.run_path({str(entry)!r})\n"
        f"boundary['adapter'].run(lambda: exec({source!r}, "
        "{'__name__': '__main__'}))\n"
    )


@contextmanager
def _owned_harness(
    command: command_process.ProcessCommand,
) -> Iterator[command_process.OwnedProcess]:
    registry = runtime_test_owner._current()
    deadline = time.monotonic() + 10
    observation = registry.admit(recovery=False, deadline=deadline)
    owner: command_process.OwnedProcess | None = None

    def cleanup(deadline: float) -> None:
        assert owner is not None
        if not command_process.terminate_owned_tree(owner, deadline=deadline):
            raise RuntimeError("Harness process tree remains owned")
        if not command_process.close_owner(owner, deadline=deadline):
            raise RuntimeError("Harness process owner could not close")
        observation.complete(deadline=deadline)

    try:
        try:
            owner = command_process.start_owned_process(
                command,
                deadline=deadline,
                signal_controller=None,
            )
            observation.bind(cleanup, deadline=deadline)
        finally:
            observation.finish_start(deadline=deadline)
        yield owner
    finally:
        if owner is not None and not observation.completed:
            cleanup(registry.cleanup_deadline(time.monotonic() + 6))


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.02)
    return not _pid_exists(pid)


@contextmanager
def _private_directory_fixture(
    *, prefix: str = "hbrowser-private-fixture-", delete: bool = True
) -> Iterator[str]:
    # Keep the underlying TemporaryDirectory from deleting unproven state even
    # when a failed assertion unwinds this context or its finalizer later runs.
    with tempfile.TemporaryDirectory(prefix=prefix, delete=False) as directory:
        captured: list[process_module._PrivateDirectory] = []
        original_capture = process_module._PrivateDirectory.capture

        def capture(
            path: Path, *, platform_name: str | None = None
        ) -> process_module._PrivateDirectory:
            guard = original_capture(path, platform_name=platform_name)
            captured.append(guard)
            return guard

        try:
            with patch.object(
                process_module._PrivateDirectory, "capture", side_effect=capture
            ):
                yield directory
        finally:
            with process_module._PRIVATE_DIRECTORY_CLEANUP_LOCK:
                unresolved = any(
                    process_module._private_directory_key(guard)
                    in process_module._PENDING_PRIVATE_DIRECTORIES
                    or process_module._private_directory_key(guard)
                    in process_module._ACTIVE_PRIVATE_DIRECTORY_CLEANUPS
                    for guard in captured
                )
            if unresolved:
                raise RuntimeError(
                    f"Retain private fixture with unresolved ownership: {directory}"
                )
            # These are fixture-held handles only. Production completion and
            # retention assertions have already run before this rescue boundary.
            for guard in captured:
                guard._retire()
            if delete and Path(directory).exists():
                shutil.rmtree(directory)


@contextmanager
def _claimed_namespace(
    guard: process_module._PrivateDirectory,
) -> Iterator[process_module._PrivateDirectory]:
    namespace_path = guard.path.parent / f".hbrowser-cleanup-test-{id(guard)}"
    namespace_path.mkdir(mode=0o700)
    namespace = None
    try:
        namespace = process_module._PrivateDirectory.capture(namespace_path)
        guard._claim_inline(namespace, confirmed=False, deadline=time.monotonic() + 2)
        yield namespace
    finally:
        # Fixture rescue only: runtime never deletes an unverified claim.
        guard._retire()
        if namespace is not None:
            namespace._retire()
        if namespace_path.exists():
            shutil.rmtree(namespace_path)


def _remove_private_inline(
    guard: process_module._PrivateDirectory, *, deadline: float
) -> None:
    if not guard._assert_identity_or_absence():
        return
    with _claimed_namespace(guard) as namespace:
        guard._remove_inline(namespace=namespace, deadline=deadline)


def _release_private_fixture(
    guard: process_module._PrivateDirectory, root: Path
) -> None:
    active = process_module._active_private_directory_cleanup(guard)
    if active is not None:
        process_module._force_reap_private_directory_cleanup(
            active, deadline=_process_observation.cleanup_deadline(time.monotonic() + 3)
        )
    if process_module._active_private_directory_cleanup(guard) is not None:
        raise RuntimeError(
            "Retain fixture while private cleanup ownership is unresolved"
        )
    state = process_module._PENDING_PRIVATE_DIRECTORIES.get(
        process_module._private_directory_key(guard)
    )
    process_module._release_pending_private_directory(guard)
    guard._retire()
    if state is not None and state.namespace is not None:
        state.namespace._retire()
    shutil.rmtree(root)


def _windows_cleanup_error(code: int) -> PermissionError:
    error = PermissionError(f"Windows cleanup error {code}")
    cast(Any, error).winerror = code
    return error


def _shutdown_controller(
    intent: control_module.ControlIntent = control_module.ControlIntent.TERMINATE,
) -> supervisor_module._ShutdownController:
    now_ns = time.monotonic_ns()
    controller = supervisor_module._ShutdownController()
    controller.apply(
        control_module.ControlRequest(
            intent=intent,
            phase_deadline_ns=now_ns + 5_000_000_000,
            overall_deadline_ns=now_ns + 10_000_000_000,
        )
    )
    return controller


class ProcessControlProtocolTests(unittest.TestCase):
    def test_ready_receipt_round_trips_exactly(self) -> None:
        receipt = control_module.ReadyReceipt(101, 0, 202, 303)
        self.assertEqual(receipt.encode(), b"hbrowser-ready-v1 101 0 202 303\n")
        self.assertEqual(control_module.ReadyReceipt.parse(receipt.encode()), receipt)

    def test_ready_receipt_rejects_unknown_legacy_and_malformed_frames(self) -> None:
        frames = (
            b"ready 303\n",
            b"hbrowser-ready-v2 101 0 202 303\n",
            b"hbrowser-ready-v1 101 0 202 303",
            b"hbrowser-ready-v1 101 0 202 303\n\n",
            b"hbrowser-ready-v1 101 0 202 303\r\n",
            b"hbrowser-ready-v1 0101 0 202 303\n",
            b"hbrowser-ready-v1 101 0 202 -303\n",
            b"hbrowser-ready-v1 0 0 202 303\n",
            b"hbrowser-ready-v1 101 0 202 0\n",
            b"hbrowser-ready-v1 101 0 202 101\n",
            b"hbrowser-ready-v1 101 0 202 303 extra\n",
            b"hbrowser-ready-v1 101 0 202 " + b"3" * 200 + b"\n",
        )
        for frame in frames:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                control_module.ReadyReceipt.parse(frame)

    def test_start_request_round_trips_exactly(self) -> None:
        request = control_module.StartRequest(deadline_ns=123)

        frame = request.encode()

        self.assertLessEqual(len(frame), control_module.MAX_START_LINE_BYTES)
        self.assertEqual(control_module.StartRequest.parse(frame), request)

    def test_start_request_rejects_legacy_malformed_and_noncanonical_frames(
        self,
    ) -> None:
        invalid_frames = (
            b"start\n",
            b"hbrowser-start-v0 1\n",
            b"hbrowser-start-v1\n",
            b"hbrowser-start-v1 1",
            b"hbrowser-start-v1 01\n",
            b"hbrowser-start-v1 -1\n",
            b"hbrowser-start-v1 1\r\n",
            b"hbrowser-start-v1 1\ntrailing\n",
            b"x" * (control_module.MAX_START_LINE_BYTES + 1),
        )
        for frame in invalid_frames:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                control_module.StartRequest.parse(frame)

    def test_control_request_round_trips_exactly(self) -> None:
        for request in (
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                10,
                None,
            ),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                20,
                30,
                allow_immediate=True,
            ),
        ):
            frame = request.encode()
            self.assertLessEqual(len(frame), control_module.MAX_CONTROL_LINE_BYTES)
            self.assertEqual(control_module.ControlRequest.parse(frame), request)

    def test_control_request_rejects_noncanonical_or_invalid_frames(self) -> None:
        invalid_frames = (
            b"terminate\n",
            b"kill\n",
            b"hbrowser-control-v0 terminate 1 none 0\n",
            b"hbrowser-control-v1 stop 1 none 0\n",
            b"hbrowser-control-v1 TERMINATE 1 none 0\n",
            b"hbrowser-control-v1 terminate 01 none 0\n",
            b"hbrowser-control-v1 terminate 1 none 2\n",
            b"hbrowser-control-v1 terminate 1 none 0\r\n",
            b"x" * (control_module.MAX_CONTROL_LINE_BYTES + 1),
        )
        for frame in invalid_frames:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                control_module.ControlRequest.parse(frame)

    def test_control_merge_escalates_only_and_preserves_phase_contract(self) -> None:
        terminate = control_module.ControlRequest(
            control_module.ControlIntent.TERMINATE,
            10,
            100,
        )
        later_terminate = control_module.ControlRequest(
            control_module.ControlIntent.TERMINATE,
            8,
            90,
        )
        kill = control_module.ControlRequest(
            control_module.ControlIntent.KILL,
            40,
            80,
        )
        later_kill = control_module.ControlRequest(
            control_module.ControlIntent.KILL,
            30,
            70,
        )

        self.assertEqual(
            terminate.merge(later_terminate),
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                8,
                90,
            ),
        )
        self.assertEqual(
            terminate.merge(kill),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                40,
                80,
            ),
        )
        self.assertEqual(
            kill.merge(later_kill),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                30,
                70,
            ),
        )
        self.assertEqual(
            kill.merge(later_terminate),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                40,
                80,
            ),
        )

    def test_deadline_conversion_floors_the_exact_float(self) -> None:
        deadline = math.nextafter(1.0, math.inf)
        numerator, denominator = deadline.as_integer_ratio()
        expected = numerator * 1_000_000_000 // denominator
        self.assertEqual(
            control_module.deadline_to_monotonic_ns(deadline),
            expected,
        )

    def test_control_reader_accepts_fragmented_and_escalating_frames(self) -> None:
        now_ns = time.monotonic_ns()
        terminate = control_module.ControlRequest(
            control_module.ControlIntent.TERMINATE,
            now_ns + 1_000_000_000,
            now_ns + 4_000_000_000,
        ).encode()
        kill = control_module.ControlRequest(
            control_module.ControlIntent.KILL,
            now_ns + 2_000_000_000,
            now_ns + 3_000_000_000,
        ).encode()
        controller = supervisor_module._ShutdownController()
        with patch.object(
            os,
            "read",
            side_effect=(terminate[:7], terminate[7:] + kill, b""),
        ):
            supervisor_module._watch_control_pipe(42, controller)

        request = controller.snapshot()
        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        self.assertFalse(controller.protocol_failed)

    def test_control_reader_rejects_truncation_oversize_and_read_error(self) -> None:
        cases: tuple[object, ...] = (
            (b"hbrowser-control-v1 terminate", b""),
            (b"x" * (control_module.MAX_CONTROL_LINE_BYTES + 1),),
            (OSError("control read failed"),),
        )
        for side_effect in cases:
            with self.subTest(side_effect=side_effect):
                controller = supervisor_module._ShutdownController()
                with patch.object(os, "read", side_effect=side_effect):
                    supervisor_module._watch_control_pipe(42, controller)
                self.assertTrue(controller.protocol_failed)
                request = controller.snapshot()
                assert request is not None
                self.assertEqual(request.intent, control_module.ControlIntent.KILL)

    def test_expired_direct_terminate_parks_until_explicit_kill(self) -> None:
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                0,
                None,
            )
        )
        now_ns = time.monotonic_ns()

        def deliver_kill(
            _: supervisor_module._ControllerState,
            __: float,
        ) -> supervisor_module._ControllerState:
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.KILL,
                    now_ns + 1_000_000_000,
                    now_ns + 2_000_000_000,
                )
            )
            return controller.state()

        with (
            patch.object(
                controller,
                "wait_for_change",
                side_effect=deliver_kill,
            ) as wait,
            patch.object(
                controller,
                "promote_planned_kill",
                wraps=controller.promote_planned_kill,
            ) as promote,
        ):
            kill_state = supervisor_module._wait_for_kill_request(controller)

        assert kill_state is not None
        request = kill_state.request
        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        wait.assert_called_once_with(ANY, supervisor_module._POLL_SECONDS)
        promote.assert_called_once_with()

    def test_finite_terminate_plan_promotes_to_kill_without_parent_round_trip(
        self,
    ) -> None:
        now_ns = time.monotonic_ns()
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                0,
                now_ns + 1_000_000_000,
                allow_immediate=True,
            )
        )

        kill_state = supervisor_module._wait_for_kill_request(controller)

        assert kill_state is not None
        request = kill_state.request
        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        self.assertEqual(request.phase_deadline_ns, now_ns + 1_000_000_000)
        self.assertFalse(request.allow_immediate)

    def test_zero_kill_budget_promotes_with_one_immediate_force_action(self) -> None:
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                0,
                0,
            )
        )

        kill_state = supervisor_module._wait_for_kill_request(controller)

        assert kill_state is not None
        request = kill_state.request
        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        self.assertTrue(request.allow_immediate)

    def test_expired_kill_fallback_starts_a_fresh_bounded_phase(self) -> None:
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                0,
                0,
            )
        )
        now_ns = time.monotonic_ns()

        with patch.object(time, "monotonic_ns", return_value=now_ns):
            controller.request_fallback(control_module.ControlIntent.KILL)

        request = controller.snapshot()
        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        self.assertGreater(request.phase_deadline_ns, now_ns)
        self.assertEqual(request.phase_deadline_ns, request.overall_deadline_ns)

    def test_owner_death_renews_a_kill_that_expires_before_consumption(self) -> None:
        now_ns = time.monotonic_ns()
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                now_ns + 10,
                now_ns + 10,
            )
        )
        with patch.object(time, "monotonic_ns", return_value=now_ns):
            controller.request_fallback(control_module.ControlIntent.TERMINATE)
        pending = controller.snapshot()
        assert pending is not None
        self.assertEqual(pending.phase_deadline_ns, now_ns + 10)

        with patch.object(time, "monotonic_ns", return_value=now_ns + 11):
            renewed_state = supervisor_module._wait_for_kill_request(controller)

        assert renewed_state is not None
        renewed = renewed_state.request
        assert renewed is not None
        self.assertEqual(renewed.intent, control_module.ControlIntent.KILL)
        self.assertGreater(renewed.phase_deadline_ns, now_ns + 11)

    def test_controller_renews_only_an_exhausted_same_intent_phase(self) -> None:
        now_ns = time.monotonic_ns()
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                now_ns + 10,
                now_ns + 10,
            )
        )
        later = control_module.ControlRequest(
            control_module.ControlIntent.KILL,
            now_ns + 30,
            now_ns + 30,
        )

        with patch.object(time, "monotonic_ns", return_value=now_ns):
            controller.apply(later)
        active = controller.snapshot()
        assert active is not None
        self.assertEqual(active.phase_deadline_ns, now_ns + 10)

        with patch.object(time, "monotonic_ns", return_value=now_ns + 11):
            controller.apply(later)
        self.assertEqual(controller.snapshot(), later)

    def test_due_finite_terminate_cannot_be_downgraded_by_later_terminate(
        self,
    ) -> None:
        controller = supervisor_module._ShutdownController()
        finite_terminate = control_module.ControlRequest(
            control_module.ControlIntent.TERMINATE,
            10,
            20,
        )
        standalone_terminate = control_module.ControlRequest(
            control_module.ControlIntent.TERMINATE,
            30,
            None,
        )
        with patch.object(time, "monotonic_ns", return_value=0):
            controller.apply(finite_terminate)
        with patch.object(time, "monotonic_ns", return_value=11):
            controller.apply(standalone_terminate)
            request = controller.snapshot()

        assert request is not None
        self.assertEqual(request.intent, control_module.ControlIntent.KILL)
        self.assertEqual(request.phase_deadline_ns, 20)
        self.assertEqual(request.overall_deadline_ns, 20)

    def test_fresh_kill_renews_an_exhausted_finite_terminate_plan(self) -> None:
        controller = supervisor_module._ShutdownController()
        with patch.object(time, "monotonic_ns", return_value=0):
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.TERMINATE,
                    10,
                    20,
                )
            )
        fresh_kill = control_module.ControlRequest(
            control_module.ControlIntent.KILL,
            40,
            40,
        )
        with patch.object(time, "monotonic_ns", return_value=21):
            controller.apply(fresh_kill)
            request = controller.snapshot()

        self.assertEqual(request, fresh_kill)

    def test_pending_immediate_kill_survives_same_intent_replacement(self) -> None:
        controller = supervisor_module._ShutdownController()
        with patch.object(time, "monotonic_ns", return_value=1):
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.KILL,
                    0,
                    0,
                    allow_immediate=True,
                )
            )
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.KILL,
                    0,
                    0,
                )
            )
            request = controller.snapshot()

        assert request is not None
        self.assertTrue(request.allow_immediate)

    def test_wait_for_change_does_not_sleep_after_an_already_applied_request(
        self,
    ) -> None:
        controller = supervisor_module._ShutdownController()
        observed = controller.state()
        now_ns = time.monotonic_ns()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                now_ns + 1_000_000_000,
                now_ns + 1_000_000_000,
            )
        )

        with patch.object(controller._condition, "wait_for") as wait_for:
            changed = controller.wait_for_change(observed, 1)

        wait_for.assert_not_called()
        self.assertGreater(changed.revision, observed.revision)

    def test_start_barrier_rejects_exact_deadline_and_queued_control(self) -> None:
        deadline_ns = 100
        for request in (
            None,
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                200,
                None,
            ),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                200,
                200,
            ),
        ):
            with self.subTest(request=request):
                controller = supervisor_module._ShutdownController()
                if request is not None:
                    with patch.object(time, "monotonic_ns", return_value=0):
                        controller.apply(request)
                start = Mock()
                now_ns = deadline_ns if request is None else 99
                with patch.object(time, "monotonic_ns", return_value=now_ns):
                    target, error = controller.start_target_if_authorized(
                        deadline_ns=deadline_ns,
                        action=start,
                    )

                self.assertIsNone(target)
                self.assertEqual(
                    error,
                    (
                        "StartupDeadlineExpired"
                        if request is None
                        else "StartupCancelled"
                    ),
                )
                start.assert_not_called()


@unittest.skipUnless(os.name == "posix", "POSIX process ownership contract")
class PosixOwnedProcessTests(unittest.TestCase):
    def test_supervisor_preflight_tests_preserve_process_signal_handlers(self) -> None:
        for name in (
            "test_missing_waitid_primitive_fails_before_target_popen",
            "test_missing_trusted_ps_fails_before_target_popen",
            "test_unusable_ps_fails_before_target_popen",
        ):
            with self.subTest(test=name):
                previous = {
                    signum: signal.getsignal(signum)
                    for signum in (signal.SIGINT, signal.SIGTERM)
                }
                try:
                    result = unittest.TestResult()
                    PosixOwnedProcessTests(name).run(result)
                    self.assertTrue(result.wasSuccessful(), result.errors)
                    for signum, handler in previous.items():
                        self.assertIs(signal.getsignal(signum), handler)
                finally:
                    for signum, handler in previous.items():
                        signal.signal(signum, handler)

    def test_start_reader_leaves_same_write_control_tail_for_cancel_barrier(
        self,
    ) -> None:
        deadline_ns = time.monotonic_ns() + 5_000_000_000
        start_frame = control_module.StartRequest(deadline_ns).encode()
        requests = (
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                deadline_ns,
                None,
            ),
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                deadline_ns,
                deadline_ns,
            ),
            None,
        )
        for request in requests:
            with self.subTest(request=request):
                read_descriptor, write_descriptor = os.pipe()
                controller = supervisor_module._ShutdownController()
                try:
                    tail = b"" if request is None else request.encode()
                    os.write(write_descriptor, start_frame + tail)
                    if request is None:
                        os.close(write_descriptor)
                        write_descriptor = -1

                    self.assertEqual(
                        supervisor_module._read_start_gate(read_descriptor),
                        start_frame,
                    )
                    drained = supervisor_module._start_control_watcher(
                        read_descriptor,
                        controller,
                    )
                    self.assertTrue(drained.wait(timeout=1))
                    start = Mock()
                    target, error = controller.start_target_if_authorized(
                        deadline_ns=deadline_ns,
                        action=start,
                    )

                    self.assertIsNone(target)
                    self.assertEqual(error, "StartupCancelled")
                    start.assert_not_called()
                finally:
                    if write_descriptor >= 0:
                        os.close(write_descriptor)
                    os.close(read_descriptor)

    def test_waitid_preflight_requires_callable_runtime_support(self) -> None:
        with patch.object(os, "waitid", None):
            self.assertFalse(supervisor_module._posix_exit_receipts_supported())
        with (
            patch.object(os, "fork", return_value=123),
            patch.object(os, "waitid", side_effect=OSError("unsupported")),
            patch.object(os, "kill") as kill,
            patch.object(os, "waitpid", side_effect=((0, 0), (123, 0))) as waitpid,
        ):
            self.assertFalse(supervisor_module._posix_exit_receipts_supported())
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(
            waitpid.call_args_list,
            [call(123, os.WNOHANG), call(123, 0)],
        )
        with (
            patch.object(os, "fork", return_value=123),
            patch.object(
                os,
                "waitid",
                side_effect=(Mock(si_pid=123), Mock(si_pid=123)),
            ) as waitid,
            patch.object(os, "waitpid", return_value=(123, 0)) as waitpid,
        ):
            self.assertTrue(supervisor_module._posix_exit_receipts_supported())
        self.assertEqual(waitid.call_count, 2)
        waitpid.assert_called_once_with(123, 0)

    def test_waitid_preflight_never_signals_after_identity_may_be_reaped(
        self,
    ) -> None:
        with (
            patch.object(os, "fork", return_value=123),
            patch.object(
                os,
                "waitid",
                side_effect=(Mock(si_pid=123), ChildProcessError()),
            ),
            patch.object(os, "waitpid", side_effect=ChildProcessError) as waitpid,
            patch.object(os, "kill") as kill,
        ):
            self.assertFalse(supervisor_module._posix_exit_receipts_supported())

        waitpid.assert_called_once_with(123, os.WNOHANG)
        kill.assert_not_called()

    def test_preflight_cannot_inherit_target_only_capabilities(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-preflight-capability-test-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            initial_drain = threading.Event()
            initial_drain.set()
            target = Mock(pid=123)
            target_environment: dict[str, str] = {}
            forwarding_token = f"test-{time.monotonic_ns()}"

            def assert_sanitized_preflight() -> bool:
                self.assertNotIn("HBROWSER_LOG_FORWARD_ENDPOINT", os.environ)
                self.assertNotIn("HBROWSER_LOG_FORWARD_TOKEN", os.environ)
                return True

            def launch_target(*_: object, **options: object) -> Mock:
                environment = cast(dict[str, str], options["env"])
                target_environment.update(environment)
                return target

            with (
                patch.dict(
                    os.environ,
                    {
                        "HBROWSER_LOG_FORWARD_ENDPOINT": "local-endpoint",
                        "HBROWSER_LOG_FORWARD_TOKEN": forwarding_token,
                    },
                ),
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest((1 << 63) - 1).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    side_effect=assert_sanitized_preflight,
                ),
                patch.object(
                    supervisor_module,
                    "_posix_process_group_snapshots_supported",
                    side_effect=assert_sanitized_preflight,
                ),
                patch.object(signal, "signal"),
                patch.object(
                    supervisor_module,
                    "_start_control_watcher",
                    return_value=initial_drain,
                ),
                _mock_popen(side_effect=launch_target),
                patch.object(
                    supervisor_module,
                    "_posix_target_has_exit_receipt",
                    return_value=True,
                ),
                patch.object(
                    supervisor_module,
                    "_complete_posix_cleanup",
                    return_value=False,
                ),
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                target_environment["HBROWSER_LOG_FORWARD_ENDPOINT"],
                "local-endpoint",
            )
            self.assertEqual(
                target_environment["HBROWSER_LOG_FORWARD_TOKEN"],
                forwarding_token,
            )

    def test_preflight_failure_clears_retained_target_environment(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-preflight-environment-clear-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            forwarding_token = f"test-{time.monotonic_ns()}"
            retained_environment = {
                "HBROWSER_LOG_FORWARD_ENDPOINT": "local-endpoint",
                "HBROWSER_LOG_FORWARD_TOKEN": forwarding_token,
            }
            with (
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest((1 << 63) - 1).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_take_target_environment",
                    return_value=retained_environment,
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    return_value=False,
                ),
                patch.object(signal, "signal"),
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(
                result,
                control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
            )
            self.assertEqual(retained_environment, {})

    def test_malformed_ps_output_never_proves_a_group_clean(self) -> None:
        result = Mock(stdout="123 123\nmalformed\n", returncode=0)
        with (
            patch.object(subprocess, "run", return_value=result) as run,
            self.assertRaisesRegex(
                supervisor_module._PhaseDeadlineExpired,
                "malformed output",
            ),
        ):
            supervisor_module._process_group_members(
                123,
                deadline_ns=time.monotonic_ns() + 1_000_000_000,
            )
        run.assert_called_once_with(
            [supervisor_module._POSIX_PS_EXECUTABLE, "-axo", "pid=,pgid="],
            check=True,
            capture_output=True,
            env={"LC_ALL": "C", "PATH": os.defpath},
            text=True,
            timeout=ANY,
        )

    def test_incomplete_ps_snapshot_never_proves_target_cleanup(self) -> None:
        target = Mock(pid=123)
        for members in ((), (999,), (123, 123)):
            self.assertFalse(
                supervisor_module._snapshot_proves_pinned_group(
                    members,
                    target.pid,
                )
            )
            with (
                self.subTest(members=members),
                patch.object(
                    supervisor_module,
                    "_posix_target_has_exit_receipt",
                    return_value=True,
                ),
                patch.object(
                    supervisor_module,
                    "_process_group_members",
                    return_value=members,
                ),
            ):
                self.assertFalse(supervisor_module._try_settle_posix_target(target))
        self.assertTrue(
            supervisor_module._snapshot_proves_pinned_group(
                (target.pid,),
                target.pid,
            )
        )
        target.wait.assert_not_called()

    def test_missing_waitid_primitive_fails_before_target_popen(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-posix-preflight-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            with (
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest((1 << 63) - 1).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    return_value=False,
                ),
                patch.object(signal, "signal"),
                _mock_popen() as popen,
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(
                result,
                control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
            )
            self.assertEqual(
                status_path.read_text(encoding="utf-8"),
                "error UnsupportedOwnershipPrimitive\n",
            )
            popen.assert_not_called()

    def test_missing_trusted_ps_fails_before_target_popen(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-posix-ps-preflight-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            with (
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest((1 << 63) - 1).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    return_value=True,
                ),
                patch.object(supervisor_module, "_POSIX_PS_EXECUTABLE", None),
                patch.object(signal, "signal"),
                _mock_popen() as popen,
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(
                result,
                control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
            )
            self.assertEqual(
                status_path.read_text(encoding="utf-8"),
                "error UnsupportedOwnershipPrimitive\n",
            )
            popen.assert_not_called()

    def test_unusable_ps_fails_before_target_popen(self) -> None:
        failures: tuple[object, ...] = (
            subprocess.CalledProcessError(1, ["ps"]),
            Mock(stdout="123 123\nmalformed\n", returncode=0),
        )
        for failure in failures:
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory(
                    prefix="hbrowser-posix-ps-functional-preflight-"
                ) as directory,
            ):
                status_path = Path(directory) / "status"
                standard_input = Mock()
                standard_input.fileno.return_value = 42
                run_effect: object
                if isinstance(failure, BaseException):
                    run_effect = failure
                else:
                    run_effect = None
                with (
                    patch.object(sys, "stdin", standard_input),
                    patch.object(
                        supervisor_module,
                        "_read_start_gate",
                        return_value=control_module.StartRequest(
                            (1 << 63) - 1
                        ).encode(),
                    ),
                    patch.object(
                        supervisor_module,
                        "_posix_exit_receipts_supported",
                        return_value=True,
                    ),
                    patch.object(
                        subprocess,
                        "run",
                        return_value=(None if run_effect is not None else failure),
                        side_effect=run_effect,
                    ),
                    patch.object(signal, "signal"),
                    _mock_popen() as popen,
                ):
                    result = supervisor_module.main(
                        (str(status_path), "--", sys.executable, "-c", "pass")
                    )

                self.assertEqual(
                    result,
                    control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
                )
                self.assertEqual(
                    status_path.read_text(encoding="utf-8"),
                    "error UnsupportedOwnershipPrimitive\n",
                )
                popen.assert_not_called()

    def test_main_body_failure_returns_only_after_posix_cleanup_proof(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-posix-main-containment-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            target = Mock(pid=123)
            initial_drain = threading.Event()
            initial_drain.set()
            with (
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest((1 << 63) - 1).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    return_value=True,
                ),
                patch.object(
                    supervisor_module,
                    "_posix_process_group_snapshots_supported",
                    return_value=True,
                ),
                patch.object(signal, "signal"),
                patch.object(
                    supervisor_module,
                    "_start_control_watcher",
                    return_value=initial_drain,
                ),
                _mock_popen(return_value=target),
                patch.object(
                    supervisor_module,
                    "_write_status",
                    side_effect=RuntimeError("READY publication failed"),
                ),
                patch.object(
                    supervisor_module,
                    "_complete_posix_cleanup",
                    return_value=False,
                ) as cleanup,
                patch.object(supervisor_module, "_report_cleanup_failure") as report,
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(
                result,
                control_module.PROVEN_CLEANUP_FAILURE_EXIT_CODE,
            )
            cleanup.assert_called_once_with(target, controller=ANY)
            report.assert_called_once()

    def test_slow_preflight_crossing_start_deadline_never_calls_target_popen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-start-deadline-preflight-"
        ) as directory:
            status_path = Path(directory) / "status"
            standard_input = Mock()
            standard_input.fileno.return_value = 42
            clock = [9]

            def finish_slow_preflight() -> bool:
                clock[0] = 10
                return True

            with (
                patch.object(sys, "stdin", standard_input),
                patch.object(
                    supervisor_module,
                    "_read_start_gate",
                    return_value=control_module.StartRequest(10).encode(),
                ),
                patch.object(
                    supervisor_module,
                    "_posix_exit_receipts_supported",
                    side_effect=finish_slow_preflight,
                ),
                patch.object(
                    supervisor_module,
                    "_posix_process_group_snapshots_supported",
                    return_value=True,
                ),
                patch.object(time, "monotonic_ns", side_effect=lambda: clock[0]),
                patch.object(signal, "signal"),
                patch.object(supervisor_module, "_start_control_watcher") as watcher,
                _mock_popen() as popen,
            ):
                result = supervisor_module.main(
                    (str(status_path), "--", sys.executable, "-c", "pass")
                )

            self.assertEqual(
                result,
                control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
            )
            self.assertEqual(
                status_path.read_text(encoding="utf-8"),
                "error StartupDeadlineExpired\n",
            )
            watcher.assert_not_called()
            popen.assert_not_called()

    def test_short_lived_target_does_not_require_parent_identity_probe(self) -> None:
        with (
            patch(
                "hbrowser.gallery.browser.process.os.getpgid",
                side_effect=AssertionError("parent must trust the READY receipt"),
            ),
            patch(
                "hbrowser.gallery.browser.process.os.getsid",
                side_effect=AssertionError("parent must trust the READY receipt"),
            ),
        ):
            process = process_module.start_owned_process(
                sys.executable,
                ["-c", "pass"],
            )
            self.assertEqual(process.wait(timeout=5), 0)

    def _write_startup_child_hook(self, directory: Path) -> None:
        (directory / "sitecustomize.py").write_text(
            "import os, pathlib, subprocess, sys\n"
            "child = subprocess.Popen(\n"
            "    [sys.executable, '-E', '-S', '-c', 'import time; time.sleep(30)'],\n"
            "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL)\n"
            "pathlib.Path(__file__).with_name(f'startup-{os.getpid()}').write_text(\n"
            "    str(child.pid), encoding='utf-8')\n",
            encoding="utf-8",
        )

    def _reap_startup_probe_children(self, directory: Path) -> None:
        # Old-code negative controls deliberately reveal unowned helpers. Only
        # the exact PIDs recorded by this disposable hook are cleaned here.
        for marker in directory.glob("startup-*"):
            pid = int(marker.read_text(encoding="utf-8"))
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.assertTrue(_wait_for_pid_exit(pid), f"Probe helper {pid} survived")

    def test_supervisor_and_cleanup_skip_python_startup_hooks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-startup-hooks-") as directory:
            root = Path(directory)
            self._write_startup_child_hook(root)
            with patch.dict(os.environ, {"PYTHONPATH": str(root)}):
                owner = process_module.start_owned_process(
                    sys.executable,
                    ["-E", "-S", "-c", "pass"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    self.assertEqual(owner.wait(timeout=5), 0)
                    unexpected = list(root.glob("startup-*"))
                    live = [
                        int(path.read_text(encoding="utf-8"))
                        for path in unexpected
                        if _pid_exists(int(path.read_text(encoding="utf-8")))
                    ]
                    self.assertEqual(
                        unexpected, [], f"Unowned startup children: {live}"
                    )
                finally:
                    if not owner._closed:
                        owner.shutdown(
                            graceful_timeout=0, terminate_timeout=3, kill_timeout=3
                        )
                    self._reap_startup_probe_children(root)

    def test_target_startup_hooks_keep_their_environment_and_owned_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-target-hooks-") as directory:
            root = Path(directory)
            self._write_startup_child_hook(root)
            with patch.dict(os.environ, {"PYTHONPATH": str(root)}):
                owner = process_module.start_owned_process(
                    sys.executable,
                    ["-c", "import time; time.sleep(30)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    marker = root / f"startup-{owner.target_pid}"
                    deadline = time.monotonic() + 5
                    while not marker.is_file() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(marker.is_file(), "Target lost its Python config")
                    helper_pid = int(marker.read_text(encoding="utf-8"))
                    self.assertEqual(os.getpgid(helper_pid), owner.target_pid)
                    owner.shutdown(
                        graceful_timeout=0, terminate_timeout=3, kill_timeout=3
                    )
                    self.assertTrue(_wait_for_pid_exit(helper_pid))
                    self.assertEqual(list(root.glob("startup-*")), [marker])
                finally:
                    if not owner._closed:
                        owner.shutdown(
                            graceful_timeout=0, terminate_timeout=3, kill_timeout=3
                        )
                    self._reap_startup_probe_children(root)

    def test_target_inherits_sanitized_supervisor_environment(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-environment-test-"
        ) as directory:
            environment_path = Path(directory) / "environment"
            script = (
                "import os,pathlib,sys;"
                "keys=('HBROWSER_PROCESS_LOG_FILE','HBROWSER_LOG_DIR',"
                "'HBROWSER_TEST_SENTINEL','AWS_SECRET_ACCESS_KEY',"
                "'BATTLE_SECRET','EH_COOKIE','HBROWSER_LOG_FORWARD_TOKEN');"
                "pathlib.Path(sys.argv[1]).write_text("
                "'|'.join(os.environ.get(key,'missing') for key in keys),"
                "encoding='utf-8')"
            )
            with patch.dict(
                os.environ,
                {
                    "HBROWSER_PROCESS_LOG_FILE": "/tmp/legacy.log",
                    "HBROWSER_LOG_DIR": "/tmp/logs",
                    "HBROWSER_TEST_SENTINEL": "retained",
                    "AWS_SECRET_ACCESS_KEY": "credential",
                    "BATTLE_SECRET": "control",
                    "EH_COOKIE": "cookie",
                    "HBROWSER_LOG_FORWARD_TOKEN": "capability",
                },
            ):
                process = process_module.start_owned_process(
                    sys.executable,
                    ["-c", script, str(environment_path)],
                )
                self.assertEqual(process.wait(timeout=5), 0)

            self.assertEqual(
                environment_path.read_text(encoding="utf-8"),
                "missing|missing|missing|missing|missing|missing|missing",
            )

    def test_unavailable_forwarding_capability_does_not_block_child_start(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-forward-fallback-test-"
        ) as directory:
            marker = Path(directory) / "business-result"
            script = (
                "import os,pathlib,sys,time;"
                "assert 'HBROWSER_LOG_FORWARD_ENDPOINT' not in os.environ;"
                "assert 'HBROWSER_LOG_FORWARD_TOKEN' not in os.environ;"
                "pathlib.Path(sys.argv[1]).write_text('completed',encoding='utf-8');"
                "time.sleep(0.2)"
            )
            with patch.object(
                process_module,
                "_forwarding_environment_for_owned_child",
                return_value={},
            ) as forwarding_environment:
                process = process_module.start_owned_process(
                    sys.executable,
                    ["-c", script, str(marker)],
                    forward_logging=True,
                )
                self.assertEqual(process.wait(timeout=5), 0)

            forwarding_environment.assert_called_once_with()
            self.assertEqual(marker.read_text(encoding="utf-8"), "completed")

    def test_forwarding_capability_reaches_only_explicitly_opted_in_child(
        self,
    ) -> None:
        capability = {
            "HBROWSER_LOG_FORWARD_ENDPOINT": "127.0.0.1:43210",
            "HBROWSER_LOG_FORWARD_TOKEN": "a" * 64,
        }
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-forward-capability-test-"
        ) as directory:
            root = Path(directory)
            ordinary_marker = root / "ordinary"
            forwarded_marker = root / "forwarded"
            script = (
                "import os,pathlib,sys,time;"
                "pathlib.Path(sys.argv[1]).write_text("
                "os.environ.get('HBROWSER_LOG_FORWARD_ENDPOINT','missing')+'|' +"
                "os.environ.get('HBROWSER_LOG_FORWARD_TOKEN','missing'),"
                "encoding='utf-8');time.sleep(0.2)"
            )
            with patch.object(
                process_module,
                "_forwarding_environment_for_owned_child",
                return_value=capability,
            ) as forwarding_environment:
                ordinary = process_module.start_owned_process(
                    sys.executable,
                    ["-c", script, str(ordinary_marker)],
                )
                forwarded = process_module.start_owned_process(
                    sys.executable,
                    ["-c", script, str(forwarded_marker)],
                    environment={
                        "HBROWSER_LOG_FORWARD_ENDPOINT": "caller-value",
                        "HBROWSER_LOG_FORWARD_TOKEN": "caller-value",
                    },
                    forward_logging=True,
                )
                self.assertEqual(ordinary.wait(timeout=5), 0)
                self.assertEqual(forwarded.wait(timeout=5), 0)

            forwarding_environment.assert_called_once_with()
            self.assertEqual(
                ordinary_marker.read_text(encoding="utf-8"), "missing|missing"
            )
            self.assertEqual(
                forwarded_marker.read_text(encoding="utf-8"),
                "127.0.0.1:43210|" + ("a" * 64),
            )

    def test_supervisor_and_target_have_distinct_owned_groups(self) -> None:
        process = process_module.start_owned_process(
            sys.executable,
            ["-c", "import time; time.sleep(30)"],
        )
        try:
            self.assertEqual(os.getpgid(process.pid), process.pid)
            self.assertEqual(os.getsid(process.pid), process.pid)
            self.assertIsNotNone(process.target_pid)
            assert process.target_pid is not None
            self.assertEqual(os.getpgid(process.target_pid), process.target_pid)
            self.assertEqual(os.getsid(process.target_pid), process.pid)
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_direct_terminate_reaps_a_target_that_exits_after_its_phase(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-late-term-exit-"
        ) as directory:
            ready_path = Path(directory) / "ready"
            term_path = Path(directory) / "term"
            release_path = Path(directory) / "release"
            exited_path = Path(directory) / "exited"
            script = (
                "import pathlib,signal,sys,time;"
                "ready=pathlib.Path(sys.argv[1]);"
                "term=pathlib.Path(sys.argv[2]);"
                "release=pathlib.Path(sys.argv[3]);"
                "exited=pathlib.Path(sys.argv[4])\n"
                "def finish(*_):\n"
                " term.write_text('received',encoding='utf-8')\n"
                " while not release.is_file(): time.sleep(0.01)\n"
                " exited.write_text('clean',encoding='utf-8')\n"
                " raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM,finish);"
                "ready.write_text('ready',encoding='utf-8');"
                "time.sleep(30)"
            )
            process = process_module.start_owned_process(
                sys.executable,
                [
                    "-c",
                    script,
                    str(ready_path),
                    str(term_path),
                    str(release_path),
                    str(exited_path),
                ],
            )
            try:
                ready_deadline = time.monotonic() + 2
                while not ready_path.is_file() and time.monotonic() < ready_deadline:
                    time.sleep(0.01)
                self.assertTrue(ready_path.is_file())

                terminate_deadline = time.monotonic() + 0.5
                process.terminate(deadline=terminate_deadline)
                term_deadline = time.monotonic() + 2
                while not term_path.is_file() and time.monotonic() < term_deadline:
                    time.sleep(0.01)
                self.assertTrue(term_path.is_file())
                while time.monotonic() <= terminate_deadline:
                    time.sleep(0.01)
                release_path.write_text("release", encoding="utf-8")

                self.assertEqual(process.wait(timeout=2), 0)
                self.assertEqual(
                    exited_path.read_text(encoding="utf-8"),
                    "clean",
                )
            finally:
                if process.poll() is None:
                    process.kill(deadline=time.monotonic() + 2)
                    process.wait(timeout=2)

    def test_fresh_standalone_terminate_revision_sends_term_again(self) -> None:
        target = Mock(pid=123)
        controller = supervisor_module._ShutdownController()
        now_ns = time.monotonic_ns()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.TERMINATE,
                now_ns + 1_000_000_000,
                None,
            )
        )
        delivered_fresh_term = False

        def deliver_fresh_term(
            _: supervisor_module._ControllerState,
            __: float,
        ) -> supervisor_module._ControllerState:
            nonlocal delivered_fresh_term
            self.assertFalse(delivered_fresh_term)
            delivered_fresh_term = True
            fresh_now_ns = time.monotonic_ns()
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.TERMINATE,
                    fresh_now_ns + 1_000_000_000,
                    None,
                )
            )
            return controller.state()

        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, False, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(
                controller,
                "wait_for_change",
                side_effect=deliver_fresh_term,
            ) as wait_for_change,
            patch.object(os, "killpg") as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(
            kill_process_group.call_args_list,
            [
                call(target.pid, signal.SIGTERM),
                call(target.pid, signal.SIGTERM),
            ],
        )
        wait_for_change.assert_called_once_with(ANY, ANY)
        target.wait.assert_called_once_with(timeout=ANY)

    def test_malformed_control_frame_reports_error_after_tree_proof(self) -> None:
        process = process_module.start_owned_process(
            sys.executable,
            ["-c", "import time; time.sleep(30)"],
        )
        try:
            control_pipe = process._supervisor.stdin
            assert control_pipe is not None
            control_pipe.write(b"terminate\n")
            control_pipe.flush()

            self.assertEqual(
                process.wait(timeout=5),
                control_module.PROVEN_PROTOCOL_FAILURE_EXIT_CODE,
            )
        finally:
            if process.poll() is None:
                process.kill(deadline=time.monotonic() + 2)
                process.wait(timeout=2)

    def test_tree_cleanup_kills_descendant_without_touching_unrelated_process(
        self,
    ) -> None:
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        with tempfile.TemporaryDirectory(prefix="hbrowser-owner-test-") as directory:
            identity_path = Path(directory) / "identity"
            script = (
                "import os,pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)']);"
                "pathlib.Path(sys.argv[1]).write_text("
                "f'{os.getpid()} {child.pid}',encoding='utf-8');"
                "time.sleep(30)"
            )
            process = process_module.start_owned_process(
                sys.executable,
                ["-c", script, str(identity_path)],
            )
            try:
                deadline = time.monotonic() + 5
                while not identity_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                target_pid, descendant_pid = (
                    int(value)
                    for value in identity_path.read_text(encoding="utf-8").split()
                )
                process.terminate()
                process.wait(timeout=5)
                self.assertTrue(_wait_for_pid_exit(target_pid))
                self.assertTrue(_wait_for_pid_exit(descendant_pid))
                self.assertIsNone(unrelated.poll())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_normal_leader_exit_still_kills_lingering_descendant(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-leader-exit-test-"
        ) as directory:
            descendant_path = Path(directory) / "descendant"
            script = (
                "import pathlib,subprocess,sys;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "time.sleep(30)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8')"
            )
            process = process_module.start_owned_process(
                sys.executable,
                ["-c", script, str(descendant_path)],
            )
            try:
                deadline = time.monotonic() + 2
                while not descendant_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                descendant_pid = int(descendant_path.read_text(encoding="utf-8"))

                self.assertEqual(process.wait(timeout=5), 0)
                self.assertTrue(_wait_for_pid_exit(descendant_pid))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_target_exit_between_waitid_and_identity_probe_is_reaped(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, True, True),
            ) as target_exited,
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ) as process_group_members,
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.killpg"
            ) as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(target_exited.call_count, 3)
        process_group_members.assert_called_once_with(target.pid, deadline_ns=ANY)
        target.wait.assert_called_once_with()
        kill_process_group.assert_not_called()

    def test_target_exit_after_successful_identity_probe_is_reaped(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, True, True),
            ) as target_exited,
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ) as process_group_members,
            patch.object(os, "killpg") as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(target_exited.call_count, 3)
        process_group_members.assert_called_once_with(target.pid, deadline_ns=ANY)
        target.wait.assert_called_once_with()
        kill_process_group.assert_not_called()

    def test_target_exit_after_identity_with_descendant_cleans_group(self) -> None:
        target = Mock(pid=123)
        actions = Mock()
        target.wait.side_effect = actions.wait
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, True, True, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                side_effect=((target.pid, 456), (target.pid,)),
            ),
            patch.object(os, "killpg", side_effect=actions.killpg),
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(
            actions.mock_calls,
            [call.killpg(target.pid, signal.SIGTERM), call.wait(timeout=ANY)],
        )

    def test_target_exit_race_with_descendant_cleans_pinned_group(self) -> None:
        target = Mock(pid=123)
        actions = Mock()
        target.wait.side_effect = actions.wait
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, True, True, True),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
                side_effect=((target.pid, 456), (target.pid,)),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.killpg",
                side_effect=actions.killpg,
            ),
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(
            actions.mock_calls,
            [call.killpg(target.pid, signal.SIGTERM), call.wait(timeout=ANY)],
        )

    def test_missing_identity_after_deadline_retains_until_exit_proof(self) -> None:
        target = Mock(pid=123)
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                intent=control_module.ControlIntent.KILL,
                phase_deadline_ns=0,
                overall_deadline_ns=0,
            )
        )
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, True, True),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ) as process_group_members,
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.killpg"
            ) as kill_process_group,
            patch.object(
                controller,
                "wait_for_change",
                return_value=controller.state(),
            ) as wait,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        wait.assert_called_once_with(ANY, supervisor_module._POLL_SECONDS)
        process_group_members.assert_called_once_with(target.pid, deadline_ns=ANY)
        target.wait.assert_called_once_with()
        kill_process_group.assert_not_called()

    def test_identity_lookup_waits_for_delayed_exit_proof(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, True, True),
            ) as target_exited,
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(
                controller,
                "wait_for_change",
                return_value=controller.state(),
            ) as wait,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(target_exited.call_count, 4)
        wait.assert_called_once_with(ANY, supervisor_module._POLL_SECONDS)
        target.wait.assert_called_once_with()

    def test_reaped_identity_after_lookup_race_fails_closed(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller()
        with (
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.waitid",
                side_effect=(None, ChildProcessError),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
            ) as process_group_members,
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.killpg"
            ) as kill_process_group,
            self.assertRaisesRegex(RuntimeError, "reaped unexpectedly"),
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        process_group_members.assert_not_called()
        target.wait.assert_not_called()
        kill_process_group.assert_not_called()

    def test_session_lookup_exit_race_is_reaped(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller()
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, True, True),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getpgid",
                return_value=target.pid,
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.getsid",
                side_effect=ProcessLookupError,
            ),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch(
                "hbrowser.gallery.browser._process_supervisor.os.killpg"
            ) as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        target.wait.assert_called_once_with()
        kill_process_group.assert_not_called()

    def test_wrong_process_group_or_session_never_signals(self) -> None:
        supervisor_session = os.getsid(0)
        for process_group, session_id in ((999, supervisor_session), (123, 999)):
            with self.subTest(process_group=process_group, session_id=session_id):
                target = Mock(pid=123)
                controller = _shutdown_controller(control_module.ControlIntent.KILL)

                def session_for_pid(
                    pid: int,
                    expected_session: int = session_id,
                    target_pid: int = target.pid,
                ) -> int:
                    return expected_session if pid == target_pid else supervisor_session

                with (
                    patch.object(
                        supervisor_module,
                        "_posix_target_has_exit_receipt",
                        return_value=False,
                    ),
                    patch.object(os, "getpgid", return_value=process_group),
                    patch.object(
                        os,
                        "getsid",
                        side_effect=session_for_pid,
                    ),
                    patch.object(os, "killpg") as kill_process_group,
                    self.assertRaisesRegex(RuntimeError, "escaped"),
                ):
                    supervisor_module._terminate_posix_target(
                        target,
                        controller=controller,
                    )

                kill_process_group.assert_not_called()
                target.wait.assert_not_called()

    def test_force_request_uses_only_sigkill_after_identity_proof(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(os, "killpg") as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        kill_process_group.assert_called_once_with(target.pid, signal.SIGKILL)
        target.wait.assert_called_once_with()

    def test_denied_kill_retries_only_after_a_fresh_request(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)

        def deliver_fresh_kill(
            _: supervisor_module._ControllerState,
            __: float,
        ) -> supervisor_module._ControllerState:
            now_ns = time.monotonic_ns()
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.KILL,
                    now_ns + 1_000_000_000,
                    now_ns + 1_000_000_000,
                )
            )
            return controller.state()

        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                return_value=False,
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_try_settle_posix_target",
                side_effect=(False, True),
            ) as settle,
            patch.object(
                controller,
                "wait_for_change",
                side_effect=deliver_fresh_kill,
            ) as wait_for_change,
            patch.object(
                os,
                "killpg",
                side_effect=(PermissionError, None),
            ) as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(kill_process_group.call_count, 2)
        wait_for_change.assert_called_once_with(ANY, ANY)
        self.assertEqual(settle.call_count, 2)

    def test_absent_group_enters_proof_only_settlement(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                return_value=False,
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_try_settle_posix_target",
                return_value=True,
            ) as settle,
            patch.object(os, "killpg", side_effect=ProcessLookupError) as kill_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        kill_group.assert_called_once_with(target.pid, signal.SIGKILL)
        settle.assert_called_once_with(target)

    def test_cleanup_exception_returns_only_after_independent_proof(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)
        with (
            patch.object(
                supervisor_module,
                "_terminate_posix_target",
                side_effect=OSError("cleanup probe failed"),
            ),
            patch.object(
                supervisor_module,
                "_try_settle_posix_target",
                return_value=True,
            ) as settle,
            patch.object(supervisor_module, "_report_cleanup_failure") as report,
        ):
            cleanup_failed = supervisor_module._complete_posix_cleanup(
                target,
                controller=controller,
            )

        self.assertTrue(cleanup_failed)
        settle.assert_called_once_with(target)
        report.assert_called_once()

    def test_invalid_identity_parks_without_fallback_signalling(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)

        class Parked(Exception):
            pass

        ownership_error = supervisor_module._OwnershipProofInvalid("escaped")
        with (
            patch.object(
                supervisor_module,
                "_terminate_posix_target",
                side_effect=ownership_error,
            ),
            patch.object(
                supervisor_module,
                "_park_unresolved_posix_cleanup",
                side_effect=Parked,
            ) as park,
            patch.object(os, "killpg") as kill_group,
            self.assertRaises(Parked),
        ):
            supervisor_module._complete_posix_cleanup(
                target,
                controller=controller,
            )

        park.assert_called_once_with(controller, ownership_error)
        kill_group.assert_not_called()

    def test_expired_force_request_waits_for_a_fresh_kill_phase(self) -> None:
        target = Mock(pid=123)
        actions = Mock()
        target.wait.side_effect = actions.wait
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                0,
                0,
            )
        )

        def renew_kill(
            _: supervisor_module._ControllerState,
            __: float,
        ) -> supervisor_module._ControllerState:
            actions.control_wait()
            now_ns = time.monotonic_ns()
            controller.apply(
                control_module.ControlRequest(
                    control_module.ControlIntent.KILL,
                    now_ns + 1_000_000_000,
                    now_ns + 1_000_000_000,
                )
            )
            return controller.state()

        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, False, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(os, "killpg", side_effect=actions.killpg),
            patch.object(controller, "wait_for_change", side_effect=renew_kill),
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(
            actions.mock_calls,
            [
                call.control_wait(),
                call.killpg(target.pid, signal.SIGKILL),
                call.wait(),
            ],
        )

    def test_kill_deadline_shrink_before_signal_reenters_pending_state(self) -> None:
        target = Mock(pid=123)
        controller = _shutdown_controller(control_module.ControlIntent.KILL)
        calls = 0

        def next_kill_request(
            *_: object,
            **__: object,
        ) -> supervisor_module._ControllerState:
            nonlocal calls
            calls += 1
            if calls == 1:
                stale = controller.state()
                controller.apply(
                    control_module.ControlRequest(
                        control_module.ControlIntent.KILL,
                        0,
                        0,
                    )
                )
                return stale
            now_ns = time.monotonic_ns()
            fresh = control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                now_ns + 1_000_000_000,
                now_ns + 1_000_000_000,
            )
            controller.apply(fresh)
            return controller.state()

        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(
                supervisor_module,
                "_wait_for_posix_control_request",
                side_effect=next_kill_request,
            ),
            patch.object(os, "killpg") as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        self.assertEqual(calls, 2)
        kill_process_group.assert_called_once_with(target.pid, signal.SIGKILL)
        target.wait.assert_called_once_with()

    def test_zero_budget_force_retains_ownership_until_proof(self) -> None:
        target = Mock(pid=123)
        controller = supervisor_module._ShutdownController()
        controller.apply(
            control_module.ControlRequest(
                control_module.ControlIntent.KILL,
                0,
                0,
                allow_immediate=True,
            )
        )
        with (
            patch.object(
                supervisor_module,
                "_posix_target_has_exit_receipt",
                side_effect=(False, False, True),
            ),
            patch.object(os, "getpgid", return_value=target.pid),
            patch.object(os, "getsid", return_value=os.getsid(0)),
            patch.object(
                supervisor_module,
                "_process_group_members",
                return_value=(target.pid,),
            ),
            patch.object(os, "killpg") as kill_process_group,
        ):
            supervisor_module._terminate_posix_target(
                target,
                controller=controller,
            )

        kill_process_group.assert_called_once_with(target.pid, signal.SIGKILL)
        target.wait.assert_called_once_with()

    def test_real_child_exit_between_probe_and_identity_is_reaped_safely(
        self,
    ) -> None:
        read_descriptor, write_descriptor = os.pipe()
        target = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,sys; os.read(int(sys.argv[1]), 1)",
                str(read_descriptor),
            ],
            close_fds=True,
            pass_fds=(read_descriptor,),
            process_group=0,
        )
        os.close(read_descriptor)
        controller = _shutdown_controller()
        real_receipt = supervisor_module._posix_target_has_exit_receipt
        real_getpgid = os.getpgid
        first_probe = True

        def race_probe(process: subprocess.Popen[bytes]) -> bool:
            nonlocal first_probe
            if first_probe:
                first_probe = False
                os.write(write_descriptor, b"x")
                os.close(write_descriptor)
                return False
            return real_receipt(process)

        def identity_after_exit(pid: int) -> int:
            deadline = time.monotonic() + 2
            while not real_receipt(target) and time.monotonic() < deadline:
                time.sleep(0.01)
            return real_getpgid(pid)

        try:
            with (
                patch.object(
                    supervisor_module,
                    "_posix_target_has_exit_receipt",
                    side_effect=race_probe,
                ),
                patch.object(os, "getpgid", side_effect=identity_after_exit),
                patch.object(os, "killpg") as kill_process_group,
            ):
                supervisor_module._terminate_posix_target(
                    target,
                    controller=controller,
                )
            self.assertEqual(target.returncode, 0)
            kill_process_group.assert_not_called()
        finally:
            try:
                os.close(write_descriptor)
            except OSError:
                pass
            if target.poll() is None:
                target.kill()
                target.wait(timeout=2)

    def test_zendriver_global_launcher_is_never_modified(self) -> None:
        original = zd.util._start_process
        process = process_module.start_owned_process(
            sys.executable,
            ["-c", "import time; time.sleep(30)"],
        )
        try:
            self.assertIs(zd.util._start_process, original)
        finally:
            process.terminate()
            process.wait(timeout=5)
        self.assertIs(zd.util._start_process, original)

    def test_nonzero_supervisor_exit_keeps_private_directories_owned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-group-test-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=7,
            )
            supervisor.wait.return_value = 7
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            with self.assertRaises(process_module.ProcessOwnershipError):
                owner.wait(timeout=1)
            self.assertTrue(private_directory.is_dir())
            self.assertTrue(status_directory.is_dir())

    def test_proven_protocol_failure_releases_owned_private_directories(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-protocol-proof-test-"
        ) as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            returncode = control_module.PROVEN_PROTOCOL_FAILURE_EXIT_CODE
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=returncode,
            )
            supervisor.wait.return_value = returncode
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            self.assertEqual(owner.wait(timeout=1), returncode)
            self.assertFalse(private_directory.exists())
            self.assertFalse(status_directory.exists())

    def test_proven_cleanup_failure_releases_owned_private_directories(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-cleanup-proof-test-"
        ) as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            returncode = control_module.PROVEN_CLEANUP_FAILURE_EXIT_CODE
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=returncode,
            )
            supervisor.wait.return_value = returncode
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            self.assertEqual(owner.wait(timeout=1), returncode)
            self.assertFalse(private_directory.exists())
            self.assertFalse(status_directory.exists())

    def test_proven_cleanup_failure_releases_unbound_owner_after_ready_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-unbound-cleanup-proof-test-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            returncode = control_module.PROVEN_CLEANUP_FAILURE_EXIT_CODE
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=returncode,
            )
            supervisor.wait.return_value = returncode
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=None,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )

            self.assertEqual(owner.wait(timeout=1), returncode)
            self.assertFalse(status_directory.exists())

    def test_target_not_started_receipt_releases_owner_without_status_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-unbound-not-started-proof-test-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            returncode = control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=returncode,
            )
            supervisor.wait.return_value = returncode
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=None,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )

            self.assertEqual(owner.wait(timeout=1), returncode)
            self.assertFalse(status_directory.exists())

    def test_terminal_group_sigint_reaches_harness_but_not_owned_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-signal-test-") as directory:
            root = Path(directory)
            identity = root / "identity"
            harness_signal = root / "harness-sigint"
            target_signal = root / "target-sigint"
            stop_file = root / "stop"
            harness_script = """
import os
import pathlib
import signal
import sys
import time
from hbrowser.gallery.browser.process import start_owned_process

identity, harness_signal, target_signal, stop_file = map(pathlib.Path, sys.argv[1:])
target_script = (
    "import pathlib,signal,sys,time;"
    "signal.signal(signal.SIGINT,lambda *_:"
    "pathlib.Path(sys.argv[1]).write_text('sigint'));"
    "time.sleep(30)"
)
owner = start_owned_process(sys.executable, ["-c", target_script, str(target_signal)])
signal.signal(signal.SIGINT, lambda *_: harness_signal.write_text("sigint"))
identity.write_text(f"{owner.pid} {owner.target_pid}")
try:
    while not stop_file.exists():
        time.sleep(0.02)
finally:
    owner.terminate()
    owner.wait(timeout=5)
"""
            command = command_process.ProcessCommand(
                "terminal SIGINT runtime harness",
                (
                    sys.executable,
                    "-c",
                    _runtime_harness_source(harness_script),
                    str(identity),
                    str(harness_signal),
                    str(target_signal),
                    str(stop_file),
                ),
                Path(__file__).resolve().parents[1],
                environment=dict(os.environ),
            )
            with _owned_harness(command) as harness_owner:
                harness = harness_owner.process
                deadline = time.monotonic() + 5
                while not identity.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                owner_pid, target_pid = (
                    int(value) for value in identity.read_text(encoding="utf-8").split()
                )
                os.killpg(harness.pid, signal.SIGINT)
                deadline = time.monotonic() + 2
                while not harness_signal.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(harness_signal.is_file())
                self.assertFalse(target_signal.exists())
                self.assertTrue(_pid_exists(owner_pid))
                self.assertTrue(_pid_exists(target_pid))
                stop_file.write_text("stop", encoding="utf-8")
                self.assertEqual(harness.wait(timeout=7), 0)
                self.assertTrue(_wait_for_pid_exit(owner_pid))
                self.assertTrue(_wait_for_pid_exit(target_pid))

    def test_target_launch_failure_proves_absence_and_removes_private_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-launch-test-") as directory:
            private_directory = Path(directory) / "private"
            private_directory.mkdir()

            with self.assertRaisesRegex(
                RuntimeError,
                "could not launch target: FileNotFoundError",
            ):
                process_module.start_owned_process(
                    Path(directory) / "missing-browser",
                    [],
                    cleanup_paths=(private_directory,),
                )

            self.assertFalse(private_directory.exists())

    def test_parent_shutdown_never_signals_cached_target_process_group(self) -> None:
        process = process_module.start_owned_process(
            sys.executable,
            ["-c", "import time; time.sleep(30)"],
        )
        with patch.object(os, "killpg") as kill_process_group:
            process.terminate()
            process.wait(timeout=5)

        kill_process_group.assert_not_called()


@unittest.skipUnless(os.name == "posix", "Late READY reconciles POSIX tree proof")
class LateReadyReceiptTests(unittest.TestCase):
    @contextmanager
    def _owner(
        self, *, returncode: int = 0
    ) -> Iterator[tuple[process_module.OwnedProcess, Path, Mock]]:
        with (
            tempfile.TemporaryDirectory(
                prefix="hbrowser-late-ready-proof-"
            ) as directory,
            patch.object(atexit, "register"),
            patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: _remove_private_inline(
                    guard, deadline=deadline
                ),
            ),
        ):
            root = Path(directory)
            (root / "status").mkdir()
            (root / "private").mkdir()
            supervisor = Mock(pid=101, stdin=Mock(), stdout=None, stderr=None)
            supervisor.wait.return_value = returncode
            observation = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=None,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=root / "status",
                cleanup_paths=(root / "private",),
                observation=observation,
            )
            try:
                yield owner, root, observation
            finally:
                # These owners contain only Mock processes. Explicit fixture
                # rescue releases the pinned fd without inventing runtime proof.
                for guard in [owner._status_directory, *owner._cleanup_paths]:
                    if guard is not None:
                        guard._retire()

    def _frame(self, owner: process_module.OwnedProcess) -> bytes:
        guard = owner._status_directory
        assert guard is not None
        return control_module.ReadyReceipt(
            owner.pid, guard.device, guard.inode, 202
        ).encode()

    def test_late_ready_accepts_only_proven_supervisor_exit(self) -> None:
        for returncode in (0, control_module.PROVEN_PROTOCOL_FAILURE_EXIT_CODE):
            with (
                self.subTest(returncode=returncode),
                self._owner(returncode=returncode) as (
                    owner,
                    root,
                    observation,
                ),
            ):
                (root / "status/status").write_bytes(self._frame(owner))
                with patch.object(atexit, "unregister") as unregister:
                    self.assertEqual(owner.wait(timeout=2), returncode)
                self.assertEqual(owner.target_pid, 202)
                self.assertTrue(owner._closed)
                self.assertIsNone(owner._status_directory)
                self.assertFalse((root / "status").exists())
                self.assertFalse((root / "private").exists())
                observation.complete.assert_called_once()
                unregister.assert_called_once_with(owner._atexit_cleanup)

    def test_unknown_exit_cannot_borrow_a_valid_ready_receipt(self) -> None:
        with self._owner(returncode=7) as (owner, root, observation):
            (root / "status/status").write_bytes(self._frame(owner))
            with (
                patch.object(owner, "_read_startup_status") as read_status,
                self.assertRaises(process_module.ProcessOwnershipError),
            ):
                owner.wait(timeout=1)
            read_status.assert_not_called()
            self.assertFalse(owner._tree_reaped)
            self.assertTrue((root / "private").exists())
            observation.complete.assert_not_called()

    def test_ready_without_supervisor_reap_never_proves_cleanup(self) -> None:
        with self._owner() as (owner, root, observation):
            (root / "status/status").write_bytes(self._frame(owner))
            cast(Mock, owner._supervisor.wait).side_effect = subprocess.TimeoutExpired(
                "supervisor", 1
            )
            with (
                patch.object(owner, "_read_startup_status") as read_status,
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                owner.wait(timeout=1)
            read_status.assert_not_called()
            self.assertFalse(owner._supervisor_reaped)
            self.assertFalse(owner._tree_reaped)
            observation.complete.assert_not_called()

    def test_late_ready_past_cleanup_deadline_retains_proof_for_explicit_retry(
        self,
    ) -> None:
        with self._owner() as (owner, root, observation):
            (root / "status/status").write_bytes(self._frame(owner))
            original_read = owner._read_status_frame
            now = [100.0]

            def late_read(
                guard: process_module._PrivateDirectory,
                descriptor: int | None,
                status_path: Path,
            ) -> bytes | None:
                frame = original_read(guard, descriptor, status_path)
                now[0] = 101.0
                return frame

            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                patch.object(owner, "_read_status_frame", side_effect=late_read),
                self.assertRaises(process_module.ProcessOwnershipError),
            ):
                owner.wait(timeout=1)
            self.assertTrue(owner._supervisor_reaped)
            self.assertFalse(owner._tree_reaped)
            self.assertIsNone(owner.target_pid)
            self.assertIsNotNone(owner._status_directory)
            self.assertTrue((root / "private").exists())
            observation.complete.assert_not_called()
            self.assertEqual(owner.wait(timeout=2), 0)
            self.assertTrue(owner._closed)
            self.assertIsNone(owner._status_directory)
            observation.complete.assert_called_once()

    def test_late_ready_rejects_wrong_identity_and_malformed_receipts(self) -> None:
        for variant in (
            "missing",
            "legacy",
            "supervisor",
            "device",
            "inode",
            "negative",
            "oversized",
            "error",
            "symlink",
            "fifo",
        ):
            with (
                self.subTest(variant=variant),
                self._owner() as (owner, root, observation),
            ):
                frame = self._frame(owner)
                fields = frame[:-1].split(b" ")
                match variant:
                    case "legacy":
                        frame = b"ready 202\n"
                    case "supervisor" | "device" | "inode":
                        index = {"supervisor": 1, "device": 2, "inode": 3}[variant]
                        fields[index] = str(int(fields[index]) + 1).encode("ascii")
                        frame = b" ".join(fields) + b"\n"
                    case "negative":
                        fields[4] = b"-1"
                        frame = b" ".join(fields) + b"\n"
                    case "oversized":
                        frame += b"x" * control_module.MAX_STATUS_LINE_BYTES
                    case "error":
                        frame = b"error StartupDeadlineExpired\n"
                status = root / "status/status"
                if variant == "symlink":
                    (root / "other").write_bytes(frame)
                    status.symlink_to(root / "other")
                elif variant == "fifo":
                    os.mkfifo(status)
                elif variant != "missing":
                    status.write_bytes(frame)
                with (
                    patch.object(atexit, "unregister") as unregister,
                    self.assertRaises(process_module.ProcessOwnershipError),
                ):
                    owner.wait(timeout=1)
                self.assertFalse(owner._tree_reaped)
                self.assertIsNone(owner.target_pid)
                self.assertTrue((root / "private").is_dir())
                observation.complete.assert_not_called()
                unregister.assert_not_called()

    def test_replaced_directory_cannot_replay_the_pinned_generation(self) -> None:
        with self._owner() as (owner, root, observation):
            frame = self._frame(owner)
            (root / "status").rename(root / "original")
            (root / "status").mkdir()
            (root / "status/status").write_bytes(frame)
            with self.assertRaises(process_module.ProcessOwnershipError):
                owner.wait(timeout=1)
            observation.complete.assert_not_called()
            self.assertFalse(owner._tree_reaped)
            self.assertTrue((root / "private").exists())
            guard = owner._status_directory
            assert guard is not None
            descriptor = guard._pin.duplicate_descriptor()
            try:
                self.assertEqual(
                    os.fstat(descriptor).st_ino, (root / "original").stat().st_ino
                )
            finally:
                os.close(descriptor)

    def test_bound_target_cannot_accept_another_target_receipt(self) -> None:
        with self._owner() as (owner, root, _):
            (root / "status/status").write_bytes(self._frame(owner))
            owner.bind_target_process_group(303)
            with self.assertRaisesRegex(
                process_module.ProcessOwnershipError, "identity mismatch"
            ):
                owner._read_startup_status(
                    root / "status/status", deadline=time.monotonic() + 1
                )

    def test_late_ready_cleanup_failure_retains_then_releases_the_same_owner(
        self,
    ) -> None:
        with self._owner() as (owner, root, observation):
            (root / "status/status").write_bytes(self._frame(owner))
            cleanup_error = OSError("private cleanup blocked")
            with (
                patch.object(
                    process_module,
                    "_remove_private_directory_owned",
                    side_effect=cleanup_error,
                ),
                self.assertRaises(OSError) as raised,
            ):
                owner.wait(timeout=1)
            self.assertIs(raised.exception, cleanup_error)
            self.assertTrue(owner._tree_reaped)
            self.assertFalse(owner._closed)
            self.assertIsNotNone(owner._status_directory)
            observation.complete.assert_not_called()
            with patch.object(
                owner,
                "_read_startup_status",
                side_effect=AssertionError("tree already proven"),
            ):
                self.assertEqual(owner.wait(timeout=2), 0)
            self.assertTrue(owner._closed)
            self.assertIsNone(owner._status_directory)
            observation.complete.assert_called_once()

    def test_real_ready_timeout_reaps_target_tree_and_preserves_original_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-ready-timeout-") as directory:
            root = Path(directory)
            private = root / "private"
            private.mkdir()
            marker = root / "target-pids"
            original_read = process_module._read_supervisor_status
            captured: list[process_module.OwnedProcess] = []
            startup_error = TimeoutError("READY receipt missed startup deadline")
            deadline = time.monotonic() + 10

            def late_read(
                owner: process_module.OwnedProcess,
                status_path: Path,
                *,
                deadline: float,
            ) -> process_module._SupervisorStatus:
                captured.append(owner)
                ready = original_read(owner, status_path, deadline=deadline)
                self.assertIsInstance(ready, process_module._SupervisorReady)
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(
                    marker.exists(), "target must actually run before READY expires"
                )
                time.sleep(max(0, deadline - time.monotonic()))
                self.assertGreaterEqual(time.monotonic(), deadline)
                self.assertIsNone(owner.target_pid)
                raise startup_error

            source = (
                "import os, pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(marker)!r}).write_text("
                "f'{os.getpid()} {child.pid}')\n"
                "time.sleep(60)\n"
            )
            try:
                with (
                    patch.object(
                        process_module, "_read_supervisor_status", side_effect=late_read
                    ),
                    patch.object(
                        atexit, "unregister", wraps=atexit.unregister
                    ) as unregister,
                    self.assertRaises(TimeoutError) as raised,
                ):
                    process_module.start_owned_process(
                        sys.executable,
                        ["-c", source],
                        cleanup_paths=(private,),
                        startup_timeout=2,
                        deadline=deadline,
                    )
                self.assertIs(raised.exception, startup_error)
                owner = captured[0]
                self.assertEqual(owner._supervisor_returncode, 0)
                self.assertTrue(owner._tree_reaped)
                self.assertTrue(owner._closed)
                self.assertIsNone(owner._status_directory)
                self.assertFalse(private.exists())
                self.assertLess(time.monotonic(), deadline)
                unregister.assert_any_call(owner._atexit_cleanup)
                self.assertIsNotNone(owner._observation)
                for raw_pid in marker.read_text().split():
                    self.assertTrue(_wait_for_pid_exit(int(raw_pid)))
            finally:
                # A failed regression must still settle every real observed owner.
                for owner in captured:
                    if not owner._closed:
                        owner.shutdown(
                            graceful_timeout=0,
                            terminate_timeout=1,
                            kill_timeout=2,
                            cleanup_timeout=2,
                        )


class PrivateDirectoryCleanupSetTests(unittest.TestCase):
    def test_owned_process_shares_duplicate_paths_and_supplied_guards(self) -> None:
        for representation in ("paths", "mixed", "guards"):
            with self.subTest(representation=representation):
                with _private_directory_fixture() as directory:
                    root = Path(directory)
                    private = root / "private"
                    private.mkdir()
                    status = root / "status"
                    status.mkdir()
                    first = process_module._PrivateDirectory.capture(private)
                    second = process_module._PrivateDirectory.capture(private)
                    paths: tuple[Path | process_module._PrivateDirectory, ...]
                    if representation == "paths":
                        first._retire()
                        second._retire()
                        paths = (private, private)
                    elif representation == "mixed":
                        first._retire()
                        paths = (private, second)
                    else:
                        paths = (first, second)
                    owner = process_module.OwnedProcess(
                        Mock(),
                        target_process_group=None,
                        windows_job=None,
                        stdout_drain=None,
                        stderr_drain=None,
                        status_directory=status,
                        cleanup_paths=paths,
                    )
                    guards = [
                        guard for guard in owner._cleanup_paths if guard.path == private
                    ]
                    owner._tree_reaped = True
                    try:
                        self.assertEqual(len(guards), 2)
                        self.assertIs(
                            guards[0]._pin.authority, guards[1]._pin.authority
                        )
                        for supplied in paths:
                            if isinstance(supplied, process_module._PrivateDirectory):
                                self.assertTrue(
                                    any(guard is supplied for guard in guards)
                                )
                        owner._release_ownership(deadline=time.monotonic() + 5)
                        self.assertTrue(owner._closed)
                        self.assertTrue(all(guard._pin.retired for guard in guards))
                        self.assertFalse(private.exists())
                        self.assertFalse(status.exists())
                        private.mkdir()
                        (private / "foreign").write_text("keep")
                        for guard in guards:
                            guard.remove(deadline=time.monotonic() + 1)
                        self.assertEqual((private / "foreign").read_text(), "keep")
                    finally:
                        if not owner._closed:
                            owner._release_ownership(deadline=time.monotonic() + 5)

    def test_ancestor_cleanup_uses_verified_paths_in_both_orders(self) -> None:
        for representation in ("parent-first", "child-first", "dotdot-parent"):
            with self.subTest(representation=representation):
                with _private_directory_fixture() as directory:
                    root = Path(directory)
                    parent = root / "parent"
                    child = parent / "child"
                    child.mkdir(parents=True)
                    (parent / "data").write_text("owned")
                    status = root / "status"
                    status.mkdir()
                    parent_input = (
                        child / ".." if representation == "dotdot-parent" else parent
                    )
                    parent_guard = process_module._PrivateDirectory.capture(
                        parent_input
                    )
                    child_guard = process_module._PrivateDirectory.capture(child)
                    paths = (
                        (child_guard, parent_guard)
                        if representation == "child-first"
                        else (parent_guard, child_guard)
                    )
                    owner = process_module.OwnedProcess(
                        Mock(),
                        target_process_group=None,
                        windows_job=None,
                        stdout_drain=None,
                        stderr_drain=None,
                        status_directory=status,
                        cleanup_paths=paths,
                    )
                    owner._tree_reaped = True
                    try:
                        self.assertIs(owner._cleanup_paths[0], child_guard)
                        self.assertEqual(parent_guard.path, parent_input.absolute())
                        self.assertEqual(parent_guard.cleanup_path, parent.resolve())
                        with self.assertRaises(process_module.ProcessOwnershipError):
                            owner._release_ownership(deadline=time.monotonic() - 1)
                        self.assertIs(owner._cleanup_paths[0], child_guard)
                        self.assertFalse(parent_guard._pin.retired)
                        self.assertFalse(child_guard._pin.retired)
                        owner._release_ownership(deadline=time.monotonic() + 5)
                        self.assertTrue(owner._closed)
                        self.assertTrue(parent_guard._pin.retired)
                        self.assertTrue(child_guard._pin.retired)
                        self.assertFalse(child.exists())
                        self.assertFalse(parent.exists())
                    finally:
                        if not owner._closed:
                            owner._release_ownership(deadline=time.monotonic() + 5)

    @unittest.skipUnless(os.name == "posix", "Fixture creates a parent symlink")
    def test_parent_symlink_retarget_does_not_change_fixed_cleanup_generation(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            original = root / "original"
            foreign = root / "foreign"
            for fixture_parent in (original, foreign):
                (fixture_parent / "private" / "child").mkdir(parents=True)
            (foreign / "private" / "marker").write_text("keep")
            alias = root / "alias"
            alias.symlink_to(original, target_is_directory=True)
            parent = process_module._PrivateDirectory.capture(alias / "private")
            child = process_module._PrivateDirectory.capture(
                alias / "private" / "child"
            )
            key = process_module._private_directory_key(parent)
            alias.unlink()
            alias.symlink_to(foreign, target_is_directory=True)
            guards = process_module._normalize_private_directories((parent, child))
            self.assertIs(guards[0], child)
            self.assertEqual(process_module._private_directory_key(parent), key)
            self.assertEqual(parent.path, alias / "private")
            for guard in guards:
                guard.remove(deadline=time.monotonic() + 5)
            self.assertFalse((original / "private").exists())
            self.assertEqual((foreign / "private" / "marker").read_text(), "keep")
            self.assertTrue(parent._pin.retired)
            self.assertTrue(child._pin.retired)

    def test_start_success_and_failure_keep_duplicate_ancestor_cleanup_contract(
        self,
    ) -> None:
        for failure in ("none", "before-spawn", "partial-capture", "constructor"):
            with self.subTest(failure=failure):
                with _private_directory_fixture() as directory:
                    root = Path(directory)
                    parent = root / "parent"
                    child = parent / "child"
                    child.mkdir(parents=True)
                    first = process_module._PrivateDirectory.capture(parent)
                    second = process_module._PrivateDirectory.capture(parent)
                    child_guard = process_module._PrivateDirectory.capture(child)
                    paths: list[Path | process_module._PrivateDirectory] = [
                        first,
                        second,
                        child_guard,
                    ]
                    if failure == "partial-capture":
                        paths.append(root / "absent")
                    owner = None
                    with ExitStack() as patches:
                        if failure == "constructor":
                            patches.enter_context(
                                patch.object(
                                    process_module,
                                    "OwnedProcess",
                                    side_effect=RuntimeError("constructor failed"),
                                )
                            )
                        try:
                            if failure == "none":
                                owner = process_module.start_owned_process(
                                    sys.executable, ["-c", "pass"], cleanup_paths=paths
                                )
                                self.assertEqual(owner.wait(timeout=5), 0)
                                self.assertTrue(owner._closed)
                            else:
                                expected = (
                                    ValueError
                                    if failure == "before-spawn"
                                    else FileNotFoundError
                                    if failure == "partial-capture"
                                    else RuntimeError
                                )
                                with self.assertRaises(expected):
                                    process_module.start_owned_process(
                                        sys.executable,
                                        ["-c", "pass"],
                                        cleanup_paths=paths,
                                        environment={"INVALID=KEY": "value"}
                                        if failure == "before-spawn"
                                        else None,
                                    )
                            self.assertTrue(first._pin.retired)
                            self.assertTrue(second._pin.retired)
                            self.assertTrue(child_guard._pin.retired)
                            self.assertFalse(parent.exists())
                            self.assertFalse(child.exists())
                            for guard in (first, second, child_guard):
                                key = process_module._private_directory_key(guard)
                                self.assertNotIn(
                                    key, process_module._PENDING_PRIVATE_DIRECTORIES
                                )
                                self.assertIsNone(
                                    process_module._active_private_directory_cleanup(
                                        guard
                                    )
                                )
                        finally:
                            if owner is not None and not owner._closed:
                                owner.shutdown(
                                    graceful_timeout=0,
                                    terminate_timeout=1,
                                    kill_timeout=2,
                                    cleanup_timeout=5,
                                )

    def test_failed_constructor_does_not_retire_supplied_shared_authority(self) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            private = root / "private"
            private.mkdir()
            status = root / "status"
            status.mkdir()
            supplied = process_module._PrivateDirectory.capture(private)
            failure = RuntimeError("atexit registration failed")
            with (
                patch.object(atexit, "register", side_effect=failure),
                self.assertRaises(RuntimeError) as raised,
            ):
                process_module.OwnedProcess(
                    Mock(),
                    target_process_group=None,
                    windows_job=None,
                    stdout_drain=None,
                    stderr_drain=None,
                    status_directory=status,
                    cleanup_paths=(private, supplied),
                )
            self.assertIs(raised.exception, failure)
            self.assertFalse(supplied._pin.retired)
            self.assertTrue(supplied._assert_identity_or_absence())
            supplied.remove(deadline=time.monotonic() + 5)
            self.assertFalse(private.exists())

    def test_alias_normalization_keeps_each_claim_lock_and_prior_worker(self) -> None:
        with _private_directory_fixture() as directory:
            private = Path(directory) / "private"
            private.mkdir()
            guards = tuple(
                process_module._PrivateDirectory.capture(private) for _ in range(2)
            )
            claims = [
                process_module._register_pending_private_directory(guard)
                for guard in guards
            ]
            locks = [
                process_module._private_directory_operation_lock(guard)
                for guard in guards
            ]
            actives = [
                process_module._ActivePrivateDirectoryCleanup(guard) for guard in guards
            ]
            for active in actives:
                process_module._register_active_private_directory_cleanup(active)
            normalized = process_module._normalize_private_directories(guards)
            for index, guard in enumerate(guards):
                self.assertIs(normalized[index], guard)
                self.assertIs(
                    process_module._register_pending_private_directory(guard),
                    claims[index],
                )
                self.assertIs(
                    process_module._private_directory_operation_lock(guard),
                    locks[index],
                )
                self.assertIs(
                    process_module._active_private_directory_cleanup(guard),
                    actives[index],
                )
            process_module._settle_private_directory_workers(
                normalized, deadline=time.monotonic() + 2
            )
            for guard in guards:
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
            for guard in normalized:
                guard.remove(deadline=time.monotonic() + 5)
            self.assertFalse(private.exists())
            for guard in guards:
                self.assertTrue(guard._pin.retired)
                self.assertNotIn(
                    process_module._private_directory_key(guard),
                    process_module._PENDING_PRIVATE_DIRECTORIES,
                )

    def test_ambiguous_redundant_close_keeps_failed_wrapper_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = process_module._PrivateDirectory.capture(path)
            second = process_module._PrivateDirectory.capture(path)
            descriptor = second._pin._descriptor
            assert descriptor is not None
            failure = OSError("redundant close outcome unknown")
            try:
                with (
                    patch.object(os, "close", side_effect=failure) as close,
                    self.assertRaises(OSError) as raised,
                ):
                    process_module._normalize_private_directories((first, second))
                self.assertIs(raised.exception, failure)
                close.assert_called_once_with(descriptor)
                self.assertIsNot(first._pin.authority, second._pin.authority)
                self.assertFalse(second._pin.retired)
                self.assertFalse(first._pin.retired)
                with self.assertRaises(identity_module.DirectoryIdentityError):
                    second._pin.duplicate_descriptor()
                for guard in (first, second):
                    self.assertIn(
                        process_module._private_directory_key(guard),
                        process_module._PENDING_PRIVATE_DIRECTORIES,
                    )
            finally:
                # The injected close did not touch this exact fixture-owned fd;
                # no worker was admitted. This is rescue after the retention proof.
                os.close(descriptor)
                first._retire()
                for guard in (first, second):
                    self.assertIsNone(
                        process_module._active_private_directory_cleanup(guard)
                    )
                    process_module._release_pending_private_directory(guard)


class PrivateDirectoryNativeIdentityTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "Darwin capture-time vnode deletion")
    def test_native_delete_event_is_distinct_from_rename_and_stays_proven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            pin = identity_module.DirectoryPin.capture(path)
            watch = pin._deletion_watch
            assert watch is not None
            descriptor = pin._descriptor
            assert descriptor is not None
            try:
                self.assertFalse(os.get_inheritable(watch.fileno()))
                self.assertFalse(pin.deleted())
                renamed = path.with_name("renamed")
                path.rename(renamed)
                self.assertFalse(pin.deleted())
                self.assertEqual(os.fstat(descriptor).st_ino, pin.inode)
                unrelated = SimpleNamespace(
                    ident=descriptor,
                    filter=select.KQ_FILTER_VNODE,
                    flags=0,
                    fflags=select.KQ_NOTE_LINK,
                )
                with patch.object(
                    pin,
                    "_deletion_watch",
                    Mock(control=Mock(return_value=[unrelated])),
                ):
                    self.assertFalse(pin.deleted())
                renamed.rmdir()
                # This exercises real NOTE_DELETE, including APFS where the
                # still-open directory can retain a nonzero st_nlink.
                self.assertTrue(pin.deleted())
                with patch.object(pin, "_deletion_watch", Mock()) as unread:
                    self.assertTrue(pin.deleted())
                    unread.control.assert_not_called()
            finally:
                pin.retire()
            self.assertTrue(watch.closed)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin independent vnode watches")
    def test_native_deletion_survives_duplicate_and_alias_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            pin = identity_module.DirectoryPin.capture(path)
            other = identity_module.DirectoryPin.inherit(pin.duplicate_descriptor())
            first_watch, second_watch = pin._deletion_watch, other._deletion_watch
            assert first_watch is not None and second_watch is not None
            try:
                self.assertNotEqual(first_watch.fileno(), second_watch.fileno())
                self.assertFalse(os.get_inheritable(second_watch.fileno()))
                path.rmdir()
                self.assertTrue(other.deleted())
                pin.share_authority(other)
                self.assertTrue(second_watch.closed)
                self.assertFalse(first_watch.closed)
                self.assertIs(other.authority, pin)
                self.assertTrue(other.deleted())
                self.assertTrue(pin.deleted())
            finally:
                other.retire()
                pin.retire()
            self.assertTrue(first_watch.closed)

    @unittest.skipUnless(
        sys.platform == "darwin", "Darwin late registration is not proof"
    )
    def test_late_native_registration_cannot_reconstruct_a_missing_delete_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            pin = identity_module.DirectoryPin.capture(path)
            descriptor = pin.duplicate_descriptor()
            late = None
            try:
                path.rmdir()
                self.assertTrue(pin.deleted())
                late = identity_module.DirectoryPin.inherit(descriptor)
                descriptor = -1
                self.assertFalse(late.deleted())
                guard = process_module._PrivateDirectory.from_pin(
                    path, late, platform_name="posix"
                )
                with self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "native deletion proof"
                ):
                    guard._assert_identity_or_absence()
                self.assertFalse(late.retired)
                self.assertTrue(pin.deleted())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if late is not None:
                    late.retire()
                pin.retire()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin event error boundary")
    def test_native_watch_errors_and_revoke_remain_unproven(self) -> None:
        for fault in (
            "read",
            "revoke",
            "revoke-delete",
            "wrong-descriptor",
            "event-error",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                pin = identity_module.DirectoryPin.capture(Path(directory))
                watch = pin._deletion_watch
                assert watch is not None
                descriptor = pin._descriptor
                assert descriptor is not None
                fake = Mock(wraps=watch)
                event = SimpleNamespace(
                    ident=descriptor + (1 if fault == "wrong-descriptor" else 0),
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ERROR if fault == "event-error" else 0,
                    fflags=(select.KQ_NOTE_REVOKE if fault.startswith("revoke") else 0)
                    | (select.KQ_NOTE_DELETE if fault != "revoke" else 0),
                )
                if fault == "read":
                    fake.control.side_effect = OSError("vnode watch read failed")
                else:
                    fake.control.return_value = [event]
                try:
                    with patch.object(pin, "_deletion_watch", fake):
                        with self.assertRaises(identity_module.DirectoryIdentityError):
                            pin.deleted()
                        fake.control.assert_called_once_with(None, 1, 0)
                    # Consuming a bad event cannot turn the next empty poll into
                    # authority to adopt a path or silently retry the failure.
                    with self.assertRaisesRegex(
                        identity_module.DirectoryIdentityError, "unproven"
                    ):
                        pin.deleted()
                    self.assertFalse(pin.retired)
                finally:
                    pin.retire()
                self.assertTrue(watch.closed)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin acquisition cleanup")
    def test_native_watch_registration_failure_closes_queue_and_descriptor(
        self,
    ) -> None:
        for fault in ("create", "inheritability", "register"):
            with self.subTest(fault=fault):
                self._assert_native_watch_registration_failure(fault)

    def _assert_native_watch_registration_failure(self, fault: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            acquired: list[int] = []
            queues: list[Any] = []
            original_open, original_queue = os.open, select.kqueue
            original_inheritable = os.set_inheritable
            failure = OSError(f"watch {fault} failed")

            def open_directory(*args: Any, **kwargs: Any) -> int:
                descriptor = original_open(*args, **kwargs)
                acquired.append(descriptor)
                return descriptor

            def create_queue() -> Any:
                if fault == "create":
                    raise failure
                queue = original_queue()
                queues.append(queue)
                wrapped = Mock(wraps=queue)
                if fault == "register":
                    wrapped.control.side_effect = failure
                return wrapped

            def inheritable(descriptor: int, value: bool) -> None:
                if (
                    fault == "inheritability"
                    and queues
                    and descriptor == queues[0].fileno()
                ):
                    raise failure
                original_inheritable(descriptor, value)

            with (
                patch.object(os, "open", side_effect=open_directory),
                patch.object(select, "kqueue", side_effect=create_queue),
                patch.object(os, "set_inheritable", side_effect=inheritable),
                self.assertRaises(OSError) as raised,
            ):
                identity_module.DirectoryPin.capture(Path(directory))
            self.assertIs(raised.exception, failure)
            self.assertEqual(len(acquired), 1)
            with self.assertRaises(OSError):
                os.fstat(acquired[0])
            self.assertTrue(all(queue.closed for queue in queues))
            replacement = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                self.assertEqual(replacement, acquired[0])
                failure.__traceback__ = None
                gc.collect()
                self.assertTrue(stat.S_ISDIR(os.fstat(replacement).st_mode))
            finally:
                os.close(replacement)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin detached close ownership")
    def test_native_watch_close_failure_does_not_skip_or_repeat_other_closes(
        self,
    ) -> None:
        for operation in ("retire", "share"):
            with self.subTest(operation=operation):
                self._assert_native_watch_close_failure(operation)

    def _assert_native_watch_close_failure(self, operation: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            authority = identity_module.DirectoryPin.capture(Path(directory))
            pin = (
                identity_module.DirectoryPin.inherit(authority.duplicate_descriptor())
                if operation == "share"
                else authority
            )
            watch = pin._deletion_watch
            descriptor = pin._descriptor
            assert watch is not None and descriptor is not None
            failure = OSError("queue close outcome is unknown")
            wrapped = Mock(wraps=watch)

            def close_after_effect() -> None:
                watch.close()
                raise failure

            wrapped.close.side_effect = close_after_effect
            pin._deletion_watch = wrapped
            replacement = None
            try:
                with self.assertRaises(OSError) as raised:
                    if operation == "share":
                        authority.share_authority(pin)
                    else:
                        pin.retire()
                self.assertIs(raised.exception, failure)
                self.assertTrue(watch.closed)
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                self.assertFalse(pin.retired)
                self.assertIs(pin.authority, pin)
                replacement = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                self.assertEqual(replacement, descriptor)
                with self.assertRaises(identity_module.DirectoryIdentityError):
                    pin.retire()
                pin.__del__()
                wrapped.close.assert_called_once_with()
                self.assertTrue(stat.S_ISDIR(os.fstat(replacement).st_mode))
                if operation == "share":
                    self.assertFalse(authority.retired)
                    self.assertFalse(authority.deleted())
            finally:
                if replacement is not None:
                    os.close(replacement)
                if operation == "share":
                    authority.retire()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin completed-root handoff")
    def test_retry_transfers_prior_native_delete_proof_to_late_cleanup_worker(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            claim = process_module._register_pending_private_directory(guard)
            claim.path.mkdir()
            namespace = process_module._PrivateDirectory.capture(claim.path)
            claim.namespace = namespace
            guard._claim_inline(
                namespace, confirmed=False, deadline=time.monotonic() + 2
            )
            claim.confirmed = True
            claim.started = True
            (namespace.path / "tree").rmdir()
            self.assertTrue(guard._pin.deleted())
            original_spawn = process_module._spawn_private_directory_cleanup
            with patch.object(
                process_module, "_spawn_private_directory_cleanup", wraps=original_spawn
            ) as spawned:
                guard.remove(deadline=time.monotonic() + 5)
            spawned.assert_called_once()
            self.assertTrue(guard._pin.retired)
            self.assertTrue(namespace._pin.retired)
            self.assertFalse(namespace.path.exists())
            self.assertNotIn(
                process_module._private_directory_key(guard),
                process_module._PENDING_PRIVATE_DIRECTORIES,
            )

    def test_renamed_live_generation_keeps_its_exact_cleanup_claim(self) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            renamed = Path(directory) / "renamed"
            path.mkdir()
            (path / "secret").write_bytes(b"private material")
            guard = process_module._PrivateDirectory.capture(path)
            key = process_module._private_directory_key(guard)
            path.rename(renamed)
            try:
                with self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "native deletion proof"
                ):
                    guard.remove(deadline=time.monotonic() + 3)
                self.assertEqual((renamed / "secret").read_bytes(), b"private material")
                self.assertFalse(guard._pin.retired)
                self.assertFalse(guard._pin.deleted())
                claim = process_module._PENDING_PRIVATE_DIRECTORIES[key]
                self.assertIs(claim.guard, guard)
                self.assertFalse(claim.started)
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
            finally:
                # Restore this test's exact generation; no path is recaptured.
                renamed.rename(path)
                guard.remove(deadline=time.monotonic() + 5)
            self.assertTrue(guard._pin.retired)
            self.assertNotIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)
            self.assertFalse(path.exists())

    def test_renamed_claim_or_namespace_cannot_supply_deletion_proof(self) -> None:
        for moved in ("root", "namespace", "root-and-deleted-namespace"):
            with self.subTest(moved=moved), _private_directory_fixture() as directory:
                base = Path(directory)
                path = base / "private"
                path.mkdir()
                (path / "secret").write_bytes(b"retained")
                guard = process_module._PrivateDirectory.capture(path)
                with _claimed_namespace(guard) as namespace:
                    claim = process_module._PrivateDirectoryClaim(
                        guard, namespace.path, namespace, confirmed=True, started=True
                    )
                    displaced = base / "displaced"
                    retained_root = base / "retained-root"
                    root_moved = False
                    namespace_moved = False
                    try:
                        if moved == "namespace":
                            # Windows rejects ancestor renames with an open
                            # descendant. Keep both pins alive, but first move
                            # the root out so the namespace rename is possible.
                            (namespace.path / "tree").rename(retained_root)
                            root_moved = True
                            namespace.path.rename(displaced)
                            namespace_moved = True
                        else:
                            (namespace.path / "tree").rename(displaced)
                            if moved == "root-and-deleted-namespace":
                                namespace.path.rmdir()
                        with self.assertRaises(
                            process_module.ProcessOwnershipError
                        ) as raised:
                            if moved == "root":
                                guard._remove_inline(
                                    namespace=namespace, deadline=time.monotonic() + 2
                                )
                            else:
                                claim.complete()
                        root_path = retained_root if moved == "namespace" else displaced
                        self.assertEqual(
                            (root_path / "secret").read_bytes(), b"retained"
                        )
                        metadata = root_path.stat()
                        self.assertEqual(
                            (metadata.st_dev, metadata.st_ino),
                            (guard.device, guard.inode),
                        )
                        self.assertFalse(guard._pin.retired)
                        self.assertFalse(guard._pin.deleted())
                        if moved == "namespace":
                            self.assertIn(
                                "native deletion proof", str(raised.exception)
                            )
                            metadata = displaced.stat()
                            self.assertEqual(
                                (metadata.st_dev, metadata.st_ino),
                                (namespace.device, namespace.inode),
                            )
                            self.assertFalse(namespace._pin.retired)
                            self.assertFalse(namespace._pin.deleted())
                    finally:
                        if moved == "namespace":
                            if namespace_moved:
                                displaced.rename(namespace.path)
                            if root_moved:
                                retained_root.rename(namespace.path / "tree")
                        elif moved == "root":
                            displaced.rename(namespace.path / "tree")
                        # A genuinely removed namespace cannot be recaptured as
                        # its old identity. The fixture owns the displaced tree.

    @unittest.skipUnless(os.name == "posix", "POSIX unlinked directory pin")
    def test_unlinked_generation_cannot_reclaim_reused_file_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "private"
            path.mkdir()
            original_metadata = path.lstat()
            original_identity = (original_metadata.st_dev, original_metadata.st_ino)
            actual_open = os.open
            actual_fstat = os.fstat
            actual_lstat = Path.lstat
            acquired: list[int] = []

            def capture_open(*args: Any, **kwargs: Any) -> int:
                descriptor = actual_open(*args, **kwargs)
                metadata = actual_fstat(descriptor)
                if (metadata.st_dev, metadata.st_ino) == original_identity:
                    acquired.append(descriptor)
                return descriptor

            def still_pinned() -> bool:
                for descriptor in acquired:
                    try:
                        metadata = actual_fstat(descriptor)
                    except OSError:
                        continue
                    if (metadata.st_dev, metadata.st_ino) == original_identity:
                        return True
                return False

            with patch.object(os, "open", side_effect=capture_open):
                guard = process_module._PrivateDirectory.capture(path)
            path.rmdir()
            path.mkdir()
            foreign = path / "foreign"
            foreign.write_bytes(b"preserve")
            held_during_replacement = still_pinned()
            if held_during_replacement:
                self.assertNotEqual(path.lstat().st_ino, original_metadata.st_ino)

            def recycled_numbers(
                target: Path, *args: Any, **kwargs: Any
            ) -> os.stat_result:
                # Recycling is possible only after every capture fd for the old
                # object has gone. A live candidate pin receives real metadata;
                # old stat-only code gets the reachable recycled-number model.
                if target == path and not still_pinned():
                    return original_metadata
                return actual_lstat(target, *args, **kwargs)

            refused = False
            try:
                with (
                    patch.object(Path, "lstat", recycled_numbers),
                    patch.object(
                        process_module,
                        "_spawn_private_directory_cleanup",
                        side_effect=AssertionError("recycled identity admitted"),
                    ) as spawn,
                ):
                    try:
                        guard.remove(deadline=time.monotonic() + 2)
                    except process_module.ProcessOwnershipError as error:
                        self.assertIn("directory identity changed", str(error))
                        refused = True
                spawn.assert_not_called()
                self.assertTrue(held_during_replacement)
                self.assertEqual(foreign.read_bytes(), b"preserve")
                key = process_module._private_directory_key(guard)
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
                if refused:
                    # APFS may retain nlink=2 after rmdir. Refusal preserves the
                    # exact pin and pending claim; it is not successful cleanup.
                    self.assertTrue(still_pinned())
                    self.assertIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)
                else:
                    self.assertFalse(still_pinned())
                    self.assertNotIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)
                    guard.remove(deadline=time.monotonic() + 2)
                    self.assertEqual(foreign.read_bytes(), b"preserve")
            finally:
                process_module._release_pending_private_directory(guard)
                # The test itself removed the empty original and no worker was
                # admitted. Release only that fixture authority after the oracle.
                release_pin = getattr(guard, "_retire", None)
                if callable(release_pin):
                    release_pin()

    @unittest.skipUnless(os.name == "posix", "POSIX native directory descriptors")
    def test_native_pin_survives_unlink_and_is_not_inherited_by_default(self) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            descriptor = guard._pin.duplicate_descriptor()
            try:
                self.assertTrue(guard._assert_identity_or_absence())
                self.assertFalse(os.get_inheritable(descriptor))
                path.rmdir()
                path.mkdir()
                foreign = path / "foreign"
                foreign.write_bytes(b"preserve")
                metadata = os.fstat(descriptor)
                self.assertEqual(
                    (metadata.st_dev, metadata.st_ino), (guard.device, guard.inode)
                )
                self.assertNotEqual(path.stat().st_ino, guard.inode)
                try:
                    self.assertFalse(guard._assert_identity_or_absence())
                except process_module.ProcessOwnershipError as error:
                    self.assertIn("directory identity changed", str(error))
                self.assertFalse(guard._pin.retired)
                self.assertEqual(foreign.read_bytes(), b"preserve")
            finally:
                os.close(descriptor)
                guard._retire()

    def test_capture_failure_closes_only_its_acquired_descriptor(self) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory)
            pin = identity_module.DirectoryPin.capture(path)
            descriptor = pin.duplicate_descriptor()
            failure = OSError("fstat failed")
            with (
                patch.object(os, "fstat", side_effect=failure),
                self.assertRaises(OSError) as raised,
            ):
                identity_module.DirectoryPin._adopt_descriptor(descriptor)
            self.assertIs(raised.exception, failure)
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            self.assertFalse(pin.retired)
            pin.retire()

    @unittest.skipUnless(os.name == "posix", "Host descriptor failure injection")
    def test_failed_pin_construction_closes_handoff_before_traceback_release(
        self,
    ) -> None:
        for entry in ("capture", "inherit", "receive", "from-worker"):
            for phase in ("allocate", "lock"):
                with self.subTest(entry=entry, phase=phase):
                    self._assert_failed_pin_construction(entry, phase)

    def _assert_failed_pin_construction(self, entry: str, phase: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pin = identity_module.DirectoryPin.capture(path)
            acquired: list[int] = []
            original_open = os.open
            failure = MemoryError(f"{entry} {phase} failed")
            caught: BaseException | None = None
            replacement: int | None = None

            def acquire(*args: Any, **kwargs: Any) -> int:
                descriptor = original_open(*args, **kwargs)
                acquired.append(descriptor)
                return descriptor

            try:
                with ExitStack() as stack:
                    close = stack.enter_context(
                        patch.object(os, "close", wraps=os.close)
                    )
                    if entry == "capture":
                        stack.enter_context(
                            patch.object(os, "open", side_effect=acquire)
                        )
                        invoke = partial(identity_module.DirectoryPin.capture, path)
                    else:
                        descriptor = pin.duplicate_descriptor()
                        acquired.append(descriptor)
                        if entry == "inherit":
                            invoke = partial(
                                identity_module.DirectoryPin.inherit, descriptor
                            )
                        elif entry == "receive":
                            channel = Mock()
                            channel.recvmsg.return_value = (
                                b"P",
                                [
                                    (
                                        socket.SOL_SOCKET,
                                        socket.SCM_RIGHTS,
                                        array.array("i", [descriptor]).tobytes(),
                                    )
                                ],
                                0,
                                None,
                            )
                            invoke = partial(identity_module.receive_pin, channel)
                        else:
                            api = Mock()
                            api.duplicate.return_value = 123
                            stack.enter_context(
                                patch.object(
                                    identity_module,
                                    "os",
                                    SimpleNamespace(**dict(vars(os), name="nt")),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    identity_module,
                                    "_WindowsDirectoryApi",
                                    return_value=api,
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    identity_module,
                                    "_windows_descriptor",
                                    return_value=descriptor,
                                )
                            )
                            invoke = partial(
                                identity_module.DirectoryPin.from_worker, 123, 456
                            )
                    if phase == "allocate":
                        stack.enter_context(
                            patch.object(
                                identity_module.DirectoryPin,
                                "__new__",
                                side_effect=failure,
                            )
                        )
                    else:
                        stack.enter_context(
                            patch.object(threading, "RLock", side_effect=failure)
                        )
                    try:
                        invoke()
                    except MemoryError as error:
                        caught = error
                    self.assertIs(caught, failure)
                    self.assertIsNotNone(failure.__traceback__)
                    self.assertEqual(len(acquired), 1)
                    close.assert_called_once_with(acquired[0])
                # Keep the failing constructor's traceback alive while checking
                # immediate closure, then recycle that exact test-owned fd.
                with self.assertRaises(OSError):
                    os.fstat(acquired[0])
                replacement = pin.duplicate_descriptor()
                self.assertEqual(replacement, acquired[0])
                failure.__traceback__ = None
                caught = None
                gc.collect()
                self.assertEqual(os.fstat(replacement).st_ino, pin.inode)
                self.assertFalse(pin.retired)
            finally:
                if replacement is not None:
                    os.close(replacement)
                pin.retire()

    def test_windows_conversion_import_failure_closes_already_acquired_handle(
        self,
    ) -> None:
        api = Mock()
        with (
            patch.dict(sys.modules, {"msvcrt": None}),
            self.assertRaises(ModuleNotFoundError),
        ):
            identity_module._windows_descriptor(123, api)
        api.close.assert_called_once_with(123)

    def test_windows_api_preparation_precedes_native_acquisition(self) -> None:
        failure = RuntimeError("Windows API preparation failed")
        windows_os = SimpleNamespace(**dict(vars(os), name="nt"))
        with (
            patch.object(identity_module, "os", windows_os),
            patch.object(identity_module, "_WindowsDirectoryApi", side_effect=failure),
            patch.object(identity_module, "_windows_descriptor") as convert,
        ):
            attempts: tuple[Callable[[], identity_module.DirectoryPin], ...] = (
                partial(identity_module.DirectoryPin.capture, Path("private")),
                partial(identity_module.DirectoryPin.from_worker, 123, 456),
            )
            for invoke in attempts:
                with (
                    self.subTest(invoke=invoke),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    invoke()
                self.assertIs(raised.exception, failure)
            convert.assert_not_called()

    def test_capture_rejects_directory_replacement_after_native_open(self) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            path = root / "private"
            path.mkdir()
            original_capture = identity_module.DirectoryPin.capture
            opened: list[identity_module.DirectoryPin] = []

            def replace(target: Path) -> identity_module.DirectoryPin:
                pin = original_capture(target)
                opened.append(pin)
                target.rename(root / "original")
                target.mkdir()
                (target / "foreign").write_text("keep")
                return pin

            with (
                patch.object(
                    identity_module.DirectoryPin, "capture", side_effect=replace
                ),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "identity changed"
                ),
            ):
                process_module._PrivateDirectory.capture(path)
            self.assertTrue(opened[0].retired)
            self.assertEqual((path / "foreign").read_text(), "keep")
            self.assertTrue((root / "original").is_dir())

    def test_claim_alias_and_cleanup_retry_share_one_native_generation(self) -> None:
        with _private_directory_fixture(delete=False) as directory:
            root = Path(directory)
            path = root / "private"
            path.mkdir()
            (path / "owned").write_text("retained until cleanup")
            guard = process_module._PrivateDirectory.capture(path)
            state = process_module._register_pending_private_directory(guard)
            state.path.mkdir(mode=0o700)
            state.namespace = process_module._PrivateDirectory.capture(state.path)
            try:
                guard._claim_inline(
                    state.namespace, confirmed=False, deadline=time.monotonic() + 2
                )
                state.confirmed = True
                state.started = True
                claimed = guard._claimed_directory(state.namespace)
                self.assertIs(claimed._pin, guard._pin)
                self.assertTrue(claimed._assert_identity_or_absence())
                with self.assertRaises(process_module.ProcessOwnershipError):
                    guard._assert_identity_or_absence()
                self.assertFalse(guard._pin.deleted())
                self.assertEqual(
                    (claimed.path / "owned").read_text(), "retained until cleanup"
                )
                with self.assertRaises(process_module.ProcessOwnershipError):
                    guard.remove(deadline=time.monotonic() - 1)
                self.assertFalse(guard._pin.retired)
                self.assertFalse(state.namespace._pin.retired)
                self.assertIs(
                    process_module._PENDING_PRIVATE_DIRECTORIES[
                        process_module._private_directory_key(guard)
                    ],
                    state,
                )
                self.assertEqual(
                    (claimed.path / "owned").read_text(), "retained until cleanup"
                )
                guard.remove(deadline=time.monotonic() + 3)
                self.assertTrue(guard._pin.retired)
                self.assertTrue(claimed._pin.retired)
                self.assertTrue(state.namespace._pin.retired)
                self.assertFalse(state.path.exists())
                self.assertNotIn(
                    process_module._private_directory_key(guard),
                    process_module._PENDING_PRIVATE_DIRECTORIES,
                )
                path.mkdir()
                (path / "foreign").write_text("keep")
                guard.remove(deadline=time.monotonic() + 1)
                self.assertEqual((path / "foreign").read_text(), "keep")
            finally:
                _release_private_fixture(guard, root)

    @unittest.skipUnless(os.name == "posix", "SCM_RIGHTS is POSIX-only")
    def test_namespace_transfer_pins_identity_after_sender_releases_handles(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "namespace"
            path.mkdir()
            pin = identity_module.DirectoryPin.capture(path)
            sender, receiver = socket.socketpair(type=socket.SOCK_DGRAM)
            received = None
            try:
                identity_module.send_pin(sender, pin)
                sender.close()
                pin.retire()
                path.rmdir()
                path.mkdir()
                marker = path / "foreign"
                marker.write_bytes(b"preserve")
                received = identity_module.receive_pin(receiver)
                self.assertEqual(
                    (received.device, received.inode), (pin.device, pin.inode)
                )
                self.assertNotEqual(path.stat().st_ino, pin.inode)
                duplicate = received.duplicate_descriptor()
                try:
                    metadata = os.fstat(duplicate)
                    self.assertEqual(
                        (metadata.st_dev, metadata.st_ino), (pin.device, pin.inode)
                    )
                    self.assertFalse(os.get_inheritable(duplicate))
                    self.assertFalse(received.retired)
                finally:
                    os.close(duplicate)
                self.assertEqual(marker.read_bytes(), b"preserve")
                received.retire()
                self.assertTrue(received.retired)
                with self.assertRaises(identity_module.DirectoryIdentityError):
                    received.duplicate_descriptor()
            finally:
                sender.close()
                receiver.close()
                pin.retire()
                if received is not None:
                    received.retire()

    @unittest.skipUnless(os.name == "posix", "SCM_RIGHTS is POSIX-only")
    def test_malformed_native_transfer_closes_every_received_descriptor(self) -> None:
        with _private_directory_fixture() as directory:
            pin = identity_module.DirectoryPin.capture(Path(directory))
            for data, flags in ((b"X", 0), (b"P", socket.MSG_CTRUNC)):
                with self.subTest(data=data, flags=flags):
                    duplicate = pin.duplicate_descriptor()
                    payload = array.array("i", [duplicate]).tobytes()
                    channel = Mock()
                    channel.recvmsg.return_value = (
                        data,
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, payload)],
                        flags,
                        None,
                    )
                    with self.assertRaises(identity_module.DirectoryIdentityError):
                        identity_module.receive_pin(channel)
                    with self.assertRaises(OSError):
                        os.fstat(duplicate)
            pin.retire()

    def test_constructor_failure_releases_new_pins_but_not_supplied_authority(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            status = root / "status"
            private = root / "private"
            status.mkdir()
            private.mkdir()
            captured: list[process_module._PrivateDirectory] = []
            capture = process_module._PrivateDirectory.capture

            def observe(path: Path) -> process_module._PrivateDirectory:
                guard = capture(path)
                captured.append(guard)
                return guard

            failure = RuntimeError("atexit registration failed")
            with (
                patch.object(
                    process_module._PrivateDirectory, "capture", side_effect=observe
                ),
                patch.object(atexit, "register", side_effect=failure),
                self.assertRaises(RuntimeError) as raised,
            ):
                process_module.OwnedProcess(
                    Mock(),
                    target_process_group=None,
                    windows_job=None,
                    stdout_drain=None,
                    stderr_drain=None,
                    status_directory=status,
                    cleanup_paths=(private,),
                )
            self.assertIs(raised.exception, failure)
            self.assertEqual(len(captured), 2)
            self.assertTrue(all(guard._pin.retired for guard in captured))
            self.assertTrue(status.is_dir())
            self.assertTrue(private.is_dir())
            supplied = capture(status)
            try:
                with (
                    patch.object(atexit, "register", side_effect=failure),
                    self.assertRaises(RuntimeError),
                ):
                    process_module.OwnedProcess(
                        Mock(),
                        target_process_group=None,
                        windows_job=None,
                        stdout_drain=None,
                        stderr_drain=None,
                        status_directory=supplied,
                        cleanup_paths=(),
                    )
                self.assertFalse(supplied._pin.retired)
                self.assertTrue(supplied._assert_identity_or_absence())
            finally:
                supplied._retire()

    def test_ambiguous_close_never_retries_a_recyclable_descriptor(self) -> None:
        with _private_directory_fixture() as directory:
            pin = identity_module.DirectoryPin.capture(Path(directory))
            descriptor = pin._descriptor
            assert descriptor is not None
            failure = OSError("close outcome unknown")
            try:
                with patch.object(os, "close", side_effect=failure) as close:
                    with self.assertRaises(OSError) as raised:
                        pin.retire()
                    self.assertIs(raised.exception, failure)
                    self.assertFalse(pin.retired)
                    with self.assertRaises(identity_module.DirectoryIdentityError):
                        pin.retire()
                    with self.assertRaises(identity_module.DirectoryIdentityError):
                        pin.duplicate_descriptor()
                    close.assert_called_once_with(descriptor)
            finally:
                # The injected close never touched this exact test descriptor.
                os.close(descriptor)

    def test_windows_conversion_closes_handle_only_when_transfer_fails(self) -> None:
        failure = OSError("CRT conversion failed")
        convert = Mock(side_effect=failure)
        api = Mock()
        fake_crt = SimpleNamespace(open_osfhandle=convert)
        windows_os = SimpleNamespace(O_RDONLY=os.O_RDONLY, O_NOINHERIT=0x80)
        with (
            patch.dict(sys.modules, {"msvcrt": fake_crt}),
            patch.object(identity_module, "os", windows_os),
            patch.object(
                identity_module,
                "_WindowsDirectoryApi",
                side_effect=AssertionError("must reuse prepared Windows API"),
            ),
        ):
            with self.assertRaises(OSError) as raised:
                identity_module._windows_descriptor(123, api)
            self.assertIs(raised.exception, failure)
            api.close.assert_called_once_with(123)
            convert.side_effect = None
            convert.return_value = 321
            api.close.reset_mock()
            self.assertEqual(identity_module._windows_descriptor(123, api), 321)
            api.close.assert_not_called()
            convert.assert_called_with(123, os.O_RDONLY | 0x80)

    @unittest.skipUnless(os.name == "posix", "POSIX inherited descriptor validation")
    def test_worker_rejects_wrong_native_pin_and_closes_only_its_copy(self) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            pin = identity_module.DirectoryPin.capture(path)
            duplicate = pin.duplicate_descriptor()
            sender, receiver = socket.socketpair(type=socket.SOCK_DGRAM)
            channel = os.dup(sender.fileno())
            try:
                with self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "Inherited private root identity",
                ):
                    cleanup_worker_module._parse_arguments(
                        (
                            str(path),
                            str(pin.device),
                            str(pin.inode + 1),
                            "posix",
                            repr(time.monotonic() + 1),
                            str(path.parent / ".hbrowser-cleanup-test"),
                            "-1",
                            "-1",
                            "0",
                            str(duplicate),
                            "-1",
                            str(channel),
                        )
                    )
                with self.assertRaises(OSError):
                    os.fstat(duplicate)
                self.assertFalse(pin.retired)
                self.assertFalse(pin.deleted())
            finally:
                sender.close()
                receiver.close()
                pin.retire()

    def test_failed_worker_keeps_published_pins_until_process_exit(self) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            path = root / "private"
            path.mkdir()
            namespace_path = root / ".hbrowser-cleanup-test"
            namespace_path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            namespace = process_module._PrivateDirectory.capture(namespace_path)
            retained: list[identity_module.DirectoryPin] = []
            output = io.BytesIO()
            stdout = io.TextIOWrapper(output, encoding="ascii", newline="\r\n")
            failure = RuntimeError("failed after ready")
            incoming = None
            sender = receiver = None
            channel = -1
            if os.name == "posix":
                sender, receiver = socket.socketpair(type=socket.SOCK_DGRAM)
                channel = os.dup(sender.fileno())
            try:
                with (
                    patch.object(
                        cleanup_worker_module, "_PROCESS_LIFETIME_PINS", retained
                    ),
                    patch.object(
                        cleanup_worker_module,
                        "_parse_arguments",
                        return_value=(
                            guard,
                            time.monotonic() + 2,
                            namespace_path,
                            namespace,
                            False,
                            channel,
                        ),
                    ),
                    patch.object(
                        cleanup_worker_module,
                        "_read_start_gate",
                        return_value=b"start\n",
                    ),
                    patch.object(
                        cleanup_worker_module,
                        "_wait_permission",
                        side_effect=failure,
                    ),
                    patch.object(cleanup_worker_module, "_report_failure") as report,
                    patch.object(threading, "Thread"),
                    patch.object(sys, "stdin", Mock(fileno=Mock(return_value=0))),
                    patch.object(sys, "stdout", stdout),
                ):
                    self.assertEqual(cleanup_worker_module.main(()), 1)
                self.assertTrue(output.getvalue().startswith(b"ready "))
                report.assert_called_once()
                self.assertIs(report.call_args.args[1], failure)
                self.assertEqual(retained, [guard._pin, namespace._pin])
                self.assertFalse(guard._pin.retired)
                self.assertFalse(namespace._pin.retired)
                if receiver is not None:
                    incoming = identity_module.receive_pin(receiver)
                else:
                    incoming = identity_module.DirectoryPin.from_worker(
                        namespace._pin.native_handle(), -1
                    )
                self.assertEqual(
                    (incoming.device, incoming.inode),
                    (namespace.device, namespace.inode),
                )
                self.assertTrue(path.is_dir())
            finally:
                stdout.close()
                if sender is not None:
                    sender.close()
                if receiver is not None:
                    receiver.close()
                if incoming is not None:
                    incoming.retire()
                guard._retire()
                namespace._retire()

    def test_late_native_release_retains_terminal_claim_for_explicit_retry(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            path.rmdir()
            now = [0.0]
            retire = guard._pin.retire

            def late_release() -> None:
                retire()
                now[0] = 1.0

            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                patch.object(guard._pin, "retire", side_effect=late_release),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "native release completed after",
                ),
            ):
                guard.remove(deadline=1.0)
            self.assertTrue(guard._pin.retired)
            key = process_module._private_directory_key(guard)
            self.assertIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)
            with patch.object(
                process_module, "_spawn_private_directory_cleanup"
            ) as spawn:
                guard.remove(deadline=time.monotonic() + 1)
            spawn.assert_not_called()
            self.assertNotIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)

    def test_windows_identity_requires_full_file_id_without_64_bit_fallback(
        self,
    ) -> None:
        kernel = Mock()
        with patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
            api = identity_module._WindowsDirectoryApi()
        kernel.GetFileInformationByHandleEx.return_value = False
        with self.assertRaisesRegex(OSError, "full directory identity"):
            api.identity(123)
        identifier = (1 << 100) + 7

        def full_identity(_handle: int, kind: int, output: Any, _size: int) -> bool:
            self.assertEqual(kind, 18)
            output._obj.VolumeSerialNumber = 55
            output._obj.FileId[:] = identifier.to_bytes(16, "little")
            return True

        kernel.GetFileInformationByHandleEx.side_effect = full_identity
        self.assertEqual(api.identity(123), (55, identifier))

    def test_windows_deletion_proof_requires_success_and_directory_flag(self) -> None:
        kernel = Mock()
        with patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
            api = identity_module._WindowsDirectoryApi()
        kernel.GetFileInformationByHandleEx.return_value = False
        with self.assertRaisesRegex(OSError, "directory deletion"):
            api.deleted(123)

        def not_a_directory(_handle: int, kind: int, output: Any, _size: int) -> bool:
            self.assertEqual(kind, 1)
            output._obj.Directory = 0
            output._obj.DeletePending = 1
            return True

        kernel.GetFileInformationByHandleEx.side_effect = not_a_directory
        with self.assertRaisesRegex(
            identity_module.DirectoryIdentityError, "not a directory"
        ):
            api.deleted(123)

    @unittest.skipUnless(os.name == "nt", "Native Windows delete-pending contract")
    def test_windows_pin_retains_deleted_generation_until_explicit_release(
        self,
    ) -> None:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            try:
                path.rmdir()
                self.assertTrue(guard._pin.deleted())
                self.assertFalse(guard._assert_identity_or_absence())
                self.assertFalse(guard._pin.retired)
                recreated_while_pinned = True
                try:
                    path.mkdir()
                except OSError:
                    self.assertTrue(guard._pin.deleted())
                    self.assertFalse(guard._pin.retired)
                    recreated_while_pinned = False
                    guard._retire()
                    path.mkdir()
                replacement = process_module._PrivateDirectory.capture(path)
                if recreated_while_pinned:
                    self.assertNotEqual(
                        (replacement.device, replacement.inode),
                        (guard.device, guard.inode),
                    )
                    self.assertTrue(guard._pin.deleted())
                    self.assertFalse(guard._assert_identity_or_absence())
                    self.assertFalse(guard._pin.retired)
                (path / "foreign").write_text("keep")
                guard._retire()
                self.assertTrue(guard._pin.retired)
                guard.remove(deadline=time.monotonic() + 1)
                self.assertTrue(replacement._assert_identity_or_absence())
                self.assertEqual((path / "foreign").read_text(), "keep")
            finally:
                guard._retire()

    def test_windows_release_handshake_requires_both_native_deletion_proofs(
        self,
    ) -> None:
        for outcome in ("complete", "root-not-deleted", "namespace-not-deleted"):
            with self.subTest(outcome=outcome):
                self._assert_windows_release_handshake(outcome)

    def _assert_windows_release_handshake(self, outcome: str) -> None:
        with _private_directory_fixture() as directory:
            root = Path(directory)
            path = root / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            namespace_path = root / ".hbrowser-cleanup-test"
            namespace_path.mkdir()
            namespace = process_module._PrivateDirectory.capture(namespace_path)
            state = process_module._PrivateDirectoryClaim(guard, namespace_path)
            received_pin = identity_module.DirectoryPin._adopt_descriptor(
                namespace._pin.duplicate_descriptor()
            )
            read_fd, write_fd = os.pipe()
            commands: list[bytes] = []
            delete_pending = {"root": False, "namespace": False}
            handles = {
                guard._pin._descriptor: "root",
                namespace._pin._descriptor: "namespace",
                received_pin._descriptor: "namespace",
            }
            kernel = Mock()
            with patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
                api = identity_module._WindowsDirectoryApi()

            def standard_info(handle: int, kind: int, output: Any, _size: int) -> bool:
                self.assertEqual(kind, 1)
                output._obj.Directory = 1
                output._obj.DeletePending = delete_pending[handles[handle]]
                return True

            kernel.GetFileInformationByHandleEx.side_effect = standard_info

            def command(frame: bytes) -> int:
                commands.append(frame)
                if frame == b"claim\n":
                    guard._claim_inline(
                        namespace, confirmed=False, deadline=time.monotonic() + 1
                    )
                    os.write(write_fd, b"claimed\n")
                elif frame == b"remove\n":
                    if outcome != "root-not-deleted":
                        shutil.rmtree(namespace.path / "tree")
                        delete_pending["root"] = True
                    os.write(write_fd, b"removed\n")
                elif frame == b"release-root\n":
                    if outcome != "namespace-not-deleted":
                        namespace.path.rmdir()
                        delete_pending["namespace"] = True
                    os.write(write_fd, b"finished\n")
                return len(frame)

            try:
                os.write(
                    write_fd,
                    f"ready {namespace.device} {namespace.inode} 123\n".encode(),
                )
                with os.fdopen(read_fd, "rb") as output:
                    child = Mock(
                        args=["worker"], stdout=output, stdin=Mock(), _handle=456
                    )
                    child.stdin.write.side_effect = command
                    active = process_module._ActivePrivateDirectoryCleanup(
                        guard=guard, process=child
                    )
                    # Test only the protocol's release ordering on this host;
                    # the native Windows DeletePending API is exercised on CI.
                    windows_os = SimpleNamespace(**dict(vars(os), name="nt"))
                    with (
                        patch.object(process_module, "os", windows_os),
                        patch.object(identity_module, "os", windows_os),
                        patch.object(
                            identity_module, "_WindowsDirectoryApi", return_value=api
                        ),
                        patch.object(
                            identity_module,
                            "_windows_handle",
                            side_effect=lambda descriptor: descriptor,
                        ),
                        patch.object(
                            identity_module.DirectoryPin,
                            "from_worker",
                            return_value=received_pin,
                        ),
                    ):
                        if outcome == "complete":
                            process_module._acknowledge_private_directory_claim(
                                active, state, deadline=time.monotonic() + 1
                            )
                        else:
                            with self.assertRaisesRegex(
                                process_module.ProcessOwnershipError,
                                "native deletion proof",
                            ):
                                process_module._acknowledge_private_directory_claim(
                                    active, state, deadline=time.monotonic() + 1
                                )
                self.assertEqual(guard._pin.retired, outcome != "root-not-deleted")
                self.assertEqual(received_pin.retired, outcome == "complete")
                if outcome == "root-not-deleted":
                    self.assertEqual(commands, [b"claim\n", b"remove\n"])
                    self.assertTrue((namespace.path / "tree").is_dir())
                elif outcome == "namespace-not-deleted":
                    self.assertEqual(commands[-1], b"release-root\n")
                    self.assertTrue(namespace.path.is_dir())
                else:
                    self.assertEqual(
                        commands,
                        [
                            b"claim\n",
                            b"remove\n",
                            b"release-root\n",
                            b"release-namespace\n",
                        ],
                    )
            finally:
                os.close(write_fd)
                guard._retire()
                namespace._retire()
                received_pin.retire()


class PinnedPrivateFileCreationTests(unittest.TestCase):
    @contextmanager
    def _directory(self) -> Iterator[process_module._PrivateDirectory]:
        with _private_directory_fixture() as directory:
            path = Path(directory) / "private"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            try:
                yield guard
            finally:
                guard.remove(deadline=time.monotonic() + 5)

    def test_new_file_completes_partial_writes_and_is_not_inherited(self) -> None:
        with self._directory() as guard:
            original_write = os.write
            descriptors: list[int] = []

            def write(descriptor: int, data: bytes) -> int:
                self.assertFalse(os.get_inheritable(descriptor))
                descriptors.append(descriptor)
                return original_write(descriptor, data[:2])

            with patch.object(os, "write", write):
                guard.create_file("private-data", b"abcdefg")
            self.assertEqual((guard.path / "private-data").read_bytes(), b"abcdefg")
            self.assertEqual(len(descriptors), 4)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])
            if os.name == "posix":
                self.assertEqual(
                    (guard.path / "private-data").stat().st_mode & 0o777, 0o600
                )

    def test_existing_file_directory_and_symlink_are_not_overwritten(self) -> None:
        with self._directory() as guard:
            target = guard.path / "private-data"
            target.write_bytes(b"keep")
            with self.assertRaises((OSError, identity_module.DirectoryIdentityError)):
                guard._pin.create_file(target.name, b"secret")
            self.assertEqual(target.read_bytes(), b"keep")
            target.unlink()
            target.mkdir()
            with self.assertRaises((OSError, identity_module.DirectoryIdentityError)):
                guard._pin.create_file(target.name, b"secret")
            self.assertTrue(target.is_dir())
            target.rmdir()
            outside = guard.path.parent / "outside"
            outside.write_bytes(b"untouched")
            target.symlink_to(outside)
            with self.assertRaises((OSError, identity_module.DirectoryIdentityError)):
                guard._pin.create_file(target.name, b"secret")
            self.assertTrue(target.is_symlink())
            self.assertEqual(outside.read_bytes(), b"untouched")
            # Cleanup has its own no-follow policy; this test owns this link.
            target.unlink()

    def test_names_cannot_escape_the_pinned_directory(self) -> None:
        with self._directory() as guard:
            with patch.object(os, "open", side_effect=AssertionError("must not open")):
                for name in (
                    "",
                    ".",
                    "..",
                    "../x",
                    "x/y",
                    "x\\y",
                    "/x",
                    "C:x",
                    "x\0y",
                    "x.",
                    "x ",
                ):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        guard.create_file(name, b"secret")
            self.assertEqual(list(guard.path.iterdir()), [])

    def test_replacement_at_native_create_still_targets_the_original_generation(
        self,
    ) -> None:
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                self._assert_replacement_at_native_create(symlink)

    def _assert_replacement_at_native_create(self, symlink: bool) -> None:
        with self._directory() as guard:
            original = guard.path.with_name("original")
            replacement = guard.path.with_name("replacement")
            replacement.mkdir()
            (replacement / "keep").write_bytes(b"unrelated")

            def replace() -> None:
                guard.path.rename(original)
                if symlink:
                    guard.path.symlink_to(replacement, target_is_directory=True)
                else:
                    replacement.rename(guard.path)

            original_open = os.open
            original_create = identity_module._WindowsDirectoryApi.create_file

            def posix_create(*args: Any, **kwargs: Any) -> int:
                replace()
                return original_open(*args, **kwargs)

            def windows_create(
                api: identity_module._WindowsDirectoryApi, native: int, name: str
            ) -> int:
                replace()
                return original_create(api, native, name)

            try:
                with (
                    patch.object(os, "open", posix_create)
                    if os.name == "posix"
                    else patch.object(
                        identity_module._WindowsDirectoryApi,
                        "create_file",
                        windows_create,
                    )
                ):
                    guard.create_file("private-data", b"secret")
                self.assertEqual((original / "private-data").read_bytes(), b"secret")
                self.assertFalse((guard.path / "private-data").exists())
                self.assertEqual((guard.path / "keep").read_bytes(), b"unrelated")
            finally:
                if original.exists():
                    if symlink:
                        guard.path.unlink()
                    else:
                        guard.path.rename(replacement)
                    original.rename(guard.path)

    def test_partial_write_error_closes_once_and_preserves_original_failure(
        self,
    ) -> None:
        for close_fails in (False, True):
            with self.subTest(close_fails=close_fails):
                self._assert_partial_write_failure(close_fails)

    def _assert_partial_write_failure(self, close_fails: bool) -> None:
        with self._directory() as guard:
            original_write, original_close = os.write, os.close
            failure = OSError("synthetic partial write failure")
            calls: list[int] = []
            closed: list[int] = []

            def write(descriptor: int, data: bytes) -> int:
                calls.append(descriptor)
                if len(calls) == 1:
                    return original_write(descriptor, data[:2])
                raise failure

            def close(descriptor: int) -> None:
                closed.append(descriptor)
                original_close(descriptor)
                if close_fails:
                    raise OSError("synthetic close failure")

            with (
                patch.object(os, "write", write),
                patch.object(os, "close", close),
                self.assertRaises(OSError) as raised,
            ):
                guard.create_file("partial", b"secret")
            self.assertIs(raised.exception, failure)
            self.assertEqual(closed, [calls[0]])
            self.assertEqual((guard.path / "partial").read_bytes(), b"se")
            with self.assertRaises(OSError):
                os.fstat(calls[0])
            if close_fails:
                self.assertIn(
                    "Private file close also failed: OSError", failure.__notes__
                )

    def test_zero_write_fails_and_releases_the_file_descriptor(self) -> None:
        with self._directory() as guard:
            descriptors: list[int] = []

            def no_progress(descriptor: int, _data: bytes) -> int:
                descriptors.append(descriptor)
                return 0

            with patch.object(os, "write", no_progress), self.assertRaises(OSError):
                guard.create_file("empty", b"secret")
            self.assertEqual((guard.path / "empty").read_bytes(), b"")
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])

    def test_close_error_does_not_retry_a_reused_descriptor(self) -> None:
        with self._directory() as guard:
            original_close = os.close
            reused: list[int] = []
            closed: list[int] = []
            failure = OSError("synthetic ambiguous close")

            def close(descriptor: int) -> None:
                original_close(descriptor)
                closed.append(descriptor)
                reused.append(os.open(guard.path / "private-data", os.O_RDONLY))
                raise failure

            try:
                with (
                    patch.object(os, "close", close),
                    self.assertRaises(OSError) as raised,
                ):
                    guard.create_file("private-data", b"secret")
                self.assertIs(raised.exception, failure)
                self.assertEqual(len(closed), 1)
                self.assertEqual(os.read(reused[0], 6), b"secret")
            finally:
                for descriptor in reused:
                    original_close(descriptor)

    def test_delegated_pin_creates_once_and_retired_pin_cannot_create(self) -> None:
        with self._directory() as guard:
            alias = identity_module.DirectoryPin.capture(guard.path)
            try:
                guard._pin.share_authority(alias)
                alias.create_file("from-alias", b"shared authority")
                self.assertEqual(
                    (guard.path / "from-alias").read_bytes(), b"shared authority"
                )
            finally:
                if alias.authority is alias:
                    alias.retire()
        # The real guard's ordinary cleanup has now retired their shared pin.
        for pin in (guard._pin, alias):
            with self.assertRaises(identity_module.DirectoryIdentityError):
                pin.create_file("too-late", b"secret")
        self.assertFalse(guard.path.exists())

    def test_windows_relative_create_uses_native_exclusive_no_reparse_contract(
        self,
    ) -> None:
        kernel, native = Mock(), Mock()
        with patch.object(ctypes, "WinDLL", side_effect=[kernel, native], create=True):
            api = identity_module._WindowsDirectoryApi()
        name = "private-\U0001f512"

        def create(
            output: Any,
            access: int,
            attributes: Any,
            status: Any,
            allocation: Any,
            flags: int,
            sharing: int,
            disposition: int,
            options: int,
            ea: Any,
            ea_length: int,
        ) -> int:
            value = attributes._obj
            self.assertEqual(value.RootDirectory, 321)
            self.assertEqual(value.Attributes, 0x1040)
            native_name = value.ObjectName.contents
            self.assertEqual(
                ctypes.string_at(native_name.Buffer, native_name.Length),
                name.encode("utf-16-le"),
            )
            self.assertEqual(access, 0x100002)
            self.assertEqual(
                (flags, sharing, disposition, options), (0x80, 0, 2, 0x200060)
            )
            self.assertEqual((allocation, ea, ea_length), (None, None, 0))
            output._obj.value = 123
            status._obj.Information = 2
            return 0

        native.NtCreateFile.side_effect = create
        self.assertEqual(api.create_file(321, name), 123)
        kernel.CreateFileW.assert_not_called()
        api.close(123)
        kernel.CloseHandle.assert_called_once_with(123)

    def test_windows_native_failure_never_falls_back_to_path_creation(self) -> None:
        kernel, native = Mock(), Mock()
        with patch.object(ctypes, "WinDLL", side_effect=[kernel, native], create=True):
            api = identity_module._WindowsDirectoryApi()
        # Collision, reparse encountered, and unsupported-parameter status.
        for status in (0xC0000035, 0xC000050B, 0xC000000D):
            native.NtCreateFile.return_value = ctypes.c_int32(status).value
            with (
                self.subTest(status=status),
                self.assertRaisesRegex(
                    identity_module.DirectoryIdentityError, f"{status:#x}"
                ),
            ):
                api.create_file(321, "private-data")
        self.assertEqual(native.NtCreateFile.call_count, 3)
        kernel.CreateFileW.assert_not_called()
        kernel.CloseHandle.assert_not_called()

    def test_windows_file_descriptor_handoff_is_writable_and_consumed_once(
        self,
    ) -> None:
        convert = Mock(return_value=987)
        api = Mock(spec=identity_module._WindowsDirectoryApi)
        failure = OSError("synthetic fd conversion failure")
        with (
            patch.dict(
                sys.modules, {"msvcrt": SimpleNamespace(open_osfhandle=convert)}
            ),
            patch.object(os, "O_NOINHERIT", 0x80, create=True),
            patch.object(os, "O_BINARY", 0x8000, create=True),
        ):
            self.assertEqual(
                identity_module._windows_descriptor(123, api, writable=True), 987
            )
            convert.assert_called_once_with(123, os.O_WRONLY | 0x8000 | 0x80)
            api.close.assert_not_called()
            convert.side_effect = failure
            with self.assertRaises(OSError) as raised:
                identity_module._windows_descriptor(456, api, writable=True)
            self.assertIs(raised.exception, failure)
            api.close.assert_called_once_with(456)


class PrivateDirectoryCleanupTests(unittest.TestCase):
    def test_owned_worker_removes_exact_private_directory(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-owned-remove-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            (private_directory / "state").write_text("owned", encoding="utf-8")
            guard = process_module._PrivateDirectory.capture(private_directory)

            guard.remove(deadline=time.monotonic() + 3)

            self.assertFalse(private_directory.exists())
            with process_module._PRIVATE_DIRECTORY_CLEANUP_LOCK:
                key = process_module._private_directory_key(guard)
                self.assertNotIn(key, process_module._PENDING_PRIVATE_DIRECTORIES)
                self.assertNotIn(
                    key,
                    process_module._ACTIVE_PRIVATE_DIRECTORY_CLEANUPS,
                )

    def test_removed_directory_retry_must_reap_its_existing_cleanup_owner(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-removed-retry-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(private_directory)
            process = Mock(stdin=Mock(), stderr=None, returncode=0)
            process.poll.return_value = 0
            process.wait.return_value = 0
            job = Mock()
            failure = process_module.ProcessOwnershipError("Job still owns a child")
            job.wait_empty.side_effect = [failure, failure, None]
            active = process_module._ActivePrivateDirectoryCleanup(
                guard=guard,
                process=process,
                windows_job=job,
                windows_job_assigned=True,
            )
            process_module._register_pending_private_directory(guard)
            process_module._register_active_private_directory_cleanup(active)
            private_directory.rmdir()  # The old worker finished its filesystem work.
            try:
                with self.assertRaises(process_module.ProcessOwnershipError):
                    process_module._release_reaped_private_directory_cleanup(
                        active, deadline=time.monotonic() + 2
                    )
                with (
                    patch.object(
                        process_module, "_spawn_private_directory_cleanup"
                    ) as spawn,
                    self.assertRaises(process_module.ProcessOwnershipError) as raised,
                ):
                    guard.remove(deadline=time.monotonic() + 2)
                self.assertIs(raised.exception, failure)
                self.assertIs(
                    process_module._active_private_directory_cleanup(guard), active
                )
                self.assertIn(
                    process_module._private_directory_key(guard),
                    process_module._PENDING_PRIVATE_DIRECTORIES,
                )
                job.close.assert_not_called()
                spawn.assert_not_called()

                with patch.object(
                    process_module, "_spawn_private_directory_cleanup"
                ) as spawn:
                    guard.remove(deadline=time.monotonic() + 2)
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
                self.assertNotIn(
                    process_module._private_directory_key(guard),
                    process_module._PENDING_PRIVATE_DIRECTORIES,
                )
                self.assertEqual(job.wait_empty.call_count, 3)
                job.close.assert_called_once_with()
                self.assertEqual(process.wait.call_count, 2)
                spawn.assert_not_called()
            finally:
                process_module._release_active_private_directory_cleanup(active)
                process_module._release_pending_private_directory(guard)

    def test_parent_channel_eof_is_a_terminal_worker_signal(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        os.close(write_descriptor)
        try:
            with (
                patch(
                    "hbrowser.gallery.browser._directory_cleanup_worker.os._exit",
                    side_effect=SystemExit(5),
                ) as exit_process,
                self.assertRaises(SystemExit),
            ):
                cleanup_worker_module._watch_parent_channel(
                    read_descriptor, threading.Event(), threading.Event()
                )
        finally:
            os.close(read_descriptor)

        exit_process.assert_called_once_with(5)

    def test_worker_failure_diagnostic_is_byte_bounded(self) -> None:
        with tempfile.TemporaryFile() as diagnostic_file:
            with patch.object(sys, "stderr", diagnostic_file):
                cleanup_worker_module._report_failure(
                    "directory-removal",
                    RuntimeError("鎖定" * 1000),
                )
            diagnostic_file.seek(0)
            diagnostic = diagnostic_file.read()

        self.assertLessEqual(
            len(diagnostic),
            cleanup_worker_module._DIAGNOSTIC_LIMIT,
        )
        self.assertTrue(diagnostic.endswith(b"\n"))
        self.assertIn(b"stage=directory-removal", diagnostic)
        self.assertTrue(diagnostic.isascii())

    def test_worker_failure_reports_stage_from_direct_script(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-worker-error-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(private_directory)
            error_pipe = Mock()
            error_pipe.read.return_value = (
                b"stage=directory-removal error=PermissionError winerror=5\n"
            )
            process = Mock(
                args=["cleanup-worker"],
                stdin=Mock(),
                stderr=error_pipe,
                returncode=1,
            )
            process.wait.return_value = 1
            try:
                with (
                    patch.object(
                        process_module,
                        "_ownership_platform",
                        return_value="posix",
                    ),
                    _mock_popen(return_value=process) as popen,
                    patch.object(
                        process_module,
                        "_acknowledge_private_directory_claim",
                        side_effect=process_module.ProcessOwnershipError(
                            "worker failed"
                        ),
                    ),
                    self.assertRaises(process_module.ProcessOwnershipError) as raised,
                ):
                    guard.remove(deadline=time.monotonic() + 2)

                command = popen.call_args.args[0]
                self.assertEqual(
                    command[1:4],
                    ["-E", "-S", str(process_module._PRIVATE_CLEANUP_WORKER_PATH)],
                )
                self.assertNotIn("-m", command)
                self.assertIn(
                    "stage=directory-removal error=PermissionError winerror=5",
                    "\n".join(raised.exception.__notes__),
                )
                error_pipe.close.assert_called_once_with()
            finally:
                process_module._release_pending_private_directory(guard)

    def test_worker_assignment_failure_kills_ungated_exact_process(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-worker-assign-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(private_directory)
            process = Mock(
                args=["cleanup-worker"],
                stdin=Mock(),
                returncode=None,
            )
            process.poll.return_value = None

            def kill() -> None:
                process.returncode = -9

            process.kill.side_effect = kill
            process.wait.return_value = -9
            job = Mock()
            job.assign.side_effect = OSError("assignment failed")

            with (
                patch.object(process_module, "_ownership_platform", return_value="nt"),
                patch.object(
                    process_module,
                    "_supervisor_launch_context",
                    return_value=("python", None),
                ),
                patch.object(
                    process_module,
                    "_supervisor_creation_options",
                    return_value={},
                ),
                _mock_popen(return_value=process),
                patch.object(process_module._WindowsJob, "create", return_value=job),
                self.assertRaisesRegex(OSError, "assignment failed"),
            ):
                process_module._spawn_private_directory_cleanup(
                    guard,
                    claim=process_module._PrivateDirectoryClaim(
                        guard, Path(directory) / ".hbrowser-cleanup-test"
                    ),
                    work_deadline=time.monotonic() + 1,
                    ownership_deadline=time.monotonic() + 2,
                )

            process.stdin.write.assert_not_called()
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=ANY)
            job.terminate.assert_not_called()
            job.close.assert_called_once_with()
            with process_module._PRIVATE_DIRECTORY_CLEANUP_LOCK:
                self.assertNotIn(
                    process_module._private_directory_key(guard),
                    process_module._ACTIVE_PRIVATE_DIRECTORY_CLEANUPS,
                )

    def test_worker_diagnostic_is_collected_only_after_job_release(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-worker-diagnostic-"
        ) as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(private_directory)
            error_pipe = Mock()
            error_pipe.read.return_value = b"stage=directory-removal error=OSError\n"
            process = Mock(stdin=Mock(), stderr=error_pipe)
            job = Mock()
            job.close.side_effect = [OSError("close failed"), None]
            active = process_module._ActivePrivateDirectoryCleanup(
                guard=guard,
                process=process,
                windows_job=job,
                windows_job_assigned=True,
            )
            process_module._register_active_private_directory_cleanup(active)
            try:
                with self.assertRaisesRegex(OSError, "close failed"):
                    process_module._release_reaped_private_directory_cleanup(
                        active,
                        deadline=time.monotonic() + 1,
                    )

                error_pipe.read.assert_not_called()
                self.assertIs(
                    process_module._active_private_directory_cleanup(guard),
                    active,
                )

                process_module._release_reaped_private_directory_cleanup(
                    active,
                    deadline=time.monotonic() + 1,
                )

                error_pipe.read.assert_called_once_with(
                    process_module._PRIVATE_CLEANUP_WORKER_DIAGNOSTIC_BYTES
                )
                error_pipe.close.assert_called_once_with()
                self.assertEqual(
                    active.worker_diagnostic,
                    "stage=directory-removal error=OSError",
                )
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
                self.assertEqual(job.wait_empty.call_count, 2)
                self.assertEqual(job.close.call_count, 2)
            finally:
                process_module._release_active_private_directory_cleanup(active)

    def test_timeout_reaps_worker_before_background_mutation_can_continue(
        self,
    ) -> None:
        with _private_directory_fixture(prefix="hbrowser-worker-timeout-") as directory:
            root = Path(directory)
            private_directory = root / "profile"
            private_directory.mkdir()
            marker = root / "mutations"
            guard = process_module._PrivateDirectory.capture(private_directory)
            stop = threading.Event()
            mutation_thread: threading.Thread | None = None

            def mutate() -> None:
                while not stop.wait(0.001):
                    with marker.open("a", encoding="utf-8") as output:
                        output.write("x")

            class Gate:
                def write(self, value: bytes) -> int:
                    nonlocal mutation_thread
                    self_outer.assertEqual(value, b"start\n")
                    mutation_thread = threading.Thread(target=mutate, daemon=True)
                    mutation_thread.start()
                    return len(value)

                def flush(self) -> None:
                    pass

                def close(self) -> None:
                    pass

            class SlowProcess:
                args: ClassVar[list[str]] = ["cleanup-worker"]
                returncode: int | None = None
                stdin = Gate()
                stdout = None

                def poll(self) -> int | None:
                    return self.returncode

                def wait(self, *, timeout: float) -> int:
                    if self.returncode is None:
                        raise subprocess.TimeoutExpired(self.args, timeout)
                    assert mutation_thread is not None
                    mutation_thread.join(timeout=timeout)
                    return self.returncode

                def kill(self) -> None:
                    stop.set()
                    self.returncode = -9

            self_outer = self
            slow_process = SlowProcess()
            try:
                with (
                    patch.object(
                        process_module,
                        "_ownership_platform",
                        return_value="posix",
                    ),
                    _mock_popen(return_value=slow_process),
                    patch.object(
                        process_module, "_acknowledge_private_directory_claim"
                    ),
                    self.assertRaisesRegex(
                        process_module.ProcessOwnershipError,
                        "exceeded its work deadline during filesystem cleanup",
                    ),
                ):
                    guard.remove(deadline=time.monotonic() + 2)

                assert mutation_thread is not None
                self.assertFalse(mutation_thread.is_alive())
                observed = marker.read_text(encoding="utf-8") if marker.exists() else ""
                time.sleep(0.02)
                current = marker.read_text(encoding="utf-8") if marker.exists() else ""
                self.assertEqual(current, observed)
                self.assertEqual(slow_process.returncode, -9)
                key = process_module._private_directory_key(guard)
                with process_module._PRIVATE_DIRECTORY_CLEANUP_LOCK:
                    self.assertNotIn(
                        key,
                        process_module._ACTIVE_PRIVATE_DIRECTORY_CLEANUPS,
                    )
                    self.assertIn(
                        key,
                        process_module._PENDING_PRIVATE_DIRECTORIES,
                    )
            finally:
                stop.set()
                if mutation_thread is not None:
                    mutation_thread.join(timeout=1)
                process_module._release_pending_private_directory(guard)

    def test_windows_sharing_violation_retries_and_removes_same_directory(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-private-retry-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory, platform_name="nt"
            )
            original = process_module._PrivateDirectory._remove_claimed_contents
            attempts = 0

            def transient(
                instance: process_module._PrivateDirectory,
                namespace: process_module._PrivateDirectory,
                *,
                deadline: float,
            ) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise _windows_cleanup_error(32)
                original(instance, namespace, deadline=deadline)

            with (
                _claimed_namespace(guard) as namespace,
                patch.object(
                    process_module._PrivateDirectory,
                    "_remove_claimed_contents",
                    transient,
                ),
                patch.object(time, "sleep") as sleep,
            ):
                guard._remove_inline(namespace=namespace, deadline=time.monotonic() + 1)
            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(
                process_module._PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
            )
            self.assertFalse(private_directory.exists())

    def test_persistent_windows_sharing_violation_stops_at_deadline(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-private-deadline-"
        ) as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory, platform_name="nt"
            )
            cleanup_error = _windows_cleanup_error(33)
            with _claimed_namespace(guard) as namespace:
                with (
                    patch.object(
                        process_module._PrivateDirectory,
                        "_remove_claimed_contents",
                        side_effect=cleanup_error,
                    ) as remove,
                    patch.object(time, "monotonic", side_effect=[0.0, 0.1, 0.3]),
                    patch.object(time, "sleep") as sleep,
                    self.assertRaises(process_module.ProcessOwnershipError) as raised,
                ):
                    guard._remove_inline(namespace=namespace, deadline=0.25)
                self.assertIs(raised.exception.__cause__, cleanup_error)
                self.assertIn(
                    "winerror=33 attempts=1", "\n".join(raised.exception.__notes__)
                )
                remove.assert_called_once_with(namespace, deadline=0.25)
                sleep.assert_called_once_with(
                    process_module._PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
                )
                self.assertTrue((namespace.path / "tree").is_dir())

    def test_non_windows_or_non_sharing_errors_fail_without_retry(self) -> None:
        for platform_name, error_code in (("posix", 32), ("nt", 5)):
            with (
                self.subTest(platform=platform_name, code=error_code),
                _private_directory_fixture(
                    prefix="hbrowser-private-policy-"
                ) as directory,
            ):
                private_directory = Path(directory) / "profile"
                private_directory.mkdir()
                guard = process_module._PrivateDirectory.capture(
                    private_directory, platform_name=platform_name
                )
                cleanup_error = _windows_cleanup_error(error_code)
                with _claimed_namespace(guard) as namespace:
                    with (
                        patch.object(
                            process_module._PrivateDirectory,
                            "_remove_claimed_contents",
                            side_effect=cleanup_error,
                        ) as remove,
                        patch.object(time, "sleep") as sleep,
                        self.assertRaises(PermissionError) as raised,
                    ):
                        guard._remove_inline(
                            namespace=namespace, deadline=time.monotonic() + 1
                        )
                    self.assertTrue((namespace.path / "tree").is_dir())
                self.assertIs(raised.exception, cleanup_error)
                remove.assert_called_once()
                sleep.assert_not_called()

    def test_retry_revalidates_identity_before_deleting_replacement(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-private-identity-"
        ) as directory:
            root = Path(directory)
            private_directory = root / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory, platform_name="nt"
            )
            original = process_module._PrivateDirectory._remove_claimed_contents
            calls = 0
            with _claimed_namespace(guard) as namespace:
                claimed = namespace.path / "tree"

                def transient(
                    instance: process_module._PrivateDirectory,
                    target: process_module._PrivateDirectory,
                    *,
                    deadline: float,
                ) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise _windows_cleanup_error(32)
                    original(instance, target, deadline=deadline)

                def replace(_: float) -> None:
                    claimed.rename(root / "displaced")
                    claimed.mkdir()
                    (claimed / "foreign").write_text("preserve", encoding="utf-8")

                with (
                    patch.object(
                        process_module._PrivateDirectory,
                        "_remove_claimed_contents",
                        transient,
                    ),
                    patch.object(time, "sleep", side_effect=replace),
                    self.assertRaisesRegex(
                        process_module.ProcessOwnershipError, "identity changed"
                    ),
                ):
                    guard._remove_inline(
                        namespace=namespace, deadline=time.monotonic() + 1
                    )
                self.assertEqual(
                    (claimed / "foreign").read_text(encoding="utf-8"), "preserve"
                )
                self.assertTrue((root / "displaced").is_dir())

    def test_namespace_retry_revalidates_before_root_removal(self) -> None:
        self._assert_namespace_retry_revalidates(parent_alias=False)

    @unittest.skipUnless(os.name == "posix", "Fixture creates a parent symlink")
    def test_namespace_retry_revalidates_with_parent_alias(self) -> None:
        self._assert_namespace_retry_revalidates(parent_alias=True)

    def _assert_namespace_retry_revalidates(self, *, parent_alias: bool) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-namespace-retry-"
        ) as directory:
            root = Path(directory).resolve()
            parent = root / "parent"
            parent.mkdir()
            if parent_alias:
                alias = root / "alias"
                alias.symlink_to(parent, target_is_directory=True)
                parent = alias
            path = parent / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path, platform_name="nt")
            with _claimed_namespace(guard) as namespace:
                canonical_namespace = namespace.cleanup_path
                if parent_alias:
                    self.assertNotEqual(namespace.path, canonical_namespace)
                    self.assertTrue(namespace.path.samefile(canonical_namespace))
                else:
                    self.assertEqual(namespace.path, canonical_namespace)
                original_rmdir = Path.rmdir
                removal_attempts: list[Path] = []
                replacements = 0

                def locked_namespace(target: Path) -> None:
                    if target == canonical_namespace:
                        removal_attempts.append(target)
                        raise _windows_cleanup_error(32)
                    original_rmdir(target)

                def replace(_: float) -> None:
                    nonlocal replacements
                    replacements += 1
                    self.assertEqual(replacements, 1)
                    canonical_namespace.rename(root / "displaced-namespace")
                    canonical_namespace.mkdir()
                    (canonical_namespace / "foreign").write_text(
                        "keep", encoding="utf-8"
                    )

                with (
                    patch.object(Path, "rmdir", locked_namespace),
                    patch.object(time, "sleep", side_effect=replace) as sleep,
                    self.assertRaisesRegex(
                        process_module.ProcessOwnershipError, "identity changed"
                    ),
                ):
                    guard._remove_inline(
                        namespace=namespace, deadline=time.monotonic() + 1
                    )
                self.assertEqual(removal_attempts, [canonical_namespace])
                sleep.assert_called_once_with(ANY)
                self.assertEqual(replacements, 1)
                self.assertEqual(
                    (canonical_namespace / "foreign").read_text(encoding="utf-8"),
                    "keep",
                )
                displaced = (root / "displaced-namespace").stat()
                self.assertEqual(
                    (displaced.st_dev, displaced.st_ino),
                    (namespace.device, namespace.inode),
                )

    def test_directory_removed_during_backoff_is_idempotent_success(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-private-absence-"
        ) as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory, platform_name="nt"
            )
            original = process_module._PrivateDirectory._remove_claimed_contents
            calls = 0
            with _claimed_namespace(guard) as namespace:

                def transient(
                    instance: process_module._PrivateDirectory,
                    target: process_module._PrivateDirectory,
                    *,
                    deadline: float,
                ) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise _windows_cleanup_error(32)
                    original(instance, target, deadline=deadline)

                with (
                    patch.object(
                        process_module._PrivateDirectory,
                        "_remove_claimed_contents",
                        transient,
                    ),
                    patch.object(
                        time,
                        "sleep",
                        side_effect=lambda _: shutil.rmtree(namespace.path / "tree"),
                    ),
                ):
                    guard._remove_inline(
                        namespace=namespace, deadline=time.monotonic() + 1
                    )
                self.assertFalse(namespace.path.exists())

    def test_replacement_after_check_is_claimed_but_never_deleted(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-claim-race-") as directory:
            root = Path(directory)
            path = root / "profile"
            path.mkdir()
            (path / "owned").write_text("original", encoding="utf-8")
            guard = process_module._PrivateDirectory.capture(path)
            namespace_path = root / ".hbrowser-cleanup-test"
            namespace_path.mkdir()
            namespace = process_module._PrivateDirectory.capture(namespace_path)
            original_check = (
                process_module._PrivateDirectory._assert_identity_or_absence
            )
            injected = False

            def replace(instance: process_module._PrivateDirectory) -> bool:
                nonlocal injected
                present = original_check(instance)
                if instance is guard and present and not injected:
                    injected = True
                    path.rename(root / "original")
                    path.mkdir()
                    (path / "foreign").write_text("keep", encoding="utf-8")
                return present

            with (
                patch.object(
                    process_module._PrivateDirectory,
                    "_assert_identity_or_absence",
                    replace,
                ),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "identity changed"
                ) as raised,
            ):
                guard._claim_inline(
                    namespace, confirmed=False, deadline=time.monotonic() + 1
                )
            self.assertTrue(injected)
            self.assertEqual(
                (namespace.path / "tree/foreign").read_text(encoding="utf-8"), "keep"
            )
            self.assertEqual(
                (root / "original/owned").read_text(encoding="utf-8"), "original"
            )
            self.assertIn(
                str(namespace.cleanup_path / "tree"),
                "\n".join(raised.exception.__notes__),
            )

    def test_original_path_replacement_is_untouched_after_claim(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-claimed-replacement-"
        ) as directory:
            path = Path(directory) / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            with _claimed_namespace(guard) as namespace:
                path.mkdir()
                (path / "foreign").write_text("keep", encoding="utf-8")
                guard._remove_inline(namespace=namespace, deadline=time.monotonic() + 1)
            self.assertEqual((path / "foreign").read_text(encoding="utf-8"), "keep")

    def test_unconfirmed_missing_claim_never_counts_as_complete(self) -> None:
        with _private_directory_fixture(prefix="hbrowser-missing-claim-") as directory:
            path = Path(directory) / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            namespace_path = Path(directory) / ".hbrowser-cleanup-test"
            namespace_path.mkdir()
            namespace = process_module._PrivateDirectory.capture(namespace_path)
            state = process_module._PrivateDirectoryClaim(
                guard, namespace_path, namespace, started=True
            )
            path.rmdir()
            self.assertFalse(state.complete())
            with self.assertRaisesRegex(
                process_module.ProcessOwnershipError, "lost both"
            ):
                guard._claim_inline(
                    namespace, confirmed=False, deadline=time.monotonic() + 1
                )
            namespace_path.rmdir()
            with self.assertRaisesRegex(
                process_module.ProcessOwnershipError, "Unconfirmed"
            ):
                state.complete()

    def test_malformed_or_missing_handshake_never_authorizes_deletion(self) -> None:
        frames = (b"", b"wrong\n", b"ready -1 2\n", b"ready 1 2 extra\n", b"x" * 129)
        with _private_directory_fixture(prefix="hbrowser-handshake-") as directory:
            path = Path(directory) / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            for frame in frames:
                with self.subTest(frame=frame):
                    state = process_module._PrivateDirectoryClaim(
                        guard, Path(directory) / ".hbrowser-cleanup-test"
                    )
                    read_fd, write_fd = os.pipe()
                    try:
                        os.write(write_fd, frame)
                    finally:
                        os.close(write_fd)
                    with os.fdopen(read_fd, "rb") as output:
                        child = Mock(args=["worker"], stdout=output, stdin=Mock())
                        active = process_module._ActivePrivateDirectoryCleanup(
                            guard=guard, process=child
                        )
                        with self.assertRaises(process_module.ProcessOwnershipError):
                            process_module._acknowledge_private_directory_claim(
                                active, state, deadline=time.monotonic() + 1
                            )
                        child.stdin.write.assert_not_called()
                    self.assertFalse(state.confirmed)
                    self.assertIsNone(state.namespace)
                    self.assertTrue(path.is_dir())

    def test_claimed_frame_without_matching_root_never_authorizes_deletion(
        self,
    ) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-handshake-missing-"
        ) as directory:
            path = Path(directory) / "profile"
            path.mkdir()
            (path / "owned").write_text("retain original")
            guard = process_module._PrivateDirectory.capture(path)
            namespace_path = Path(directory) / ".hbrowser-cleanup-test"
            namespace_path.mkdir()
            (namespace_path / "foreign").write_text("retain namespace")
            namespace = process_module._PrivateDirectory.capture(namespace_path)
            state = process_module._PrivateDirectoryClaim(guard, namespace_path)
            read_fd, write_fd = os.pipe()
            try:
                with ExitStack() as resources:
                    channel = None
                    native_suffix = ""
                    if os.name == "posix":
                        sender, channel = socket.socketpair(type=socket.SOCK_DGRAM)
                        resources.callback(sender.close)
                        resources.callback(channel.close)
                        identity_module.send_pin(sender, namespace._pin)
                    else:
                        native_suffix = f" {namespace._pin.native_handle()}"
                    os.write(
                        write_fd,
                        (
                            f"ready {namespace.device} {namespace.inode}"
                            f"{native_suffix}\n"
                        ).encode(),
                    )
                    with os.fdopen(read_fd, "rb") as output:
                        # -1 is the current-process pseudo HANDLE on Windows.
                        child = Mock(
                            args=["worker"], stdout=output, stdin=Mock(), _handle=-1
                        )
                        child.stdin.write.side_effect = lambda _: os.write(
                            write_fd, b"claimed\n"
                        )
                        active = process_module._ActivePrivateDirectoryCleanup(
                            guard=guard, process=child, identity_channel=channel
                        )
                        with self.assertRaises(process_module.ProcessOwnershipError):
                            process_module._acknowledge_private_directory_claim(
                                active, state, deadline=time.monotonic() + 1
                            )
                        child.stdin.write.assert_called_once_with(b"claim\n")
                    self.assertFalse(state.confirmed)
                    self.assertTrue(path.is_dir())
                    self.assertEqual((path / "owned").read_text(), "retain original")
                    self.assertEqual(
                        (namespace_path / "foreign").read_text(), "retain namespace"
                    )
                    self.assertTrue(guard._assert_identity_or_absence())
                    self.assertFalse(guard._pin.retired)
                    self.assertFalse(namespace._pin.retired)
                    self.assertIsNotNone(state.namespace)
                    assert state.namespace is not None
                    self.assertTrue(state.namespace._assert_identity_or_absence())
                    self.assertFalse(state.namespace._pin.retired)
            finally:
                os.close(write_fd)
                guard._retire()
                namespace._retire()
                if state.namespace is not None:
                    state.namespace._retire()

    def test_failed_worker_does_not_publish_success_after_material_removal(
        self,
    ) -> None:
        with _private_directory_fixture(prefix="hbrowser-failed-removed-") as directory:
            path = Path(directory) / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            child = Mock(
                args=["worker"], stdin=Mock(), stdout=Mock(), stderr=None, returncode=1
            )
            child.poll.return_value = 1
            child.wait.return_value = 1
            job = Mock(spec=process_module._WindowsJob)

            def acknowledge(
                active: process_module._ActivePrivateDirectoryCleanup,
                state: process_module._PrivateDirectoryClaim,
                *,
                deadline: float,
            ) -> None:
                state.path.mkdir()
                state.namespace = process_module._PrivateDirectory.capture(state.path)
                guard._claim_inline(state.namespace, confirmed=False, deadline=deadline)
                state.confirmed = True
                guard._remove_inline(namespace=state.namespace, deadline=deadline)

            with (
                _mock_popen(return_value=child),
                patch.object(
                    process_module._WindowsJob, "create", return_value=job
                ) as create_job,
                patch.object(
                    process_module, "_acknowledge_private_directory_claim", acknowledge
                ),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "exit_code=1"
                ),
            ):
                guard.remove(deadline=time.monotonic() + 2)
            if os.name == "nt":
                create_job.assert_called_once_with()
                job.assign.assert_called_once_with(child)
                job.wait_empty.assert_called_once_with(timeout=ANY)
                job.close.assert_called_once_with()
                job.terminate.assert_not_called()
            else:
                create_job.assert_not_called()
            self.assertFalse(path.exists())
            self.assertIsNone(process_module._active_private_directory_cleanup(guard))
            self.assertNotIn(
                process_module._private_directory_key(guard),
                process_module._PENDING_PRIVATE_DIRECTORIES,
            )

    def test_worker_interruption_preserves_claim_and_public_replacement(
        self,
    ) -> None:
        for phase in ("after-claim", "before-delete", "after-delete"):
            with (
                self.subTest(phase=phase),
                _private_directory_fixture(
                    prefix="hbrowser-worker-phase-", delete=False
                ) as directory,
            ):
                root = Path(directory)
                path = root / "profile"
                path.mkdir()
                (path / "owned").write_text("original", encoding="utf-8")
                guard = process_module._PrivateDirectory.capture(path)
                marker = root / "phase"
                worker = root / "worker.py"
                runtime_worker = process_module._PRIVATE_CLEANUP_WORKER_PATH
                target = (
                    "_claim_inline"
                    if phase == "after-claim"
                    else "_remove_claimed_contents"
                )
                worker.write_text(
                    "import sys, runpy, time\nfrom pathlib import Path\n"
                    f"sys.path.insert(0, {str(runtime_worker.parent)!r})\n"
                    "import process\n"
                    f"original = process._PrivateDirectory.{target}\n"
                    "def interrupted(self, *args, **kwargs):\n"
                    + (
                        "    original(self, *args, **kwargs)\n"
                        if phase != "before-delete"
                        else ""
                    )
                    + f"    Path({str(marker)!r}).write_text('entered')\n"
                    "    time.sleep(30)\n"
                    f"process._PrivateDirectory.{target} = interrupted\n"
                    f"runpy.run_path({str(runtime_worker)!r}, run_name='__main__')\n",
                    encoding="utf-8",
                )
                try:
                    with (
                        patch.object(
                            process_module, "_PRIVATE_CLEANUP_WORKER_PATH", worker
                        ),
                        self.assertRaises(process_module.ProcessOwnershipError),
                    ):
                        guard.remove(deadline=time.monotonic() + 1)
                    self.assertTrue(
                        marker.is_file(),
                        "worker must reach the intended interruption boundary",
                    )
                    self.assertIsNone(
                        process_module._active_private_directory_cleanup(guard)
                    )
                    state = process_module._PENDING_PRIVATE_DIRECTORIES[
                        process_module._private_directory_key(guard)
                    ]
                    self.assertIsNotNone(state.namespace)
                    self.assertEqual(state.confirmed, phase != "after-claim")
                    path.mkdir()
                    (path / "foreign").write_text("keep", encoding="utf-8")
                    guard.remove(deadline=time.monotonic() + 3)
                    self.assertEqual(
                        (path / "foreign").read_text(encoding="utf-8"), "keep"
                    )
                    self.assertFalse(state.path.exists())
                    self.assertNotIn(
                        process_module._private_directory_key(guard),
                        process_module._PENDING_PRIVATE_DIRECTORIES,
                    )
                finally:
                    _release_private_fixture(guard, root)

    def test_interrupted_setup_without_identity_is_not_adopted_on_retry(self) -> None:
        with _private_directory_fixture(
            prefix="hbrowser-worker-setup-", delete=False
        ) as directory:
            root = Path(directory)
            path = root / "profile"
            path.mkdir()
            guard = process_module._PrivateDirectory.capture(path)
            worker = root / "worker.py"
            runtime_worker = process_module._PRIVATE_CLEANUP_WORKER_PATH
            marker = root / "setup"
            worker.write_text(
                "import sys, runpy, time\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(runtime_worker.parent)!r})\n"
                "original = Path.mkdir\n"
                "def stalled(path, *args, **kwargs):\n"
                "    original(path, *args, **kwargs)\n"
                f"    Path({str(marker)!r}).write_text('entered')\n"
                "    time.sleep(30)\n"
                "Path.mkdir = stalled\n"
                f"runpy.run_path({str(runtime_worker)!r}, run_name='__main__')\n",
                encoding="utf-8",
            )
            try:
                with (
                    patch.object(
                        process_module, "_PRIVATE_CLEANUP_WORKER_PATH", worker
                    ),
                    self.assertRaises(process_module.ProcessOwnershipError),
                ):
                    guard.remove(deadline=time.monotonic() + 1)
                self.assertTrue(marker.is_file())
                state = process_module._PENDING_PRIVATE_DIRECTORIES[
                    process_module._private_directory_key(guard)
                ]
                self.assertIsNone(state.namespace)
                self.assertTrue(state.path.is_dir())
                self.assertTrue(path.is_dir())
                with self.assertRaises(process_module.ProcessOwnershipError):
                    guard.remove(deadline=time.monotonic() + 2)
                self.assertIsNone(state.namespace)
                self.assertTrue(state.path.is_dir())
                self.assertTrue(path.is_dir())
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
            finally:
                _release_private_fixture(guard, root)


class ProcessPolicyTests(unittest.TestCase):
    def _closed_owner(self, directory: str) -> process_module.OwnedProcess:
        status_directory = Path(directory) / "status"
        status_directory.mkdir()
        supervisor = Mock(
            pid=100,
            stdin=Mock(),
            stdout=None,
            stderr=None,
            returncode=0,
        )
        owner = process_module.OwnedProcess(
            supervisor,
            target_process_group=200,
            windows_job=None,
            stdout_drain=None,
            stderr_drain=None,
            status_directory=status_directory,
            cleanup_paths=(),
        )
        owner._closed = True
        owner._supervisor_returncode = 0
        return owner

    def test_wait_rejects_cached_success_acquired_after_positive_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-wait-late-") as directory:
            owner = self._closed_owner(directory)
            now = [100.0]
            late_lock = Mock()

            def acquire(*, timeout: float) -> bool:
                self.assertGreater(timeout, 0)
                now[0] = 102.0
                return True

            late_lock.acquire.side_effect = acquire
            owner._shutdown_lock = late_lock
            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "expired while waiting",
                ),
            ):
                owner.wait(timeout=1)

            late_lock.release.assert_called_once_with()

    def test_shutdown_rejects_cached_success_acquired_after_positive_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-shutdown-late-") as directory:
            owner = self._closed_owner(directory)
            now = [100.0]
            late_lock = Mock()

            def acquire(*, timeout: float) -> bool:
                self.assertGreater(timeout, 0)
                now[0] = 102.0
                return True

            late_lock.acquire.side_effect = acquire
            owner._shutdown_lock = late_lock
            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "expired while waiting",
                ),
            ):
                owner.shutdown(
                    graceful_timeout=1,
                    terminate_timeout=0,
                    kill_timeout=0,
                    cleanup_timeout=0,
                )

            late_lock.release.assert_called_once_with()

    def test_zero_timeout_preserves_nonblocking_cached_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-zero-wait-") as directory:
            owner = self._closed_owner(directory)

            self.assertEqual(owner.wait(timeout=0), 0)
            self.assertEqual(
                owner.shutdown(
                    graceful_timeout=0,
                    terminate_timeout=0,
                    kill_timeout=0,
                    cleanup_timeout=0,
                ),
                0,
            )

    def test_zero_cleanup_budget_rejects_uncached_private_release(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-zero-private-release-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=0,
            )
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )
            owner._supervisor_reaped = True
            owner._supervisor_returncode = 0
            owner._tree_reaped = True

            with self.assertRaises(process_module.ProcessOwnershipError):
                owner.shutdown(
                    graceful_timeout=0,
                    terminate_timeout=0,
                    kill_timeout=0,
                    cleanup_timeout=0,
                )

            self.assertTrue(status_directory.is_dir())

    def test_stalled_control_write_does_not_hold_state_past_wait_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-control-write-lock-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            write_started = threading.Event()
            release_write = threading.Event()

            def blocked_write(_: bytes) -> int:
                write_started.set()
                self.assertTrue(release_write.wait(timeout=1))
                return 10

            control_pipe = Mock()
            control_pipe.fileno.return_value = Mock()
            control_pipe.write.side_effect = blocked_write
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.poll.return_value = None
            supervisor.wait.side_effect = subprocess.TimeoutExpired("supervisor", 0.05)
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )
            terminate_thread = threading.Thread(target=owner.terminate, daemon=True)
            terminate_thread.start()
            self.assertTrue(write_started.wait(timeout=0.5))

            started_at = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                owner.wait(timeout=0.05)
            elapsed = time.monotonic() - started_at

            release_write.set()
            terminate_thread.join(timeout=1)
            self.assertFalse(terminate_thread.is_alive())
            self.assertLess(elapsed, 0.25)

    def test_direct_terminate_control_lock_obeys_its_explicit_deadline(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-control-lock-deadline-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            control_pipe = Mock()
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )
            owner._control_write_lock.acquire()
            started_at = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "control writer",
                ):
                    owner.terminate(deadline=started_at + 0.05)
            finally:
                owner._control_write_lock.release()

            self.assertLess(time.monotonic() - started_at, 0.25)

    def test_immediate_control_backpressure_never_sleeps_past_zero_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-zero-control-write-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            control_pipe = Mock()
            control_pipe.fileno.return_value = 42
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )

            with (
                patch.object(os, "set_blocking"),
                patch.object(os, "write", side_effect=BlockingIOError),
                patch.object(time, "sleep") as sleep,
            ):
                owner._request_supervisor_shutdown(
                    control_module.ControlIntent.KILL,
                    phase_deadline=0,
                    overall_deadline=0,
                    allow_immediate=True,
                )

            sleep.assert_not_called()
            control_pipe.close.assert_called_once_with()

    def test_real_control_descriptor_is_written_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-control-write-nonblocking-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            control_pipe = Mock()
            control_pipe.fileno.return_value = 42
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.poll.return_value = None
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )

            with (
                patch.object(os, "set_blocking") as set_blocking,
                patch.object(
                    os,
                    "write",
                    side_effect=lambda _, data: len(data),
                ) as write,
            ):
                owner.terminate()

            set_blocking.assert_called_once_with(42, False)
            frame = write.call_args.args[1]
            self.assertEqual(write.call_args.args[0], 42)
            self.assertEqual(
                control_module.ControlRequest.parse(frame).intent,
                control_module.ControlIntent.TERMINATE,
            )
            control_pipe.write.assert_not_called()

    def test_partial_control_write_failure_closes_the_control_pipe(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-control-write-partial-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            control_pipe = Mock()
            control_pipe.fileno.return_value = 42
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )

            with (
                patch.object(os, "set_blocking"),
                patch.object(os, "write", side_effect=(3, OSError("closed"))) as write,
            ):
                owner.terminate()

            self.assertEqual(write.call_count, 2)
            first_frame = write.call_args_list[0].args[1]
            second_fragment = write.call_args_list[1].args[1]
            self.assertEqual(second_fragment, first_frame[3:])
            control_pipe.close.assert_called_once_with()

    def test_private_release_serializes_control_pipe_close_with_writer(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-control-close-race-"
        ) as directory:
            status_directory = Path(directory) / "status"
            status_directory.mkdir()
            write_started = threading.Event()
            release_write = threading.Event()
            errors: list[BaseException] = []

            def blocked_write(data: bytes) -> int:
                write_started.set()
                self.assertTrue(release_write.wait(timeout=1))
                return len(data)

            control_pipe = Mock()
            control_pipe.fileno.return_value = Mock()
            control_pipe.write.side_effect = blocked_write
            supervisor = Mock(
                pid=100,
                stdin=control_pipe,
                stdout=None,
                stderr=None,
                returncode=None,
            )
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(),
            )
            owner._tree_reaped = True

            def write_control() -> None:
                try:
                    deadline = time.monotonic() + 1
                    owner._request_supervisor_shutdown(
                        control_module.ControlIntent.KILL,
                        phase_deadline=deadline,
                        overall_deadline=deadline,
                    )
                except BaseException as error:
                    errors.append(error)

            def release_owner() -> None:
                try:
                    owner._release_ownership(deadline=time.monotonic() + 1)
                except BaseException as error:
                    errors.append(error)

            writer = threading.Thread(target=write_control, daemon=True)
            releaser = threading.Thread(target=release_owner, daemon=True)
            writer.start()
            self.assertTrue(write_started.wait(timeout=0.5))
            releaser.start()
            time.sleep(0.05)
            control_pipe.close.assert_not_called()

            release_write.set()
            writer.join(timeout=1)
            releaser.join(timeout=1)

            self.assertFalse(writer.is_alive())
            self.assertFalse(releaser.is_alive())
            self.assertEqual(errors, [])
            control_pipe.close.assert_called_once_with()

    def test_supervisor_ready_receipt_after_absolute_deadline_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-late-ready-") as directory:
            status_path = Path(directory) / "status"
            now = [100.0]

            def late_read(*_: object, **__: object) -> process_module._SupervisorReady:
                now[0] = 101.0
                return process_module._SupervisorReady(123)

            owner = Mock(spec=process_module.OwnedProcess)
            owner._read_startup_status.side_effect = late_read
            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                self.assertRaisesRegex(TimeoutError, "READY deadline"),
            ):
                process_module._read_supervisor_status(
                    owner,
                    status_path,
                    deadline=101.0,
                )

    def test_concurrent_waiters_share_lock_wait_deadline(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-concurrent-wait-test-"
        ) as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=0,
            )
            supervisor.wait.return_value = 0
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )
            errors = list[BaseException]()

            def wait_for_owner(timeout: float) -> None:
                try:
                    owner.wait(timeout=timeout)
                except BaseException as error:
                    errors.append(error)

            owner._shutdown_lock.acquire()
            try:
                first = threading.Thread(
                    target=wait_for_owner,
                    args=(0.2,),
                    daemon=True,
                )
                first.start()
                deadline = time.monotonic() + 1
                while (
                    owner._shutdown_attempt_users != 1 and time.monotonic() < deadline
                ):
                    time.sleep(0.001)
                self.assertEqual(owner._shutdown_attempt_users, 1)

                second = threading.Thread(
                    target=wait_for_owner,
                    args=(1.0,),
                    daemon=True,
                )
                second.start()
                deadline = time.monotonic() + 1
                while (
                    owner._shutdown_attempt_users != 2 and time.monotonic() < deadline
                ):
                    time.sleep(0.001)
                self.assertEqual(owner._shutdown_attempt_users, 2)
                first.join(timeout=0.5)
                second.join(timeout=0.5)

                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                self.assertEqual(len(errors), 2)
                self.assertTrue(
                    all(
                        isinstance(error, process_module.ProcessOwnershipError)
                        for error in errors
                    )
                )
            finally:
                owner._shutdown_lock.release()

            with patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: _remove_private_inline(
                    guard, deadline=deadline
                ),
            ):
                owner.wait(timeout=1)

    def test_supervisor_protocol_contains_no_parent_pid(self) -> None:
        status_path, command = supervisor_module._parse_arguments(
            ("status", "--", "browser", "--flag")
        )

        self.assertEqual(status_path, Path("status"))
        self.assertEqual(command, ("browser", "--flag"))
        with self.assertRaisesRegex(ValueError, "invalid supervisor arguments"):
            supervisor_module._parse_arguments(("status", "123", "--", "browser"))

    def test_supervisor_consumes_forwarding_capability_for_target_only(self) -> None:
        capability = {
            "HBROWSER_LOG_FORWARD_ENDPOINT": "127.0.0.1:43210",
            "HBROWSER_LOG_FORWARD_TOKEN": "a" * 64,
        }
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin:/bin", **capability},
            clear=True,
        ):
            target_environment = supervisor_module._take_target_environment()

            self.assertEqual(os.environ, {"PATH": "/usr/bin:/bin"})
            self.assertEqual(
                target_environment,
                {"PATH": "/usr/bin:/bin", **capability},
            )

    def test_windows_venv_supervisor_bypasses_python_redirector(self) -> None:
        venv_python = r"C:\workspace\.venv\Scripts\python.exe"
        base_python = r"C:\Python314\python.exe"
        with (
            patch.object(sys, "executable", venv_python),
            patch.object(sys, "_base_executable", base_python),
            patch.object(Path, "is_file", return_value=True),
            patch.dict(
                os.environ,
                {
                    "HBROWSER_PROCESS_LOG_FILE": r"C:\logs\legacy.log",
                    "HBROWSER_LOG_DIR": r"C:\logs",
                    "HBROWSER_TEST_SENTINEL": "retained",
                },
                clear=True,
            ),
        ):
            executable, environment = process_module._supervisor_launch_context("nt")

        self.assertEqual(executable, base_python)
        self.assertEqual(
            environment,
            {"__PYVENV_LAUNCHER__": venv_python},
        )

    def test_windows_non_venv_supervisor_uses_current_interpreter(self) -> None:
        executable_path = r"C:\Python314\python.exe"
        with (
            patch.object(sys, "executable", executable_path),
            patch.object(sys, "_base_executable", r"c:\python314\PYTHON.EXE"),
            patch.dict(
                os.environ,
                {
                    "HBROWSER_PROCESS_LOG_FILE": r"C:\logs\legacy.log",
                    "HBROWSER_LOG_DIR": r"C:\logs",
                    "HBROWSER_TEST_SENTINEL": "retained",
                },
                clear=True,
            ),
        ):
            executable, environment = process_module._supervisor_launch_context("nt")

        self.assertEqual(executable, executable_path)
        self.assertEqual(environment, {})

    def test_posix_supervisor_uses_explicit_sanitized_environment(self) -> None:
        with (
            patch.object(sys, "executable", "/usr/bin/python3"),
            patch.dict(
                os.environ,
                {
                    "HBROWSER_PROCESS_LOG_FILE": "/tmp/legacy.log",
                    "HBROWSER_LOG_DIR": "/tmp/logs",
                    "HBROWSER_TEST_SENTINEL": "retained",
                },
                clear=True,
            ),
        ):
            executable, environment = process_module._supervisor_launch_context("posix")

        self.assertEqual(executable, "/usr/bin/python3")
        self.assertEqual(environment, {})

    def test_environment_allowlist_and_explicit_overlay_strip_reserved_keys(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "PATH": "/runtime/bin",
                "HOME": "/profile",
                "LC_ALL": "C.UTF-8",
                "XDG_CACHE_HOME": "/cache",
                "AWS_SECRET_ACCESS_KEY": "credential",
                "ARBITRARY_SENTINEL": "sentinel",
                "BATTLE_COOKIE": "control",
                "EH_PASSWORD": "control",
                "HBROWSER_LOG_DIR": "/private/log",
                "HBROWSER_LOG_FORWARD_TOKEN": "inherited-capability",
            },
            clear=True,
        ):
            environment = process_module._build_child_environment(
                "posix",
                explicit_environment={
                    "APPLICATION_MODE": "worker",
                    "BATTLE_EXPLICIT": "removed",
                    "EH_EXPLICIT": "removed",
                    "HBROWSER_LOG_FORWARD_ENDPOINT": "removed",
                },
            )

        self.assertEqual(
            environment,
            {
                "PATH": "/runtime/bin",
                "HOME": "/profile",
                "LC_ALL": "C.UTF-8",
                "XDG_CACHE_HOME": "/cache",
                "APPLICATION_MODE": "worker",
            },
        )

    def test_forwarding_capability_is_injected_only_after_sanitization(self) -> None:
        with (
            patch.object(sys, "executable", "/usr/bin/python3"),
            patch.dict(
                os.environ,
                {"HBROWSER_LOG_FORWARD_TOKEN": "inherited", "PATH": "/bin"},
                clear=True,
            ),
        ):
            _, ordinary = process_module._supervisor_launch_context("posix")
            _, forwarded = process_module._supervisor_launch_context(
                "posix",
                environment={
                    "HBROWSER_LOG_FORWARD_ENDPOINT": "caller",
                    "HBROWSER_LOG_FORWARD_TOKEN": "caller",
                },
                forwarding_environment={
                    "HBROWSER_LOG_FORWARD_ENDPOINT": "127.0.0.1:43210",
                    "HBROWSER_LOG_FORWARD_TOKEN": "a" * 64,
                },
            )

        self.assertEqual(ordinary, {"PATH": "/bin"})
        self.assertEqual(
            forwarded,
            {
                "PATH": "/bin",
                "HBROWSER_LOG_FORWARD_ENDPOINT": "127.0.0.1:43210",
                "HBROWSER_LOG_FORWARD_TOKEN": "a" * 64,
            },
        )

    def test_child_environment_rejects_invalid_entries(self) -> None:
        for environment in (
            {"BAD=KEY": "value"},
            {"BAD\0KEY": "value"},
            {"KEY": "bad\0value"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                process_module._build_child_environment(
                    "posix",
                    explicit_environment=environment,
                )

    def test_missing_windows_base_interpreter_fails_closed(self) -> None:
        with (
            patch.object(
                sys,
                "executable",
                r"C:\workspace\.venv\Scripts\python.exe",
            ),
            patch.object(sys, "_base_executable", None),
            self.assertRaisesRegex(
                process_module.ProcessOwnershipError,
                "base executable path",
            ),
        ):
            process_module._supervisor_launch_context("nt")

    def test_unknown_platform_fails_closed(self) -> None:
        with (
            patch.object(os, "name", "unsupported"),
            self.assertRaisesRegex(RuntimeError, "Unsupported"),
        ):
            process_module._supervisor_creation_options()

    def test_browser_process_wrapper_cannot_receive_logging_capability(self) -> None:
        owner = Mock()
        with patch.object(
            process_module,
            "start_owned_process",
            return_value=owner,
        ) as start_process:
            result = process_module.start_owned_browser_process(
                "chrome",
                ["--headless"],
            )

        self.assertIs(result, owner)
        _, keyword_arguments = start_process.call_args
        self.assertNotIn("environment", keyword_arguments)
        self.assertNotIn("forward_logging", keyword_arguments)

    def test_windows_job_assignment_precedes_start_gate(self) -> None:
        events: list[str] = []
        launched: list[tuple[object, dict[str, object]]] = []
        supervisor = Mock(pid=101, stdout=None, stderr=None, returncode=None)
        supervisor.poll.return_value = None
        supervisor.wait.return_value = 0
        supervisor.stdin.write.side_effect = lambda _: events.append("start")
        job = Mock()
        job.assign.side_effect = lambda _: events.append("assign")
        job.wait_empty.return_value = None

        def launch_supervisor(
            command: object,
            **options: object,
        ) -> Mock:
            events.append("popen")
            launched.append((command, options))
            return supervisor

        def create_job() -> Mock:
            events.append("job")
            return job

        def report_ready(*_: object, **__: object) -> object:
            events.append("ready")
            return process_module._SupervisorReady(202)

        with (
            patch.object(process_module, "_ownership_platform", return_value="nt"),
            patch.object(
                process_module,
                "_supervisor_creation_options",
                return_value={"creationflags": 512},
            ),
            patch.object(
                process_module,
                "_supervisor_launch_context",
                return_value=(
                    r"C:\Python314\python.exe",
                    {"__PYVENV_LAUNCHER__": (r"C:\workspace\.venv\Scripts\python.exe")},
                ),
            ),
            _mock_popen(side_effect=launch_supervisor),
            patch.object(
                process_module._WindowsJob,
                "create",
                side_effect=create_job,
            ),
            patch.object(
                process_module,
                "_read_supervisor_status",
                side_effect=report_ready,
            ) as read_status,
            patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: _remove_private_inline(
                    guard, deadline=deadline
                ),
            ),
        ):
            owner = process_module.start_owned_process("browser", [])
            owner.terminate()
            owner.wait(timeout=1)

        self.assertEqual(events[:5], ["popen", "job", "assign", "start", "ready"])
        command, options = launched[0]
        assert isinstance(command, list)
        self.assertEqual(command[0], r"C:\Python314\python.exe")
        self.assertEqual(
            command[1:5],
            ["-E", "-S", str(process_module._SUPERVISOR_PATH), ANY],
        )
        self.assertEqual(command[5:], ["--", "browser"])
        self.assertEqual(
            options["env"],
            {"__PYVENV_LAUNCHER__": (r"C:\workspace\.venv\Scripts\python.exe")},
        )
        start_request = control_module.StartRequest.parse(
            supervisor.stdin.write.call_args_list[0].args[0]
        )
        ready_deadline = cast(
            float,
            read_status.call_args.kwargs["deadline"],
        )
        self.assertEqual(
            start_request.deadline_ns,
            control_module.deadline_to_monotonic_ns(ready_deadline),
        )
        terminate_request = control_module.ControlRequest.parse(
            supervisor.stdin.write.call_args_list[1].args[0]
        )
        self.assertEqual(
            terminate_request.intent,
            control_module.ControlIntent.TERMINATE,
        )
        self.assertIsNone(terminate_request.overall_deadline_ns)
        job.terminate.assert_not_called()
        job.wait_empty.assert_called_once_with(timeout=ANY)
        job.close.assert_called_once_with()

    def test_windows_assignment_failure_never_opens_start_gate(self) -> None:
        supervisor = Mock(pid=101, stdout=None, stderr=None)
        supervisor.wait.return_value = 0
        supervisor.poll.side_effect = [None, 0]
        job = Mock()
        job.assign.side_effect = OSError("assignment failed")

        with (
            patch.object(process_module, "_ownership_platform", return_value="nt"),
            patch.object(
                process_module, "_supervisor_creation_options", return_value={}
            ),
            _mock_popen(return_value=supervisor),
            patch.object(process_module._WindowsJob, "create", return_value=job),
            patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: _remove_private_inline(
                    guard, deadline=deadline
                ),
            ),
            self.assertRaisesRegex(OSError, "assignment failed"),
        ):
            process_module.start_owned_process("browser", [])

        supervisor.stdin.write.assert_not_called()
        supervisor.terminate.assert_called_once_with()
        supervisor.wait.assert_called_once_with(timeout=ANY)
        wait_timeout = supervisor.wait.call_args.kwargs["timeout"]
        self.assertGreater(wait_timeout, 0)
        self.assertLessEqual(wait_timeout, 10)
        job.close.assert_called_once_with()

    def test_startup_failure_cleanup_uses_the_same_absolute_deadline(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-start-deadline-test-"
        ) as directory:
            supervisor = Mock(
                pid=101,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            expires_at = time.monotonic() + 2.0
            with (
                patch.object(tempfile, "mkdtemp", return_value=directory),
                patch.object(
                    process_module, "_ownership_platform", return_value="posix"
                ),
                patch.object(
                    process_module, "_supervisor_creation_options", return_value={}
                ),
                _mock_popen(return_value=supervisor),
                patch.object(atexit, "register"),
                patch.object(
                    process_module,
                    "_read_supervisor_status",
                    side_effect=TimeoutError("READY timed out"),
                ) as read_status,
                patch.object(
                    process_module.OwnedProcess,
                    "shutdown",
                    autospec=True,
                    return_value=0,
                ) as shutdown,
                self.assertRaisesRegex(TimeoutError, "READY timed out"),
            ):
                process_module.start_owned_process(
                    "browser",
                    [],
                    startup_timeout=1.0,
                    deadline=expires_at,
                )

        read_deadline = read_status.call_args.kwargs["deadline"]
        self.assertGreater(read_deadline, time.monotonic())
        self.assertLessEqual(read_deadline, expires_at)
        self.assertLessEqual(read_deadline - time.monotonic(), 1.0)
        shutdown.assert_called_once()
        shutdown_kwargs = shutdown.call_args.kwargs
        self.assertEqual(shutdown_kwargs["deadline"], expires_at)
        self.assertLessEqual(shutdown_kwargs["terminate_timeout"], 2.0)
        self.assertLessEqual(shutdown_kwargs["kill_timeout"], 2.0)
        self.assertLessEqual(shutdown_kwargs["cleanup_timeout"], 2.0)

    def test_startup_timeout_consumes_late_not_started_receipt_and_preserves_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-late-not-started-receipt-"
        ) as directory:
            status_directory = Path(directory)
            returncode = control_module.PROVEN_TARGET_NOT_STARTED_EXIT_CODE
            supervisor = Mock(
                pid=101,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.return_value = returncode
            ready_timeout = TimeoutError("READY timed out")
            expires_at = time.monotonic() + 2.0
            with (
                patch.object(tempfile, "mkdtemp", return_value=directory),
                patch.object(
                    process_module,
                    "_ownership_platform",
                    return_value="posix",
                ),
                patch.object(
                    process_module,
                    "_supervisor_creation_options",
                    return_value={},
                ),
                _mock_popen(return_value=supervisor),
                patch.object(atexit, "register"),
                patch.object(
                    process_module,
                    "_read_supervisor_status",
                    side_effect=ready_timeout,
                ),
                patch.object(
                    process_module,
                    "_remove_private_directory_owned",
                    side_effect=lambda guard, *, deadline: _remove_private_inline(
                        guard, deadline=deadline
                    ),
                ),
                self.assertRaises(TimeoutError) as raised,
            ):
                process_module.start_owned_process(
                    "browser",
                    [],
                    startup_timeout=1.0,
                    deadline=expires_at,
                )

            self.assertIs(raised.exception, ready_timeout)
            self.assertFalse(status_directory.exists())
            supervisor.wait.assert_called_once_with(timeout=ANY)
            supervisor.stdin.close.assert_called_once_with()

    def test_pre_transfer_cleanup_failure_remains_in_durable_registry(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-provisional-owner-test-"
        ) as directory:
            supervisor = Mock(
                pid=101,
                stdin=Mock(),
                stdout=None,
                stderr=None,
            )
            job = Mock()
            job.assign.side_effect = OSError("assignment failed")
            job.terminate.side_effect = OSError("termination failed")
            try:
                with (
                    patch.object(tempfile, "mkdtemp", return_value=directory),
                    patch.object(
                        process_module, "_ownership_platform", return_value="nt"
                    ),
                    patch.object(
                        process_module,
                        "_supervisor_creation_options",
                        return_value={},
                    ),
                    _mock_popen(return_value=supervisor),
                    patch.object(
                        process_module._WindowsJob,
                        "create",
                        return_value=job,
                    ),
                    self.assertRaises(process_module.ProcessOwnershipError),
                ):
                    process_module.start_owned_process("browser", [])

                with process_module._PROVISIONAL_OWNERS_LOCK:
                    self.assertEqual(len(process_module._PROVISIONAL_OWNERS), 1)
            finally:
                # This test deliberately injects an unreapable fake. Do not
                # leave that fake in the interpreter's real atexit registry.
                with process_module._PROVISIONAL_OWNERS_LOCK:
                    process_module._PROVISIONAL_OWNERS.clear()

    def test_provisional_cleanup_cannot_borrow_a_long_caller_deadline(self) -> None:
        with _private_directory_fixture() as directory:
            guard = process_module._PrivateDirectory.capture(Path(directory))
            supervisor = Mock(pid=101, stdin=Mock(), stdout=None, stderr=None)
            supervisor.poll.side_effect = (None, 0)
            supervisor.wait.side_effect = (
                subprocess.TimeoutExpired("supervisor", 5),
                0,
            )
            owner = process_module._ProvisionalProcessOwner(
                supervisor=supervisor,
                status_guard=guard,
                cleanup_guards=(),
            )
            process_module._register_provisional_owner(owner)
            try:
                with patch.object(
                    process_module._PrivateDirectory, "remove", autospec=True
                ) as remove:
                    process_module._cleanup_provisional_owner(
                        owner, deadline=time.monotonic() + 120
                    )
                remove.assert_called_once()
                self.assertIs(remove.call_args.args[0], guard)
                for wait_call in supervisor.wait.call_args_list:
                    self.assertLessEqual(wait_call.kwargs["timeout"], 5.0)
                cleanup_deadline = remove.call_args.kwargs["deadline"]
                self.assertLessEqual(cleanup_deadline - time.monotonic(), 5.0)
                # Only the remover was mocked. The fixture still owns this pin.
                self.assertFalse(guard._pin.retired)
            finally:
                process_module._release_provisional_owner(owner)
                guard._retire()

    def test_windows_job_termination_and_close_failures_retain_handle(self) -> None:
        accounting_type = type(
            "Accounting",
            (ctypes.Structure,),
            {"_fields_": [("ActiveProcesses", ctypes.c_ulong)]},
        )
        kernel32 = Mock()
        kernel32.TerminateJobObject.return_value = False
        kernel32.CloseHandle.return_value = False
        job = process_module._WindowsJob(kernel32, 77, accounting_type)

        with (
            patch.object(ctypes, "get_last_error", return_value=5, create=True),
            self.assertRaises(OSError),
        ):
            job.terminate()
        self.assertEqual(job._handle, 77)

        with (
            patch.object(ctypes, "get_last_error", return_value=6, create=True),
            self.assertRaises(OSError),
        ):
            job.close()
        self.assertEqual(job._handle, 77)

    def test_windows_active_tree_timeout_keeps_private_paths_owned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-job-test-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(pid=100, stdin=Mock(), stdout=None, stderr=None)
            supervisor.wait.return_value = 0
            job = Mock()
            job.wait_empty.side_effect = [TimeoutError("tree alive"), None]
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            with self.assertRaisesRegex(TimeoutError, "tree alive"):
                owner.wait(timeout=1)
            self.assertTrue(private_directory.is_dir())
            self.assertTrue(status_directory.is_dir())
            job.close.assert_not_called()

            with patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: _remove_private_inline(
                    guard, deadline=deadline
                ),
            ):
                owner.wait(timeout=1)
            self.assertFalse(private_directory.exists())
            self.assertFalse(status_directory.exists())
            job.close.assert_called_once_with()

    def test_shutdown_allows_natural_exit_without_signalling_windows_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-natural-exit-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.return_value = 0
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=process_module._PrivateDirectory.capture(
                    status_directory,
                    platform_name="nt",
                ),
                cleanup_paths=(
                    process_module._PrivateDirectory.capture(
                        private_directory,
                        platform_name="nt",
                    ),
                ),
            )

            owner.shutdown(
                graceful_timeout=1,
                terminate_timeout=1,
                kill_timeout=1,
            )

            supervisor.wait.assert_called_once_with(timeout=ANY)
            supervisor.stdin.write.assert_not_called()
            job.terminate.assert_not_called()
            job.wait_empty.assert_called_once_with(timeout=ANY)
            job.close.assert_called_once_with()

    def test_shutdown_terminates_only_after_natural_exit_timeout(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-terminate-phase-"
        ) as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.side_effect = [
                subprocess.TimeoutExpired("supervisor", 1),
                0,
            ]
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            owner.shutdown(
                graceful_timeout=1,
                terminate_timeout=1,
                kill_timeout=1,
            )

            self.assertEqual(supervisor.wait.call_count, 2)
            terminate_request = control_module.ControlRequest.parse(
                supervisor.stdin.write.call_args.args[0]
            )
            self.assertEqual(
                terminate_request.intent,
                control_module.ControlIntent.TERMINATE,
            )
            self.assertIsNotNone(terminate_request.overall_deadline_ns)
            job.terminate.assert_not_called()
            job.close.assert_called_once_with()

    def test_shutdown_kills_job_only_after_terminate_timeout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-kill-phase-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.side_effect = [
                subprocess.TimeoutExpired("supervisor", 1),
                subprocess.TimeoutExpired("supervisor", 1),
                1,
            ]
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            owner.shutdown(
                graceful_timeout=1,
                terminate_timeout=1,
                kill_timeout=1,
            )

            self.assertEqual(supervisor.wait.call_count, 3)
            terminate_request = control_module.ControlRequest.parse(
                supervisor.stdin.write.call_args.args[0]
            )
            self.assertEqual(
                terminate_request.intent,
                control_module.ControlIntent.TERMINATE,
            )
            self.assertIsNotNone(terminate_request.overall_deadline_ns)
            job.terminate.assert_called_once_with()
            job.wait_empty.assert_called_once_with(timeout=ANY)
            job.close.assert_called_once_with()

    def test_zero_kill_budget_performs_one_immediate_windows_job_action(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-zero-windows-kill-"
        ) as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.side_effect = (
                subprocess.TimeoutExpired("supervisor", 0),
                subprocess.TimeoutExpired("supervisor", 0),
                0,
            )
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            result = owner.shutdown(
                graceful_timeout=0,
                terminate_timeout=0,
                kill_timeout=0,
                cleanup_timeout=1,
            )

            self.assertEqual(result, 0)
            self.assertEqual(supervisor.wait.call_count, 3)
            job.terminate.assert_called_once_with()
            job.wait_empty.assert_called_once_with(timeout=ANY)
            job.close.assert_called_once_with()

    def test_private_cleanup_retry_never_resignals_reaped_windows_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-release-retry-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.return_value = 0
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=process_module._PrivateDirectory.capture(
                    status_directory,
                    platform_name="nt",
                ),
                cleanup_paths=(
                    process_module._PrivateDirectory.capture(
                        private_directory,
                        platform_name="nt",
                    ),
                ),
            )
            cleanup_error = _windows_cleanup_error(32)

            remove_calls = 0

            def release_with_one_failure(
                guard: process_module._PrivateDirectory,
                *,
                deadline: float,
            ) -> None:
                nonlocal remove_calls
                remove_calls += 1
                if guard.path == private_directory and remove_calls == 2:
                    raise process_module.ProcessOwnershipError(
                        "private cleanup timed out"
                    ) from cleanup_error
                _remove_private_inline(guard, deadline=deadline)

            with (
                patch.object(
                    process_module._PrivateDirectory,
                    "remove",
                    autospec=True,
                    side_effect=release_with_one_failure,
                ),
                self.assertRaises(process_module.ProcessOwnershipError) as raised,
            ):
                owner.shutdown(
                    graceful_timeout=0,
                    terminate_timeout=0,
                    kill_timeout=0,
                    cleanup_timeout=1,
                )

            self.assertIs(raised.exception.__cause__, cleanup_error)
            self.assertTrue(private_directory.is_dir())
            job.close.assert_not_called()

            owner.shutdown(
                graceful_timeout=0,
                terminate_timeout=0,
                kill_timeout=0,
            )

            supervisor.wait.assert_called_once_with(timeout=ANY)
            job.wait_empty.assert_called_once_with(timeout=ANY)
            supervisor.stdin.write.assert_not_called()
            job.terminate.assert_not_called()
            job.close.assert_called_once_with()
            self.assertFalse(private_directory.exists())

    def test_poll_remains_nonblocking_during_private_directory_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-release-poll-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=None,
            )
            supervisor.wait.return_value = 0
            supervisor.poll.return_value = 0
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )
            release_started = threading.Event()
            allow_release = threading.Event()
            shutdown_errors: list[BaseException] = []

            def blocking_profile_release(
                guard: process_module._PrivateDirectory,
                *,
                deadline: float,
            ) -> None:
                if guard.path == private_directory:
                    release_started.set()
                    if not allow_release.wait(timeout=2):
                        raise TimeoutError("test did not release private cleanup")
                _remove_private_inline(guard, deadline=deadline)

            def shutdown_owner() -> None:
                try:
                    owner.shutdown(
                        graceful_timeout=0,
                        terminate_timeout=0,
                        kill_timeout=0,
                    )
                except BaseException as error:
                    shutdown_errors.append(error)

            with patch.object(
                process_module._PrivateDirectory,
                "remove",
                autospec=True,
                side_effect=blocking_profile_release,
            ):
                shutdown_thread = threading.Thread(target=shutdown_owner, daemon=True)
                shutdown_thread.start()
                self.assertTrue(release_started.wait(timeout=1))

                poll_result: list[int | None] = []
                poll_finished = threading.Event()

                def poll_owner() -> None:
                    poll_result.append(owner.poll())
                    poll_finished.set()

                poll_thread = threading.Thread(target=poll_owner, daemon=True)
                poll_thread.start()
                try:
                    self.assertTrue(
                        poll_finished.wait(timeout=0.5),
                        "poll blocked behind private directory cleanup",
                    )
                finally:
                    allow_release.set()
                    poll_thread.join(timeout=1)
                    shutdown_thread.join(timeout=2)

            self.assertEqual(poll_result, [0])
            self.assertEqual(shutdown_errors, [])
            self.assertFalse(shutdown_thread.is_alive())
            job.close.assert_called_once_with()

    def test_gate_flush_failure_keeps_unproven_private_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-gate-test-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=101,
                stdout=None,
                stderr=None,
                returncode=7,
            )
            supervisor.stdin.flush.side_effect = OSError("gate flush failed")
            supervisor.wait.return_value = 7

            with (
                patch.object(
                    tempfile,
                    "mkdtemp",
                    return_value=str(status_directory),
                ),
                patch.object(
                    process_module, "_ownership_platform", return_value="posix"
                ),
                patch.object(
                    process_module, "_supervisor_creation_options", return_value={}
                ),
                _mock_popen(return_value=supervisor),
                patch.object(os, "kill"),
                patch.object(atexit, "register"),
                self.assertRaises(process_module.ProcessOwnershipError) as raised,
            ):
                process_module.start_owned_process(
                    "browser",
                    [],
                    cleanup_paths=(private_directory,),
                )

            self.assertTrue(status_directory.is_dir())
            self.assertTrue(private_directory.is_dir())
            self.assertIn("Startup failure type: OSError", raised.exception.__notes__)

    def test_malformed_status_keeps_unproven_private_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-status-test-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=101,
                stdout=None,
                stderr=None,
                returncode=0,
            )
            supervisor.wait.return_value = 0

            with (
                patch.object(
                    tempfile,
                    "mkdtemp",
                    return_value=str(status_directory),
                ),
                patch.object(
                    process_module, "_ownership_platform", return_value="posix"
                ),
                patch.object(
                    process_module, "_supervisor_creation_options", return_value={}
                ),
                _mock_popen(return_value=supervisor),
                patch.object(
                    process_module,
                    "_read_supervisor_status",
                    side_effect=RuntimeError("invalid status"),
                ),
                patch.object(atexit, "register"),
                self.assertRaises(process_module.ProcessOwnershipError),
            ):
                process_module.start_owned_process(
                    "browser",
                    [],
                    cleanup_paths=(private_directory,),
                )

            self.assertTrue(status_directory.is_dir())
            self.assertTrue(private_directory.is_dir())

    def test_reaped_tree_does_not_signal_again_when_private_cleanup_retries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-retry-test-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=0,
            )
            supervisor.wait.return_value = 0
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=None,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            with patch.object(
                process_module._PrivateDirectory,
                "remove",
                autospec=True,
                side_effect=[None, RuntimeError("remove failed"), None],
            ) as remove:
                with self.assertRaisesRegex(RuntimeError, "remove failed"):
                    owner.wait(timeout=1)
                owner.terminate()
                owner.kill()
                owner.wait(timeout=1)

            self.assertEqual(supervisor.wait.call_count, 1)
            self.assertEqual(remove.call_count, 3)
            supervisor.stdin.write.assert_not_called()

    def test_empty_windows_job_stays_bound_until_private_cleanup_succeeds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-job-retry-") as directory:
            root = Path(directory)
            status_directory = root / "status"
            private_directory = root / "private"
            status_directory.mkdir()
            private_directory.mkdir()
            supervisor = Mock(
                pid=100,
                stdin=Mock(),
                stdout=None,
                stderr=None,
                returncode=0,
            )
            supervisor.wait.return_value = 0
            job = Mock()
            owner = process_module.OwnedProcess(
                supervisor,
                target_process_group=200,
                windows_job=job,
                stdout_drain=None,
                stderr_drain=None,
                status_directory=status_directory,
                cleanup_paths=(private_directory,),
            )

            with patch.object(
                process_module._PrivateDirectory,
                "remove",
                autospec=True,
                side_effect=[None, RuntimeError("remove failed"), None],
            ):
                with self.assertRaisesRegex(RuntimeError, "remove failed"):
                    owner.wait(timeout=1)
                job.close.assert_not_called()
                owner.terminate()
                owner.wait(timeout=1)

            job.wait_empty.assert_called_once_with(timeout=ANY)
            job.terminate.assert_not_called()
            job.close.assert_called_once_with()


@unittest.skipUnless(os.name == "nt", "Windows file-sharing contract")
class WindowsPrivateDirectoryIntegrationTests(unittest.TestCase):
    def test_owned_job_releases_real_message_database_lock_before_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-windows-lock-") as directory:
            root = Path(directory)
            profile = root / "profile"
            message_database = profile / "Default" / "Collaboration" / "MessageDB"
            message_database.parent.mkdir(parents=True)
            message_database.write_bytes(b"locked")
            lock_ready = root / "message-db-lock-ready"
            lock_script = r"""
import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
import pathlib
import sys
import time
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
kernel32.CreateFileW.restype = wintypes.HANDLE
handle = kernel32.CreateFileW(
    sys.argv[1],
    0x80000000,
    0x00000001 | 0x00000002,
    None,
    3,
    0x00000080,
    None,
)
if handle in (None, ctypes.c_void_p(-1).value):
    raise ctypes.WinError(ctypes.get_last_error())
pathlib.Path(sys.argv[2]).write_text("ready", encoding="utf-8")
time.sleep(30)
"""
            base_executable = cast(str, cast(Any, sys)._base_executable)
            owner = process_module.start_owned_process(
                base_executable,
                [
                    "-c",
                    lock_script,
                    str(message_database),
                    str(lock_ready),
                ],
                cleanup_paths=(profile,),
            )
            try:
                ready_deadline = time.monotonic() + 3
                while not lock_ready.is_file() and time.monotonic() < ready_deadline:
                    time.sleep(0.01)
                self.assertTrue(lock_ready.is_file())
                with self.assertRaises(OSError) as locked:
                    message_database.unlink()
                self.assertEqual(getattr(locked.exception, "winerror", None), 32)

                owner.kill()
                owner.wait(timeout=3)
            finally:
                if not owner._closed:
                    owner.shutdown(
                        graceful_timeout=0,
                        terminate_timeout=3,
                        kill_timeout=3,
                        cleanup_timeout=3,
                    )

            self.assertFalse(profile.exists())
            self.assertTrue(owner._closed)
            self.assertIsNone(owner._windows_job)


@unittest.skipUnless(os.name == "nt", "Windows venv ownership contract")
class WindowsVenvOwnedProcessTests(unittest.TestCase):
    def test_real_venv_redirector_keeps_supervisor_parent_alive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-windows-venv-") as directory:
            venv_path = Path(directory) / "venv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    "--without-pip",
                    str(venv_path),
                ],
                check=True,
                timeout=60,
            )
            venv_python = venv_path / "Scripts" / "python.exe"
            marker = Path(directory) / "startup-hook"
            hook_root = Path(directory) / "python-startup"
            hook_root.mkdir()
            (hook_root / "sitecustomize.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
                encoding="utf-8",
            )
            harness = """
import os
import pathlib
import sys

from hbrowser.gallery.browser.process import start_owned_process

assert sys.executable.lower() != sys._base_executable.lower()
os.environ["PYTHONPATH"] = sys.argv[1]
owner = start_owned_process(
    sys._base_executable,
    ["-E", "-S", "-c", "import time; time.sleep(30)"],
)
assert owner.target_pid is not None
owner.terminate()
owner.wait(timeout=10)
assert not pathlib.Path(sys.argv[2]).exists()
"""
            with tempfile.TemporaryFile() as output:
                command = command_process.ProcessCommand(
                    "Windows venv runtime harness",
                    (
                        str(venv_python),
                        "-c",
                        _runtime_harness_source(harness),
                        str(hook_root),
                        str(marker),
                    ),
                    Path(__file__).resolve().parents[1],
                    environment=dict(os.environ),
                    stdout=output,
                    stderr=output,
                )
                with _owned_harness(command) as owner:
                    returncode = owner.process.wait(timeout=30)
                output.seek(0)
                diagnostic = output.read().decode("utf-8", errors="replace")

        self.assertEqual(returncode, 0, diagnostic)


if __name__ == "__main__":
    unittest.main()


class PrivateCleanupSpawnHandoffTests(unittest.TestCase):
    def test_pre_spawn_registration_failure_completes_original_observation(
        self,
    ) -> None:
        for registered in (False, True):
            with self.subTest(registered=registered):
                self._assert_registration_failure(registered=registered)

    def _assert_registration_failure(self, *, registered: bool) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        captured: list[process_module._ActivePrivateDirectoryCleanup] = []
        actual_register = process_module._register_active_private_directory_cleanup

        def fail_registration(
            active: process_module._ActivePrivateDirectoryCleanup,
        ) -> None:
            captured.append(active)
            observation = active.observation
            assert isinstance(observation, runtime_test_owner._Observation)
            self.assertIn(observation, registry.observations)
            self.assertTrue(observation.starting)
            self.assertIsNotNone(observation.callback)
            self.assertFalse(active.spawn_attempted)
            if registered:
                actual_register(active)
            raise OSError("registration interrupted")

        with tempfile.TemporaryDirectory(prefix="hbrowser-cleanup-admission-") as name:
            private = Path(name) / "private"
            private.mkdir()
            guard = process_module._PrivateDirectory.capture(private)
            claim = process_module._register_pending_private_directory(guard)
            deadline = time.monotonic() + 10
            try:
                with (
                    patch.object(
                        process_module,
                        "_register_active_private_directory_cleanup",
                        side_effect=fail_registration,
                    ),
                    patch.object(subprocess, "Popen") as popen,
                    self.assertRaisesRegex(OSError, "registration interrupted"),
                ):
                    process_module._spawn_private_directory_cleanup(
                        guard,
                        claim=claim,
                        work_deadline=deadline,
                        ownership_deadline=deadline,
                    )
                popen.assert_not_called()
                self.assertEqual(len(captured), 1)
                self.assertIsNone(
                    process_module._active_private_directory_cleanup(guard)
                )
                self.assertTrue(private.exists())
                self.assertEqual(set(registry.observations), observations_before)
            finally:
                guard.remove(deadline=deadline)

    def test_unknown_private_spawn_retains_exact_owner_and_foreign_directory(
        self,
    ) -> None:
        for fault in ("popen return", "handle assignment"):
            with self.subTest(fault=fault):
                self._assert_unknown_private_spawn(fault)

    def _assert_unknown_private_spawn(self, fault: str) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        actual_popen = subprocess.Popen
        active_type = process_module._ActivePrivateDirectoryCleanup
        actual_setattr = active_type.__setattr__
        children: list[subprocess.Popen[bytes]] = []
        captured: list[process_module._ActivePrivateDirectoryCleanup] = []
        actual_register = process_module._register_active_private_directory_cleanup
        failure = OSError("private worker handle handoff interrupted")

        def record_original_owner(
            active: process_module._ActivePrivateDirectoryCleanup,
        ) -> None:
            captured.append(active)
            actual_register(active)

        def lose_returned_handle(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            child = actual_popen(*args, **kwargs)
            children.append(child)
            raise failure

        def lose_assigned_handle(
            active: process_module._ActivePrivateDirectoryCleanup,
            name: str,
            value: Any,
        ) -> None:
            if name == "process" and value is not None:
                children.append(value)
                raise failure
            actual_setattr(active, name, value)

        with tempfile.TemporaryDirectory(prefix="hbrowser-cleanup-handoff-") as name:
            root = Path(name)
            private = root / "private"
            private.mkdir()
            guard = process_module._PrivateDirectory.capture(private)
            claim = process_module._register_pending_private_directory(guard)
            deadline = time.monotonic() + 10
            original = root / "original"
            foreign = root / "foreign"
            retained = None
            try:
                with ExitStack() as patches:
                    patches.enter_context(
                        patch.object(
                            process_module,
                            "_register_active_private_directory_cleanup",
                            side_effect=record_original_owner,
                        )
                    )
                    patches.enter_context(
                        patch.object(
                            subprocess, "Popen", side_effect=lose_returned_handle
                        )
                        if fault == "popen return"
                        else patch.object(
                            active_type, "__setattr__", new=lose_assigned_handle
                        )
                    )
                    with self.assertRaises(process_module.ProcessOwnershipError):
                        process_module._spawn_private_directory_cleanup(
                            guard,
                            claim=claim,
                            work_deadline=deadline,
                            ownership_deadline=deadline,
                        )
                self.assertEqual(len(children), 1)
                retained = process_module._active_private_directory_cleanup(guard)
                self.assertIsNotNone(retained)
                assert retained is not None
                self.assertTrue(retained.spawn_attempted)
                self.assertIsNone(retained.process)
                observation = retained.observation
                assert isinstance(observation, runtime_test_owner._Observation)
                self.assertFalse(observation.starting)
                self.assertFalse(observation.completed)
                assert observation.callback is not None
                private.rename(original)
                private.mkdir()
                (private / "marker").write_text("preserve", encoding="utf-8")
                with self.assertRaisesRegex(
                    process_module.ProcessOwnershipError, "spawn outcome is unknown"
                ):
                    observation.callback(deadline)
                self.assertIn(observation, registry.observations)
                self.assertIsNone(children[0].poll())
                self.assertEqual((private / "marker").read_text(), "preserve")
                self.assertFalse(guard._pin.retired)
                if os.name == "posix":
                    assert retained.identity_channel is not None
                    self.assertGreaterEqual(retained.identity_channel.fileno(), 0)
            finally:
                # This controlled injector alone retained the actual handle.
                # Production may not rediscover authority from a PID or path.
                if children:
                    # Keep the test safe even when a regression incorrectly
                    # erased the registry entry before this assertion failed.
                    retained = captured[0]
                    retained.process = children[0]
                    observation = retained.observation
                    assert isinstance(observation, runtime_test_owner._Observation)
                    if observation.callback is not None:
                        observation.callback(deadline)
                    else:
                        process_module._force_reap_private_directory_cleanup(
                            retained, deadline=deadline
                        )
                if original.exists():
                    private.rename(foreign)
                    original.rename(private)
                guard.remove(deadline=deadline)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual((foreign / "marker").read_text(), "preserve")
            self.assertFalse(private.exists())
            self.assertEqual(set(registry.observations), observations_before)


class CapturedPrivatePathLaunchTests(unittest.TestCase):
    def test_allocated_private_startup_keeps_one_provisional_owner(self) -> None:
        for fault in (
            "allocation",
            "capture",
            "parameters",
            "late-parameters",
            "late-parameters-return",
            "spawn",
        ):
            with self.subTest(fault=fault):
                self._assert_allocated_startup_failure(fault)

    def _assert_allocated_startup_failure(self, fault: str) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        owners_before = set(process_module._PROVISIONAL_OWNERS)
        guards: list[process_module._PrivateDirectory] = []
        owners: list[process_module._ProvisionalProcessOwner] = []
        children: list[subprocess.Popen[bytes]] = []
        original_mkdtemp = tempfile.mkdtemp
        original_capture = process_module._PrivateDirectory.capture
        original_register = process_module._register_provisional_owner
        original_popen = subprocess.Popen
        failure = OSError(f"private launch {fault} failed")
        deadline = time.monotonic() + (0.3 if fault.startswith("late-") else 5)
        prefix = "hbrowser-owned-additional-"

        def register(owner: process_module._ProvisionalProcessOwner) -> None:
            owners.append(owner)
            original_register(owner)

        def allocate(*args: Any, **kwargs: Any) -> str:
            value = original_mkdtemp(*args, **kwargs)
            assert isinstance(value, str)
            if kwargs.get("prefix") == prefix and fault == "allocation":
                # This fault injector alone keeps the exact native identity
                # when allocation fails before returning its path to runtime.
                guards.append(original_capture(Path(value)))
                raise failure
            return value

        def capture(
            path: Path, *, platform_name: str | None = None
        ) -> process_module._PrivateDirectory:
            guard = original_capture(path, platform_name=platform_name)
            if path.name.startswith(prefix):
                guards.append(guard)
                if fault == "capture":
                    raise failure
            return guard

        def parameters(path: Path) -> list[str]:
            self.assertEqual(len(owners), 1)
            owner = owners[0]
            self.assertIs(owner.private_guard, guards[0])
            self.assertIn(guards[0], owner.cleanup_guards)
            self.assertIn(id(owner), process_module._PROVISIONAL_OWNERS)
            self.assertFalse(owner.spawn_attempted)
            (path / "private").write_bytes(b"launch material")
            if fault.startswith("late-"):
                time.sleep(max(0, deadline - time.monotonic()) + 0.01)
            if fault in ("parameters", "late-parameters"):
                raise failure
            return ["-c", "pass"]

        def spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            self.assertFalse(
                fault.startswith("late-"), "late startup launched a process"
            )
            child = original_popen(*args, **kwargs)
            if fault == "spawn" and not children:
                children.append(child)
                raise failure
            return child

        try:
            with (
                patch.object(tempfile, "mkdtemp", side_effect=allocate),
                patch.object(
                    process_module._PrivateDirectory, "capture", side_effect=capture
                ),
                patch.object(
                    process_module, "_register_provisional_owner", side_effect=register
                ),
                patch.object(subprocess, "Popen", side_effect=spawn),
                self.assertRaises(
                    OSError
                    if fault == "parameters"
                    else process_module.ProcessOwnershipError
                ),
            ):
                process_module._start_owned_process_with_private_directory(
                    sys.executable,
                    parameters,
                    directory_prefix=prefix,
                    deadline=deadline,
                )
            self.assertEqual(len(owners), 1)
            self.assertEqual(len(guards), 1)
            owner = owners[0]
            guard = guards[0]
            if fault == "parameters":
                self.assertTrue(guard._pin.retired)
                self.assertFalse(guard.path.exists())
                self.assertNotIn(id(owner), process_module._PROVISIONAL_OWNERS)
            else:
                self.assertIs(self._retained_owner(owners_before), owner)
                observation = owner.observation
                assert isinstance(observation, runtime_test_owner._Observation)
                assert observation.callback is not None
                self.assertFalse(observation.starting)
                self.assertFalse(observation.completed)
                self.assertFalse(guard._pin.retired)
                self.assertTrue(guard.path.exists())
                if fault in ("allocation", "capture", "spawn"):
                    with self.assertRaisesRegex(
                        process_module.ProcessOwnershipError,
                        "spawn outcome is unknown"
                        if fault == "spawn"
                        else "no captured native identity",
                    ):
                        observation.callback(time.monotonic() + 3)
                    self.assertIn(observation, registry.observations)
                    self.assertFalse(guard._pin.retired)
                if fault == "spawn":
                    self.assertTrue(owner.spawn_attempted)
                    self.assertIsNone(owner.supervisor)
                    self.assertIsNone(children[0].poll())
                elif fault.startswith("late-"):
                    self.assertFalse(owner.spawn_attempted)
                    self.assertIsNone(owner.supervisor)
        finally:
            for owner in owners:
                if id(owner) not in process_module._PROVISIONAL_OWNERS:
                    continue
                # Only controlled injectors restore their exact lost authority;
                # runtime must never rediscover it by numeric identity or path.
                if guards and owner.private_guard is None:
                    owner.private_guard = guards[0]
                    owner.private_directory = guards[0].path
                    owner.cleanup_guards = (guards[0],)
                if children and owner.supervisor is None:
                    owner.supervisor = children[0]
                process_module._cleanup_provisional_owner(
                    owner, deadline=time.monotonic() + 5
                )
        self.assertEqual(set(process_module._PROVISIONAL_OWNERS), owners_before)
        self.assertEqual(set(registry.observations), observations_before)
        self.assertTrue(all(guard._pin.retired for guard in guards))
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_replaced_captured_directory_is_rejected_before_target_launch(self) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        owners_before = set(process_module._PROVISIONAL_OWNERS)
        with _private_directory_fixture() as directory:
            root = Path(directory)
            private = root / "private"
            private.mkdir()
            guard = process_module._PrivateDirectory.capture(private)
            original = root / "original"
            foreign = root / "foreign"
            private.rename(original)
            private.mkdir()
            marker = private / "marker"
            marker.write_text("preserve", encoding="utf-8")
            target_marker = root / "launched"
            deadline = time.monotonic() + 10
            started = None
            try:
                with self.assertRaises(process_module.ProcessOwnershipError):
                    started = process_module.start_owned_process(
                        sys.executable,
                        [
                            "-c",
                            "from pathlib import Path; import sys; "
                            "Path(sys.argv[1]).touch()",
                            str(target_marker),
                        ],
                        cleanup_paths=(guard,),
                        deadline=deadline,
                    )
                self.assertEqual(marker.read_text(), "preserve")
                self.assertFalse(target_marker.exists())
                retained = self._retained_owner(owners_before)
                self.assertFalse(retained.spawn_attempted)
                self.assertIsNone(retained.supervisor)
                self.assertIn(guard, retained.cleanup_guards)
                observation = retained.observation
                self.assertIsNotNone(observation)
                self.assertIn(observation, registry.observations)
                assert isinstance(observation, runtime_test_owner._Observation)
                self.assertFalse(observation.starting)
                self.assertFalse(observation.completed)
                self.assertIsNotNone(observation.callback)
                # The same owner refuses the foreign generation on a retry too.
                assert observation.callback is not None
                with self.assertRaises(process_module.ProcessOwnershipError):
                    observation.callback(deadline)
                self.assertEqual(marker.read_text(), "preserve")
            finally:
                if started is not None:
                    started.shutdown(
                        graceful_timeout=0,
                        terminate_timeout=1,
                        kill_timeout=2,
                        deadline=deadline,
                    )
                # Restore the exact fixture-owned generation before asking the
                # original callback to retry. No replacement guard is captured.
                if original.exists():
                    private.rename(foreign)
                    original.rename(private)
                self._finish_retained(owners_before, deadline=deadline)
            self.assertEqual((foreign / "marker").read_text(), "preserve")
            self.assertFalse(private.exists())
            self.assertTrue(guard._pin.retired)
            self.assertEqual(set(registry.observations), observations_before)
            self.assertEqual(set(process_module._PROVISIONAL_OWNERS), owners_before)

    def test_unknown_spawn_retains_obligation_until_exact_handle_is_recovered(
        self,
    ) -> None:
        for fault in ("popen return", "handle assignment"):
            with self.subTest(fault=fault):
                self._assert_unknown_spawn_retained(fault)

    def _assert_unknown_spawn_retained(self, fault: str) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        owners_before = set(process_module._PROVISIONAL_OWNERS)
        actual_popen = subprocess.Popen
        children: list[subprocess.Popen[bytes]] = []
        failure = OSError("launch returned an unknown outcome")
        deadline = time.monotonic() + 10
        owner_type = process_module._ProvisionalProcessOwner
        actual_setattr = owner_type.__setattr__

        def lose_returned_handle(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            # The fault injector retains the actual Popen object. Production
            # never receives it, so None cannot prove that no child was launched.
            child = actual_popen(*args, **kwargs)
            children.append(child)
            raise failure

        def lose_assigned_handle(
            owner: process_module._ProvisionalProcessOwner, name: str, value: Any
        ) -> None:
            if name == "supervisor" and value is not None:
                children.append(value)
                raise failure
            actual_setattr(owner, name, value)

        try:
            with (
                (
                    patch.object(subprocess, "Popen", side_effect=lose_returned_handle)
                    if fault == "popen return"
                    else patch.object(
                        owner_type, "__setattr__", new=lose_assigned_handle
                    )
                ),
                self.assertRaises(process_module.ProcessOwnershipError),
            ):
                process_module.start_owned_process(
                    sys.executable, ["-c", "pass"], deadline=deadline
                )
            self.assertEqual(len(children), 1)
            retained = self._retained_owner(owners_before)
            self.assertTrue(retained.spawn_attempted)
            self.assertIsNone(retained.supervisor)
            observation = retained.observation
            assert isinstance(observation, runtime_test_owner._Observation)
            self.assertFalse(observation.starting)
            self.assertFalse(observation.completed)
            assert observation.callback is not None
            with self.assertRaisesRegex(
                process_module.ProcessOwnershipError, "spawn outcome is unknown"
            ):
                observation.callback(deadline)
            self.assertIn(observation, registry.observations)
            self.assertIsNone(children[0].poll())
            assert retained.status_directory is not None
            self.assertTrue(retained.status_directory.exists())
        finally:
            # Recover only the exact handle retained by this controlled fault.
            # Never discover a replacement owner from its numeric PID or path.
            if children:
                retained = self._retained_owner(owners_before)
                retained.supervisor = children[0]
                self._finish_retained(owners_before, deadline=deadline)
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(set(registry.observations), observations_before)
        self.assertEqual(set(process_module._PROVISIONAL_OWNERS), owners_before)

    def test_status_acquisition_failure_keeps_original_provisional_authority(
        self,
    ) -> None:
        for phase in ("creation", "capture"):
            with self.subTest(phase=phase):
                self._assert_failed_status_handoff(phase)

    def _assert_failed_status_handoff(self, phase: str) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        owners_before = set(process_module._PROVISIONAL_OWNERS)
        acquired: list[process_module._PrivateDirectory] = []
        actual_mkdtemp = tempfile.mkdtemp
        actual_capture = process_module._PrivateDirectory.capture
        deadline = time.monotonic() + 10
        failure = OSError("status acquisition handoff failed")

        def check_pre_effect_owner() -> None:
            retained = self._retained_owner(owners_before)
            observation = retained.observation
            assert isinstance(observation, runtime_test_owner._Observation)
            self.assertIn(observation, registry.observations)
            self.assertTrue(observation.starting)
            self.assertIsNotNone(observation.callback)
            self.assertFalse(retained.spawn_attempted)

        def lose_created_path(*args: Any, **kwargs: Any) -> str:
            check_pre_effect_owner()
            path = actual_mkdtemp(*args, **kwargs)
            acquired.append(actual_capture(Path(path)))
            raise failure

        def lose_captured_guard(path: Path) -> process_module._PrivateDirectory:
            check_pre_effect_owner()
            acquired.append(actual_capture(path))
            raise failure

        try:
            with ExitStack() as patches:
                patches.enter_context(
                    patch.object(tempfile, "mkdtemp", side_effect=lose_created_path)
                    if phase == "creation"
                    else patch.object(
                        process_module._PrivateDirectory,
                        "capture",
                        side_effect=lose_captured_guard,
                    )
                )
                with self.assertRaises(process_module.ProcessOwnershipError):
                    process_module.start_owned_process(
                        sys.executable, ["-c", "pass"], deadline=deadline
                    )
            self.assertEqual(len(acquired), 1)
            retained = self._retained_owner(owners_before)
            self.assertFalse(retained.spawn_attempted)
            self.assertIsNone(retained.status_guard)
            self.assertTrue(retained.status_creation_attempted)
            observation = retained.observation
            assert isinstance(observation, runtime_test_owner._Observation)
            assert observation.callback is not None
            with self.assertRaisesRegex(
                process_module.ProcessOwnershipError,
                "no captured native identity",
            ):
                observation.callback(deadline)
            self.assertFalse(observation.completed)
            self.assertTrue(acquired[0]._assert_identity_or_absence())
        finally:
            if acquired:
                retained = self._retained_owner(owners_before)
                # The controlled failure retained this exact native pin;
                # cleanup must not recapture a path after failed handoff.
                retained.status_guard = acquired[0]
                retained.status_directory = acquired[0].path
                self._finish_retained(owners_before, deadline=deadline)
        self.assertTrue(acquired[0]._pin.retired)
        self.assertEqual(set(registry.observations), observations_before)
        self.assertEqual(set(process_module._PROVISIONAL_OWNERS), owners_before)

    def test_provisional_cleanup_cannot_extend_current_invocation_deadline(
        self,
    ) -> None:
        observation = Mock()
        retained = process_module._ProvisionalProcessOwner(observation=observation)
        with self.assertRaisesRegex(
            process_module.ProcessOwnershipError, "cleanup deadline expired"
        ):
            process_module._cleanup_provisional_owner(
                retained, deadline=time.monotonic() - 1
            )
        observation.complete.assert_not_called()

    def test_independent_exit_cleanup_retries_same_owner_after_startup_deadline(
        self,
    ) -> None:
        registry = runtime_test_owner._current()
        observations_before = set(registry.observations)
        owners_before = set(process_module._PROVISIONAL_OWNERS)
        self.assertFalse(owners_before)
        deadline = time.monotonic() + 1
        captured: list[process_module._ProvisionalProcessOwner] = []
        failure = RuntimeError("constructor failed after the original deadline")

        def fail_late(*_args: Any, **_kwargs: Any) -> None:
            retained = self._retained_owner(owners_before)
            captured.append(retained)
            self.assertIsNotNone(retained.supervisor)
            self.assertIsNotNone(retained.status_guard)
            time.sleep(max(0, deadline - time.monotonic()) + 0.01)
            raise failure

        try:
            with (
                patch.object(process_module, "OwnedProcess", side_effect=fail_late),
                self.assertRaises((TimeoutError, process_module.ProcessOwnershipError)),
            ):
                process_module.start_owned_process(
                    sys.executable, ["-c", "pass"], deadline=deadline
                )
            self.assertGreaterEqual(time.monotonic(), deadline)
            self.assertEqual(len(captured), 1)
            retained = self._retained_owner(owners_before)
            self.assertIs(retained, captured[0])
            supervisor = retained.supervisor
            guard = retained.status_guard
            assert supervisor is not None
            assert guard is not None
            self.assertIsNone(supervisor.poll())
            self.assertFalse(guard._pin.retired)
            self.assertIn(retained.observation, registry.observations)
            # This is the existing independent exit-cleanup invocation. It gets
            # its own 15-second total budget; it does not extend the failed call.
            process_module._cleanup_provisional_owners_at_exit()
            self.assertIs(retained.supervisor, supervisor)
            self.assertIs(retained.status_guard, guard)
            self.assertIsNotNone(supervisor.poll())
            self.assertTrue(guard._pin.retired)
            self.assertNotIn(id(retained), process_module._PROVISIONAL_OWNERS)
            self.assertEqual(set(registry.observations), observations_before)
        finally:
            # Retain the failed assertion, but rescue only this test's known
            # supervisor/pins under an explicit separate fixture cleanup budget.
            for retained in captured:
                if id(retained) in process_module._PROVISIONAL_OWNERS:
                    process_module._cleanup_provisional_owner(
                        retained, deadline=time.monotonic() + 5
                    )

    def _retained_owner(
        self, owners_before: set[int]
    ) -> process_module._ProvisionalProcessOwner:
        retained = [
            owner
            for identity, owner in process_module._PROVISIONAL_OWNERS.items()
            if identity not in owners_before
        ]
        self.assertEqual(len(retained), 1)
        return retained[0]

    def _finish_retained(self, owners_before: set[int], *, deadline: float) -> None:
        retained = self._retained_owner(owners_before)
        observation = retained.observation
        assert isinstance(observation, runtime_test_owner._Observation)
        assert observation.callback is not None
        observation.callback(deadline)
