import atexit
import ctypes
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import ANY, Mock, call, patch

import zendriver as zd

from hbrowser.gallery.browser import _directory_cleanup_worker as cleanup_worker_module
from hbrowser.gallery.browser import _process_control as control_module
from hbrowser.gallery.browser import _process_supervisor as supervisor_module
from hbrowser.gallery.browser import process as process_module


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
                patch.object(subprocess, "Popen", side_effect=launch_target),
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
                patch.object(subprocess, "Popen") as popen,
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
                patch.object(subprocess, "Popen") as popen,
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
                    patch.object(subprocess, "Popen") as popen,
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
                patch.object(subprocess, "Popen", return_value=target),
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
                patch.object(subprocess, "Popen") as popen,
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
            harness = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    harness_script,
                    str(identity),
                    str(harness_signal),
                    str(target_signal),
                    str(stop_file),
                ],
                start_new_session=True,
            )
            try:
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
            finally:
                if harness.poll() is None:
                    harness.terminate()
                    harness.wait(timeout=7)

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


class PrivateDirectoryCleanupTests(unittest.TestCase):
    def test_owned_worker_removes_exact_private_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-owned-remove-") as directory:
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
                cleanup_worker_module._exit_when_parent_channel_closes(read_descriptor)
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
        with tempfile.TemporaryDirectory(prefix="hbrowser-worker-error-") as directory:
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
                    patch.object(subprocess, "Popen", return_value=process) as popen,
                    self.assertRaises(process_module.ProcessOwnershipError) as raised,
                ):
                    guard.remove(deadline=time.monotonic() + 2)

                command = popen.call_args.args[0]
                self.assertEqual(
                    command[1],
                    str(process_module._PRIVATE_CLEANUP_WORKER_PATH),
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
        with tempfile.TemporaryDirectory(prefix="hbrowser-worker-assign-") as directory:
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
                patch.object(subprocess, "Popen", return_value=process),
                patch.object(process_module._WindowsJob, "create", return_value=job),
                self.assertRaisesRegex(OSError, "assignment failed"),
            ):
                process_module._spawn_private_directory_cleanup(
                    guard,
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
        with tempfile.TemporaryDirectory(
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
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-worker-timeout-"
        ) as directory:
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
                    patch.object(subprocess, "Popen", return_value=slow_process),
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

    def test_windows_sharing_violation_retries_and_removes_same_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="hbrowser-private-retry-") as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory,
                platform_name="nt",
            )
            original_rmtree = shutil.rmtree
            attempts = 0

            def transient_rmtree(path: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise _windows_cleanup_error(32)
                original_rmtree(path)

            with (
                patch.object(
                    shutil,
                    "rmtree",
                    side_effect=transient_rmtree,
                ),
                patch.object(time, "sleep") as sleep,
            ):
                guard._remove_inline(deadline=time.monotonic() + 1)

            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(
                process_module._PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
            )
            self.assertFalse(private_directory.exists())

    def test_persistent_windows_sharing_violation_stops_at_deadline(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-private-deadline-"
        ) as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory,
                platform_name="nt",
            )
            cleanup_error = _windows_cleanup_error(33)

            with (
                patch.object(
                    shutil,
                    "rmtree",
                    side_effect=cleanup_error,
                ) as rmtree,
                patch.object(
                    time,
                    "monotonic",
                    side_effect=[0.0, 0.0, 0.1, 0.3],
                ),
                patch.object(time, "sleep") as sleep,
                self.assertRaises(process_module.ProcessOwnershipError) as raised,
            ):
                guard._remove_inline(deadline=0.25)

            self.assertIs(raised.exception.__cause__, cleanup_error)
            self.assertIn(
                "winerror=33 attempts=1",
                "\n".join(raised.exception.__notes__),
            )
            rmtree.assert_called_once_with(private_directory)
            sleep.assert_called_once_with(
                process_module._PRIVATE_CLEANUP_RETRY_INITIAL_SECONDS
            )
            self.assertTrue(private_directory.is_dir())

    def test_non_windows_or_non_sharing_errors_fail_without_retry(self) -> None:
        cases = (("posix", 32), ("nt", 5))
        for platform_name, error_code in cases:
            with self.subTest(platform_name=platform_name, error_code=error_code):
                with tempfile.TemporaryDirectory(
                    prefix="hbrowser-private-policy-"
                ) as directory:
                    private_directory = Path(directory) / "profile"
                    private_directory.mkdir()
                    guard = process_module._PrivateDirectory.capture(
                        private_directory,
                        platform_name=platform_name,
                    )
                    cleanup_error = _windows_cleanup_error(error_code)

                    with (
                        patch.object(
                            shutil,
                            "rmtree",
                            side_effect=cleanup_error,
                        ) as rmtree,
                        patch.object(time, "sleep") as sleep,
                        self.assertRaises(PermissionError) as raised,
                    ):
                        guard._remove_inline(deadline=time.monotonic() + 1)

                    self.assertIs(raised.exception, cleanup_error)
                    rmtree.assert_called_once_with(private_directory)
                    sleep.assert_not_called()
                    self.assertTrue(private_directory.is_dir())

    def test_retry_revalidates_identity_before_deleting_replacement(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-private-identity-"
        ) as directory:
            root = Path(directory)
            private_directory = root / "profile"
            displaced_directory = root / "original-profile"
            replacement_marker = private_directory / "replacement"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory,
                platform_name="nt",
            )

            def replace_during_backoff(_: float) -> None:
                private_directory.rename(displaced_directory)
                private_directory.mkdir()
                replacement_marker.write_text("keep", encoding="utf-8")

            with (
                patch.object(
                    shutil,
                    "rmtree",
                    side_effect=_windows_cleanup_error(32),
                ) as rmtree,
                patch.object(
                    time,
                    "sleep",
                    side_effect=replace_during_backoff,
                ),
                self.assertRaisesRegex(
                    process_module.ProcessOwnershipError,
                    "identity changed",
                ),
            ):
                guard._remove_inline(deadline=time.monotonic() + 1)

            rmtree.assert_called_once_with(private_directory)
            self.assertEqual(
                replacement_marker.read_text(encoding="utf-8"),
                "keep",
            )
            self.assertTrue(displaced_directory.is_dir())

    def test_directory_removed_during_backoff_is_idempotent_success(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbrowser-private-absence-"
        ) as directory:
            private_directory = Path(directory) / "profile"
            private_directory.mkdir()
            guard = process_module._PrivateDirectory.capture(
                private_directory,
                platform_name="nt",
            )
            original_rmtree = shutil.rmtree

            with (
                patch.object(
                    shutil,
                    "rmtree",
                    side_effect=_windows_cleanup_error(32),
                ) as rmtree,
                patch.object(
                    time,
                    "sleep",
                    side_effect=lambda _: original_rmtree(private_directory),
                ),
            ):
                guard._remove_inline(deadline=time.monotonic() + 1)

            rmtree.assert_called_once_with(private_directory)
            self.assertFalse(private_directory.exists())


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
            status_path.write_text("ready 123", encoding="utf-8")
            now = [100.0]

            def late_read(*_: object, **__: object) -> str:
                now[0] = 101.0
                return "ready 123"

            with (
                patch.object(time, "monotonic", side_effect=lambda: now[0]),
                patch.object(Path, "read_text", side_effect=late_read),
                self.assertRaisesRegex(TimeoutError, "READY deadline"),
            ):
                process_module._read_supervisor_status(
                    Mock(spec=process_module.OwnedProcess),
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
                side_effect=lambda guard, *, deadline: guard._remove_inline(
                    deadline=deadline
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
            patch.object(subprocess, "Popen", side_effect=launch_supervisor),
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
                side_effect=lambda guard, *, deadline: guard._remove_inline(
                    deadline=deadline
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
            command[1:4],
            ["-m", "hbrowser.gallery.browser._process_supervisor", ANY],
        )
        self.assertEqual(command[4:], ["--", "browser"])
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
            patch.object(subprocess, "Popen", return_value=supervisor),
            patch.object(process_module._WindowsJob, "create", return_value=job),
            patch.object(
                process_module,
                "_remove_private_directory_owned",
                side_effect=lambda guard, *, deadline: guard._remove_inline(
                    deadline=deadline
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
                patch.object(subprocess, "Popen", return_value=supervisor),
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
                patch.object(subprocess, "Popen", return_value=supervisor),
                patch.object(atexit, "register"),
                patch.object(
                    process_module,
                    "_read_supervisor_status",
                    side_effect=ready_timeout,
                ),
                patch.object(
                    process_module,
                    "_remove_private_directory_owned",
                    side_effect=lambda guard, *, deadline: guard._remove_inline(
                        deadline=deadline
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
                    patch.object(subprocess, "Popen", return_value=supervisor),
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
        supervisor = Mock(pid=101, stdin=Mock(), stdout=None, stderr=None)
        supervisor.poll.side_effect = (None, 0)
        supervisor.wait.side_effect = (
            subprocess.TimeoutExpired("supervisor", 5),
            0,
        )
        guard = Mock()
        owner = process_module._ProvisionalProcessOwner(
            supervisor=supervisor,
            status_guard=guard,
            cleanup_guards=(),
        )
        process_module._register_provisional_owner(owner)
        try:
            process_module._cleanup_provisional_owner(
                owner,
                deadline=time.monotonic() + 120,
            )
        finally:
            process_module._release_provisional_owner(owner)

        for wait_call in supervisor.wait.call_args_list:
            self.assertLessEqual(wait_call.kwargs["timeout"], 5.0)
        cleanup_deadline = guard.remove.call_args.kwargs["deadline"]
        self.assertLessEqual(cleanup_deadline - time.monotonic(), 5.0)

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
                side_effect=lambda guard, *, deadline: guard._remove_inline(
                    deadline=deadline
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
                guard._remove_inline(deadline=deadline)

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
                guard._remove_inline(deadline=deadline)

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
                patch.object(subprocess, "Popen", return_value=supervisor),
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
                patch.object(subprocess, "Popen", return_value=supervisor),
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
            harness = """
import sys

from hbrowser.gallery.browser.process import start_owned_process

assert sys.executable.lower() != sys._base_executable.lower()
owner = start_owned_process(
    sys._base_executable,
    ["-c", "import time; time.sleep(30)"],
)
assert owner.target_pid is not None
owner.terminate()
owner.wait(timeout=10)
"""
            result = subprocess.run(
                [str(venv_python), "-c", harness],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
