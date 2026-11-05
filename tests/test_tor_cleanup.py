import atexit
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

from hbrowser.gallery.browser import tor
from hbrowser.gallery.browser.process import (
    OwnedProcess,
    ProcessOwnershipError,
    start_owned_process,
)
from hbrowser.gallery.utils import Deadline


class TorBootstrapDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0.0
        self.deadline = Deadline(20.0)
        # Twenty seconds leaves the existing half-budget cleanup reserve: the
        # bootstrap deadline is ten, and owner cleanup must keep deadline twenty.
        self.owner = Mock(spec=OwnedProcess)
        self.owner.stdout = [b"Bootstrapped 100%: Done\n"]
        self.owner.poll.return_value = None
        self.lines: deque[str] = deque(maxlen=256)
        self.enterContext(patch.object(time, "monotonic", side_effect=lambda: self.now))
        self.enterContext(patch.object(tor, "_find_tor_binary", return_value="/tor"))
        self.enterContext(
            patch.object(tempfile, "mkdtemp", return_value="/unused/mock-tor-data")
        )
        self.start_process = self.enterContext(
            patch.object(tor, "start_owned_process", return_value=self.owner)
        )
        self.register = self.enterContext(patch.object(atexit, "register"))
        self.unregister = self.enterContext(patch.object(atexit, "unregister"))
        self.enterContext(patch.object(tor, "deque", return_value=self.lines))
        self.enterContext(
            patch.object(threading, "Thread", side_effect=self._immediate_reader)
        )
        self.log_info = self.enterContext(patch.object(tor.logger, "info"))
        self.log_debug = self.enterContext(patch.object(tor.logger, "debug"))
        self.addCleanup(self._release_mock_owner)

    def _immediate_reader(self, *, target: Callable[[], None], daemon: bool) -> Mock:
        self.assertTrue(daemon)
        thread = Mock()
        thread.start.side_effect = target
        return thread

    def _release_mock_owner(self) -> None:
        # Assertions below run before this fallback. No real process is started;
        # failed negative controls still release their mock atexit registration.
        self.owner.shutdown.side_effect = None
        tor.terminate_tor_process(self.owner, deadline=Deadline.after(20))

    def _assert_startup_cleaned_owner(self) -> None:
        self.owner.shutdown.assert_called_once_with(
            graceful_timeout=0,
            terminate_timeout=5,
            kill_timeout=5,
            deadline=self.deadline.expires_at,
        )
        cleanup = getattr(self.owner, tor._TOR_PROCESS_CLEANUP_ATTRIBUTE)
        self.assertIsNone(cleanup._tor_process)
        self.assertFalse(cleanup._registered)
        self.unregister.assert_called_once_with(cleanup)

    def test_continuous_output_checks_deadline_within_the_batch(self) -> None:
        self.owner.stdout = [b"Bootstrapped 10%: progress\n"] * 256

        def refill(_message: str) -> None:
            self.now += 0.5
            self.lines.append("Bootstrapped 10%: more progress")
            if self.log_debug.call_count > 256:
                raise AssertionError("bootstrap kept draining a replenished queue")

        self.log_debug.side_effect = refill
        with self.assertRaisesRegex(RuntimeError, "failed to bootstrap"):
            tor._start_tor_process(9150, deadline=self.deadline)

        self.assertEqual(self.log_debug.call_count, 20)
        self.assertEqual(len(self.lines), 256)
        self.assertEqual(self.now, 10.0)
        self._assert_startup_cleaned_owner()

    def test_continuous_output_returns_to_process_exit_observation(self) -> None:
        self.owner.stdout = [b"progress one\n", b"progress two\n"]
        self.owner.poll.side_effect = [None, 7]
        self.owner.returncode = 7

        def refill(_message: str) -> None:
            self.lines.append("more progress")
            if self.log_debug.call_count > 2:
                raise AssertionError("bootstrap did not return to process polling")

        self.log_debug.side_effect = refill
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly with code 7"):
            tor._start_tor_process(9150, deadline=self.deadline)

        self.assertEqual(self.log_debug.call_count, 2)
        self.assertEqual(self.owner.poll.call_count, 2)
        self._assert_startup_cleaned_owner()

    def test_blank_output_cannot_bypass_the_per_line_deadline(self) -> None:
        case = self

        class RefilledBlankLines(deque[str]):
            def popleft(self) -> str:
                line = super().popleft()
                case.now += 0.5
                self.append("")
                if case.now > 10.0:
                    raise AssertionError("blank output bypassed the shared deadline")
                return line

        lines = RefilledBlankLines(maxlen=256)
        self.owner.stdout = [b"\n"] * 256
        with (
            patch.object(tor, "deque", return_value=lines),
            self.assertRaisesRegex(RuntimeError, "failed to bootstrap"),
        ):
            tor._start_tor_process(9150, deadline=self.deadline)

        self.assertEqual(self.now, 10.0)
        self.log_debug.assert_not_called()
        self._assert_startup_cleaned_owner()

    def test_success_queued_after_slow_progress_does_not_steal_cleanup_reserve(
        self,
    ) -> None:
        self.owner.stdout = [b"progress\n", b"Bootstrapped 100%: Done\n"]

        def exhaust_bootstrap(_message: str) -> None:
            self.now = 10.0

        self.log_debug.side_effect = exhaust_bootstrap
        with self.assertRaisesRegex(RuntimeError, "failed to bootstrap"):
            tor._start_tor_process(9150, deadline=self.deadline)

        self.assertEqual(list(self.lines), ["Bootstrapped 100%: Done"])
        self.assertFalse(
            any(
                "completed successfully" in str(call)
                for call in self.log_info.call_args_list
            )
        )
        self._assert_startup_cleaned_owner()

    def test_success_logging_at_or_after_bootstrap_deadline_is_rejected(self) -> None:
        for completion_time in (10.0, 10.5):
            with self.subTest(completion_time=completion_time):
                self.now = 0.0
                self.owner.shutdown.reset_mock()
                self.unregister.reset_mock()

                def finish_late(
                    message: str, *, finished_at: float = completion_time
                ) -> None:
                    if "completed successfully" in message:
                        self.now = finished_at

                self.log_info.side_effect = finish_late
                with self.assertRaisesRegex(RuntimeError, "failed to bootstrap"):
                    tor.start_tor_with_retry(
                        9150, max_retries=1, deadline=self.deadline
                    )
                self._assert_startup_cleaned_owner()

    def test_success_before_bootstrap_deadline_retains_the_live_owner(self) -> None:
        def finish_in_time(message: str) -> None:
            if "completed successfully" in message:
                self.now = 9.5

        self.log_info.side_effect = finish_in_time
        result = tor.start_tor_with_retry(9150, deadline=self.deadline)

        self.assertIs(result, self.owner)
        cleanup = getattr(self.owner, tor._TOR_PROCESS_CLEANUP_ATTRIBUTE)
        self.assertIs(cleanup._tor_process, self.owner)
        self.assertTrue(cleanup._registered)
        self.owner.shutdown.assert_not_called()
        self.unregister.assert_not_called()
        self.assertEqual(self.start_process.call_args.kwargs["deadline"], 10.0)
        tor.terminate_tor_process(result, deadline=self.deadline)
        self._assert_startup_cleaned_owner()

    def test_late_success_with_unproven_cleanup_retains_owner_and_stops_retries(
        self,
    ) -> None:
        cleanup_error = ProcessOwnershipError("owner receipt unavailable")
        self.owner.shutdown.side_effect = cleanup_error

        def finish_late(message: str) -> None:
            if "completed successfully" in message:
                self.now = 10.0

        self.log_info.side_effect = finish_late
        with self.assertRaises(tor._TorProcessCleanupError) as raised:
            tor.start_tor_with_retry(9150, max_retries=3, deadline=self.deadline)

        self.assertIs(raised.exception.__cause__, cleanup_error)
        self.assertEqual(
            raised.exception.__notes__, ["Bootstrap failure type: RuntimeError"]
        )
        self.start_process.assert_called_once()
        cleanup = getattr(self.owner, tor._TOR_PROCESS_CLEANUP_ATTRIBUTE)
        self.assertIs(cleanup._tor_process, self.owner)
        self.assertTrue(cleanup._registered)
        self.unregister.assert_not_called()
        self.owner.shutdown.side_effect = None
        tor.terminate_tor_process(self.owner, deadline=self.deadline)
        self.assertEqual(self.owner.shutdown.call_count, 2)
        self.assertIsNone(cleanup._tor_process)
        self.unregister.assert_called_once_with(cleanup)

    def test_retry_entry_rejects_late_handoff_without_releasing_owner(self) -> None:
        original_start = tor._start_tor_process

        def delayed_handoff(socks_port: int, *, deadline: Deadline) -> OwnedProcess:
            owner = original_start(socks_port, deadline=deadline)
            self.now = deadline.expires_at
            return owner

        with (
            patch.object(
                tor, "_start_tor_process", side_effect=delayed_handoff
            ) as start,
            self.assertRaises(tor._TorProcessCleanupError) as raised,
        ):
            tor.start_tor_with_retry(9150, max_retries=3, deadline=self.deadline)

        self.assertIsInstance(raised.exception.__cause__, ProcessOwnershipError)
        self.assertIn("deadline expired", str(raised.exception.__cause__))
        start.assert_called_once()
        self.start_process.assert_called_once()
        cleanup = getattr(self.owner, tor._TOR_PROCESS_CLEANUP_ATTRIBUTE)
        self.assertIs(cleanup._tor_process, self.owner)
        self.assertTrue(cleanup._registered)
        self.owner.shutdown.assert_not_called()
        self.unregister.assert_not_called()
        # A later explicit cleanup has its own caller deadline; startup never
        # extends its exhausted deadline or discards this exact retained owner.
        tor.terminate_tor_process(self.owner, deadline=Deadline.after(1))
        self.assertIsNone(cleanup._tor_process)
        self.unregister.assert_called_once_with(cleanup)


class TorCleanupDeadlineTests(unittest.TestCase):
    def test_concurrent_cleanup_honors_short_deadline_without_cancelling_owner(
        self,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        second_done = threading.Event()
        errors: dict[str, BaseException] = {}
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "owned-data"
            data.mkdir()
            owner = start_owned_process(
                sys.executable,
                ["-c", "import time; time.sleep(60)"],
                cleanup_paths=(data,),
            )
            cleanup = tor._TorProcessAtexitCleanup(owner)
            cleanup.register()
            original_release = owner._release_ownership

            def release_after_barrier(*, deadline: float) -> None:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test did not release the cleanup barrier")
                original_release(deadline=deadline)

            def invoke(name: str, seconds: float) -> None:
                try:
                    cleanup.cleanup(deadline=Deadline.after(seconds))
                except BaseException as error:
                    errors[name] = error
                finally:
                    if name == "second":
                        second_done.set()

            first = threading.Thread(target=invoke, args=("first", 10.0))
            second = threading.Thread(target=invoke, args=("second", 0.05))
            try:
                with patch.object(owner, "_release_ownership", release_after_barrier):
                    first.start()
                    self.assertTrue(entered.wait(5), "owner never reached cleanup")
                    second.start()
                    self.assertTrue(
                        second_done.wait(1),
                        "short caller waited for the other caller's cleanup",
                    )
                    self.assertIsInstance(errors.get("second"), ProcessOwnershipError)
                    self.assertTrue(first.is_alive())
                    self.assertTrue(data.exists())
                    self.assertIs(cleanup._tor_process, owner)
                    self.assertTrue(cleanup._registered)
            finally:
                release.set()
                first.join(10)
                if second.ident is not None:
                    second.join(10)
                cleanup.cleanup()
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertNotIn("first", errors)
            self.assertIsNone(cleanup._tor_process)
            self.assertFalse(cleanup._registered)
            self.assertFalse(data.exists())
            self.assertIsNotNone(owner.poll())
            cleanup.cleanup(deadline=Deadline.after(1))

    def test_state_lock_wait_expires_without_releasing_reference(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        errors: list[BaseException] = []
        finished = threading.Event()

        def invoke() -> None:
            try:
                cleanup.cleanup(deadline=Deadline.after(0.05))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with (
            patch.object(atexit, "register"),
            patch.object(atexit, "unregister") as unregister,
        ):
            cleanup.register()
            cleanup._lock.acquire()
            caller = threading.Thread(target=invoke)
            try:
                caller.start()
                self.assertTrue(finished.wait(1), "state lock ignored caller deadline")
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ProcessOwnershipError)
                self.assertIs(cleanup._tor_process, owner)
                self.assertTrue(cleanup._registered)
                owner.shutdown.assert_not_called()
                unregister.assert_not_called()
            finally:
                cleanup._lock.release()
                caller.join(2)
            self.assertFalse(caller.is_alive())
            cleanup.cleanup()
            self.assertIsNone(cleanup._tor_process)
            unregister.assert_called_once_with(cleanup)

    def test_deadline_rechecked_after_state_lock_admission(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        now = [0.0]
        real_lock = cleanup._lock

        def late_acquire(*, timeout: float) -> bool:
            acquired = real_lock.acquire(timeout=timeout)
            now[0] = 1.0
            return acquired

        with (
            patch.object(atexit, "register"),
            patch.object(atexit, "unregister") as unregister,
            patch.object(time, "monotonic", side_effect=lambda: now[0]),
        ):
            cleanup.register()
            with patch.object(cleanup, "_lock", wraps=real_lock) as state_lock:
                state_lock.acquire.side_effect = late_acquire
                with self.assertRaises(ProcessOwnershipError):
                    cleanup.cleanup(deadline=Deadline(1.0))
            self.assertIs(cleanup._tor_process, owner)
            self.assertTrue(cleanup._registered)
            owner.shutdown.assert_not_called()
            unregister.assert_not_called()
            self.assertFalse(real_lock.locked())
        cleanup.cleanup()

    def test_late_shutdown_preserves_registration_until_a_successful_retry(
        self,
    ) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        now = [0.0]

        def finish_late(**_arguments: object) -> None:
            now[0] = 1.0

        owner.shutdown.side_effect = finish_late
        with (
            patch.object(atexit, "register"),
            patch.object(atexit, "unregister") as unregister,
            patch.object(time, "monotonic", side_effect=lambda: now[0]),
        ):
            cleanup.register()
            with self.assertRaises(ProcessOwnershipError):
                cleanup.cleanup(deadline=Deadline(1.0))
            self.assertIs(cleanup._tor_process, owner)
            self.assertTrue(cleanup._registered)
            unregister.assert_not_called()
            owner.shutdown.side_effect = None
            cleanup.cleanup(deadline=Deadline(2.0))
            self.assertIsNone(cleanup._tor_process)
            self.assertFalse(cleanup._registered)
            unregister.assert_called_once_with(cleanup)

    def test_default_budget_includes_state_admission(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        now = [100.0]
        real_lock = cleanup._lock

        def admit_later(*, timeout: float) -> bool:
            now[0] += 1.0
            return real_lock.acquire(timeout=timeout)

        with (
            patch.object(time, "monotonic", side_effect=lambda: now[0]),
            patch.object(cleanup, "_lock", wraps=real_lock) as state_lock,
        ):
            state_lock.acquire.side_effect = admit_later
            cleanup.cleanup()
        owner.shutdown.assert_called_once_with(
            graceful_timeout=0, terminate_timeout=5, kill_timeout=5, deadline=115.0
        )

    def test_interpreter_cleanup_does_not_change_atexit_registration(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        with (
            patch.object(atexit, "register"),
            patch.object(atexit, "unregister") as unregister,
        ):
            cleanup.register()
            cleanup()
            cleanup()
            self.assertIsNone(cleanup._tor_process)
            self.assertTrue(cleanup._registered)
            unregister.assert_not_called()
            cleanup.cleanup()
            cleanup.cleanup()
            unregister.assert_called_once_with(cleanup)
            self.assertFalse(cleanup._registered)
        owner.shutdown.assert_called_once()

    def test_completion_state_lock_timeout_retains_retryable_registration(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        entered = threading.Event()
        finished = threading.Event()
        errors: list[BaseException] = []

        def finish_shutdown(**_arguments: object) -> None:
            if not cleanup._lock.acquire(timeout=0.5):
                raise AssertionError("shutdown retained the Tor state lock")
            entered.set()

        def invoke() -> None:
            try:
                cleanup.cleanup(deadline=Deadline.after(0.05))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        owner.shutdown.side_effect = finish_shutdown
        with (
            patch.object(atexit, "register"),
            patch.object(atexit, "unregister") as unregister,
        ):
            cleanup.register()
            caller = threading.Thread(target=invoke)
            caller.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(finished.wait(1), "completion lock ignored deadline")
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ProcessOwnershipError)
                self.assertIs(cleanup._tor_process, owner)
                self.assertTrue(cleanup._registered)
                unregister.assert_not_called()
            finally:
                if entered.is_set():
                    cleanup._lock.release()
                caller.join(2)
            self.assertFalse(caller.is_alive())
            owner.shutdown.side_effect = None
            cleanup.cleanup()
            self.assertIsNone(cleanup._tor_process)
            unregister.assert_called_once_with(cleanup)

    def test_expired_caller_cannot_claim_cached_cleanup_success(self) -> None:
        owner = Mock(spec=OwnedProcess)
        cleanup = tor._TorProcessAtexitCleanup(owner)
        cleanup.cleanup()
        self.assertIsNone(cleanup._tor_process)
        with self.assertRaises(ProcessOwnershipError):
            cleanup.cleanup(deadline=Deadline(0.0))
        owner.shutdown.assert_called_once()
