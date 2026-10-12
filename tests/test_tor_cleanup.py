import atexit
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hbrowser.gallery.browser import tor
from hbrowser.gallery.browser.process import (
    OwnedProcess,
    ProcessOwnershipError,
    start_owned_process,
)
from hbrowser.gallery.utils import Deadline


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
