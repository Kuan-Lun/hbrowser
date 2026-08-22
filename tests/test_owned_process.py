import atexit
import ctypes
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
from typing import cast
from unittest.mock import ANY, Mock, call, patch

import zendriver as zd

from hbrowser.gallery.browser import _directory_cleanup_worker as cleanup_worker_module
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
    setattr(error, "winerror", code)
    return error


@unittest.skipUnless(os.name == "posix", "POSIX process ownership contract")
class PosixOwnedProcessTests(unittest.TestCase):
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
                args = ["cleanup-worker"]
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
                    side_effect=BlockingIOError("pipe full"),
                ) as write,
            ):
                owner.terminate()

            set_blocking.assert_called_once_with(42, False)
            write.assert_called_once_with(42, b"terminate\n")
            control_pipe.write.assert_not_called()

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

    def test_windows_venv_supervisor_bypasses_python_redirector(self) -> None:
        venv_python = r"C:\workspace\.venv\Scripts\python.exe"
        base_python = r"C:\Python314\python.exe"
        with (
            patch.object(sys, "executable", venv_python),
            patch.object(sys, "_base_executable", base_python),
            patch.object(Path, "is_file", return_value=True),
            patch.dict(
                os.environ,
                {"HBROWSER_TEST_SENTINEL": "retained"},
                clear=True,
            ),
        ):
            executable, environment = process_module._supervisor_launch_context("nt")

        self.assertEqual(executable, base_python)
        self.assertEqual(
            environment,
            {
                "HBROWSER_TEST_SENTINEL": "retained",
                "__PYVENV_LAUNCHER__": venv_python,
            },
        )

    def test_windows_non_venv_supervisor_uses_current_interpreter(self) -> None:
        executable_path = r"C:\Python314\python.exe"
        with (
            patch.object(sys, "executable", executable_path),
            patch.object(sys, "_base_executable", r"c:\python314\PYTHON.EXE"),
        ):
            executable, environment = process_module._supervisor_launch_context("nt")

        self.assertEqual(executable, executable_path)
        self.assertIsNone(environment)

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
            ),
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
        self.assertEqual(
            supervisor.stdin.write.call_args_list,
            [call(b"start\n"), call(b"terminate\n")],
        )
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
            supervisor.stdin.write.assert_called_once_with(b"terminate\n")
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
            supervisor.stdin.write.assert_called_once_with(b"terminate\n")
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
            base_executable = cast(str, getattr(sys, "_base_executable"))
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
