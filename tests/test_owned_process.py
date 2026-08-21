import atexit
import ctypes
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import zendriver as zd

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


class ProcessPolicyTests(unittest.TestCase):
    def test_unknown_platform_fails_closed(self) -> None:
        with (
            patch.object(os, "name", "unsupported"),
            self.assertRaisesRegex(RuntimeError, "Unsupported"),
        ):
            process_module._supervisor_creation_options()

    def test_windows_job_assignment_precedes_start_gate(self) -> None:
        events: list[str] = []
        supervisor = Mock(pid=101, stdout=None, stderr=None)
        supervisor.poll.return_value = None
        supervisor.wait.return_value = 0
        supervisor.stdin.write.side_effect = lambda _: events.append("start")
        job = Mock()
        job.assign.side_effect = lambda _: events.append("assign")
        job.wait_empty.return_value = None

        def launch_supervisor(*_: object, **__: object) -> Mock:
            events.append("popen")
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
        ):
            owner = process_module.start_owned_process("browser", [])
            owner.terminate()
            owner.wait(timeout=1)

        self.assertEqual(events[:5], ["popen", "job", "assign", "start", "ready"])
        job.terminate.assert_called_once_with()
        job.wait_empty.assert_called_once_with(timeout=ANY)
        job.close.assert_called_once_with()

    def test_windows_assignment_failure_never_opens_start_gate(self) -> None:
        supervisor = Mock(pid=101, stdout=None, stderr=None)
        supervisor.wait.return_value = 0
        job = Mock()
        job.assign.side_effect = OSError("assignment failed")

        with (
            patch.object(process_module, "_ownership_platform", return_value="nt"),
            patch.object(
                process_module, "_supervisor_creation_options", return_value={}
            ),
            patch.object(subprocess, "Popen", return_value=supervisor),
            patch.object(process_module._WindowsJob, "create", return_value=job),
            self.assertRaisesRegex(OSError, "assignment failed"),
        ):
            process_module.start_owned_process("browser", [])

        supervisor.stdin.write.assert_not_called()
        supervisor.terminate.assert_called_once_with()
        supervisor.wait.assert_called_once_with(timeout=5)
        job.close.assert_called_once_with()

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

            owner.wait(timeout=1)
            self.assertFalse(private_directory.exists())
            self.assertFalse(status_directory.exists())
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


if __name__ == "__main__":
    unittest.main()
