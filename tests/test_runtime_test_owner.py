from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ENTRY = _ROOT / "scripts/pytest-source.py"
_BOOTSTRAP = (
    f"import runpy\nstate = runpy.run_path({str(_ENTRY)!r})\na = state['adapter']\n"
)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Every automatic subprocess retains the parent capability. Its adapter
    # registers before runtime sessions exist, including when the whole gate
    # is cancelled while this test's finally block cannot run.
    environment["PYTHONPATH"] = str(_ROOT)
    return environment


def _run(source: str, *, timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-c", _BOOTSTRAP + dedent(source)),
        cwd=_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _ok(source: str) -> None:
    result = _run(source)
    assert result.returncode == 0, result.stdout + result.stderr


def test_closed_invocation_rejects_real_background_admission() -> None:
    _ok("""
        import threading
        from hbrowser.gallery.browser import process
        a.run(lambda: None)
        failures = []
        def late():
            try:
                process.start_owned_process('/definitely/not/a/program', [])
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=late)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()
        assert failures == ['Runtime test owner has closed process admission']
        try:
            a.run(lambda: None)
        except RuntimeError as error:
            assert 'only run once' in str(error)
        else:
            raise AssertionError('invocation was reused')
    """)


def test_expired_admission_and_stable_callback_transfer_are_rejected() -> None:
    _ok("""
        import time
        registry = a._current()
        try:
            registry.admit(recovery=False, deadline=time.monotonic() - 1)
        except TimeoutError:
            pass
        else:
            raise AssertionError('late admission succeeded')
        token = registry.admit(recovery=False, deadline=time.monotonic()+2)
        calls = []
        token.bind(lambda d: calls.append('provisional'), deadline=time.monotonic()+2)
        def final(deadline):
            calls.append('final')
            token.complete(deadline=deadline)
        token.bind(final, deadline=time.monotonic()+2)
        token.finish_start(deadline=time.monotonic()+2)
        try:
            token.bind(lambda d: None, deadline=time.monotonic()+2)
        except RuntimeError:
            pass
        else:
            raise AssertionError('stable callback overwritten')
        registry.close()
        assert calls == ['final']
    """)


def test_starting_background_owner_handoffs_before_cleanup() -> None:
    _ok("""
        import os, sys, threading, time
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        registry.install()
        entered = threading.Event()
        release = threading.Event()
        original = process._read_supervisor_status
        owners = []
        failures = []
        def pause(*args, **kwargs):
            if threading.current_thread().name == 'background-start':
                entered.set()
                assert release.wait(2)
            return original(*args, **kwargs)
        def start():
            try:
                owners.append(process.start_owned_process(
                    sys.executable, ['-c', 'import time; time.sleep(30)']))
            except BaseException as error:
                failures.append(error)
        with patch.object(process, '_read_supervisor_status', side_effect=pause):
            thread = threading.Thread(target=start, name='background-start')
            thread.start()
            assert entered.wait(2)
            assert len(registry.observations) == 1
            token = next(iter(registry.observations))
            assert token.starting and token.callback is not None
            timer = threading.Timer(.05, release.set)
            timer.start()
            registry.close()
            thread.join(2)
            timer.join(2)
        assert not failures, failures
        assert len(owners) == 1 and owners[0]._closed
        assert not thread.is_alive()
        try:
            os.kill(owners[0].pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('supervisor survived proof')
    """)


def test_mock_owner_real_private_cleanup_remains_observed() -> None:
    _ok("""
        import tempfile, time
        from pathlib import Path
        from unittest.mock import Mock, patch
        from hbrowser.gallery.browser import process, _process_observation
        registry = a._current()
        registry.install()
        admissions = []
        original = registry.admit
        def admit(**kwargs):
            admissions.append(kwargs['recovery'])
            return original(**kwargs)
        registry.admit = admit
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private'
            path.mkdir()
            with patch.object(process.subprocess, 'Popen', return_value=Mock()):
                with patch.object(_process_observation, 'observer', None):
                    mock_owner = process.OwnedProcess(
                        Mock(pid=123, stdin=None), target_process_group=None,
                        windows_job=None, stdout_drain=None, stderr_drain=None,
                        status_directory=path, cleanup_paths=())
                    mock_owner._tree_reaped = True
            mock_owner._release_ownership(deadline=time.monotonic()+3)
            assert mock_owner._closed and not path.exists()
        assert admissions == [True], admissions
        assert not registry.observations
        registry.close()
    """)


def test_windows_dispatch_does_not_create_posix_scope() -> None:
    _ok("""
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        a.os = SimpleNamespace(name='nt', environ=os.environ)
        with patch.object(a.PosixScope, 'create', side_effect=AssertionError('POSIX')):
            assert a.run(lambda: 7) == 7
            assert a._current().scope is None
    """)


def test_pytest_term_stops_later_tests_and_waits_for_runtime_receipts(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX nested session cancellation contract")
    identity = tmp_path / "identity.json"
    later = tmp_path / "later"
    test = tmp_path / "test_cancel.py"
    test.write_text(
        dedent(f"""
        import json, os, signal, sys, time
        from pathlib import Path
        from hbrowser.gallery.browser.process import start_owned_process
        def test_a_cancel():
            target = ("import signal,time;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)")
            owner = start_owned_process(sys.executable, ['-c', target])
            Path({str(identity)!r}).write_text(json.dumps({{
                'pytest': os.getpid(), 'supervisor': owner.pid,
                'target': owner.target_pid}}))
            time.sleep(60)
        def test_b_must_not_run():
            Path({str(later)!r}).touch()
    """)
    )
    command = (
        sys.executable,
        str(_ENTRY),
        "-o",
        "addopts=",
        "-q",
        "-p",
        "no:cacheprovider",
        str(test),
    )
    # The real outer command owns pytest's group and waits for the adapter's
    # durable nested scope. A plain Popen receipt is not the cleanup oracle.
    owner_script = _BOOTSTRAP + dedent(f"""
        from _shared.h2h_dev_tools import process as p
        from pathlib import Path
        try:
            phase = p.ProcessCommand(
                'pytest cancellation', {command!r}, Path({str(_ROOT)!r}))
            result = p.run_command(phase, hard_deadline=None)
        except p.RunnerSignalInterrupt as error:
            result = 128 + error.signal_number
        raise SystemExit(result)
    """)
    with (tmp_path / "owner.log").open("w+") as output:
        outer = subprocess.Popen(
            (sys.executable, "-c", owner_script),
            cwd=_ROOT,
            env=_environment(),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while not identity.exists():
                assert outer.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            ids = json.loads(identity.read_text())
            os.kill(outer.pid, signal.SIGTERM)
            assert outer.wait(timeout=9) == 143
            assert not later.exists()
            for pid in ids.values():
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            output.seek(0)
            assert "owned_tree_empty=True" in output.read()
        finally:
            if outer.poll() is None:
                outer.terminate()
                outer.wait(timeout=9)


def test_signal_during_starting_defers_until_runtime_owner_is_bound() -> None:
    _ok("""
        import os, signal, sys
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        original = process._read_supervisor_status
        pids = []
        def cancel(owner, *args, **kwargs):
            pids.append(owner.pid)
            signal.raise_signal(signal.SIGTERM)
            assert a._current().cancelled == signal.SIGTERM
            return original(owner, *args, **kwargs)
        with patch.object(process, '_read_supervisor_status', side_effect=cancel):
            try:
                a.run(lambda: process.start_owned_process(
                    sys.executable, ['-c', 'import time; time.sleep(30)']))
            except SystemExit as error:
                assert error.code == 143
            else:
                raise AssertionError('startup cancellation was lost')
        assert pids and a._current().finished
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError('startup owner survived cleanup')
    """)


def test_pre_spawn_registration_failure_releases_only_new_no_child_token() -> None:
    _ok("""
        import tempfile, time
        from pathlib import Path
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        registry.install()
        with tempfile.TemporaryDirectory() as directory:
            guard = process._PrivateDirectory.capture(Path(directory))
            prior = process._ActivePrivateDirectoryCleanup(guard=guard)
            process._register_active_private_directory_cleanup(prior)
            try:
                with patch.object(process.subprocess, 'Popen') as popen:
                    try:
                        process._spawn_private_directory_cleanup(
                            guard, work_deadline=time.monotonic()+1,
                            ownership_deadline=time.monotonic()+2)
                    except process.ProcessOwnershipError:
                        pass
                    else:
                        raise AssertionError('duplicate worker was admitted')
                    popen.assert_not_called()
                assert not registry.observations
                assert process._active_private_directory_cleanup(guard) is prior
            finally:
                process._release_active_private_directory_cleanup(prior)
        registry.close()
    """)


def test_sealed_scope_blocks_atexit_retry_after_reaped_cleanup_failure() -> None:
    _ok("""
        import tempfile, time
        from pathlib import Path
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        registry.install()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / 'private'
            private.mkdir()
            worker = root / 'failed-worker.py'
            worker.write_text(
                "import sys; sys.stdin.readline(); raise SystemExit(9)\\n")
            guard = process._PrivateDirectory.capture(private)
            with patch.object(process, '_PRIVATE_CLEANUP_WORKER_PATH', worker):
                try:
                    guard.remove(deadline=time.monotonic()+3)
                except process.ProcessOwnershipError:
                    pass
                else:
                    raise AssertionError('failed private deletion was accepted')
            assert private.is_dir()
            assert not registry.observations
            assert process._active_private_directory_cleanup(guard) is None
            key = process._private_directory_key(guard)
            assert key in process._PENDING_PRIVATE_DIRECTORIES
            registry.close()
            with patch.object(process.subprocess, 'Popen') as spawn:
                process._cleanup_pending_private_directories_at_exit()
                spawn.assert_not_called()
            assert key in process._PENDING_PRIVATE_DIRECTORIES
            assert private.is_dir()
    """)
