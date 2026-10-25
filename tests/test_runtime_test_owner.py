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


@pytest.mark.parametrize("entry", ["run", "plugin"])
@pytest.mark.parametrize(
    "phase, signal_number, uncertain",
    [
        ("scope", signal.SIGINT, False),
        ("scope", signal.SIGTERM, False),
        ("handler-before", 0, False),
        ("handler-after", 0, False),
        ("handoff", signal.SIGTERM, False),
        ("projection", signal.SIGTERM, False),
        ("scope", signal.SIGTERM, True),
        ("projection", signal.SIGTERM, True),
        ("scope-cleanup", signal.SIGINT, False),
        ("projection-cleanup", signal.SIGTERM, False),
        ("projection-cleanup", signal.SIGTERM, True),
        ("install-error-cleanup", signal.SIGTERM, False),
        ("normal-cleanup", signal.SIGTERM, False),
        ("normal-cleanup", signal.SIGTERM, True),
    ],
)
def test_registry_entry_owns_setup_until_cleanup_is_proven(
    tmp_path: Path, entry: str, phase: str, signal_number: int, uncertain: bool
) -> None:
    if os.name != "posix":
        pytest.skip("Real POSIX scope and signal setup boundary")
    evidence = tmp_path / "before-rescue.json"
    _ok(f"""
        import json, os, signal, time
        from pathlib import Path
        from unittest.mock import patch
        from _shared.h2h_dev_tools import process as outer_process
        phase, number, uncertain = {phase!r}, {int(signal_number)!r}, {uncertain!r}
        original_create = a.PosixScope.create
        original_close = a.PosixScope.close
        original_signal = signal.signal
        original_registry_close = a._Registry.close
        original_current = a._current
        original_projection = a._WorkerSources
        scopes, projections, closes = [], [], []
        cleanup_handoff_deadlines = []
        deadline = time.monotonic() + a._CLEANUP_SECONDS
        parent = original_create(deadline=deadline)
        inherited = os.environ.get(a.ENVIRONMENT_KEY)
        os.environ[a.ENVIRONMENT_KEY] = parent.capability.encode()
        failure = RuntimeError('handler entry rejected')
        cleanup_failure = OSError('scope closure is uncertain')
        def create(*, deadline):
            scope = original_create(deadline=deadline)
            scopes.append((scope, deadline))
            if phase.startswith('scope'):
                os.kill(os.getpid(), number)
            return scope
        def handler(signum, callback):
            is_setup = getattr(callback, '__self__', None).__class__ is a._Registry
            if phase == 'handler-before' and is_setup and signum == signal.SIGINT:
                raise failure
            result = original_signal(signum, callback)
            if phase == 'handler-after' and is_setup and signum == signal.SIGINT:
                raise failure
            return result
        def current():
            registry = original_current()
            if phase == 'handoff':
                os.kill(os.getpid(), number)
            return registry
        def projection():
            if phase == 'install-error-cleanup':
                raise failure
            value = original_projection()
            projections.append(value)
            if phase.startswith('projection'):
                os.kill(os.getpid(), number)
            return value
        def registry_close(registry):
            if phase.endswith('-cleanup'):
                cleanup_handoff_deadlines.append(registry.deadline)
                os.kill(os.getpid(), signal.SIGTERM)
            original_registry_close(registry)
        def close(scope, *, deadline):
            if any(scope is item for item, _ in scopes):
                closes.append(deadline)
                if uncertain and len(closes) == 1:
                    raise cleanup_failure
            original_close(scope, deadline=deadline)
        with outer_process.controlled_termination_signals():
            handlers = {{
                n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)
            }}
            caught = None
            later_errors = []
            try:
                with patch.object(a.PosixScope, 'create', create), \
                     patch.object(a.PosixScope, 'close', close), \
                     patch.object(signal, 'signal', handler), \
                     patch.object(a, '_current', current), \
                     patch.object(a, '_WorkerSources', projection), \
                     patch.object(a._Registry, 'close', registry_close):
                    try:
                        if {entry!r} == 'run':
                            a.run(lambda: None)
                        else:
                            a.pytest_addhooks()
                    except BaseException as error:
                        caught = error
                    if {entry!r} == 'plugin':
                        # Pytest may unconfigure after a plugin entry failure.
                        # Unknown cleanup must not be retried automatically.
                        for _ in range(2 if uncertain else 1):
                            try:
                                a.pytest_unconfigure()
                            except BaseException as error:
                                if caught is None:
                                    caught = error
                                else:
                                    later_errors.append(type(error).__name__)
                state = {{
                    'phase': phase, 'entry': {entry!r}, 'uncertain': uncertain,
                    'later_errors': later_errors,
                    'scope_directories': [
                        str(s.capability.directory) for s, _ in scopes],
                    'scope_exists': [
                        s.capability.directory.exists() for s, _ in scopes],
                    'creation_deadlines': [d for _, d in scopes],
                    'close_deadlines': closes,
                    'cleanup_handoff_deadlines': cleanup_handoff_deadlines,
                    'cancelled': getattr(a._registry, 'cancelled', None),
                    'parent_pending': parent.has_pending(deadline=deadline),
                    'environment_restored': (os.environ.get(a.ENVIRONMENT_KEY)
                        == parent.capability.encode()),
                    'handlers_restored': all(
                        signal.getsignal(n) == h for n, h in handlers.items()),
                    'projection_exists': [p.directory.exists() for p in projections],
                    'exception': type(caught).__name__ if caught is not None else None,
                }}
                Path({str(evidence)!r}).write_text(json.dumps(state))
                assert caught is not None, state
                assert state['environment_restored'], state
                assert state['handlers_restored'], state
                assert not later_errors, state
                if phase in ('scope-cleanup', 'projection-cleanup'):
                    assert cleanup_handoff_deadlines == closes, state
                    assert state['cancelled'] == number, state
                if uncertain:
                    assert caught is cleanup_failure, (caught, state)
                    assert len(closes) == 1, state
                    assert state['parent_pending'] and all(state['scope_exists']), state
                    assert all(state['projection_exists']), state
                else:
                    assert not state['parent_pending'], state
                    assert not any(state['scope_exists']), state
                    assert not any(state['projection_exists']), state
                    if phase.startswith('handler'):
                        assert caught is failure, (caught, state)
                    elif {entry!r} == 'run':
                        assert isinstance(caught, SystemExit), (caught, state)
                        assert caught.code == 128 + number, (caught, state)
                    elif phase == 'install-error-cleanup':
                        assert caught is failure, (caught, state)
                    else:
                        assert isinstance(caught, KeyboardInterrupt), (caught, state)
                    if phase.startswith('scope'):
                        assert closes == [scopes[0][1]], state
            finally:
                # Before-rescue state above is authoritative. These controls
                # launched no runtime children, so only their exact empty scopes
                # and projection may be rescued, within the original bound.
                for n, previous in handlers.items():
                    original_signal(n, previous)
                for scope, bound in scopes:
                    if scope.capability.directory.exists():
                        assert not scope.has_pending(deadline=min(deadline, bound))
                        scope.proven_empty = True
                        original_close(scope, deadline=min(deadline, bound))
                for value in projections:
                    if value.directory.exists():
                        value.close()
                if inherited is None:
                    os.environ.pop(a.ENVIRONMENT_KEY, None)
                else:
                    os.environ[a.ENVIRONMENT_KEY] = inherited
                assert not parent.has_pending(deadline=deadline)
                parent.proven_empty = True
                original_close(parent, deadline=deadline)
    """)


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


@pytest.mark.parametrize("failure", ["verification", "second-binding"])
def test_worker_projection_install_failure_cleans_partial_export(failure: str) -> None:
    _ok(f"""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        original = (process._SUPERVISOR_PATH, process._PRIVATE_CLEANUP_WORKER_PATH)
        original_patch = patch.object
        def bind(target, name, value):
            if name == '_PRIVATE_CLEANUP_WORKER_PATH':
                raise RuntimeError('copy failed')
            return original_patch(target, name, value)
        fault = (
            patch.object(a._WorkerSources, 'verify',
                         side_effect=RuntimeError('copy failed'))
            if {failure!r} == 'verification'
            else patch.object(a.patch, 'object', side_effect=bind))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'projection'
            destination.mkdir()
            with patch.object(a.tempfile, 'mkdtemp', return_value=str(destination)):
                with fault:
                    try:
                        registry.install()
                    except RuntimeError as error:
                        assert str(error) == 'copy failed'
                    else:
                        raise AssertionError('invalid projection accepted')
            assert not destination.exists()
        assert (process._SUPERVISOR_PATH,
                process._PRIVATE_CLEANUP_WORKER_PATH) == original
        assert not registry.installed and registry.worker_sources is None
        registry.close()
    """)


def test_worker_projection_retained_when_nested_scope_cannot_be_proven_empty() -> None:
    if os.name != "posix":
        pytest.skip("POSIX nested scope ledger contract")
    _ok("""
        import time
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        registry.install()
        projection = registry.worker_sources
        nested = a.PosixScope.create(deadline=time.monotonic()+2)
        has_pending = registry.scope.has_pending
        def reject_pending(*, deadline):
            assert has_pending(deadline=deadline)
            raise RuntimeError('nested owner remains')
        with patch.object(registry.scope, 'has_pending', side_effect=reject_pending):
            try:
                registry.close()
            except RuntimeError as error:
                assert str(error) == 'nested owner remains'
            else:
                raise AssertionError('nested scope ignored')
        assert projection.directory.exists()
        assert process._SUPERVISOR_PATH.parent == projection.directory
        # This synthetic nested scope never admitted a process. Remove its
        # actual durable obligation before retrying the outer close.
        nested.proven_empty = True
        nested.close(deadline=time.monotonic()+2)
        registry.close()
        assert not projection.directory.exists()
    """)


def test_worker_projection_retained_until_cleanup_proof_is_complete() -> None:
    _ok("""
        import time
        from hbrowser.gallery.browser import process
        registry = a._current()
        original = (process._SUPERVISOR_PATH, process._PRIVATE_CLEANUP_WORKER_PATH)
        registry.install()
        projection = registry.worker_sources
        registry.install()
        assert registry.worker_sources is projection
        token = registry.admit(recovery=False, deadline=time.monotonic()+2)
        def fail(deadline):
            token.complete(deadline=deadline)
            raise RuntimeError('cleanup proof rejected')
        token.bind(fail, deadline=time.monotonic()+2)
        token.finish_start(deadline=time.monotonic()+2)
        try:
            registry.close()
        except RuntimeError as error:
            assert str(error) == 'Runtime test cleanup failed'
        else:
            raise AssertionError('cleanup failure ignored')
        assert projection.directory.exists()
        assert process._SUPERVISOR_PATH.parent == projection.directory
        registry.close()
        assert not projection.directory.exists()
        assert (process._SUPERVISOR_PATH,
                process._PRIVATE_CLEANUP_WORKER_PATH) == original
    """)


@pytest.mark.parametrize("drift", ["bytes", "mode", "extra"])
def test_worker_projection_rejects_changed_export(drift: str) -> None:
    if drift == "mode" and os.name == "nt":
        pytest.skip("Windows chmod does not expose POSIX executable bits")
    _ok(f"""
        from hbrowser.gallery.browser import process
        registry = a._current()
        original = (process._SUPERVISOR_PATH, process._PRIVATE_CLEANUP_WORKER_PATH)
        registry.install()
        projection = registry.worker_sources
        path = projection.directory / 'process.py'
        if {drift!r} == 'bytes':
            path.write_bytes(path.read_bytes() + b'\\n')
        elif {drift!r} == 'mode':
            path.chmod(path.stat().st_mode ^ 0o100)
        else:
            (projection.directory / 'extra.py').touch()
        try:
            registry.close()
        except RuntimeError as error:
            assert str(error) == 'Runtime worker source projection changed'
        else:
            raise AssertionError('changed projection accepted')
        assert not projection.directory.exists()
        assert (process._SUPERVISOR_PATH,
                process._PRIVATE_CLEANUP_WORKER_PATH) == original
    """)


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
        import sys, threading, time
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
        assert owners[0]._supervisor.poll() is not None
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
        import signal, sys
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        original = process._read_supervisor_status
        owners = []
        def cancel(owner, *args, **kwargs):
            owners.append(owner)
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
        assert owners and a._current().finished
        for owner in owners:
            assert owner._closed
            assert owner._supervisor.poll() is not None
    """)


def test_pre_spawn_registration_failure_releases_only_new_no_child_token() -> None:
    _ok("""
        import tempfile, time
        from pathlib import Path
        from unittest.mock import patch
        from hbrowser.gallery.browser import process
        registry = a._current()
        try:
            registry.install()
            with tempfile.TemporaryDirectory() as directory:
                guard = process._PrivateDirectory.capture(Path(directory))
                claim = process._PrivateDirectoryClaim(
                    guard, guard.path.with_name(".hbrowser-cleanup-test"))
                prior = process._ActivePrivateDirectoryCleanup(guard=guard)
                process._register_active_private_directory_cleanup(prior)
                try:
                    with patch.object(process.subprocess, 'Popen') as popen:
                        try:
                            process._spawn_private_directory_cleanup(
                                guard, claim=claim, work_deadline=time.monotonic()+1,
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
        finally:
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
