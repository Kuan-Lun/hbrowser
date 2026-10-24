from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from textwrap import dedent, indent

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FILES = {
    "process.py",
    "_process_control.py",
    "_process_observation.py",
    "_process_supervisor.py",
    "_directory_cleanup_worker.py",
}


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("pytest-source.py", "runtime_test_owner.py"):
        shutil.copy2(_ROOT / "scripts" / name, scripts / name)
    for relative in ("hbrowser", "scripts/_shared"):
        shutil.copytree(
            _ROOT / relative,
            root / relative,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    configuration = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    components = configuration["tool"]["h2h-dev-tools"]["components"]
    (root / "pyproject.toml").write_text(
        "[tool.h2h-dev-tools]\ncomponents = " + json.dumps(components) + "\n"
    )
    return root


def _poison_cache(module: Path, tmp_path: Path) -> Path:
    stale = tmp_path / "stale.py"
    prefix = b"raise RuntimeError('STALE WORKER BYTECODE')\n"
    stale.write_bytes(prefix + b" " * (module.stat().st_size - len(prefix)))
    stamp = module.stat().st_mtime_ns
    os.utime(stale, ns=(stamp, stamp))
    cache = (
        module.parent
        / "__pycache__"
        / f"{module.stem}.{sys.implementation.cache_tag}.pyc"
    )
    cache.parent.mkdir(exist_ok=True)
    py_compile.compile(
        str(stale),
        cfile=str(cache),
        dfile=str(module),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    return cache


@pytest.mark.parametrize("entry", ["pytest", "harness"])
@pytest.mark.parametrize(
    "poison", ["_process_control.py", "process.py", "_process_observation.py"]
)
def test_worker_source_projection_ignores_valid_stale_sibling_cache(
    tmp_path: Path, entry: str, poison: str
) -> None:
    root = _checkout(tmp_path)
    worker_directory = root / "hbrowser/gallery/browser"
    cache = _poison_cache(worker_directory / poison, tmp_path)
    poisoned_bytes = cache.read_bytes()
    evidence = tmp_path / "projection.json"
    # No-argument probes execute the real direct-script import graph, then
    # reject arguments before creating any target/cleanup child. The old
    # implementation therefore fails safely before the genuine owner exercise.
    body = dedent(f"""
        import json, stat, subprocess, sys, time
        from pathlib import Path
        import runtime_test_owner as adapter
        from hbrowser.gallery.browser import process
        from hbrowser.gallery.browser._process_control import (
            PROVEN_TARGET_NOT_STARTED_EXIT_CODE)
        for script, expected in (
            (process._SUPERVISOR_PATH, PROVEN_TARGET_NOT_STARTED_EXIT_CODE),
            (process._PRIVATE_CLEANUP_WORKER_PATH, 1),
        ):
            probe = subprocess.run(
                [sys.executable, '-E', '-S', str(script)],
                capture_output=True, text=True, timeout=5)
            assert 'STALE WORKER BYTECODE' not in probe.stderr, probe.stderr
            assert probe.returncode == expected, probe.stderr
        registry = adapter._current()
        projection = registry.worker_sources
        assert projection is not None
        registry.install()
        assert registry.worker_sources is projection
        assert projection.directory != Path({str(worker_directory)!r})
        assert set(projection.contents) == {_FILES!r}
        for name in projection.contents:
            source = Path({str(worker_directory)!r}) / name
            exported = projection.directory / name
            assert exported.read_bytes() == source.read_bytes()
            assert stat.S_IMODE(exported.stat().st_mode) == stat.S_IMODE(
                source.stat().st_mode)
        assert process._SUPERVISOR_PATH.parent == projection.directory
        assert process._PRIVATE_CLEANUP_WORKER_PATH.parent == projection.directory
        owner = process.start_owned_process(
            sys.executable, ['-E', '-S', '-c', 'pass'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert owner.wait(timeout=5) == 0
            assert owner._closed and owner._tree_reaped
        finally:
            if not owner._closed:
                owner.shutdown(graceful_timeout=.1, terminate_timeout=.2,
                               kill_timeout=1, cleanup_timeout=2)
        private = Path({str(tmp_path / "private")!r})
        private.mkdir()
        (private / 'child').write_text('remove')
        process._PrivateDirectory.capture(private).remove(
            deadline=time.monotonic()+3)
        assert not private.exists()
        assert not registry.observations
        projection.verify()
        Path({str(evidence)!r}).write_text(json.dumps({{
            'directory': str(projection.directory),
            'source': str(projection.source), 'files': sorted(projection.contents),
        }}))
    """)
    if entry == "pytest":
        tests = root / "tests"
        tests.mkdir()
        (tests / "test_projection.py").write_text(
            "def test_projection():\n" + indent(body, "    ")
        )
        arguments = [str(root / "scripts/pytest-source.py"), "-q", "-o", "addopts="]
    else:
        script = root / "scripts/pytest-source.py"
        arguments = [
            "-c",
            f"import runpy; state = runpy.run_path({str(script)!r}); "
            f"state['adapter'].run(lambda: exec({body!r}))",
        ]
    # Normal harness startup inherits candidate source authority; only its
    # production workers deliberately ignore environment prefixes with -E -S.
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(root),
        PYTHONPYCACHEPREFIX=str(tmp_path / "controller-cache"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(evidence.read_text())
    assert not Path(proof["directory"]).exists()
    assert proof["source"] == str(worker_directory)
    assert set(proof["files"]) == _FILES
    assert cache.read_bytes() == poisoned_bytes
