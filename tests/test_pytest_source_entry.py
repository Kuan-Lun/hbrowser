from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = """import runpy
from pathlib import Path

package = Path(__file__).resolve().parents[1] / "scripts/_shared/h2h_dev_tools"
runpy.run_path(str(package / "source_loader.py"))["load_package"](
    package, "_shared.h2h_dev_tools"
)
"""
_TEST = """from pathlib import Path
from _shared.h2h_dev_tools import version_policy
import hbrowser

def test_candidate():
    assert getattr(version_policy.__loader__, "_h2h_source_registration", None)
    assert hbrowser.CANDIDATE == 'candidate runtime source'
    assert Path(hbrowser.__file__).resolve() == (
        Path(__file__).resolve().parents[1] / 'hbrowser/__init__.py'
    )
"""


def _run(
    root: Path, environment: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, *arguments),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(_ROOT / "scripts/pytest-source.py", scripts / "pytest-source.py")
    shutil.copy2(
        _ROOT / "scripts/runtime_test_owner.py", scripts / "runtime_test_owner.py"
    )
    shutil.copytree(
        _ROOT / "hbrowser",
        root / "hbrowser",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        _ROOT / "scripts/_shared",
        scripts / "_shared",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    configuration = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    components = configuration["tool"]["h2h-dev-tools"]["components"]
    (root / "pyproject.toml").write_text(
        "[tool.h2h-dev-tools]\ncomponents = " + json.dumps(components) + "\n"
    )
    runtime = root / "hbrowser/__init__.py"
    runtime.write_text(
        runtime.read_text() + "\nCANDIDATE = 'candidate runtime source'\n"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_candidate.py").write_text("def test_warmup(): pass\n")
    conftest = tests / "conftest.py"
    # Warm the real pytest assertion-rewrite cache using a previous conftest
    # without a bootstrap, then restore equal-size source with the same mtime.
    previous = "# prior conftest without snapshot bootstrap\n"
    conftest.write_text(previous + " " * (len(_BOOTSTRAP) - len(previous)), newline="")
    timestamp = conftest.stat().st_mtime_ns
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("GIT_", "PYTHON", "PYTEST_"))
    }
    environment.update(
        PYTHONPATH=str(scripts),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    warmup = _run(root, environment, "-m", "pytest", "-p", "no:cacheprovider", "-q")
    assert warmup.returncode == 0, warmup.stdout + warmup.stderr
    caches = list((tests / "__pycache__").glob("conftest.*-pytest-*.pyc"))
    assert len(caches) == 1
    conftest.write_text(_BOOTSTRAP, newline="")
    os.utime(conftest, ns=(timestamp, timestamp))
    # Match the candidate header while retaining the previous bootstrap code.
    # Pytest 9.2 checks source hashes, so matching mtime/size alone no longer
    # establishes a negative control that actually executes untrusted bytecode.
    cache = caches[0]
    poisoned = cache.read_bytes()
    flags = int.from_bytes(poisoned[4:8], "little")
    assert flags in {0, 1, 3}, f"Unknown Python cache flags: {flags}"
    if flags & 1:
        cache.write_bytes(
            poisoned[:8]
            + importlib.util.source_hash(conftest.read_bytes())
            + poisoned[16:]
        )
    (tests / "test_candidate.py").write_text(_TEST)

    for module, label in (
        (scripts / "_shared/h2h_dev_tools/version_policy.py", "SHARED"),
        (runtime, "RUNTIME"),
    ):
        stale = tmp_path / f"stale-{label.lower()}.py"
        prefix = f"raise RuntimeError('STALE {label} BYTECODE EXECUTED')\n".encode()
        stale.write_bytes(prefix.ljust(module.stat().st_size, b" "))
        stamp = module.stat().st_mtime_ns
        os.utime(stale, ns=(stamp, stamp))
        # The negative-control child has no private cache prefix, regardless of
        # the parent canonical runner's prefix. Seed that child's default cache.
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
    ordinary = _run(
        root, environment, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"
    )
    assert ordinary.returncode != 0
    assert "STALE SHARED BYTECODE EXECUTED" in ordinary.stdout + ordinary.stderr
    return root, environment


@pytest.mark.parametrize(
    "selection,expected", [("success", 0), ("failure", 1), ("empty", 5)]
)
def test_pytest_source_entry_ignores_cached_bootstrap_and_preserves_cli_status(
    tmp_path: Path, selection: str, expected: int
) -> None:
    root, environment = _checkout(tmp_path)
    if selection == "failure":
        (root / "tests/test_failure.py").write_text(
            "def test_failure():\n    assert False, 'expected fixture failure'\n"
        )
    arguments = ("-k", "no_selected_test") if selection == "empty" else ()

    result = _run(
        root,
        environment,
        "-B",
        "scripts/pytest-source.py",
        "-p",
        "no:cacheprovider",
        "-q",
        *arguments,
    )

    assert result.returncode == expected, result.stdout + result.stderr
    assert "STALE SHARED BYTECODE EXECUTED" not in result.stdout + result.stderr
    assert "STALE RUNTIME BYTECODE EXECUTED" not in result.stdout + result.stderr
    if selection == "success":
        assert "1 passed" in result.stdout
    elif selection == "failure":
        assert "1 failed, 1 passed" in result.stdout
    else:
        assert "1 deselected" in result.stdout


def test_pytest_adapter_bootstrap_never_uses_adjacent_timestamp_cache(
    tmp_path: Path,
) -> None:
    root, environment = _checkout(tmp_path)
    adapter = root / "scripts/runtime_test_owner.py"
    stale = tmp_path / "stale-adapter.py"
    prefix = b"raise RuntimeError('STALE OWNER ADAPTER')\n"
    stale.write_bytes(prefix + b" " * (adapter.stat().st_size - len(prefix)))
    timestamp = adapter.stat().st_mtime_ns
    os.utime(stale, ns=(timestamp, timestamp))
    cache = (
        adapter.parent
        / "__pycache__"
        / f"{adapter.stem}.{sys.implementation.cache_tag}.pyc"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(
        str(stale),
        cfile=str(cache),
        dfile=str(adapter),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    old = _run(root, environment, "-B", "-c", "import runtime_test_owner")
    assert old.returncode != 0
    assert "STALE OWNER ADAPTER" in old.stderr
    current = _run(
        root,
        environment,
        "-B",
        "scripts/pytest-source.py",
        "-q",
        "-p",
        "no:cacheprovider",
    )
    assert current.returncode == 0, current.stdout + current.stderr
    assert "1 passed" in current.stdout
