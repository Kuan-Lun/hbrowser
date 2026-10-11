"""Run repository pytest and local workers with fresh source import caches."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol


class _GatewaySpec(Protocol):
    @property
    def popen(self) -> object: ...

    @property
    def via(self) -> object: ...


class _Channel(Protocol):
    def waitclose(self) -> None: ...


class _Gateway(Protocol):
    @property
    def spec(self) -> _GatewaySpec: ...

    def remote_exec(self, source: str) -> _Channel: ...


def _registered_hashes(package: Path, namespace: str) -> dict[str, str]:
    registration = getattr(
        globals().get("__loader__"), "_h2h_source_registration", None
    )
    if (
        not isinstance(registration, tuple)
        or len(registration) != 3
        or not isinstance(registration[0], str)
        or Path(registration[0]).resolve() != package
        or registration[1] != namespace
        or not isinstance(registration[2], dict)
    ):
        raise RuntimeError("Pytest runner requires a verified controller source loader")
    hashes = {}
    for filename, digest in registration[2].items():
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise RuntimeError("Invalid controller source registration")
        hashes[filename] = digest
    if "source_loader.py" not in hashes:
        raise RuntimeError("Controller source registration has no bootstrap")
    return hashes


def _run_pytest(
    package: Path, namespace: str, import_root: str, hashes: dict[str, str]
) -> None:
    # Pytest is an invocation dependency, not a runtime requirement of the
    # toolkit distribution or of consumers that select other components.
    import pytest

    class WorkerSource:
        @pytest.hookimpl(optionalhook=True)
        def pytest_xdist_newgateway(self, gateway: _Gateway) -> None:
            # A popen gateway reached through another gateway may be remote.
            # Only local workers share the controller's filesystem authority.
            if not gateway.spec.popen or gateway.spec.via:
                return
            gateway.remote_exec(
                "import hashlib, stat, sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {import_root!r})\n"
                f"package = Path({str(package)!r})\n"
                f"hashes = {hashes!r}\n"
                "path = package / 'source_loader.py'\n"
                "if not stat.S_ISREG(path.lstat().st_mode):\n"
                "    raise RuntimeError("
                "'Source bootstrap differs from the controller registration')\n"
                "content = path.read_bytes()\n"
                "if hashlib.sha256(content).hexdigest() != "
                "hashes['source_loader.py']:\n"
                "    raise RuntimeError("
                "'Source bootstrap differs from the controller registration')\n"
                "bootstrap = {'__file__': str(path), "
                "'__name__': '<worker_source_bootstrap>'}\n"
                "exec(compile(content, str(path), 'exec'), bootstrap)\n"
                "bootstrap['load_package'](\n"
                f"    package, {namespace!r}, expected_hashes=hashes)\n"
            ).waitclose()

    raise SystemExit(pytest.main(plugins=[WorkerSource()]))


def main() -> None:
    """Preserve pytest arguments while isolating caches for this invocation.

    Call from a source-bootstrapped package before importing pytest or xdist.
    Local xdist workers inherit cache settings and bootstrap the same verified
    source before their pytest plugins and conftests load. Test selection,
    process ownership and deadlines stay local.
    """
    package = Path(__file__).resolve().parent
    parts = __package__.split(".")
    if tuple(package.parts[-len(parts) :]) != tuple(parts):
        raise RuntimeError("Pytest source package path does not match its namespace")
    hashes = _registered_hashes(package, __package__)
    import_root = str(package.parents[len(parts) - 1])
    environment_keys = (
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
    )
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    previous_prefix = sys.pycache_prefix
    previous_write_flag = sys.dont_write_bytecode
    previous_path = sys.path
    with tempfile.TemporaryDirectory(prefix="h2h-pytest-source-") as cache:
        try:
            sys.pycache_prefix = cache
            sys.dont_write_bytecode = True
            sys.path = [import_root, *previous_path]
            os.environ["PYTHONPYCACHEPREFIX"] = cache
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            previous_pythonpath = previous_environment["PYTHONPATH"]
            os.environ["PYTHONPATH"] = import_root + (
                os.pathsep + previous_pythonpath
                if previous_pythonpath is not None
                else ""
            )
            _run_pytest(package, __package__, import_root, hashes)
        finally:
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_write_flag
            sys.path = previous_path
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
