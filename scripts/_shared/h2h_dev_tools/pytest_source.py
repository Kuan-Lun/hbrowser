"""Run repository pytest and local workers with fresh source import caches."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .pytest_runtime import (
    RuntimeSourceError,
    RuntimeSources,
    capture_runtime_sources,
    install_runtime_sources,
)

if TYPE_CHECKING:
    from _pytest.assertion.rewrite import AssertionRewritingHook
    from _pytest.config import Config


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


def _install_pytest_sources(runtime: RuntimeSources) -> Callable[[], None]:
    # The loader and finder share a plain registration across run_path bootstrap
    # instances. Select only this invocation's verified toolkit finder, never
    # every ambient import hook or a finder inferred from a package name.
    registration = getattr(
        globals().get("__loader__"), "_h2h_source_registration", None
    )
    matching = [
        finder
        for finder in sys.meta_path
        if getattr(finder, "_h2h_source_registration", None) == registration
    ]
    if registration is None or len(matching) != 1:
        raise RuntimeSourceError(
            "Pytest requires exactly one registered toolkit source finder"
        )
    runtime_finder = install_runtime_sources(runtime)
    protected = [runtime_finder, *matching]
    try:
        # Pytest installs its rewrite finder before loading explicit/autoload
        # plugins and conftests. It directly asks PathFinder and reads source,
        # bypassing verified loaders. Config hooks would run too late here.
        from _pytest import assertion
        from _pytest.assertion.rewrite import AssertionRewritingHook

        original = getattr(assertion, "install_importhook", None)
        if not callable(original):
            raise RuntimeSourceError(
                "Unsupported pytest assertion import-hook installer"
            )

        def install(config: Config) -> AssertionRewritingHook:
            hook = original(config)
            if (
                not isinstance(hook, AssertionRewritingHook)
                or hook not in sys.meta_path
                or any(finder not in sys.meta_path for finder in protected)
            ):
                raise RuntimeSourceError(
                    "Unsupported pytest assertion import-hook registration"
                )
            for finder in protected:
                sys.meta_path.remove(finder)
            position = sys.meta_path.index(hook)
            sys.meta_path[position:position] = protected
            return hook

        assertion.install_importhook = install
    except BaseException:
        sys.meta_path.remove(runtime_finder)
        raise

    def restore() -> None:
        assertion.install_importhook = original
        sys.meta_path.remove(runtime_finder)

    return restore


def _registered_parents() -> dict[str, tuple[str, str | None]]:
    parents = getattr(globals().get("__loader__"), "_h2h_parent_sources", None)
    if not isinstance(parents, dict):
        raise RuntimeError("Pytest runner requires verified parent source authority")
    result = {}
    for name, source in parents.items():
        if (
            not isinstance(name, str)
            or not isinstance(source, tuple)
            or len(source) != 2
            or not isinstance(source[0], str)
            or (source[1] is not None and not isinstance(source[1], str))
        ):
            raise RuntimeError("Invalid controller parent source registration")
        result[name] = source
    return result


def _run_pytest(
    package: Path,
    namespace: str,
    import_root: str,
    hashes: dict[str, str],
    parents: dict[str, tuple[str, str | None]],
    cache: str,
    runtime: RuntimeSources,
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
            # Interpreter flags such as -I/-E ignore the inherited PYTHON*
            # environment. Set the controller-owned cache authority before
            # this worker loads pytest, plugins, conftests or test modules.
            gateway.remote_exec(
                "import sys\n"
                f"sys.pycache_prefix = {cache!r}\n"
                "sys.dont_write_bytecode = True\n"
                "import hashlib, stat\nfrom pathlib import Path\n"
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
                f"    package, {namespace!r}, expected_hashes=hashes,\n"
                f"    expected_parents={parents!r})\n"
                f"from {namespace}.pytest_source import _install_pytest_sources\n"
                f"_install_pytest_sources({runtime!r})\n"
            ).waitclose()

    raise SystemExit(pytest.main(plugins=[WorkerSource()]))


def main(*, runtime_sources: Mapping[str, Path]) -> None:
    """Preserve pytest arguments while isolating caches for this invocation.

    Call from a source-bootstrapped package before importing pytest or xdist.
    Local xdist workers explicitly install the same cache settings and verified
    toolkit plus declared runtime source before pytest plugins and conftests load.
    Installed dependencies retain their normal loaders. Test selection, process
    ownership and deadlines stay local.
    """
    package = Path(__file__).resolve().parent
    parts = __package__.split(".")
    if tuple(package.parts[-len(parts) :]) != tuple(parts):
        raise RuntimeError("Pytest source package path does not match its namespace")
    hashes = _registered_hashes(package, __package__)
    parents = _registered_parents()
    runtime = capture_runtime_sources(runtime_sources)
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
            restore_sources = _install_pytest_sources(runtime)
            try:
                _run_pytest(
                    package, __package__, import_root, hashes, parents, cache, runtime
                )
            finally:
                restore_sources()
        finally:
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_write_flag
            sys.path = previous_path
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
