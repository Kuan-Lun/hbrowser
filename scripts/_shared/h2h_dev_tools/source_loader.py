"""Bootstrap a verified source snapshot without consulting Python bytecode caches.

Execute this file with runpy.run_path before importing the package. A normal
import of this bootstrap would itself be subject to timestamp bytecode reuse.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import runpy
import stat
import sys
import tokenize
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any


class SourceLoadError(ImportError):
    """The requested source package is not the currently verified implementation."""


def _bytes(path: Path) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SourceLoadError(f"Source must be a regular file: {path}")
    return path.read_bytes()


def _sources(directory: Path, namespace: str) -> dict[str, str]:
    manifest = directory / "manifest.json"
    if (
        manifest.exists()
        or namespace == "_shared.h2h_dev_tools"
        or directory.parts[-3:] == ("scripts", "_shared", "h2h_dev_tools")
    ):
        # This verifier is intentionally executed from source too. Its complete
        # graph checks the loader, verifier, and every module before any import.
        verifier = runpy.run_path(str(directory / "bundle_verify.py"))
        document = verifier["verify_bundle"](directory.parents[2])
        return dict(document["files"])
    return {
        path.name: hashlib.sha256(_bytes(path)).hexdigest()
        for path in sorted(directory.iterdir())
        if path.suffix in {".py", ".sh"}
    }


class _SourceLoader(importlib.abc.SourceLoader):
    def __init__(
        self, name: str, path: Path, registration: object, digest: str
    ) -> None:
        self.name = name
        self.path = path
        self._h2h_source_registration = registration
        self.digest = digest

    def get_filename(self, fullname: str) -> str:
        return str(self.path)

    def get_data(self, path: str) -> bytes:
        if Path(path) != self.path:
            raise SourceLoadError(f"Unexpected source path: {path}")
        content = _bytes(self.path)
        if hashlib.sha256(content).hexdigest() != self.digest:
            raise SourceLoadError(f"Source changed after bootstrap: {self.path}")
        return content

    def get_code(self, fullname: str) -> CodeType:
        # SourceLoader's default implementation reads pyc; never delegate to it.
        return self.source_to_code(self.get_data(str(self.path)), str(self.path))

    def get_source(self, fullname: str) -> str:
        content = self.get_data(str(self.path))
        encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
        return content.decode(encoding)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, directory: Path, namespace: str, hashes: dict[str, str]) -> None:
        self.directory = directory
        self.namespace = namespace
        # Plain data permits run_path bootstrap re-entry without depending on the
        # identity of classes created by a preceding run_path invocation.
        self._h2h_source_registration = (str(directory), namespace, hashes)

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == self.namespace:
            filename = "__init__.py"
            package = True
        elif fullname.startswith(self.namespace + "."):
            suffix = fullname.removeprefix(self.namespace + ".")
            filename = suffix + ".py"
            package = False
        else:
            return None
        hashes = self._h2h_source_registration[2]
        if filename not in hashes:
            raise SourceLoadError(
                f"Module is outside the verified source set: {fullname}"
            )
        loader = _SourceLoader(
            fullname,
            self.directory / filename,
            self._h2h_source_registration,
            hashes[filename],
        )
        return importlib.util.spec_from_file_location(
            fullname,
            self.directory / filename,
            loader=loader,
            submodule_search_locations=[str(self.directory)] if package else None,
        )


def load_package(directory: Path, namespace: str) -> ModuleType:
    """Verify source and register its flat package; preserve identical loaded types.

    Repeated calls reverify the payload. A changed package or an earlier normal
    import is rejected rather than silently retaining stale module objects.
    """
    directory = directory.absolute()
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise SourceLoadError(f"Source package must be a real directory: {directory}")
    hashes = _sources(directory, namespace)
    if "__init__.py" not in hashes:
        raise SourceLoadError("Source package is missing __init__.py")
    registration = (str(directory), namespace, hashes)
    matching = []
    for finder in sys.meta_path:
        previous = getattr(finder, "_h2h_source_registration", None)
        if previous is not None and previous[1] == namespace:
            if previous != registration:
                raise SourceLoadError(
                    f"Source package changed after bootstrap: {namespace}"
                )
            matching.append(finder)
    for name, module in tuple(sys.modules.items()):
        if name == namespace or name.startswith(namespace + "."):
            loader = getattr(module, "__loader__", None)
            previous = getattr(loader, "_h2h_source_registration", None)
            if not matching or previous != registration:
                raise SourceLoadError(
                    f"Package was imported before source bootstrap: {name}"
                )
    if not matching:
        sys.meta_path.insert(0, _SourceFinder(directory, namespace, hashes))
    return importlib.import_module(namespace)
