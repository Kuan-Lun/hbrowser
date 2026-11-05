"""Bind explicitly declared test runtime packages before pytest imports them."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import stat
import sys
from collections.abc import Mapping
from importlib.resources.abc import TraversableResources
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

# A declaration concerns repository-owned packages only. Installed dependency
# wheels remain under the invocation's independent environment/artifact checks.
# Each worker receives the canonical root, Python source hashes and relative
# directory topology. Resource bytes deliberately remain ordinary file reads.
type RuntimeSources = dict[str, tuple[str, dict[str, str], tuple[str, ...]]]


class RuntimeSourceError(ImportError):
    """Runtime imports cannot be attributed to the declared candidate source."""


def _source_bytes(path: Path) -> bytes:
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeSourceError(f"Runtime source must be a regular file: {path}")
    return path.read_bytes()


def _inventory(directory: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise RuntimeSourceError(
            f"Runtime package must be a real directory: {directory}"
        )
    result: dict[str, str] = {}
    directories: list[str] = []
    for child in sorted(directory.iterdir()):
        if child.name == "__pycache__":
            continue
        mode = child.lstat().st_mode
        if stat.S_ISDIR(mode):
            hashes, nested = _inventory(child)
            directories.append(child.name)
            directories.extend(f"{child.name}/{name}" for name in nested)
            result.update(
                (f"{child.name}/{name}", digest) for name, digest in hashes.items()
            )
        elif child.suffix == ".py":
            result[child.name] = hashlib.sha256(_source_bytes(child)).hexdigest()
        elif stat.S_ISLNK(mode):
            raise RuntimeSourceError(
                f"Runtime package cannot contain symlinks: {child}"
            )
    return result, tuple(directories)


def capture_runtime_sources(packages: Mapping[str, Path]) -> RuntimeSources:
    """Capture pure Python package identities from explicit repository declarations."""
    result = {}
    for name, directory in packages.items():
        if not name.isidentifier() or name in sys.stdlib_module_names:
            raise RuntimeSourceError(f"Invalid runtime package declaration: {name!r}")
        path = directory
        if not path.is_absolute() or path.name != name or path.resolve() != path:
            raise RuntimeSourceError(f"Runtime package path is not canonical: {path}")
        hashes, directories = _inventory(path)
        if "__init__.py" not in hashes:
            raise RuntimeSourceError(
                f"Runtime root must be a regular Python package: {name}"
            )
        result[name] = (str(path), hashes, directories)
    return result


class _RuntimeLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname: str, path: Path, digest: str, package: bool) -> None:
        super().__init__(fullname, str(path))
        self.source = path
        self.digest = digest
        self.package = package

    def get_code(self, fullname: str) -> CodeType:
        content = _source_bytes(self.source)
        if hashlib.sha256(content).hexdigest() != self.digest:
            raise RuntimeSourceError(
                f"Runtime source changed after capture: {self.source}"
            )
        return compile(content, str(self.source), "exec", dont_inherit=True)

    def exec_module(self, module: ModuleType) -> None:
        exec(self.get_code(module.__name__), module.__dict__)
        if self.package and list(getattr(module, "__path__", ())) != [
            str(self.source.parent)
        ]:
            raise RuntimeSourceError(
                f"Runtime package redirected its source: {module.__name__}"
            )


class _RuntimeNamespaceLoader(importlib.abc.Loader, TraversableResources):
    """A single captured namespace directory with ordinary resource access."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def exec_module(self, module: ModuleType) -> None:
        self.files()
        module.__dict__["__file__"] = None

    def get_resource_reader(self, fullname: str) -> TraversableResources:
        del fullname
        return self

    def files(self) -> Path:
        directory = self.directory
        if (
            directory.resolve() != directory
            or not directory.is_dir()
            or (directory / "__init__.py").exists()
        ):
            raise RuntimeSourceError(
                f"Runtime namespace directory changed: {directory}"
            )
        return directory


class _RuntimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources: RuntimeSources) -> None:
        self.sources = sources

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        root, *parts = fullname.split(".")
        if root not in self.sources:
            return None
        directory, hashes, directories = self.sources[root]
        base = Path(directory)
        if parts:
            expected_parent = str(base.joinpath(*parts[:-1]))
            if path is None or list(path) != [expected_parent]:
                raise RuntimeSourceError(f"Runtime import path changed: {fullname}")
        stem = "/".join(parts)
        package_name = (stem + "/" if stem else "") + "__init__.py"
        module_name = stem + ".py"
        filename = package_name if package_name in hashes else module_name
        if filename in hashes:
            source = base / filename
            is_package = filename == package_name
            return importlib.util.spec_from_file_location(
                fullname,
                source,
                loader=_RuntimeLoader(fullname, source, hashes[filename], is_package),
                submodule_search_locations=[str(source.parent)] if is_package else None,
            )
        if stem in directories:
            namespace_directory = base / stem
            spec = importlib.machinery.ModuleSpec(
                fullname,
                loader=_RuntimeNamespaceLoader(namespace_directory),
                is_package=True,
            )
            spec.submodule_search_locations = [str(namespace_directory)]
            return spec
        # Never fall through to a foreign wheel, PYTHONPATH or native extension
        # for a name covered by this pure-Python declaration.
        raise ModuleNotFoundError(
            f"No candidate runtime source for {fullname}", name=fullname
        )


def install_runtime_sources(sources: RuntimeSources) -> _RuntimeFinder:
    """Revalidate exact controller source before installing an invocation finder."""
    for name, (directory, hashes, directories) in sources.items():
        if name in sys.modules or any(
            key.startswith(name + ".") for key in sys.modules
        ):
            raise RuntimeSourceError(
                f"Runtime package was imported before source binding: {name}"
            )
        if capture_runtime_sources({name: Path(directory)})[name] != (
            directory,
            hashes,
            directories,
        ):
            raise RuntimeSourceError(f"Runtime source differs from controller: {name}")
    finder = _RuntimeFinder(sources)
    sys.meta_path.insert(0, finder)
    return finder
