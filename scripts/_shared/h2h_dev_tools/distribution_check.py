"""Verify installed requirements and wheel RECORD/import origins."""

from __future__ import annotations

import importlib
import json
import sys
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement


def check_runtime_requirements(
    package: Distribution, *, prereleases: bool | None = None
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for raw in package.requires or ():
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate(
            {"extra": ""}
        ):
            continue
        try:
            dependency = distribution(requirement.name)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"{package.metadata['Name']} requires {requirement}; package is missing"
            ) from error
        if not requirement.specifier.contains(
            dependency.version, prereleases=prereleases
        ):
            raise RuntimeError(
                f"{package.metadata['Name']} requires {requirement}; "
                f"installed {requirement.name}=={dependency.version} "
                f"does not satisfy {requirement}"
            )
        selected[requirement.name] = dependency.version
    return selected


def check_wheel_origin(project: str, module_name: str, *, local: bool) -> str:
    package = distribution(project)
    module = importlib.import_module(module_name)
    if module.__file__ is None:
        raise RuntimeError(f"{project} has no import origin")
    actual = Path(module.__file__).resolve()
    record_name = module_name.replace(".", "/") + "/__init__.py"
    expected = Path(str(package.locate_file(record_name))).resolve()
    if actual != expected:
        raise RuntimeError(
            f"{project} import disagrees with installed metadata: {actual}"
        )
    if package.files is None or not any(
        str(item) == record_name for item in package.files
    ):
        raise RuntimeError(f"{project} has no installed package RECORD")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable", False):
        raise RuntimeError(
            f"{project} must come from a wheel, not an editable checkout"
        )
    if local and (
        not actual.is_relative_to(Path(sys.prefix).resolve())
        or not Path(str(package.locate_file("")))
        .resolve()
        .is_relative_to(Path(sys.prefix).resolve())
    ):
        raise RuntimeError(f"{project} must be installed inside the smoke environment")
    return str(actual)
