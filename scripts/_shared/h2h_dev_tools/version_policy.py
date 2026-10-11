#!/usr/bin/env python3
"""Validate a task-level X.Y.Z bump against an exact release candidate."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from .dependency_manifest import dependency_manifest

_VERSION_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_AUDIT_PATH = ".release/dependency-audit.json"


def _git(*arguments: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        input=input_text,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.rstrip("\n")


def _tree_file(tree: str, path: str) -> bytes:
    return subprocess.run(
        ("git", "show", f"{tree}:{path}"),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _load_toml(tree: str) -> dict[str, Any]:
    return tomllib.loads(_tree_file(tree, "pyproject.toml").decode())


def _release_metadata(document: dict[str, Any]) -> dict[str, Any]:
    project = copy.deepcopy(document.get("project", {}))
    project.pop("version", None)
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        optional.pop("dev", None)
    return {
        "build-system": document.get("build-system", {}),
        "project": project,
        "wheel": document.get("tool", {}).get("hatch", {}).get("build", {}),
    }


def _matches(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _release_patterns(document: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        document.get("tool", {})
        .get("h2h", {})
        .get("version", {})
        .get("release-paths", ())
    )


def _policy(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("tool", {}).get("h2h", {}).get("version", {})
    if not isinstance(value, dict):
        raise ValueError("tool.h2h.version must be a table")
    return value


def _development_patterns(document: dict[str, Any]) -> tuple[str, ...]:
    patterns = _policy(document).get("development-paths", ())
    if not isinstance(patterns, (list, tuple)) or not all(
        isinstance(item, str) for item in patterns
    ):
        raise ValueError("development-paths must be an array of strings")
    return tuple(patterns)


def _validate_transition(message_range: str, version: str) -> None:
    transitions = 0
    for revision in _git(
        "rev-list", "--reverse", "--no-merges", message_range
    ).splitlines():
        parent = f"{revision}^1"
        before, after = _load_toml(parent), _load_toml(revision)
        if before["project"]["version"] == after["project"]["version"]:
            continue
        transitions += 1
        if (
            _git("show", "-s", "--format=%s", revision)
            != f"chore(release): bump version to {version}"
        ):
            raise ValueError("version transition requires its dedicated release commit")
        if not set(_changed_paths(parent, revision)) <= {"pyproject.toml", _AUDIT_PATH}:
            raise ValueError("release commit may only change version and audit")
        before, after = copy.deepcopy(before), copy.deepcopy(after)
        before["project"].pop("version")
        after["project"].pop("version")
        if before != after:
            raise ValueError(
                "release commit may only change project.version in metadata"
            )
    if transitions != 1:
        raise ValueError("task must contain exactly one project-version transition")


def _changed_paths(base: str, candidate: str) -> tuple[str, ...]:
    # A move out of a public release path still retires its old surface.
    return tuple(
        path
        for path in _git(
            "diff",
            "--no-renames",
            "--name-only",
            "--diff-filter=ACDMRT",
            base,
            candidate,
        ).splitlines()
        if path
    )


def _release_changed(
    base: dict[str, Any],
    candidate: dict[str, Any],
    paths: tuple[str, ...],
    patterns: tuple[str, ...],
) -> bool:
    return _release_metadata(base) != _release_metadata(candidate) or any(
        _matches(path, patterns) for path in paths
    )


def _release_messages(message_range: str, patterns: tuple[str, ...]) -> str:
    """Classify release signals without promoting dev-only commit messages.

    The task's final diff still decides whether a single bump is required.
    Each message contributes its severity only when that commit changes a
    release surface. Both sides' declarations preserve retired public paths.
    """

    messages = []
    for revision in _git("rev-list", message_range).splitlines():
        parent = f"{revision}^1"
        base = _load_toml(parent)
        candidate = _load_toml(revision)
        commit_patterns = (
            patterns + _release_patterns(base) + _release_patterns(candidate)
        )
        if _release_changed(
            base, candidate, _changed_paths(parent, revision), commit_patterns
        ):
            messages.append(_git("show", "-s", "--format=%B", revision))
    return "\n\0\n".join(messages)


def _parse_version(value: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"project version must use X.Y.Z: {value}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _expected_version(
    base: tuple[int, int, int],
    *,
    breaking: bool,
    feature: bool,
) -> tuple[int, int, int]:
    major, minor, patch = base
    if major == 0:
        return (0, minor + 1, 0) if breaking else (0, minor, patch + 1)
    if breaking:
        return major + 1, 0, 0
    if feature:
        return major, minor + 1, 0
    return major, minor, patch + 1


def _validate_audit(tree: str, document: dict[str, Any], version: str) -> None:
    try:
        receipt = json.loads(_tree_file(tree, _AUDIT_PATH))
        package = json.loads(_tree_file(tree, "package.json"))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "version bump requires valid dependency audit "
            ".release/dependency-audit.json"
        ) from error
    digest = hashlib.sha256(dependency_manifest(document, package)).hexdigest()
    if receipt.get("schema") != "h2h.dependency-audit.v1":
        raise ValueError("dependency audit receipt has an unsupported schema")
    if receipt.get("project_version") != version:
        raise ValueError("dependency audit receipt does not match project version")
    if receipt.get("manifest_sha256") != digest:
        raise ValueError("dependency audit receipt does not match dependencies")
    review = receipt.get("review", {})
    if (
        review.get("status") != "reviewed"
        or not isinstance(review.get("note"), str)
        or not review["note"].strip()
    ):
        raise ValueError("dependency audit receipt lacks a compatibility review")
    if not isinstance(receipt.get("python"), list) or not isinstance(
        receipt.get("node"), list
    ):
        raise ValueError("dependency audit must contain Python and Node evidence")
    project = document.get("project", {})
    groups = {
        "build": document.get("build-system", {}).get("requires", []),
        "runtime": project.get("dependencies", []),
        **{
            f"optional:{name}": values
            for name, values in project.get("optional-dependencies", {}).items()
        },
        **{
            f"group:{name}": values
            for name, values in document.get("dependency-groups", {}).items()
        },
    }
    expected_python = sorted(
        (group, requirement)
        for group, requirements in groups.items()
        for requirement in requirements
        if isinstance(requirement, str)
    )
    actual_python = sorted(
        (entry["group"], entry["declared"]) for entry in receipt["python"]
    )
    if actual_python != expected_python:
        raise ValueError("dependency audit Python inventory is incomplete")
    expected_node = sorted(package.get("devDependencies", {}).items())
    actual_node = sorted(
        (entry["name"], entry["declared"]) for entry in receipt["node"]
    )
    if actual_node != expected_node:
        raise ValueError("dependency audit Node inventory is incomplete")


def _candidate(arguments: argparse.Namespace) -> tuple[str, str, str]:
    if arguments.index:
        merge_task_ref = os.environ.get("WORKFLOW_MERGE_TASK_REF")
        if merge_task_ref:
            merge_revision = _git(
                "rev-parse", "--verify", f"{merge_task_ref}^{{commit}}"
            )
        else:
            merge_head = Path(_git("rev-parse", "--git-path", "MERGE_HEAD"))
            if not merge_head.exists():
                raise ValueError(
                    "--index requires WORKFLOW_MERGE_TASK_REF or an active merge"
                )
            merge_revision = _git("rev-parse", "MERGE_HEAD")
        return "HEAD", _git("write-tree"), f"HEAD..{merge_revision}"
    if arguments.base:
        return (
            str(arguments.base),
            str(arguments.candidate),
            f"{arguments.base}..{arguments.candidate}",
        )
    primary_name = subprocess.run(
        ("scripts/detect-primary-branch.sh",),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    candidate = _git("merge-tree", "--write-tree", primary_name, "HEAD").splitlines()[0]
    return primary_name, candidate, f"{primary_name}..HEAD"


def validate(base_tree: str, candidate_tree: str, task: str) -> None:
    """Validate the exact candidate using the task's complete commit history."""
    _validate(base_tree, candidate_tree, f"{base_tree}..{task}")


def _validate(base_tree: str, candidate_tree: str, message_range: str) -> None:
    try:
        base_document = _load_toml(base_tree)
    except subprocess.CalledProcessError:
        candidate_document = _load_toml(candidate_tree)
        policy = _policy(candidate_document)
        initial = policy.get("initial-version")
        if _git("ls-tree", "-r", "--name-only", base_tree):
            raise ValueError(
                "initial release requires the explicit empty baseline"
            ) from None
        if initial is None or _parse_version(
            str(candidate_document["project"]["version"])
        ) != _parse_version(str(initial)):
            raise ValueError(f"initial release must use {initial}") from None
        _validate_audit(candidate_tree, candidate_document, str(initial))
        return
    candidate_document = _load_toml(candidate_tree)
    base_version_text = str(base_document["project"]["version"])
    candidate_version_text = str(candidate_document["project"]["version"])
    base_version = _parse_version(base_version_text)
    candidate_version = _parse_version(candidate_version_text)

    changed_paths = _changed_paths(base_tree, candidate_tree)
    release_patterns = _release_patterns(base_document) + _release_patterns(
        candidate_document
    )
    release_changed = _release_changed(
        base_document, candidate_document, changed_paths, release_patterns
    )
    unknown = [
        path
        for path in changed_paths
        if path != "pyproject.toml"
        and not _matches(path, release_patterns)
        and not _matches(path, _development_patterns(candidate_document))
    ]

    messages = _git("log", "--format=%B%x00", message_range)
    none_impact = bool(
        re.search(r"^Version-Impact:\s*none\s*$", messages, re.MULTILINE)
    )
    none_reason = re.search(r"^Version-Reason:\s*(\S.+)$", messages, re.MULTILINE)
    if none_impact != (none_reason is not None):
        raise ValueError(
            "Version-Impact: none and a nonempty Version-Reason must be used together"
        )
    if none_impact and not _policy(candidate_document).get("allow-none-impact", False):
        raise ValueError("this repository does not permit Version-Impact: none")
    if unknown and not none_impact:
        raise ValueError(
            "unclassified version impact paths: " + ", ".join(sorted(unknown))
        )

    version_changed = base_version_text != candidate_version_text
    if release_changed and not version_changed and not none_impact:
        raise ValueError("release surface changed without a project version bump")
    if version_changed and none_impact:
        raise ValueError("Version-Impact: none cannot accompany a version bump")
    if not release_changed and not unknown and version_changed:
        raise ValueError("project version changed without a release-surface change")
    if not version_changed:
        return

    release_messages = _release_messages(message_range, release_patterns)
    breaking = bool(
        re.search(r"^[a-z]+(?:\([^\n)]+\))?!:", release_messages, re.MULTILINE)
        or re.search(r"^BREAKING CHANGE:", release_messages, re.MULTILINE)
    )
    feature = bool(re.search(r"^feat(?:\([^\n)]+\))?:", release_messages, re.MULTILINE))
    expected = _expected_version(
        base_version,
        breaking=breaking,
        feature=feature,
    )
    if _policy(candidate_document).get("dedicated-release-commit", False):
        _validate_transition(message_range, candidate_version_text)
    if candidate_version != expected:
        expected_text = ".".join(str(part) for part in expected)
        raise ValueError(
            f"expected project version {expected_text}, got {candidate_version_text}"
        )
    _validate_audit(candidate_tree, candidate_document, candidate_version_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--base")
    parser.add_argument("--candidate", default="HEAD")
    arguments = parser.parse_args()
    if arguments.index and arguments.base:
        parser.error("--index and --base are mutually exclusive")
    if arguments.candidate != "HEAD" and not arguments.base:
        parser.error("--candidate requires --base")

    _validate(*_candidate(arguments))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"check-version: {error}", file=sys.stderr)
        raise SystemExit(1) from error
