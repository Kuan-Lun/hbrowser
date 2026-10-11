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
from dataclasses import dataclass
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
        if not optional:
            project.pop("optional-dependencies")
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
    # Keep names as bytes until after splitting: text mode translates CR/LF,
    # while Git's ordinary display quoting changes non-ASCII and control names.
    output = subprocess.run(
        (
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRT",
            base,
            candidate,
            "--",
        ),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not output:
        return ()
    paths = output.split(b"\0")
    if paths.pop() != b"" or not all(paths):
        raise ValueError("malformed NUL-delimited Git difference")
    return tuple(os.fsdecode(path) for path in paths)


def _release_changed(
    base: dict[str, Any],
    candidate: dict[str, Any],
    paths: tuple[str, ...],
    patterns: tuple[str, ...],
) -> bool:
    return _release_metadata(base) != _release_metadata(candidate) or any(
        _matches(path, patterns) for path in paths
    )


@dataclass(frozen=True)
class _CommitImpact:
    message: str
    paths: frozenset[str]
    metadata_changed: bool
    release_changed: bool
    exempt: bool


def _none_impact(message: str, *, allowed: bool) -> bool:
    impacts = re.findall(r"^Version-Impact:[ \t]*(.*)$", message, re.MULTILINE)
    reasons = re.findall(r"^Version-Reason:[ \t]*(.*)$", message, re.MULTILINE)
    if not impacts and not reasons:
        return False
    if (
        len(impacts) != 1
        or impacts[0].strip() != "none"
        or len(reasons) != 1
        or not reasons[0].strip()
    ):
        raise ValueError(
            "Version-Impact: none and a nonempty Version-Reason "
            "must be used together in the same commit"
        )
    if not allowed:
        raise ValueError("this repository does not permit Version-Impact: none")
    return True


def _automatic_merge(base: str, task: str) -> str:
    try:
        return _git("merge-tree", "--write-tree", base, task).splitlines()[0]
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "cannot establish Version-Impact: none scope across a conflicted merge"
        ) from error


def _commit_impacts(
    message_range: str, patterns: tuple[str, ...], *, allow_none: bool
) -> tuple[_CommitImpact, ...]:
    history = []
    for row in _git("rev-list", "--reverse", "--parents", message_range).splitlines():
        revision, *parents = row.split()
        message = _git("show", "-s", "--format=%B", revision)
        history.append(
            (revision, parents, message, _none_impact(message, allowed=allow_none))
        )
    has_exemptions = any(exempt for _, _, _, exempt in history)
    impacts = []
    for revision, parents, message, exempt in history:
        if not parents:
            raise ValueError(
                "task history contains a commit without a comparison parent"
            )
        parent = parents[0]
        if has_exemptions and len(parents) > 1:
            if len(parents) != 2:
                raise ValueError(
                    "cannot establish none-impact scope for an octopus merge"
                )
            # A normal merge only aggregates contributions already evaluated
            # below. Additional resolution changes belong to this merge itself.
            parent = _automatic_merge(*parents)
        base = _load_toml(parent)
        candidate = _load_toml(revision)
        if exempt and base["project"]["version"] != candidate["project"]["version"]:
            raise ValueError("Version-Impact: none cannot accompany a version bump")
        paths = _changed_paths(parent, revision)
        commit_patterns = (
            patterns + _release_patterns(base) + _release_patterns(candidate)
        )
        impacts.append(
            _CommitImpact(
                message=message,
                paths=frozenset(paths),
                metadata_changed=_release_metadata(base)
                != _release_metadata(candidate),
                release_changed=_release_changed(
                    base, candidate, paths, commit_patterns
                ),
                exempt=exempt,
            )
        )
    return tuple(impacts)


def _exempt_changes(
    impacts: tuple[_CommitImpact, ...],
    base: str,
    candidate: str,
    message_range: str,
    candidate_document: dict[str, Any],
) -> tuple[set[str], bool]:
    exempt_paths = {
        path for impact in impacts if impact.exempt for path in impact.paths
    }
    required_paths = {
        path for impact in impacts if not impact.exempt for path in impact.paths
    }
    exempt_paths.difference_update(required_paths)
    metadata_exempt = any(
        impact.exempt and impact.metadata_changed for impact in impacts
    ) and not any(not impact.exempt and impact.metadata_changed for impact in impacts)
    if not exempt_paths and not metadata_exempt:
        return set(), False
    _, separator, task = message_range.partition("..")
    if not separator or not task or task.startswith("."):
        raise ValueError("none-impact scope requires a two-endpoint task history")
    expected = _automatic_merge(base, task)
    # Only the annotated commits' own changes are exempt. An additional staged
    # edit or merge resolution cannot borrow a historical exception for its path.
    exempt_paths.difference_update(_changed_paths(expected, candidate))
    metadata_exempt &= _release_metadata(_load_toml(expected)) == _release_metadata(
        candidate_document
    )
    return exempt_paths, metadata_exempt


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
        base = _git("rev-parse", "--verify", "HEAD^{commit}")
        merge_head = Path(_git("rev-parse", "--git-path", "MERGE_HEAD"))
        if merge_head.exists():
            parents = merge_head.read_text(encoding="ascii").splitlines()
            if len(parents) != 1:
                raise ValueError("--index requires a two-parent merge")
            merge_parent = parents[0]
        else:
            # Git's direct pre-merge hook runs before writing MERGE_HEAD. The
            # launcher must supply its captured commit, never a moving branch.
            merge_parent = os.environ.get("WORKFLOW_MERGE_TASK_REF", "")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", merge_parent):
            raise ValueError("--index requires an immutable full task commit OID")
        merge_revision = _git("rev-parse", "--verify", f"{merge_parent}^{{commit}}")
        if merge_revision != merge_parent:
            raise ValueError("--index task OID must identify a commit, not a tag")
        return base, _git("write-tree"), f"{base}..{merge_revision}"
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
    impacts = _commit_impacts(
        message_range,
        release_patterns,
        allow_none=bool(_policy(candidate_document).get("allow-none-impact", False)),
    )
    exempt_paths, metadata_exempt = _exempt_changes(
        impacts, base_tree, candidate_tree, message_range, candidate_document
    )
    release_changed = (
        _release_metadata(base_document) != _release_metadata(candidate_document)
        and not metadata_exempt
    ) or any(
        path not in exempt_paths and _matches(path, release_patterns)
        for path in changed_paths
    )
    unknown = [
        path
        for path in changed_paths
        if path != "pyproject.toml"
        and not _matches(path, release_patterns)
        and not _matches(path, _development_patterns(candidate_document))
        and path not in exempt_paths
    ]

    if unknown:
        raise ValueError(
            "unclassified version impact paths: " + ", ".join(sorted(unknown))
        )

    version_changed = base_version_text != candidate_version_text
    if release_changed and not version_changed:
        raise ValueError("release surface changed without a project version bump")
    if not release_changed and not unknown and version_changed:
        raise ValueError("project version changed without a release-surface change")
    if not version_changed:
        return

    release_messages = tuple(
        impact.message
        for impact in impacts
        if impact.release_changed and not impact.exempt
    )
    # Commit types and the bang marker belong to the subject. The body may
    # quote older subjects; only the explicit breaking footer is independent.
    subjects = tuple(message.partition("\n")[0] for message in release_messages)
    breaking = any(
        re.match(r"[a-z]+(?:\([^\n)]+\))?!:", subject) for subject in subjects
    ) or any(
        re.search(r"^BREAKING CHANGE:", message, re.MULTILINE)
        for message in release_messages
    )
    feature = any(re.match(r"feat(?:\([^\n)]+\))?:", subject) for subject in subjects)
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
