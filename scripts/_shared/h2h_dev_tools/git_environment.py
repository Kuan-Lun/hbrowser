"""Run test children without inheriting a Git hook's repository authority."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

GIT_ENVIRONMENT_QUERY_TIMEOUT_SECONDS = 15.0
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class GitEnvironmentError(RuntimeError):
    """Git could not supply a valid repository-local environment inventory."""


def without_local_git_environment(
    environment: Mapping[str, str], *, repository: Path
) -> dict[str, str]:
    """Copy the caller's environment, removing only Git's declared local names.

    Apply this at the test-child boundary, never to a gate or Git workflow which
    must retain the real candidate's index and repository metadata.
    """
    result = dict(environment)
    try:
        output = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "--local-env-vars"),
            env=result,
            check=True,
            capture_output=True,
            timeout=GIT_ENVIRONMENT_QUERY_TIMEOUT_SECONDS,
        ).stdout.decode("ascii")
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise GitEnvironmentError(
            "could not query Git repository-local environment names"
        ) from error
    names = output.splitlines()
    if (
        not output.endswith("\n")
        or not names
        or len(names) != len(set(names))
        or any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in names)
    ):
        raise GitEnvironmentError("invalid Git repository-local environment names")
    for name in names:
        result.pop(name, None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command: list[str] = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a test command is required")
    try:
        environment = without_local_git_environment(
            os.environ, repository=arguments.repository
        )
        # Windows CRT overlay does not preserve a waitable parent for the child.
        # Wait explicitly inside the existing Job; never start another owner.
        if os.name == "nt":
            return subprocess.run(command, env=environment, check=False).returncode
        # POSIX exec preserves the surrounding owner's PID and process group.
        os.execvpe(command[0], command, environment)
    except (GitEnvironmentError, OSError) as error:
        parser.exit(1, f"git-environment: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
