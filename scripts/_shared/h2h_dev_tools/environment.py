"""Bind explicit package sources before uv resolves an environment rebuild."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname


def source_requirement(assignment: str) -> str:
    """Produce a named requirement; uv owns metadata and PEP 508 validation."""
    package, separator, source = assignment.partition("=")
    if (
        not separator
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", package)
        or not source
        or any(character in source for character in "\r\n\0")
    ):
        raise ValueError(f"Invalid dependency override: {assignment!r}")
    if source == "index":
        return package
    parsed = urlsplit(source)
    local: Path | None = None
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(
                "Local file URLs must have no remote host, query or fragment"
            )
        local = Path(url2pathname(parsed.path))
    elif not parsed.scheme or (
        len(parsed.scheme) == 1 and source[1:3] in {":/", ":\\"}
    ):
        local = Path(source)
    if local is not None:
        local = local.resolve(strict=True)
        if local.is_dir():
            if not (local / "pyproject.toml").is_file():
                raise ValueError(f"Local project has no pyproject.toml: {local}")
            return f"-e {package} @ {local.as_uri()}"
        if not local.is_file():
            raise ValueError(
                f"Local source must be a project or distribution file: {local}"
            )
        source = local.as_uri()
    return f"{package} @ {source}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--editable", action="append", default=[])
    arguments = parser.parse_args()
    try:
        requirements = [source_requirement(item) for item in arguments.source]
        for editable in arguments.editable:
            if not editable or any(character in editable for character in "\r\n\0"):
                raise ValueError("Editable requirements must be nonempty single lines")
            requirements.append(f"-e {editable}")
        if not requirements:
            raise ValueError(
                "At least one package source or editable project is required"
            )
        arguments.output.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Invalid environment source: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
