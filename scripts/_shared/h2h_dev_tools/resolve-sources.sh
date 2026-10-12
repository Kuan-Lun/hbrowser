#!/usr/bin/env bash
# Resolve named sources without modifying the environment being replaced.
set -euo pipefail

requirements=${1:?output requirements file is required}
shift
source_directory="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_directory="$(mktemp -d "${TMPDIR:-/tmp}/h2h-environment.XXXXXX")"
trap 'rm -rf "$bootstrap_directory"' EXIT

# Bootstrap works from a clean checkout and never imports the toolkit package
# or reuses the target .venv. Only uv and its Python 3.14 interpreter are needed.
uv venv --python 3.14 "$bootstrap_directory/python"
bootstrap_python="$bootstrap_directory/python/bin/python"
if [[ ! -x "$bootstrap_python" ]]; then
    bootstrap_python="$bootstrap_directory/python/Scripts/python.exe"
fi
"$bootstrap_python" "$source_directory/environment.py" --output "$requirements" "$@"

# Named direct requirements make uv reject a source whose distribution metadata
# does not match PACKAGE. Resolve the entire cohort before any --clear operation.
# The temporary plan is not a project lockfile; installation rechecks the original
# named requirements, preserving both version bounds and package identity.
uv pip compile --python "$bootstrap_python" "$requirements" \
    --no-header --no-annotate --output-file "$bootstrap_directory/resolved.txt"
