#!/usr/bin/env bash
# Resolve named sources without modifying the environment being replaced.
set -euo pipefail

requirements=${1:?output requirements file is required}
shift
source_directory="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Require the resolver version verified for encoded local project URLs before
# creating any Python environment or clearing the caller's existing environment.
uv_version="$(uv --version)" || {
    printf '%s\n' 'resolve-sources: uv >=0.13.0 is required; upgrade uv and retry.' >&2
    exit 2
}
if [[ ! "$uv_version" =~ ^uv\ ([0-9]+)\.([0-9]+)\.([0-9]+)($|[[:space:]]) ]] \
    || ! (( 10#${BASH_REMATCH[1]} > 0 || 10#${BASH_REMATCH[2]} >= 13 )); then
    printf 'resolve-sources: uv >=0.13.0 is required; found %s. Upgrade uv and retry.\n' "$uv_version" >&2
    exit 2
fi

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
