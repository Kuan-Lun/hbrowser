#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"

command -v uv >/dev/null || {
    printf 'rebuild-env: uv is required\n' >&2
    exit 1
}
command -v npm >/dev/null || {
    printf 'rebuild-env: npm is required\n' >&2
    exit 1
}

# Resolve only from project manifests; neither uv.lock nor package-lock.json is
# read or generated.
# Request the supported minor explicitly. Dependency installation below enforces
# the complete project.requires-python constraint, including excluded patches.
uv venv --python 3.14 --clear .venv
uv pip install --python .venv/bin/python --upgrade --reinstall -e '.[dev]'
npm install --package-lock=false
