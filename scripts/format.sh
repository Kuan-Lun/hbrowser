#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"

.venv/bin/ruff check --extend-exclude scripts/_shared --fix .
.venv/bin/ruff format --extend-exclude scripts/_shared .
node_modules/.bin/markdownlint-cli2 --fix
