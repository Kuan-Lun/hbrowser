#!/usr/bin/env bash

set -euo pipefail

cd -P -- "$(dirname "$0")/.."
repository_root="$PWD"
cd "$repository_root"

.venv/bin/ruff check --extend-exclude scripts/_shared --fix .
.venv/bin/ruff format --extend-exclude scripts/_shared .
node_modules/.bin/markdownlint-cli2 --fix
