#!/usr/bin/env bash

set -euo pipefail

CDPATH= cd -P -- "$(dirname "$0")/.."
repository_root="$PWD"
CDPATH= cd "$repository_root"

scripts/check-fast.sh
.venv/bin/python scripts/_shared/h2h_dev_tools/git_environment.py \
    --repository "$repository_root" -- .venv/bin/python scripts/pytest-source.py

.venv/bin/python scripts/_shared/h2h_dev_tools/offline_wheel.py \
    --repository "$repository_root" \
    --smoke-script scripts/smoke-installed-wheel.py
