#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"

scripts/check-fast.sh
.venv/bin/pytest

.venv/bin/python scripts/_shared/h2h_dev_tools/offline_wheel.py \
    --repository "$repository_root" \
    --smoke-script scripts/smoke-installed-wheel.py
