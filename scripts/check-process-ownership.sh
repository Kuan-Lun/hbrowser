#!/usr/bin/env bash
set -euo pipefail

CDPATH= cd -P -- "$(dirname "$0")/.."
repository_root="$PWD"
CDPATH= cd "$repository_root"

if [[ -n "${HBROWSER_CHECK_PYTHON:-}" ]]; then
    process_python="$HBROWSER_CHECK_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    process_python=".venv/bin/python"
else
    process_python="python"
fi

"$process_python" scripts/_shared/h2h_dev_tools/git_environment.py \
    --repository "$repository_root" -- "$process_python" scripts/pytest-source.py \
    tests/test_owned_process.py \
    tests/test_browser_factory.py \
    tests/test_runtime_test_owner.py \
    tests/test_tor_cleanup.py
