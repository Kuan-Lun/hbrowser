#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"

if [[ -n "${HBROWSER_CHECK_PYTHON:-}" ]]; then
    process_python="$HBROWSER_CHECK_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    process_python=".venv/bin/python"
else
    process_python="python"
fi

"$process_python" -m pytest \
    tests/test_owned_process.py \
    tests/test_browser_factory.py
