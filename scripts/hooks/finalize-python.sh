#!/usr/bin/env bash
# Shared agent finalizer for Python changes. Claude Code invokes this as a
# Stop hook; Codex and humans can run the same entry point directly.
#
# Pipeline:
#   1. Black                 — pre-format pass; also catches syntax errors.
#   2. Ruff `--fix`          — auto-fixes safe lints.
#   3. Black                 — stabilizes any code Ruff rewrote.
#   4. Mypy                  — checks the final formatted code.
#
# Any tool failure exits with 2 so agent runtimes can surface it as a failed
# finalization gate.

set -eu
trap 'exit 2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_FILES=()
while IFS= read -r -d '' file; do
    if [[ -f "$file" ]]; then
        PY_FILES+=("$file")
    fi
done < <(
    git ls-files --cached --others --exclude-standard -z -- '*.py' '*.pyi'
)

if [[ ${#PY_FILES[@]} -eq 0 ]]; then
    exit 0
fi

uv run black "${PY_FILES[@]}" >&2
uv run ruff check --fix "${PY_FILES[@]}" >&2
uv run black "${PY_FILES[@]}" >&2
uv run mypy "${PY_FILES[@]}" >&2
