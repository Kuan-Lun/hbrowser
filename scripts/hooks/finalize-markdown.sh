#!/usr/bin/env bash
# Shared agent finalizer for Markdown changes. Claude Code invokes this as a
# Stop hook; Codex and humans can run the same entry point directly.
#
# `pymarkdown fix` is best-effort because not every Markdown rule can be
# auto-fixed. Ruff's preview formatter then validates and formats Python code
# blocks embedded in Markdown.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MD_FILES=()
while IFS= read -r file; do
    MD_FILES+=("$file")
done < <(
    find . -maxdepth 2 -type f -name "*.md" \
        -not -path "./.venv/*" \
        -not -path "./node_modules/*" \
        -not -path "./.pytest_cache/*" \
        -not -path "./.*" \
        | sort
)

if [ ${#MD_FILES[@]} -eq 0 ]; then
    exit 0
fi

uv run pymarkdown fix "${MD_FILES[@]}" >/dev/null 2>&1 || true

if ! uv run ruff format --preview "${MD_FILES[@]}" >&2; then
    exit 2
fi
