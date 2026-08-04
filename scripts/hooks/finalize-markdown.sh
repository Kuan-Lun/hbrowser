#!/usr/bin/env bash
# Shared agent finalizer for Markdown changes. Claude Code invokes this as a
# Stop hook; Codex and humans can run the same entry point directly.
#
# `pymarkdown fix` is best-effort because not every Markdown rule can be
# auto-fixed. Ruff's preview formatter then validates and formats Python code
# blocks embedded in Markdown, and `pymarkdown scan` is the final lint gate.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MD_FILES=()
while IFS= read -r -d '' file; do
    if [[ -f "$file" ]]; then
        MD_FILES+=("$file")
    fi
done < <(
    git ls-files --cached --others --exclude-standard -z -- '*.md'
)

if [[ ${#MD_FILES[@]} -eq 0 ]]; then
    exit 0
fi

# Keep these exclusions aligned with `markdownlint.config` in VS Code. MD013
# would force hard-wrapped prose, while MD014 misreads PowerShell's `$env:`
# syntax as a shell prompt.
PYMARKDOWN_DISABLED_RULES="MD013,MD014"
uv run pymarkdown -d "$PYMARKDOWN_DISABLED_RULES" fix "${MD_FILES[@]}" \
    >/dev/null 2>&1 || true

if ! uv run ruff format --preview "${MD_FILES[@]}" >&2; then
    exit 2
fi

if ! uv run pymarkdown -d "$PYMARKDOWN_DISABLED_RULES" scan "${MD_FILES[@]}" >&2; then
    exit 2
fi
