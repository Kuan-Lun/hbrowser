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

FORMAT_PATHS=(hbrowser tests)
TYPE_PATHS=(hbrowser)

uv run black "${FORMAT_PATHS[@]}" >&2
uv run ruff check --fix "${FORMAT_PATHS[@]}" >&2
uv run black "${FORMAT_PATHS[@]}" >&2
uv run mypy "${TYPE_PATHS[@]}" >&2
