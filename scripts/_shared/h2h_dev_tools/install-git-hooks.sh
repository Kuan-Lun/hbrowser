#!/usr/bin/env bash
set -euo pipefail

repo_root=${1:?repository root is required}
resource_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

existing_hooks_path="$(git config --get core.hooksPath || true)"
if [[ -n "$existing_hooks_path" && "$existing_hooks_path" != ".githooks" ]]; then
    echo "Refusing to replace existing core.hooksPath: $existing_hooks_path" >&2
    exit 1
fi

if [[ -z "$existing_hooks_path" ]]; then
    default_hooks_directory="$(git rev-parse --git-path hooks)"
    shopt -s nullglob
    for existing_hook in "$default_hooks_directory"/*; do
        if [[ -f "$existing_hook" \
            && -x "$existing_hook" \
            && "$existing_hook" != *.sample ]]; then
            echo "Refusing to disable existing Git hook: $existing_hook" >&2
            exit 1
        fi
    done
fi

git config --local core.hooksPath .githooks
primary="$(bash "$resource_dir/primary-branch.sh" "$repo_root")"
git config --local "branch.$primary.mergeOptions" --no-ff
git config --local "branch.$primary.rebase" false
git config --local pull.rebase false
git config --local pull.ff only

printf 'Installed repository Git hooks; primary branch: %s\n' "$primary"
