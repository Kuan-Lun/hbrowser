#!/usr/bin/env bash
set -euo pipefail

repo_root=${1:?repository root is required}
resource_dir="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDPATH= cd "$repo_root"
repo_root=$PWD

# Preserve all value bytes: command substitution otherwise strips trailing LFs.
configuration_status=0
existing_hooks_path="$(
    git config --get core.hooksPath
    status=$?
    printf '.'
    exit "$status"
)" || configuration_status=$?
if [[ "$configuration_status" == 0 ]]; then
    existing_hooks_path=${existing_hooks_path%.}
    existing_hooks_path=${existing_hooks_path%$'\n'}
    if [[ "$existing_hooks_path" != ".githooks" ]]; then
        printf 'Refusing to replace existing core.hooksPath: %s\n' "$existing_hooks_path" >&2
        exit 1
    fi
elif [[ "$configuration_status" != 1 ]]; then
    printf 'Could not inspect existing core.hooksPath (Git exit %s)\n' "$configuration_status" >&2
    exit 1
fi

if [[ "$configuration_status" == 1 ]]; then
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

primary="$(bash "$resource_dir/primary-branch.sh" "$repo_root")"
git config --local core.hooksPath .githooks
git config --local "branch.$primary.mergeOptions" --no-ff
git config --local "branch.$primary.rebase" false
git config --local pull.rebase false
git config --local pull.ff only

printf 'Installed repository Git hooks; primary branch: %s\n' "$primary"
