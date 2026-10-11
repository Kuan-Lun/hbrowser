#!/usr/bin/env bash

set -euo pipefail

repository_root=${1:?repository root is required}
after_build=${2-}
shift 2
cd "$repository_root"
for executable in uv npm; do
    command -v "$executable" >/dev/null || {
        printf 'rebuild-env: %s is required\n' "$executable" >&2
        exit 1
    }
done

source_overrides=()

usage() {
    printf '%s\n' \
        'Usage: scripts/rebuild-env.sh [--source PACKAGE=SOURCE]...' \
        '' \
        'Dependencies use the package index by default. SOURCE can be index,' \
        'a local project, wheel, archive/URL, or Git requirement. Overrides' \
        'must be supplied explicitly; sibling repositories are never guessed.'
}

while (( $# )); do
    case "$1" in
        --source)
            (( $# >= 2 )) || {
                printf '%s\n' '--source requires PACKAGE=SOURCE' >&2
                exit 2
            }
            assignment=$2
            package="${assignment%%=*}"
            source="${assignment#*=}"
            [[ "$assignment" == *=* && "$package" =~ ^[A-Za-z0-9._-]+$ \
                && -n "$source" ]] || {
                printf 'Invalid dependency override: %s\n' "$assignment" >&2
                exit 2
            }
            if [[ "$source" != index ]]; then
                source_overrides+=("$source")
            fi
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

install_arguments=(-e ".[dev]")
if (( ${#source_overrides[@]} > 0 )); then
    for source in "${source_overrides[@]}"; do
        if [[ -d "$source" ]]; then
            [[ -f "$source/pyproject.toml" ]] || {
                printf 'Local project has no pyproject.toml: %s\n' "$source" >&2
                exit 1
            }
            install_arguments+=(--editable "$source")
        else
            install_arguments+=("$source")
        fi
    done
fi
uv venv --clear --python 3.14 .venv
uv pip install --refresh --python .venv/bin/python "${install_arguments[@]}"

npm install --package-lock=false --ignore-scripts --no-audit --no-fund

if [[ -n "$after_build" ]]; then
    .venv/bin/python "$after_build"
fi
