#!/usr/bin/env bash
set -euo pipefail
CDPATH= cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.."
repository_root="$PWD"
exec bash "$repository_root/scripts/_shared/h2h_dev_tools/install-git-hooks.sh" "$repository_root" "$@"
