#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.."
repository_root="$PWD"
exec bash "$repository_root/scripts/_shared/h2h_dev_tools/git-flow-merge.sh" "$repository_root" candidate scripts/review-merge-candidate.sh "$@"
