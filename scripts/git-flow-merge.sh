#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repository_root/scripts/_shared/h2h_dev_tools/git-flow-merge.sh" "$repository_root" candidate scripts/review-merge-candidate.sh "$@"
