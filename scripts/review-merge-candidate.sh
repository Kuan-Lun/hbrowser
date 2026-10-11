#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repository_root/scripts/_shared/h2h_dev_tools/review-merge-candidate.sh" "$repository_root" "$@"
