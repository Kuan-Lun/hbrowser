#!/usr/bin/env bash
set -euo pipefail
repository_root=${1:?repository root is required}
CDPATH= cd "$repository_root"
fail() { printf 'review-merge-candidate: %s\n' "$1" >&2; exit 1; }
if [[ -x .venv/bin/python ]]; then
    python=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
    python=.venv/Scripts/python.exe
else
    fail 'create the local development environment with scripts/rebuild-env.sh'
fi
scope="$("$python" scripts/check_change_scope.py --index --base HEAD)" \
    || fail 'cannot classify the merge candidate'
case "$scope" in
    documentation)
        printf '%s\n' 'Documentation-only candidate: online code review is not required.'
        ;;
    full)
        [[ -f scripts/review-code.py ]] || fail 'scripts/review-code.py is missing'
        exec "$python" scripts/review-code.py run --index \
            --timeout-seconds "${H2H_CODE_REVIEW_TIMEOUT_SECONDS:-900}"
        ;;
    *) fail "invalid check scope: $scope" ;;
esac
