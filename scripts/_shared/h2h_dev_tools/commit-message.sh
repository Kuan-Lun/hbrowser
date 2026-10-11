#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]] || [[ -z "$1" || -z "$2" || -z "$3" ]]; then
    printf 'usage: commit-message.sh REPOSITORY_ROOT MESSAGE_FILE PATTERN\n' >&2
    exit 1
fi
repository_root=$1
message_file=$2
pattern=$3

# Keep relative message paths anchored to the caller, as Git passes them.
if ! first_line="$(sed -n '1p;1q' < "$message_file")"; then
    printf 'Could not read commit message: %s\n' "$message_file" >&2
    exit 1
fi

if [[ "$first_line" =~ $pattern ]]; then
    exit 0
else
    match_status=$?
fi
if [[ "$match_status" -eq 2 ]]; then
    printf 'Invalid commit message pattern\n' >&2
    exit 1
fi

if [[ "$first_line" == Merge\ * ]] \
    && git -C "$repository_root" rev-parse --verify MERGE_HEAD >/dev/null 2>&1; then
    exit 0
fi

printf 'Commit message must follow Conventional Commits: %s\n' "$first_line" >&2
exit 1
