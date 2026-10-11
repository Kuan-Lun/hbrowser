#!/usr/bin/env bash

set -euo pipefail

fail() {
    printf 'git-flow-merge: %s\n' "$1" >&2
    exit 1
}

usage() {
    printf 'usage: scripts/git-flow-merge.sh [--cleanup-only]\n' >&2
}

assert_clean() {
    local status_output worktree=$1

    status_output="$(
        git -C "$worktree" status --porcelain=v1 --untracked-files=all
    )" || fail "could not inspect worktree status: $worktree"
    [[ -z "$status_output" ]] \
        || fail "worktree is not clean: $worktree"
}

assert_no_operation() {
    local git_path name worktree=$1

    for name in \
        MERGE_HEAD \
        CHERRY_PICK_HEAD \
        REVERT_HEAD \
        REBASE_HEAD \
        rebase-apply \
        rebase-merge \
        sequencer \
        BISECT_START; do
        git_path="$(
            git -C "$worktree" \
                rev-parse --path-format=absolute --git-path "$name"
        )" || fail "could not inspect Git operation state: $worktree"
        [[ ! -e "$git_path" ]] || fail "Git operation is active: $name"
    done
}

assert_ref_equals() {
    local actual expected=$3 ref=$2 worktree=$1

    actual="$(
        git -C "$worktree" rev-parse --verify "$ref^{commit}" 2>/dev/null
    )" || fail "Git ref is no longer available: $ref"
    [[ "$actual" == "$expected" ]] \
        || fail "Git ref changed during cleanup: $ref"
}

is_ancestor() {
    local ancestor=$2 descendant=$3 status worktree=$1

    if git -C "$worktree" merge-base --is-ancestor "$ancestor" "$descendant"; then
        return 0
    else
        status=$?
    fi
    [[ "$status" -eq 1 ]] \
        || fail "could not verify commit ancestry: $ancestor $descendant"
    return 1
}

abort_merge_if_needed() {
    local worktree=$1 merge_head

    merge_head="$(
        git -C "$worktree" \
            rev-parse --path-format=absolute --git-path MERGE_HEAD
    )" || fail "could not inspect merge state: $worktree"
    if [[ -e "$merge_head" ]]; then
        git -C "$worktree" merge --abort || return 1
    fi
}

restore_after_failure() {
    local status=$? child_status=0

    trap - EXIT
    # Repeated signals cannot cut short the owned supervisor's cleanup.
    trap '' HUP INT TERM
    if (( status != 0 )); then
        if [[ -n "$active_pid" ]]; then
            # The child owns its descendants. Wait for teardown before aborting
            # a candidate which a surviving hook could otherwise still modify.
            kill -TERM "$active_pid" 2>/dev/null || true
            wait "$active_pid" || child_status=$?
            check_owner_cleanup "$child_status"
        fi
        if [[ "$owner_cleanup_failed" -ne 0 ]]; then
            printf 'git-flow-merge: process cleanup unproven; candidate and task retained\n' >&2
        elif abort_merge_if_needed "$merge_worktree"; then
            # A signal or cleanup failure after the commit must not undo success.
            if [[ "$merge_worktree" == "$repository_root" ]] \
                && [[ "$(git -C "$merge_worktree" rev-parse HEAD)" == "$primary_oid" ]]; then
                git -C "$repository_root" switch "$task_branch" || {
                    printf 'git-flow-merge: task branch retained; could not switch to %s\n' \
                        "$task_branch" >&2
                }
            fi
        fi
    fi
    if [[ -n "$execution_directory" && "$owner_cleanup_failed" -eq 0 ]]; then
        rm -rf -- "$execution_directory"
    fi
    exit "$status"
}

check_owner_cleanup() {
    if [[ "$1" -eq 125 || ! -f "$active_cleanup_receipt" ]] \
        || [[ "$(cat "$active_cleanup_receipt")" != owned_tree_empty ]]; then
        owner_cleanup_failed=1
    fi
}

receive_signal() {
    if [[ "$starting_child" -eq 1 ]]; then
        pending_signal=$1
    else
        exit "$1"
    fi
}

prepare_owner() {
    local source_directory=$1 destination=$2 source_file owner_source_files
    mkdir -p "$destination/h2h_dev_tools"
    owner_source_files="$("$merge_python" -I -B "$source_directory/merge_command.py" --print-source-files)" \
        || fail 'could not resolve the process owner source closure'
    while IFS= read -r source_file; do
        cp "$source_directory/$source_file" "$destination/h2h_dev_tools/$source_file"
    done <<< "$owner_source_files"
    merge_launcher="$destination/h2h_dev_tools/merge_command.py"
}

run_tracked() {
    local status=0 working_directory=$2
    active_owner=$1
    shift 2
    command_number=$((command_number + 1))
    active_cleanup_receipt="$execution_directory/cleanup-$command_number"
    set -- "$working_directory" -- "$@"
    if [[ "$active_owner" == review ]]; then
        set -- --review "$@"
    fi
    set -- "$merge_python" -I -B "$merge_launcher" \
        --cleanup-receipt "$active_cleanup_receipt" "$@"
    # Do not leave a forked supervisor unregistered if a signal arrives between
    # the background launch and assignment of its PID.
    starting_child=1
    (cd "$working_directory" && exec "$@") &
    active_pid=$!
    starting_child=0
    [[ "$pending_signal" -eq 0 ]] || exit "$pending_signal"
    wait "$active_pid" || status=$?
    check_owner_cleanup "$status"
    active_pid=''
    [[ "$owner_cleanup_failed" -eq 0 ]] || return 125
    return "$status"
}

repository_root=${1:?repository root is required}
strategy=${2:?merge strategy is required}
review_command=${3-}
shift 3
[[ "$strategy" == direct || "$strategy" == candidate ]] || fail 'invalid merge strategy'

mode='finish'
if [[ $# -gt 1 ]]; then
    usage
    fail 'too many arguments'
fi
if [[ $# -eq 1 ]]; then
    [[ "$1" == '--cleanup-only' ]] || {
        usage
        fail "unsupported argument: $1"
    }
    mode='cleanup-only'
fi

cd "$repository_root"

primary="$(scripts/detect-primary-branch.sh)"
task_branch="$(git branch --show-current)"
[[ -n "$task_branch" ]] || fail 'detached HEAD is not supported'
[[ "$task_branch" != "$primary" ]] \
    || fail 'run this script from the task branch'

assert_clean "$repository_root"
assert_no_operation "$repository_root"

primary_ref="refs/heads/$primary"
task_ref="refs/heads/$task_branch"
primary_oid="$(git rev-parse --verify "$primary_ref^{commit}")" \
    || fail "primary branch is not a commit: $primary"
task_oid="$(git rev-parse --verify "$task_ref^{commit}")" \
    || fail "task branch is not a commit: $task_branch"
git merge-base "$primary_oid" "$task_oid" >/dev/null \
    || fail "task branch and primary have no common ancestor: $primary"

primary_worktree=''
candidate_worktree=''
candidate_branch=''
worktree_listing="$(git worktree list --porcelain)" \
    || fail 'could not inspect repository worktrees'
while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
        'worktree '*)
            candidate_worktree="${line#worktree }"
            candidate_branch=''
            ;;
        'branch refs/heads/'*)
            candidate_branch="${line#branch refs/heads/}"
            if [[ "$candidate_branch" == "$primary" ]]; then
                primary_worktree="$candidate_worktree"
            fi
            ;;
    esac
done <<< "$worktree_listing"

if is_ancestor "$repository_root" "$task_oid" "$primary_oid"; then
    if [[ -n "$primary_worktree" ]]; then
        assert_clean "$primary_worktree"
        assert_no_operation "$primary_worktree"
        cleanup_worktree="$primary_worktree"
    else
        cleanup_worktree="$repository_root"
    fi

    assert_ref_equals "$repository_root" "$task_ref" "$task_oid"
    assert_ref_equals "$repository_root" "$primary_ref" "$primary_oid"
    is_ancestor "$repository_root" "$task_oid" "$primary_oid" \
        || fail 'task is no longer contained in primary; branch was retained'

    if [[ "$cleanup_worktree" == "$repository_root" ]]; then
        git switch "$primary" \
            || fail "could not switch to primary; task branch was retained: $primary"
    else
        cd "$cleanup_worktree"
        git -C "$cleanup_worktree" worktree remove "$repository_root" \
            || fail "no-op task worktree remains: $repository_root"
    fi

    assert_clean "$cleanup_worktree"
    assert_no_operation "$cleanup_worktree"
    assert_ref_equals "$cleanup_worktree" "$task_ref" "$task_oid"
    assert_ref_equals "$cleanup_worktree" "$primary_ref" "$primary_oid"
    is_ancestor "$cleanup_worktree" "$task_oid" "$primary_oid" \
        || fail 'task is no longer contained in primary; branch was retained'
    git -C "$cleanup_worktree" branch -d "$task_branch" \
        || fail "no-op task branch remains: $task_branch"

    printf 'Cleaned up no-op task %s already contained in %s at %s\n' \
        "$task_branch" "$primary" "$task_oid"
    exit 0
fi

[[ "$mode" != 'cleanup-only' ]] \
    || fail 'task contains commits not present in primary; branch was retained'

hooks_path="$(git config --local --get core.hooksPath || true)"
[[ "$hooks_path" == '.githooks' ]] \
    || fail 'repository hooks are not installed; run scripts/install-git-hooks.sh'
[[ -x .githooks/pre-merge-commit ]] \
    || fail '.githooks/pre-merge-commit is missing or not executable'
if [[ "$strategy" == candidate ]]; then
    [[ -x .githooks/pre-commit ]] || fail '.githooks/pre-commit is missing or not executable'
fi

if [[ -n "$primary_worktree" ]]; then
    assert_clean "$primary_worktree"
    assert_no_operation "$primary_worktree"
    merge_worktree="$primary_worktree"
else
    merge_worktree="$repository_root"
fi

active_pid=''
active_owner=''
active_cleanup_receipt=''
command_number=0
starting_child=0
pending_signal=0
owner_cleanup_failed=0
execution_directory=''
trap restore_after_failure EXIT
trap 'receive_signal 129' HUP
trap 'receive_signal 130' INT
trap 'receive_signal 143' TERM
if [[ -x "$repository_root/.venv/bin/python" ]]; then
    merge_python="$repository_root/.venv/bin/python"
elif [[ -x "$repository_root/.venv/Scripts/python.exe" ]]; then
    merge_python="$repository_root/.venv/Scripts/python.exe"
else
    fail 'missing repository-local Python; run scripts/rebuild-env.sh'
fi
source_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
execution_directory="$(mktemp -d "${TMPDIR:-/tmp}/h2h-merge-owner.XXXXXX")"
prepare_owner "$source_directory" "$execution_directory/bootstrap"
if [[ -z "$primary_worktree" ]]; then
    git switch "$primary"
fi
assert_ref_equals "$merge_worktree" "$primary_ref" "$primary_oid"
assert_ref_equals "$merge_worktree" "$task_ref" "$task_oid"
export WORKFLOW_MERGE_TASK_REF="$task_branch"
if [[ "$strategy" == direct ]]; then
    # Git invokes pre-merge-commit for the exact candidate in these repositories.
    run_tracked process "$merge_worktree" git -C "$merge_worktree" \
        merge --no-ff --no-edit "$task_branch" \
        || fail 'merge or merge gate failed; task branch was retained'
else
    # These repositories' pre-commit hooks delegate pending merges to their gate.
    run_tracked process "$merge_worktree" git -C "$merge_worktree" \
        merge --no-ff --no-commit "$task_branch" \
        || fail 'merge failed; task branch was retained'
    # The task need not contain primary's latest tools. Export the real staged
    # merge tree after Git has combined both branches, so review/commit execute
    # the candidate's owner implementation rather than the old task copy.
    candidate_directory="$execution_directory/candidate"
    mkdir "$candidate_directory"
    source_relative="${source_directory#"$repository_root"/}"
    [[ "$source_relative" != "$source_directory" ]] \
        || fail 'process owner source is outside the task repository'
    candidate_tree="$(git -C "$merge_worktree" write-tree)"
    git -C "$merge_worktree" archive "$candidate_tree" "$source_relative" \
        | tar -x -C "$candidate_directory"
    prepare_owner "$candidate_directory/$source_relative" "$execution_directory/review"
    if [[ -n "$review_command" ]]; then
        run_tracked review "$merge_worktree" bash "$review_command" \
            || fail 'code review failed; task branch was retained'
    fi
    run_tracked process "$merge_worktree" git -C "$merge_worktree" commit --no-edit \
        || fail 'merge commit or release gate failed; task branch was retained'
fi

cd "$merge_worktree"
if [[ "$repository_root" != "$merge_worktree" ]]; then
    assert_clean "$repository_root"
    git -C "$merge_worktree" worktree remove "$repository_root" \
        || fail "merge succeeded, but task worktree remains: $repository_root"
fi

git -C "$merge_worktree" branch -d "$task_branch" \
    || fail "merge succeeded, but task branch remains: $task_branch"

merge_commit="$(git -C "$merge_worktree" rev-parse HEAD)"
printf 'Merged %s into %s at %s\n' \
    "$task_branch" "$primary" "$merge_commit"
