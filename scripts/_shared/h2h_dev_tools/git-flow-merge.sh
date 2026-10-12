#!/usr/bin/env bash

set -euo pipefail

diagnose() {
    if [[ -n "${merge_launcher-}" || "${recovering:-0}" -eq 1 ]]; then
        # A failure must reach EXIT recovery even when the caller stops reading.
        # Use the prepared owner for normal failures as well as recovery. Startup
        # failures are silent, never a blocking printf fallback. FD 3 preserves
        # the destination while startup stderr is null.
        "${merge_python-}" -I -B "${merge_launcher-}" --workflow-diagnostic "$1" \
            3>&2 >/dev/null 2>&1 || true
        return
    fi
    # Preflight has no owned child or Git mutation to recover. It must still
    # explain missing Python before a diagnostic helper is available.
    # Bash can retry buffered text from a failed builtin write in a later command
    # substitution. Isolate diagnostics so a closed pipe cannot corrupt Git data.
    (printf 'git-flow-merge: %s\n' "$1" >&2) || true
}

fail() {
    diagnose "$1"
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

assert_checked_out_ref() {
    local actual worktree=$1 expected=$2

    actual="$(git -C "$worktree" symbolic-ref --quiet HEAD)" \
        || fail "worktree no longer has its expected branch: $worktree"
    [[ "$actual" == "$expected" ]] \
        || fail "worktree branch changed during cleanup: $worktree"
}

assert_cleanup_refs() {
    local worktree=$1 expected_primary=$2

    assert_ref_equals "$worktree" "$task_ref" "$task_oid"
    assert_ref_equals "$worktree" "$primary_ref" "$expected_primary"
    is_ancestor "$worktree" "$task_oid" "$expected_primary" \
        || fail 'task is no longer contained in primary; branch was retained'
}

assert_reference_transaction_support() {
    local version

    version="$(git version)" || fail 'could not inspect Git version'
    if [[ "$version" =~ ^git\ version\ ([0-9]+)\.([0-9]+) ]] \
        && (( 10#${BASH_REMATCH[1]} > 2 \
            || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} >= 28) )); then
        return 0
    fi
    fail 'safe branch cleanup requires Git 2.28 or newer (reference-transaction hooks)'
}

delete_task_branch() {
    local worktree=$1 expected_primary=$2 cleanup_hooks original_hook status=0

    assert_clean "$worktree"
    assert_no_operation "$worktree"
    assert_checked_out_ref "$worktree" "$primary_ref"
    assert_cleanup_refs "$worktree" "$expected_primary"
    original_hook="$(git -C "$worktree" rev-parse --path-format=absolute \
        --git-path hooks/reference-transaction)" \
        || fail 'could not locate the existing reference transaction hook'
    cleanup_hooks="$(mktemp -d "${TMPDIR:-/tmp}/h2h-cleanup-hooks.XXXXXX")" \
        || fail 'could not isolate branch deletion from its upstream'
    # Keep the guard in this already-loaded function: the linked task worktree
    # may have been removed, so no implementation can be loaded from it here.
    if ! cat > "$cleanup_hooks/reference-transaction" <<'HOOK'
#!/bin/sh
set -eu
input="$H2H_CLEANUP_HOOK_DIRECTORY/input-$$"
trap 'rm -f -- "$input" || :' EXIT
cat > "$input"
# Preserve all original events and the exact input, including aborted events
# emitted by a backend's internal transaction. Its rejection takes precedence.
if [ -x "$H2H_CLEANUP_ORIGINAL_HOOK" ]; then
    "$H2H_CLEANUP_ORIGINAL_HOOK" "$@" < "$input" || exit "$?"
fi
[ "$1" = prepared ] || exit 0
while read -r old new ref; do
    [ "$ref" = "$H2H_CLEANUP_TASK_REF" ] || continue
    case "$new" in ''|*[!0]*) continue ;; esac
    # branch -d sends a zero old OID rather than requesting a compare-and-swap.
    # At prepared, Git holds the reference locks: read the actual old value now.
    actual_task=$(git rev-parse --verify "$ref^{commit}") || exit 1
    actual_primary=$(git rev-parse --verify "$H2H_CLEANUP_PRIMARY_REF^{commit}") || exit 1
    actual_head=$(git symbolic-ref --quiet HEAD) || exit 1
    if [ "$actual_task" != "$H2H_CLEANUP_TASK_OID" ] \
        || [ "$actual_primary" != "$H2H_CLEANUP_PRIMARY_OID" ] \
        || [ "$actual_head" != "$H2H_CLEANUP_PRIMARY_REF" ]; then
        printf 'git-flow-merge: ref changed while locking branch deletion; task retained\n' >&2 || true
        exit 1
    fi
    git merge-base --is-ancestor "$actual_task" "$actual_primary" || exit 1
done < "$input"
HOOK
    then
        rm -rf -- "$cleanup_hooks"
        fail 'could not write the branch deletion guard; task retained'
    fi
    if ! chmod +x "$cleanup_hooks/reference-transaction" \
        || [[ ! -x "$cleanup_hooks/reference-transaction" ]]; then
        rm -rf -- "$cleanup_hooks"
        fail 'could not enable the branch deletion guard; task retained'
    fi
    # An unconfigured remote has no fetch mapping, making primary HEAD Git's
    # ancestry authority. Do not override merge: that setting is multivalued.
    # The prepared hook supplies the locked check that branch -d itself omits.
    H2H_CLEANUP_HOOK_DIRECTORY="$cleanup_hooks" \
    H2H_CLEANUP_ORIGINAL_HOOK="$original_hook" \
    H2H_CLEANUP_TASK_REF="$task_ref" H2H_CLEANUP_TASK_OID="$task_oid" \
    H2H_CLEANUP_PRIMARY_REF="$primary_ref" H2H_CLEANUP_PRIMARY_OID="$expected_primary" \
    run_tracked process "$worktree" git -C "$worktree" -c "core.hooksPath=$cleanup_hooks" \
        -c "branch.$task_branch.remote=$cleanup_hooks" \
        branch -d "$task_branch" || status=$?
    if [[ "$owner_cleanup_failed" -ne 0 ]]; then
        diagnose "process cleanup unproven; hook files retained for inspection: $cleanup_hooks"
        return "$status"
    fi
    rm -rf -- "$cleanup_hooks" || fail 'could not remove temporary cleanup directory'
    return "$status"
}

abort_merge_if_needed() {
    local worktree=$1 merge_head

    merge_head="$(
        git -C "$worktree" \
            rev-parse --path-format=absolute --git-path MERGE_HEAD
    )" || fail "could not inspect merge state: $worktree"
    if [[ -e "$merge_head" ]]; then
        run_tracked process "$worktree" git -C "$worktree" merge --abort || return 1
    fi
}

restore_after_failure() {
    local status=$? child_status=0

    trap - EXIT
    # Repeated cancellation and closed diagnostic pipes cannot cut short cleanup
    # or replace the original exit status after the owner has proved termination.
    trap '' HUP INT TERM PIPE
    recovering=1
    if (( status != 0 )); then
        if [[ -n "$active_pid" ]]; then
            # The child owns its descendants. Wait for teardown before aborting
            # a candidate which a surviving hook could otherwise still modify.
            kill -TERM "$active_pid" 2>/dev/null || true
            wait "$active_pid" || child_status=$?
            check_owner_cleanup "$child_status"
        fi
        if [[ "$owner_cleanup_failed" -ne 0 ]]; then
            diagnose 'process cleanup unproven; candidate and task retained'
        elif [[ "$workflow_started" -eq 1 ]]; then
            if abort_merge_if_needed "$merge_worktree"; then
                # A signal or cleanup failure after commit must not undo success.
                if [[ "$merge_worktree" == "$repository_root" ]] \
                    && [[ "$(git -C "$merge_worktree" rev-parse HEAD)" == "$primary_oid" ]]; then
                    run_tracked process "$repository_root" git -C "$repository_root" switch "$task_branch" || {
                        diagnose "recovery incomplete; could not finish switching to $task_branch"
                    }
                fi
            else
                diagnose 'recovery incomplete; merge abort failed; inspect candidate and task before retrying'
            fi
        fi
        if [[ "$owner_cleanup_failed" -ne 0 ]]; then
            diagnose 'process cleanup unproven; no further Git mutation is permitted'
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

select_local_python() {
    local worktree=$1
    if [[ -x "$worktree/.venv/bin/python" ]]; then
        merge_python="$worktree/.venv/bin/python"
    elif [[ -x "$worktree/.venv/Scripts/python.exe" ]]; then
        merge_python="$worktree/.venv/Scripts/python.exe"
    else
        fail "missing repository-local Python in $worktree; run scripts/rebuild-env.sh there"
    fi
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
    if [[ "$recovering" -eq 1 ]]; then
        set -- --recovery --recovery-output-fds 3 4 "$@"
    fi
    set -- "$merge_python" -I -B "$merge_launcher" \
        --cleanup-receipt "$active_cleanup_receipt" "$@"
    # Do not leave a forked supervisor unregistered if a signal arrives between
    # the background launch and assignment of its PID.
    starting_child=1
    if [[ "$recovering" -eq 1 ]]; then
        # Preserve destinations for the nonblocking diagnostic writer only.
        # Startup errors (missing cwd/interpreter/launcher) must not block before
        # the owner can install that writer or establish its hard deadline.
        (CDPATH= cd "$working_directory" && exec "$@") 3>&1 4>&2 >/dev/null 2>&1 &
    else
        (CDPATH= cd "$working_directory" && exec "$@") &
    fi
    active_pid=$!
    starting_child=0
    if [[ "$pending_signal" -ne 0 && "$recovering" -eq 0 ]]; then
        exit "$pending_signal"
    fi
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

CDPATH= cd "$repository_root"
repository_root=$PWD

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
assert_reference_transaction_support

primary_worktree_from_refs() {
    local ref path terminator selected='' matched=0

    # Git emits ref NUL path NUL LF. Read NUL-delimited fields before Bash
    # command substitution, which cannot preserve NUL bytes. Only a fully
    # parsed exact ref is accepted, including its empty not-checked-out path.
    while IFS= read -r -d '' ref; do
        IFS= read -r -d '' path || return 1
        IFS= read -r terminator || return 1
        [[ -z "$terminator" ]] || return 1
        if [[ "$ref" == "$primary_ref" ]]; then
            [[ "$matched" -eq 0 ]] || return 1
            selected="$path"
            matched=1
        fi
    done
    [[ -z "$ref" && "$matched" -eq 1 ]] || return 1
    # A non-newline sentinel preserves a legal trailing LF in the selected path.
    printf '%s.' "$selected"
}

primary_worktree="$(
    git for-each-ref --format='%(refname)%00%(worktreepath)%00' "$primary_ref" \
        | primary_worktree_from_refs
)" || fail 'could not inspect repository worktrees'
primary_worktree="${primary_worktree%.}"

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
recovering=0
workflow_started=0
trap restore_after_failure EXIT
trap 'receive_signal 129' HUP
trap 'receive_signal 130' INT
trap 'receive_signal 143' TERM
# Pipe consumers may exit before cleanup finishes. Diagnostics must not kill
# this owner or prevent its EXIT handler from validating the cleanup receipt.
trap '' PIPE
select_local_python "$repository_root"
source_directory="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
execution_directory="$(mktemp -d "${TMPDIR:-/tmp}/h2h-merge-owner.XXXXXX")"
prepare_owner "$source_directory" "$execution_directory/bootstrap"
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

    workflow_started=1
    if [[ "$cleanup_worktree" == "$repository_root" ]]; then
        run_tracked process "$cleanup_worktree" git -C "$cleanup_worktree" switch "$primary" \
            || fail "could not switch to primary; task branch was retained: $primary"
    else
        assert_checked_out_ref "$repository_root" "$task_ref"
        select_local_python "$cleanup_worktree"
        CDPATH= cd "$cleanup_worktree"
        run_tracked process "$cleanup_worktree" git -C "$cleanup_worktree" worktree remove "$repository_root" \
            || fail "no-op task worktree remains: $repository_root"
    fi

    delete_task_branch "$cleanup_worktree" "$primary_oid" \
        || fail "no-op task branch cleanup failed; inspect refs: $task_branch"

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

workflow_started=1
if [[ -z "$primary_worktree" ]]; then
    run_tracked process "$merge_worktree" git -C "$merge_worktree" switch "$primary" \
        || fail "could not switch to primary; task branch was retained: $primary"
fi
assert_ref_equals "$merge_worktree" "$primary_ref" "$primary_oid"
assert_ref_equals "$merge_worktree" "$task_ref" "$task_oid"
export WORKFLOW_MERGE_TASK_REF="$task_oid"
if [[ "$strategy" == direct ]]; then
    # Git invokes pre-merge-commit for the exact candidate in these repositories.
    run_tracked process "$merge_worktree" git -C "$merge_worktree" \
        merge --no-ff --no-edit -m "Merge branch '$task_branch' into $primary" "$task_oid" \
        || fail 'merge or merge gate failed; task branch was retained'
else
    # These repositories' pre-commit hooks delegate pending merges to their gate.
    run_tracked process "$merge_worktree" git -C "$merge_worktree" \
        merge --no-ff --no-commit -m "Merge branch '$task_branch' into $primary" "$task_oid" \
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
    archive_paths=("$source_relative")
    if [[ "$source_relative" == scripts/_shared/h2h_dev_tools ]]; then
        # Snapshot source bootstrap also verifies the repository's explicit
        # component selection. Export that authority from this exact tree.
        archive_paths+=(pyproject.toml)
    fi
    git -C "$merge_worktree" archive "$candidate_tree" "${archive_paths[@]}" \
        | tar -x -C "$candidate_directory"
    prepare_owner "$candidate_directory/$source_relative" "$execution_directory/review"
    if [[ -n "$review_command" ]]; then
        run_tracked review "$merge_worktree" bash "$review_command" \
            || fail 'code review failed; task branch was retained'
    fi
    run_tracked process "$merge_worktree" git -C "$merge_worktree" commit --no-edit \
        || fail 'merge commit or release gate failed; task branch was retained'
fi

CDPATH= cd "$merge_worktree"
merge_commit="$(git -C "$merge_worktree" rev-parse HEAD)"
assert_checked_out_ref "$merge_worktree" "$primary_ref"
assert_cleanup_refs "$merge_worktree" "$merge_commit"
if [[ "$repository_root" != "$merge_worktree" ]]; then
    assert_clean "$repository_root"
    assert_no_operation "$repository_root"
    assert_checked_out_ref "$repository_root" "$task_ref"
    assert_cleanup_refs "$merge_worktree" "$merge_commit"
    select_local_python "$merge_worktree"
    run_tracked process "$merge_worktree" git -C "$merge_worktree" worktree remove "$repository_root" \
        || fail "merge succeeded, but task worktree remains: $repository_root"
fi

delete_task_branch "$merge_worktree" "$merge_commit" \
    || fail "merge succeeded, but task branch cleanup failed; inspect refs: $task_branch"

printf 'Merged %s into %s at %s\n' \
    "$task_branch" "$primary" "$merge_commit"
