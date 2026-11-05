#!/usr/bin/env bash

set -euo pipefail

diagnose() {
    # An uncertain owner cannot admit another writer, even to report the failure.
    [[ "${owner_cleanup_failed:-0}" -eq 0 ]] || return 0
    # Terminal release may already have removed admission, including on error.
    [[ "${terminal_release_started:-0}" -eq 0 ]] || return 0
    if [[ -n "${merge_launcher-}" || "${recovering:-0}" -eq 1 ]]; then
        # Use the same signal handoff, wait and cleanup receipt as Git owners.
        # A failed diagnostic must prevent recovery even if Git was already clean.
        run_tracked diagnostic "$PWD" "$1" || true
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
    printf 'usage: scripts/git-flow-merge.sh [--cleanup-only [task-branch]]\n' >&2
}

assert_clean() {
    local worktree=$1

    run_workflow_state check-clean --worktree "$worktree" \
        || fail "could not verify clean worktree: $worktree"
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

assert_effective_hooks_path() {
    local worktree=$1 hooks_path configuration_status=0

    # Preserve every byte of the effective value, including trailing newlines.
    # Git emits exactly one record-terminating LF; the sentinel keeps command
    # substitution from removing either that LF or newlines inside the value.
    hooks_path="$(
        git -C "$worktree" config --get core.hooksPath
        status=$?
        printf '.'
        exit "$status"
    )" || configuration_status=$?
    [[ "$configuration_status" -eq 0 ]] \
        || fail 'repository hooks are not installed; run scripts/install-git-hooks.sh'
    hooks_path="${hooks_path%.}"
    hooks_path="${hooks_path%$'\n'}"
    [[ "$hooks_path" == '.githooks' ]] \
        || fail 'repository hooks are not installed; run scripts/install-git-hooks.sh'
}

assert_merge_hooks() {
    local worktree=$1

    assert_effective_hooks_path "$worktree"
    [[ -x "$worktree/.githooks/pre-merge-commit" ]] \
        || fail '.githooks/pre-merge-commit is missing or not executable'
    [[ -x "$worktree/.githooks/pre-commit" ]] \
        || fail '.githooks/pre-commit is missing or not executable'
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

assert_git_support() {
    local version

    version="$(git version)" || fail 'could not inspect Git version'
    if [[ "$version" =~ ^git\ version\ ([0-9]+)\.([0-9]+) ]] \
        && (( 10#${BASH_REMATCH[1]} > 2 \
            || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} >= 38) )); then
        return 0
    fi
    fail 'safe merge workflow requires Git 2.38 or newer (ort preview and reference-transaction hooks)'
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

run_workflow_state() {
    # Every normal state operation uses the existing outer lifetime. Only the
    # failure handler sets recovering and supplies the ten-second owner budget.
    # Always use the original copied bootstrap, even after task removal.
    local owner=process
    case "${1-}" in
        release) owner=release; shift; set -- validate-release "$@" ;;
    esac
    run_tracked "$owner" "${merge_worktree:-$repository_root}" \
        "$merge_python" -I -B "$workflow_launcher" --workflow-state \
        "$@" --state "$workflow_state"
}

release_successful_workflow() {
    # Validation still runs under its normal owner. Once terminal release starts,
    # any failure or interruption may only finish that owner, never start recovery
    # against admission that the filesystem-only release may already have removed.
    terminal_release_started=1
    run_workflow_state release
}

restore_after_failure() {
    local status=$? child_status=0 recovery_failed=0

    trap - EXIT
    trap '' HUP INT TERM PIPE
    recovering=1
    if [[ -n "$active_pid" ]]; then
        kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" || child_status=$?
        check_owner_cleanup "$child_status"
        active_pid=''
    fi
    if [[ "$owner_cleanup_failed" -ne 0 ]]; then
        diagnose 'process cleanup unproven; candidate and task retained; workflow admission retained'
        recovery_failed=1
    elif [[ "$terminal_release_started" -ne 0 ]]; then
        [[ "$status" -eq 0 ]] || recovery_failed=1
    elif [[ -f "${workflow_state-}" ]]; then
        if (( status != 0 )); then
            if run_workflow_state recover-abort; then
                run_workflow_state recover-restore || recovery_failed=1
            else
                recovery_failed=1
            fi
        fi
        if [[ "$recovery_failed" -eq 0 && "$owner_cleanup_failed" -eq 0 ]]; then
            terminal_release_started=1
            run_workflow_state release || recovery_failed=1
        fi
        if [[ "$recovery_failed" -ne 0 || "$owner_cleanup_failed" -ne 0 ]]; then
            diagnose 'recovery ownership or cleanup unproven; candidate, task and workflow admission retained'
        fi
    fi
    if [[ "$recovery_failed" -ne 0 || "$owner_cleanup_failed" -ne 0 ]]; then
        [[ "$status" -ne 0 ]] || status=1
    elif [[ -n "$execution_directory" ]]; then
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
    # Resolve the stdlib owner closure without admitting arbitrary site startup.
    owner_source_files="$("$merge_python" -I -S -B "$source_directory/merge_command.py" --print-source-files)" \
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
    local status=0 working_directory=$2 bounded_command=$recovering output_descriptor=2
    active_owner=$1
    shift 2
    command_number=$((command_number + 1))
    active_cleanup_receipt="$execution_directory/cleanup-$command_number"
    if [[ "$active_owner" == diagnostic || "$active_owner" == completion ]]; then
        [[ "$active_owner" != completion ]] || output_descriptor=1
        # This fixed stdlib entry must not run site code before ownership begins.
        # Only its canonical bounded writer can spawn children; the receipt is
        # published after that writer is reaped and its obligation is discharged.
        set -- "$merge_python" -I -S -B "$merge_launcher" \
            --workflow-"$active_owner" "$active_cleanup_receipt" "$1"
    else
        set -- "$working_directory" -- "$@"
        if [[ "$active_owner" == review ]]; then
            set -- --review "$@"
        elif [[ "$active_owner" == release ]]; then
            set -- --release-workflow "$workflow_state" "$@"
        fi
        if [[ "$bounded_command" -eq 1 ]]; then
            set -- --recovery --recovery-output-fds 3 4 "$@"
        fi
        # Ownership and recovery deadlines start in this fixed stdlib launcher.
        # Site startup must not spawn or block before those boundaries exist.
        set -- "$merge_python" -I -S -B "$merge_launcher" \
            --cleanup-receipt "$active_cleanup_receipt" "$@"
    fi
    # Do not leave a forked supervisor unregistered if a signal arrives between
    # the background launch and assignment of its PID.
    starting_child=1
    if [[ "$active_owner" == diagnostic || "$active_owner" == completion ]]; then
        # Preserve stdout for success and stderr for failure diagnostics. Startup
        # errors stay silent; only the bounded writer may use the destination.
        (CDPATH= cd "$working_directory" && exec "$@") 3>&"$output_descriptor" >/dev/null 2>&1 &
    elif [[ "$bounded_command" -eq 1 ]]; then
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
review_command=${2:?review command is required}
shift 2

mode='finish'
cleanup_task=''
if [[ $# -gt 2 ]]; then
    usage
    fail 'too many arguments'
fi
if [[ $# -ge 1 ]]; then
    [[ "$1" == '--cleanup-only' ]] || {
        usage
        fail "unsupported argument: $1"
    }
    mode='cleanup-only'
    if [[ $# -eq 2 ]]; then
        [[ -n "$2" ]] || fail 'cleanup task branch must not be empty'
        cleanup_task=$2
    fi
fi

CDPATH= cd "$repository_root"
repository_root=$PWD

active_pid=''
active_owner=''
active_cleanup_receipt=''
command_number=0
starting_child=0
pending_signal=0
owner_cleanup_failed=0
terminal_release_started=0
execution_directory=''
recovering=0
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
workflow_launcher="$merge_launcher"
workflow_state="$execution_directory/workflow.json"
run_workflow_state acquire --repository "$repository_root" --owner-pid "$$" \
    || fail 'could not acquire exclusive merge workflow admission'

primary="$(scripts/detect-primary-branch.sh)"
current_branch="$(git branch --show-current)"
[[ -n "$current_branch" ]] || fail 'detached HEAD is not supported'
if [[ -n "$cleanup_task" ]]; then
    [[ "$current_branch" == "$primary" ]] \
        || fail 'named cleanup must run from the primary checkout'
    task_branch=$cleanup_task
    [[ "$task_branch" != "$primary" ]] || fail 'primary cannot be the cleanup task'
else
    task_branch=$current_branch
    [[ "$task_branch" != "$primary" ]] \
        || fail 'run this script from the task branch or use --cleanup-only task-branch'
fi

assert_clean "$repository_root"
assert_no_operation "$repository_root"

primary_ref="refs/heads/$primary"
task_ref="refs/heads/$task_branch"
git check-ref-format "$task_ref" >/dev/null || fail "invalid task branch: $task_branch"
primary_oid="$(git rev-parse --verify "$primary_ref^{commit}")" \
    || fail "primary branch is not a commit: $primary"
task_oid="$(git rev-parse --verify "$task_ref^{commit}")" \
    || fail "task branch is not a commit: $task_branch"
git merge-base "$primary_oid" "$task_oid" >/dev/null \
    || fail "task branch and primary have no common ancestor: $primary"
assert_git_support

worktree_from_refs() {
    local expected_ref=$1 ref path terminator selected='' matched=0

    # Git emits ref NUL path NUL LF. Read NUL-delimited fields before Bash
    # command substitution, which cannot preserve NUL bytes. Only a fully
    # parsed exact ref is accepted, including its empty not-checked-out path.
    while IFS= read -r -d '' ref; do
        IFS= read -r -d '' path || return 1
        IFS= read -r terminator || return 1
        [[ -z "$terminator" ]] || return 1
        if [[ "$ref" == "$expected_ref" ]]; then
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
        | worktree_from_refs "$primary_ref"
)" || fail 'could not inspect repository worktrees'
primary_worktree="${primary_worktree%.}"

task_worktree="$repository_root"
if [[ -n "$cleanup_task" ]]; then
    [[ "$primary_worktree" == "$repository_root" ]] \
        || fail 'primary checkout changed before named cleanup'
    task_worktree="$(
        git for-each-ref --format='%(refname)%00%(worktreepath)%00' "$task_ref" \
            | worktree_from_refs "$task_ref"
    )" || fail 'could not inspect task worktree'
    task_worktree="${task_worktree%.}"
    if [[ -n "$task_worktree" ]]; then
        assert_checked_out_ref "$task_worktree" "$task_ref"
        assert_clean "$task_worktree"
        assert_no_operation "$task_worktree"
    fi
fi

if [[ -n "$primary_worktree" ]]; then
    assert_clean "$primary_worktree"
    assert_no_operation "$primary_worktree"
    merge_worktree="$primary_worktree"
else
    merge_worktree="$repository_root"
fi

if [[ -n "$cleanup_task" ]]; then
    is_ancestor "$repository_root" "$task_oid" "$primary_oid" \
        || fail 'task contains commits not present in primary; branch was retained'
fi

run_workflow_state configure --worktree "$merge_worktree" \
    --primary-ref "$primary_ref" --primary-oid "$primary_oid" \
    --task-ref "$task_ref" --task-oid "$task_oid" \
    || fail 'could not bind merge workflow identity'

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

    run_workflow_state cleanup || fail 'could not begin owned task cleanup'
    if [[ -z "$cleanup_task" && "$cleanup_worktree" == "$repository_root" ]]; then
        run_workflow_state switch-primary \
            || fail "could not switch to primary; task branch was retained: $primary"
    elif [[ -n "$task_worktree" ]]; then
        assert_checked_out_ref "$task_worktree" "$task_ref"
        select_local_python "$cleanup_worktree"
        CDPATH= cd "$cleanup_worktree"
        run_workflow_state remove-task-worktree --task-worktree "$task_worktree" \
            || fail "no-op task worktree remains: $task_worktree; inspect untracked and ignored files, then retry from primary with --cleanup-only $task_branch"
    fi

    delete_task_branch "$cleanup_worktree" "$primary_oid" \
        || fail "no-op task branch cleanup failed; inspect refs: $task_branch"

    run_tracked completion "$cleanup_worktree" \
        "Cleaned up no-op task $task_branch already contained in $primary at $task_oid"
    release_successful_workflow
    exit 0
fi

[[ "$mode" != 'cleanup-only' ]] \
    || fail 'task contains commits not present in primary; branch was retained'

assert_effective_hooks_path "$merge_worktree"

if [[ -z "$primary_worktree" ]]; then
    run_workflow_state switch-primary \
        || fail "could not switch to primary; task branch was retained: $primary"
fi
# Branch-conditional configuration can change when primary is checked out.
assert_effective_hooks_path "$merge_worktree"
assert_ref_equals "$merge_worktree" "$primary_ref" "$primary_oid"
assert_ref_equals "$merge_worktree" "$task_ref" "$task_oid"
# These repositories' pre-commit hooks delegate pending merges to their gate.
run_workflow_state begin || fail 'could not establish safe merge inputs'
merge_status=0
run_tracked process "$merge_worktree" git -C "$merge_worktree" \
    -c core.hooksPath=.githooks -c "branch.$primary.mergeOptions=" \
    merge -s ort --no-overwrite-ignore --no-ff --no-commit -m "Merge branch '$task_branch' into $primary" "$task_oid" \
    || merge_status=$?
[[ "$owner_cleanup_failed" -eq 0 ]] || fail 'merge cleanup unproven'
run_workflow_state capture || fail 'merge outcome ownership unproven'
[[ "$merge_status" -eq 0 ]] || fail 'merge failed; task branch was retained'
# Candidate merges may introduce hooks; validate the actual combined tree.
assert_merge_hooks "$merge_worktree"
# The task need not contain primary's latest tools. Export the real staged
# merge tree after Git has combined both branches, so review/commit execute
# the candidate's owner implementation rather than the old task copy.
candidate_directory="$execution_directory/candidate"
mkdir "$candidate_directory"
source_relative="${source_directory#"$repository_root"/}"
[[ "$source_relative" != "$source_directory" ]] \
    || fail 'process owner source is outside the task repository'
candidate_tree="$(cat "${workflow_state%.json}.tree")" \
    || fail 'could not read captured merge candidate tree'
[[ -n "$candidate_tree" ]] || fail 'captured merge candidate tree is empty'
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
assert_merge_hooks "$merge_worktree"
run_tracked process "$merge_worktree" git -C "$merge_worktree" \
    -c core.hooksPath=.githooks commit --no-edit \
    || fail 'merge commit or release gate failed; task branch was retained'

run_workflow_state finish || fail 'merge commit identity changed; task retained'

CDPATH= cd "$merge_worktree"
merge_commit="$(cat "${workflow_state%.json}.commit")"
assert_checked_out_ref "$merge_worktree" "$primary_ref"
assert_cleanup_refs "$merge_worktree" "$merge_commit"
if [[ "$repository_root" != "$merge_worktree" ]]; then
    assert_clean "$repository_root"
    assert_no_operation "$repository_root"
    assert_checked_out_ref "$repository_root" "$task_ref"
    assert_cleanup_refs "$merge_worktree" "$merge_commit"
    select_local_python "$merge_worktree"
    run_workflow_state remove-task-worktree --task-worktree "$repository_root" \
        || fail "merge succeeded, but task worktree remains: $repository_root; inspect untracked and ignored files, then retry from primary with --cleanup-only $task_branch"
fi

delete_task_branch "$merge_worktree" "$merge_commit" \
    || fail "merge succeeded, but task branch cleanup failed; inspect refs: $task_branch"

run_tracked completion "$merge_worktree" \
    "Merged $task_branch into $primary at $merge_commit"
release_successful_workflow
