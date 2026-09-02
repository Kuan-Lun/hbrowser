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
        git -C "$worktree" merge --abort || true
    fi
}

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

repository_root="$(git rev-parse --show-toplevel)"
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

if [[ -n "$primary_worktree" ]]; then
    assert_clean "$primary_worktree"
    assert_no_operation "$primary_worktree"
    merge_worktree="$primary_worktree"
else
    git switch "$primary"
    merge_worktree="$repository_root"
fi

if ! WORKFLOW_MERGE_TASK_REF="$task_branch" \
    git -C "$merge_worktree" merge --no-ff --no-edit "$task_branch"; then
    abort_merge_if_needed "$merge_worktree"
    if [[ "$merge_worktree" == "$repository_root" ]]; then
        git -C "$repository_root" switch "$task_branch" || true
    fi
    fail 'merge or merge gate failed; task branch was retained'
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
