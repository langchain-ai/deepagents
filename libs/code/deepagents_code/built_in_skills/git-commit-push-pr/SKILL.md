---
name: git-commit-push-pr
description: "Deliver finished work in a git repository by verifying it against that repository's own checks, then staging, committing, pushing, and locating the pull request, and reporting the commit hash range and PR URL. Use when the user asks to: (1) commit, (2) \"commit n push\" or \"commit and push\", (3) push the branch, (4) open a PR, (5) make a PR to fix something, (6) update an existing PR, or (7) split work into a sequence of commits."
license: MIT
compatibility: designed for deepagents-code
---

# Git Commit, Push, and PR

Delivery is a bounded sequence. Run it in order and stop at the first step whose tool result did not succeed.

## Best practices

- **Never claim a step is done without an in-session tool result for that exact step.** No commit hash means no commit; no successful `git push` means not pushed; no PR URL means no PR.
- **Verify with the target repository's checks, not this package's.** Discover them from the repository being changed (its `Makefile`, `pyproject.toml`/`package.json` scripts, `AGENTS.md`, `CONTRIBUTING.md`, or CI config). Do not assume `make lint && make test` exists.
- **A failed or denied push is a blocker to surface, not a finished turn.**

## Process

1. **Establish state.** Run `git status` and `git diff` (plus `git diff --staged` if anything is already staged). Identify the current branch and its upstream — and, when updating an existing PR, the PR head that branch corresponds to.
2. **Verify.** Run the target repository's lint and test commands, scoped to what changed. Require exit code 0 before staging. On a nonzero exit, re-derive the command or path and fix the cause; never stage over a failing check.
3. **Stage explicitly.** Stage the intended paths by name rather than `git add -A`, then re-read `git diff --staged` to confirm only those changes are included. Compose the commit message in a temp file (for example `/tmp/commit-msg.txt`) and pass it with `git commit -F`, so multi-line messages and shell metacharacters survive intact.
4. **Commit.** If the commit fails because signing is unavailable, retry with `--no-gpg-sign` rather than ending the turn.
5. **Push.** Push the branch to its remote. If the push is rejected, denied, or requires credentials you do not have, stop and tell the user what blocked it.
6. **Locate the pull request.** Use `gh pr view` / `gh pr list --head <branch>` for the pushed branch, or open a PR when the user asked for one and none exists. When splitting work across several commits, keep every commit independently passing the repository's checks so the PR is bisectable.
7. **Report evidence.** Give the developer the commit hash range (`git log --oneline <base>..HEAD`), the branch name, and the PR URL.

## Common pitfalls

- Reporting success from a plan rather than from tool output — the most common failure in this workflow.
- Running this package's checks against a different target repository, and concluding the code is broken when the command simply does not exist there.
- Amending or force-pushing a branch that backs an open PR without the user asking for it.
- Committing unrelated working-tree changes because staging was done with a wildcard.
