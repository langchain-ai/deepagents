# fix(ci): consolidate PR labeling workflows to eliminate race condition

## Problem

PR [#1774](https://github.com/langchain-ai/deepagents/pull/1774) was missing its `size: S` and `internal` labels despite the size labeler workflow running successfully. Investigation of the issue timeline revealed:

```
20:12:06Z  labeled    size: S      (by size labeler)
20:12:07Z  labeled    internal     (by tag-external-contributions)
20:12:07Z  unlabeled  internal     (clobbered)
20:12:07Z  unlabeled  size: S      (clobbered)
20:12:07Z  labeled    cli          (by file/title labeler)
20:12:07Z  labeled    deepagents   (by file/title labeler)
20:12:11Z  labeled    feature      (by title labeler)
```

**Root cause:** 4 separate workflows fired concurrently on `pull_request_target`, all mutating labels on the same PR. One of them (`bcoe/conventional-release-labels@v1` in `pr_labeler_title.yml`) likely uses `setLabels` internally, which replaces _all_ labels rather than appending — clobbering labels that other workflows had already applied.

## Solution

Consolidated all PR labeling into a single sequential workflow (`pr_labeler.yml`), eliminating concurrent label mutations entirely.

## Changes

### Created

- **`.github/workflows/pr_labeler.yml`** — unified PR labeling workflow
  - Size labels (XS/S/M/L/XL based on changed lines, excluding docs + lockfiles)
  - File-based labels (maps changed file paths to package labels)
  - Title-based labels (conventional commit type + scope parsing)
  - Internal/external contributor classification (org membership check)
  - Contributor tier (`trusted-contributor` for >= 5 merged PRs)
  - Backfill job for open PRs via `workflow_dispatch`
  - All labels ensured to exist before batch `addLabels` call (prevents 422 from aborting entire batch)

### Deleted

| File | Reason |
|---|---|
| `.github/workflows/pr_size_labeler.yml` | Moved to `pr_labeler.yml` |
| `.github/workflows/pr_labeler_file.yml` | Moved to `pr_labeler.yml` |
| `.github/workflows/pr_labeler_title.yml` | Moved to `pr_labeler.yml` |
| `.github/pr-file-labeler.yml` | Config inlined into `pr_labeler.yml` |

### Modified

- **`.github/workflows/tag-external-contributions.yml` -> `tag-external-issues.yml`**
  - Renamed to reflect new scope
  - Removed `pull_request_target` trigger and all PR-related steps
  - Scoped down to issues only (labeling + backfill)
  - Removed unnecessary `pull-requests: write` permission
  - Updated Setup Requirements comment (removed stale "Pull requests (write)" App permission)
  - Added 404 guard on backfill `removeLabel`

- **`.github/workflows/require_issue_link.yml`**
  - Updated stale references from `tag-external-contributions.yml` to `pr_labeler.yml` (lines 5, 15-18)

## Design decisions

- **`external` label uses App token** (not `GITHUB_TOKEN`) so the `labeled` event propagates to `require_issue_link.yml`. Events from `GITHUB_TOKEN` don't trigger downstream workflows.
- **`internal` label uses `GITHUB_TOKEN`** — no downstream workflow needs to react to it.
- **Org membership check only runs on `opened`** — contributor doesn't change on push/reopen. App token generation is skipped for other events.
- **Size + file labels skip on `edited`** — only the title changes on edit, not the diff.
- **Title labels run on all events** — harmless no-op on `synchronize`/`reopened` since the title is unchanged.
- **`ensureLabel` called for all labels before batch `addLabels`** — GitHub API returns 422 if any label in the batch doesn't exist, which would prevent _all_ labels from being applied. The old workflows were partially resilient to this (each workflow's labels were independent), but the consolidated batch call is all-or-nothing.

## Event trigger matrix

| Label type | `opened` | `synchronize` | `reopened` | `edited` |
|---|---|---|---|---|
| Size | Y | Y | Y | - |
| File-based | Y | Y | Y | - |
| Title (type + scope) | Y | Y | Y | Y |
| Internal/external | Y | - | - | - |
| Contributor tier | Y | - | - | - |
