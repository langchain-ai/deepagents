# CLI Release Process

This document describes the release process for the CLI package (`libs/cli`) in the Deep Agents monorepo using [release-please](https://github.com/googleapis/release-please).

## Overview

CLI releases are managed via release-please, which:

1. Analyzes conventional commits on the `main` branch
2. Creates/updates a release PR with changelog and version bump
3. When merged, creates a draft GitHub release for review
4. Publishing the draft triggers PyPI publication

## How It Works

### Automatic Release PRs

When commits land on `main`, release-please analyzes them and either:

- **Creates a new release PR** if releasable changes exist
- **Updates an existing release PR** with additional changes
- **Does nothing** if no releasable commits are found (e.g. commits with type `chore`, `refactor`, etc.)

Release PRs are created on branches named `release-please--branches--main--components--<package>`.

### Triggering a Release

To release the CLI:

1. Merge conventional commits to `main` (see [Commit Format](#commit-format))
2. Wait for release-please to create/update the release PR
3. Review the generated changelog in the PR
4. Merge the release PR — this creates a **draft** GitHub release
5. Review and edit the release notes in the GitHub UI
6. Click "Publish release" — this triggers PyPI publication

### Version Bumping

Version bumps are determined by commit types:

| Commit Type                    | Version Bump  | Example                                  |
| ------------------------------ | ------------- | ---------------------------------------- |
| `fix:`                         | Patch (0.0.x) | `fix(cli): resolve config loading issue` |
| `feat:`                        | Minor (0.x.0) | `feat(cli): add new export command`      |
| `feat!:` or `BREAKING CHANGE:` | Major (x.0.0) | `feat(cli)!: redesign config format`     |

> [!NOTE]
> While version is < 1.0.0, `bump-minor-pre-major` and `bump-patch-for-minor-pre-major` are enabled, so breaking changes bump minor and features bump patch.

## Commit Format

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/) format with types and scopes defined in `.github/workflows/pr_lint.yml`:

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Examples

```bash
# Patch release
fix(cli): resolve type hinting issue

# Minor release
feat(cli): add new chat completion feature

# Major release (breaking change)
feat(cli)!: redesign configuration format

BREAKING CHANGE: Config files now use TOML instead of JSON.
```

## Configuration Files

### `release-please-config.json`

Defines release-please behavior for each package.

### `.release-please-manifest.json`

Tracks the current version of each package:

```json
{
  "libs/cli": "0.0.17"
}
```

This file is automatically updated by release-please when releases are created.

## Release Workflow

### Detection Mechanism

The release-please workflow (`.github/workflows/release-please.yml`) detects a CLI release by checking if `libs/cli/CHANGELOG.md` was modified in the commit. This file is always updated by release-please when merging a release PR.

### Lockfile Updates

When release-please creates or updates a release PR, the `update-lockfiles` job automatically regenerates `uv.lock` files since release-please updates `pyproject.toml` versions but doesn't regenerate lockfiles. An up-to-date lockfile is necessary for the cli since it depends on the SDK, and `libs/harbor` depends on the CLI.

### Release Pipeline

The release workflow (`.github/workflows/release.yml`) runs when a release PR is merged:

1. **Build** - Creates distribution package
2. **Collect Contributors** - Gathers PR authors for release notes, including social media handles. Excludes members of `langchain-ai`.
3. **Release Notes** - Extracts changelog or generates from git log
4. **Test PyPI** - Publishes to test.pypi.org for validation
5. **Pre-release Checks** - Runs tests against the built package
6. **Mark Release** - Creates a **draft** GitHub release with the built artifacts

When you publish the draft release, `.github/workflows/release-publish.yml` triggers and publishes to PyPI.

### Release PR Labels

Release-please uses labels to track the state of release PRs:

| Label | Meaning |
| ----- | ------- |
| `autorelease: pending` | Release PR has been merged but not yet tagged/released |
| `autorelease: tagged` | Release PR has been successfully tagged and released |

Because `skip-github-release: true` is set in the release-please config (we create releases via our own workflow instead of release-please), our `release.yml` workflow must update these labels manually. After successfully creating the GitHub release and tag, the `mark-release` job transitions the label from `pending` to `tagged`.

This label transition signals to release-please that the merged PR has been fully processed, allowing it to create new release PRs for subsequent commits.

## Manual Release

For hotfixes or exceptional cases, you can trigger a release manually. Use the `hotfix` commit type so as to not trigger a further PR update/version bump.

1. Go to **Actions** > **Package Release**
2. Click **Run workflow**
3. Select the package to release (`deepagents-cli` only for exception/recovery/hotfix scenarios; otherwise use release-please)
4. (Optionally enable `dangerous-nonmain-release` for hotfix branches)

> [!WARNING]
> Manual releases should be rare. Prefer the standard release-please flow for the CLI. Manual dispatch bypasses the changelog detection in `release-please.yml` and skips the lockfile update job. Only use it for recovery scenarios (e.g., the release workflow failed after the release PR was already merged).

## Troubleshooting

### "Found release tag with component X, but not configured in manifest" Warnings

You may see warnings in the release-please logs like:

```txt
⚠ Found release tag with component 'deepagents=', but not configured in manifest
```

This is **harmless**. Release-please scans existing tags in the repository and warns when it finds tags for packages that aren't in the current configuration. The `deepagents` SDK package has existing release tags (`deepagents==0.x.x`) but is not currently managed by release-please.

These warnings will disappear once the SDK is added to `release-please-config.json`. Until then, they can be safely ignored—they don't affect CLI releases.

### Unexpected Commit Authors in Release PRs

When viewing a release-please PR on GitHub, you may see commits attributed to contributors who didn't directly push to that PR. For example:

```txt
johndoe and others added 3 commits 4 minutes ago
```

This is a **GitHub UI quirk** caused by force pushes/rebasing, not actual commits to the PR branch.

**What's happening:**

1. release-please rebases its branch onto the latest `main`
2. The PR branch now includes commits from `main` as parent commits
3. GitHub's UI shows all "new" commits that appeared after the force push, including rebased parents

**The actual PR commits** are only:

- The release commit (e.g., `release(deepagents-cli): 0.0.18`)
- The lockfile update commit (e.g., `chore: update lockfiles`)

Other commits shown are just the base that the PR branch was rebased onto. This is normal behavior and doesn't indicate unauthorized access.

### Release PR Stuck with "autorelease: pending" Label

If a release PR shows `autorelease: pending` after the release workflow completed, the label update step may have failed. This can block release-please from creating new release PRs.

**To fix manually:**

```bash
# Find the PR number for the release commit
gh pr list --state merged --search "release(deepagents-cli)" --limit 5

# Update the label
gh pr edit <PR_NUMBER> --remove-label "autorelease: pending" --add-label "autorelease: tagged"
```

The label update is non-fatal in the workflow (`|| true`), so the release itself succeeded—only the label needs fixing.

### Yanking a Release

If you need to yank (retract) a release:

#### 1. Yank from PyPI

Using the PyPI web interface or a CLI tool.

#### 2. Delete GitHub Release/Tag (optional)

```bash
# Delete the GitHub release
gh release delete "deepagents-cli==<VERSION>" --yes

# Delete the git tag
git tag -d "deepagents-cli==<VERSION>"
git push origin --delete "deepagents-cli==<VERSION>"
```

#### 3. Fix the Manifest

Edit `.release-please-manifest.json` to the last good version:

```json
{
  "libs/cli": "0.0.15"
}
```

Also update `libs/cli/pyproject.toml` and `_version.py` to match.

### Release Failed: CLI SDK Pin Mismatch

If the release workflow fails at the "Verify CLI pins latest SDK version" step with:

```txt
CLI SDK pin does not match SDK version!
SDK version (libs/deepagents/pyproject.toml): 0.4.2
CLI SDK pin (libs/cli/pyproject.toml): 0.4.1
```

This means the CLI's pinned `deepagents` dependency in `libs/cli/pyproject.toml` doesn't match the current SDK version. This can happen when the SDK is released independently and the CLI's pin isn't updated before the CLI release PR is merged.

**To fix:**

1. **Hotfix the pin on `main`:**

   ```bash
   # Update the pin in libs/cli/pyproject.toml
   # e.g., change deepagents==0.4.1 to deepagents==0.4.2
   cd libs/cli && uv lock
   git add libs/cli/pyproject.toml libs/cli/uv.lock
   git commit -m "hotfix(cli): bump SDK pin to <VERSION>"
   git push origin main
   ```

2. **Manually trigger the release** (the push to `main` won't re-trigger the release because the commit doesn't modify `libs/cli/CHANGELOG.md`):
   - Go to **Actions** > **Package Release**
   - Click **Run workflow**
   - Select `main` branch and `deepagents-cli` package

3. **Publish the draft release** once the workflow completes

4. **Fix the `autorelease: pending` label** if the original automated release left it on the merged release PR. The failed workflow skipped the `mark-release` job, so the label was never swapped. See [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label) for the fix. **If you skip this step, release-please will not create new release PRs.**

### Re-releasing a Version

PyPI does not allow re-uploading the same version. If a release failed partway:

1. If already on PyPI: bump the version and release again
2. If only on test PyPI: the workflow uses `skip-existing: true`, so re-running should work
3. If the GitHub release exists but PyPI publish failed: delete the release/tag and re-run the workflow

### "Untagged, merged release PRs outstanding" Error

If release-please logs show:

```txt
⚠ There are untagged, merged release PRs outstanding - aborting
```

This means a release PR was merged but its merge commit doesn't have the expected tag. This can happen if:

- The release workflow failed and the tag was manually created on a different commit (e.g., a hotfix)
- Someone manually moved or recreated a tag

**To diagnose**, compare the tag's commit with the release PR's merge commit:

```bash
# Find what commit the tag points to
git ls-remote --tags origin | grep "deepagents-cli==<VERSION>"

# Find the release PR's merge commit
gh pr view <PR_NUMBER> --json mergeCommit --jq '.mergeCommit.oid'
```

If these differ, release-please is confused.

**To fix**, move the tag and update the GitHub release:

```bash
# 1. Delete the remote tag
git push origin :refs/tags/deepagents-cli==<VERSION>

# 2. Delete local tag if it exists
git tag -d deepagents-cli==<VERSION> 2>/dev/null || true

# 3. Create tag on the correct commit (the release PR's merge commit)
git tag deepagents-cli==<VERSION> <MERGE_COMMIT_SHA>

# 4. Push the new tag
git push origin deepagents-cli==<VERSION>

# 5. Update the GitHub release's target_commitish to match
#    (moving a tag doesn't update this field automatically)
gh api -X PATCH repos/langchain-ai/deepagents/releases/$(gh api repos/langchain-ai/deepagents/releases --jq '.[] | select(.tag_name == "deepagents-cli==<VERSION>") | .id') \
  -f target_commitish=<MERGE_COMMIT_SHA>
```

After fixing, the next push to main should properly create new release PRs.

> [!NOTE]
> Moving a tag will put the associated GitHub release back into draft state. If the package was already published to PyPI, you can safely re-publish the draft — the publish workflow uses `skip-existing: true`, so it will succeed without re-uploading.

## References

- [release-please documentation](https://github.com/googleapis/release-please)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
