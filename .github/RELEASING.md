# CLI Release Process

This document describes the release process for the CLI package (`libs/cli`) in the Deep Agents monorepo using [release-please](https://github.com/googleapis/release-please).

## Overview

CLI releases are managed via release-please, which:

1. Analyzes conventional commits on the `master` branch
2. Creates/updates a release PR with changelog and version bump
3. When merged, creates a draft GitHub release for review
4. Publishing the draft triggers PyPI publication

## How It Works

### Automatic Release PRs

When commits land on `master`, release-please analyzes them and either:

- **Creates a new release PR** if releasable changes exist
- **Updates an existing release PR** with additional changes
- **Does nothing** if no releasable commits are found (e.g. commits with type `chore`, `refactor`, etc.)

Release PRs are created on branches named `release-please--branches--master--components--<package>`.

### Triggering a Release

To release the CLI:

1. Merge conventional commits to `master` (see [Commit Format](#commit-format))
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

## Manual Release

For hotfixes or exceptional cases, you can trigger a release manually. Use the `hotfix` commit type so as to not trigger a further PR update/version bump.

1. Go to **Actions** > **Package Release**
2. Click **Run workflow**
3. Select the CLI
4. (Optionally enable `dangerous-nonmaster-release` for hotfix branches)

> [!WARNING]
> Manual releases should be rare. Prefer the standard release-please flow.

## Troubleshooting

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

After fixing, the next push to master should properly create new release PRs.

> [!NOTE]
> Moving a tag will put the associated GitHub release back into draft state. If the package was already published to PyPI, you can safely re-publish the draft — the publish workflow uses `skip-existing: true`, so it will succeed without re-uploading.

## References

- [release-please documentation](https://github.com/googleapis/release-please)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
