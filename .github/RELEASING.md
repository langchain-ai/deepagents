# Release Process

This document describes the release process for packages in the Deep Agents monorepo using [release-please](https://github.com/googleapis/release-please).

## Managed Packages

| Package | Path | Component | PyPI |
| ------- | ---- | --------- | ---- |
| `deepagents` (SDK) | `libs/deepagents` | `deepagents` | [deepagents](https://pypi.org/project/deepagents/) |
| `deepagents-cli` | `libs/cli` | `deepagents-cli` | [deepagents-cli](https://pypi.org/project/deepagents-cli/) |
| `deepagents-acp` | `libs/acp` | `deepagents-acp` | [deepagents-acp](https://pypi.org/project/deepagents-acp/) |
| `langchain-daytona` | `libs/partners/daytona` | `langchain-daytona` | [langchain-daytona](https://pypi.org/project/langchain-daytona/) |

## Overview

Releases are managed via release-please, which:

1. Analyzes conventional commits on the `main` branch
2. Creates/updates a release PR with changelog and version bump
3. When merged, creates a GitHub release and publishes to PyPI

## How It Works

### Automatic Release PRs

When commits land on `main`, release-please analyzes them and either:

- **Creates a new release PR** if releasable changes exist
- **Updates an existing release PR** with additional changes
- **Does nothing** if no releasable commits are found (e.g. commits with type `chore`, `refactor`, etc.)

Release PRs are created on branches named `release-please--branches--main--components--<package>`.

### Triggering a Release

To release a package:

1. Merge conventional commits to `main` (see [Commit Format](#commit-format))
2. Wait for release-please to create/update the release PR
3. Review the generated changelog in the PR
4. **(CLI only)** Verify the SDK pin — check that `deepagents==` in `libs/cli/pyproject.toml` is up to date. See [Release Failed: CLI SDK Pin Mismatch](#release-failed-cli-sdk-pin-mismatch) for recovery if this is missed.
5. Merge the release PR — this triggers the build, pre-release checks, PyPI publish, and GitHub release

> [!IMPORTANT]
> When developing CLI features that depend on new SDK functionality, bump the SDK pin as part of that work — don't defer it to release time. The pin should always reflect the minimum SDK version the CLI actually requires!

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
  "libs/cli": "0.0.35",
  "libs/deepagents": "0.5.0",
  "libs/acp": "0.0.5",
  "libs/partners/daytona": "0.0.5"
}
```

This file is automatically updated by release-please when releases are created.

## Release Workflow

### Detection Mechanism

The release-please workflow (`.github/workflows/release-please.yml`) detects releases by checking if a package's `CHANGELOG.md` was modified in the commit (e.g., `libs/cli/CHANGELOG.md` for the CLI, `libs/deepagents/CHANGELOG.md` for the SDK, `libs/acp/CHANGELOG.md` for ACP, `libs/partners/daytona/CHANGELOG.md` for Daytona). This file is always updated by release-please when merging a release PR.

### Lockfile Updates

When release-please creates or updates a release PR, the `update-lockfiles` job automatically regenerates `uv.lock` files since release-please updates `pyproject.toml` versions but doesn't regenerate lockfiles. An up-to-date lockfile is necessary for the cli since it depends on the SDK, and `libs/evals` depends on the CLI.

### Release Pipeline

The release workflow (`.github/workflows/release.yml`) runs when a release PR is merged:

1. **Build** - Creates distribution package
2. **Collect Contributors** - Gathers PR authors for release notes, including social media handles. Excludes members of `langchain-ai`.
3. **Release Notes** - Extracts changelog or generates from git log
4. **Test PyPI** - Publishes to test.pypi.org for validation
5. **Pre-release Checks** - Runs tests against the built package
6. **Publish** - Publishes to PyPI
7. **Mark Release** - Creates a published GitHub release with the built artifacts

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
3. Select the package to release (for release-please-managed packages, prefer the automated flow; manual dispatch is for exception/recovery/hotfix scenarios)
4. (Optionally enable `dangerous-nonmain-release` for hotfix branches)

> [!WARNING]
> Manual releases should be rare. Prefer the standard release-please flow for managed packages. Manual dispatch bypasses the changelog detection in `release-please.yml` and skips the lockfile update job. Only use it for recovery scenarios (e.g., the release workflow failed after the release PR was already merged).

## Alpha / Pre-release Versions

Alpha releases let you publish a test version to PyPI before the GA release. Users must opt in (`pip install deepagents-cli==0.0.35a1` or `--pre`); plain `pip install deepagents-cli` skips pre-releases.

### Why not use release-please for pre-releases?

release-please is SemVer-only internally. Its `prerelease` versioning strategy produces versions like `0.0.35-alpha.1`, which is **not valid [PEP 440](https://peps.python.org/pep-0440/)**. Python/PyPI requires `0.0.35a1` (no hyphen, no dot). The Python file updaters write the SemVer string verbatim and their regexes cannot round-trip PEP 440 versions, so bumping version files on `main` to a PEP 440 pre-release would break subsequent release-please runs.

### How to publish an alpha

Alpha releases use a **throwaway branch** + manual dispatch. This keeps `main`, the release-please manifest, and any pending release PR completely untouched.

1. **Create a branch from `main`:**

   ```bash
   git checkout main && git pull
   git checkout -b username/alpha/cli-<VERSION>
   ```

   Replace the username in the branch name with your GitHub username.

2. **Bump the version** in both files to a [PEP 440 pre-release](https://peps.python.org/pep-0440/#pre-releases) (e.g., `0.0.35a1`):

   - `libs/cli/pyproject.toml` — `version = "0.0.35a1"`
   - `libs/cli/deepagents_cli/_version.py` — `__version__ = "0.0.35a1"`

3. **Commit and push:**

   ```bash
   git add libs/cli/pyproject.toml libs/cli/deepagents_cli/_version.py
   git commit -m "hotfix(cli): alpha release <VERSION>"
   git push -u origin alpha/cli-<VERSION>
   ```

4. **Trigger the release workflow:**

   - Go to **Actions** > **Package Release** > **Run workflow**
   - Branch: `alpha/cli-<VERSION>`
   - Package: `deepagents-cli`
   - Enable `dangerous-nonmain-release` ✓
   - Leave `dangerous-skip-sdk-pin-check` unchecked (unless the SDK pin is intentionally behind)

5. **Clean up** — delete the branch after the workflow succeeds:

   ```bash
   git checkout main
   git branch -D alpha/cli-<VERSION>
   git push origin --delete alpha/cli-<VERSION>
   ```

### Promoting an alpha to GA

After validating the alpha, merge the pending release PR (e.g., `release(deepagents-cli): 0.0.35`) as normal — release-please handles the GA version, changelog, and tag. No extra steps needed; PyPI treats `0.0.35a1` and `0.0.35` as distinct versions and `pip install deepagents-cli` automatically prefers the GA release.

If no release PR exists yet (e.g., no releasable commits since the last GA), you can force one with a `Release-As` commit footer:

```bash
git commit --allow-empty -m "feat(cli): release 0.0.35" -m "Release-As: 0.0.35"
```

### Multiple alpha iterations

Increment the PEP 440 pre-release number on each iteration: `0.0.35a1`, `0.0.35a2`, `0.0.35a3`, etc. Each iteration follows the same branch + manual dispatch flow above.

For beta or release candidate stages, use `b` or `rc`: `0.0.35b1`, `0.0.35rc1`.

## Troubleshooting

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

- The release commit (e.g., `release(deepagents): 0.5.1` or `release(deepagents-cli): 0.0.35`)
- The lockfile update commit (e.g., `chore: update lockfiles`)

Other commits shown are just the base that the PR branch was rebased onto. This is normal behavior and doesn't indicate unauthorized access.

### Release PR Stuck with "autorelease: pending" Label

If a release PR shows `autorelease: pending` after the release workflow completed, the label update step may have failed. This can block release-please from creating new release PRs.

**To fix manually:**

```bash
# Find the PR number for the release commit (replace <PACKAGE> with deepagents, deepagents-cli, deepagents-acp, or langchain-daytona)
gh pr list --state merged --search "release(<PACKAGE>)" --limit 5

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
# Delete the GitHub release (replace <PACKAGE> with deepagents, deepagents-cli, deepagents-acp, or langchain-daytona)
gh release delete "<PACKAGE>==<VERSION>" --yes

# Delete the git tag
git tag -d "<PACKAGE>==<VERSION>"
git push origin --delete "<PACKAGE>==<VERSION>"
```

#### 3. Fix the Manifest

Edit `.release-please-manifest.json` to the last good version for the affected package, and update the corresponding `pyproject.toml` and `_version.py` to match.

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

3. **Verify the `autorelease: pending` label was swapped.** The `mark-release` job will attempt to find the release PR by label and update it automatically, even on manual dispatch. If the label wasn't swapped (e.g., the job failed), fix it manually — see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label). **If you skip this step, release-please will not create new release PRs.**

### Re-releasing a Version

PyPI does not allow re-uploading the same version. If a release failed partway:

1. If already on PyPI: bump the version and release again
2. If only on test PyPI: the workflow uses `skip-existing: true`, so re-running should work
3. If the GitHub release exists but PyPI publish failed (e.g., from a manual re-run): delete the release/tag and re-run the workflow

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
# Find what commit the tag points to (replace <PACKAGE> with deepagents, deepagents-cli, deepagents-acp, or langchain-daytona)
git ls-remote --tags origin | grep "<PACKAGE>==<VERSION>"

# Find the release PR's merge commit
gh pr view <PR_NUMBER> --json mergeCommit --jq '.mergeCommit.oid'
```

If these differ, release-please is confused.

**To fix**, move the tag and update the GitHub release:

```bash
# 1. Delete the remote tag (replace <PACKAGE> with deepagents, deepagents-cli, deepagents-acp, or langchain-daytona)
git push origin :refs/tags/<PACKAGE>==<VERSION>

# 2. Delete local tag if it exists
git tag -d <PACKAGE>==<VERSION> 2>/dev/null || true

# 3. Create tag on the correct commit (the release PR's merge commit)
git tag <PACKAGE>==<VERSION> <MERGE_COMMIT_SHA>

# 4. Push the new tag
git push origin <PACKAGE>==<VERSION>

# 5. Update the GitHub release's target_commitish to match
#    (moving a tag doesn't update this field automatically)
gh api -X PATCH repos/langchain-ai/deepagents/releases/$(gh api repos/langchain-ai/deepagents/releases --jq '.[] | select(.tag_name == "<PACKAGE>==<VERSION>") | .id') \
  -f target_commitish=<MERGE_COMMIT_SHA>
```

After fixing, the next push to main should properly create new release PRs.

> [!NOTE]
> If the package was already published to PyPI and you need to re-run the workflow, it uses `skip-existing: true` on test PyPI, so it will succeed without re-uploading.

## References

- [release-please documentation](https://github.com/googleapis/release-please)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
