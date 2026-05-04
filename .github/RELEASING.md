# Release Process

This document describes the release process for packages in the Deep Agents monorepo using [release-please](https://github.com/googleapis/release-please).

## Managed Packages

| Package | Path | Component | PyPI |
| ------- | ---- | --------- | ---- |
| `deepagents` (SDK) | `libs/deepagents` | `deepagents` | [`deepagents`](https://pypi.org/project/deepagents/) |
| `deepagents-cli` | `libs/cli` | `deepagents-cli` | [`deepagents-cli`](https://pypi.org/project/deepagents-cli/) |
| `deepagents-acp` | `libs/acp` | `deepagents-acp` | [`deepagents-acp`](https://pypi.org/project/deepagents-acp/) |
| `deepagents-code` | `libs/code` | `deepagents-code` | [`deepagents-code`](https://pypi.org/project/deepagents-code/) |
| `langchain-daytona` | `libs/partners/daytona` | `langchain-daytona` | [`langchain-daytona`](https://pypi.org/project/langchain-daytona/) |
| `langchain-modal` | `libs/partners/modal` | `langchain-modal` | [`langchain-modal`](https://pypi.org/project/langchain-modal/) |
| `langchain-runloop` | `libs/partners/runloop` | `langchain-runloop` | [`langchain-runloop`](https://pypi.org/project/langchain-runloop/) |
| `langchain-quickjs` | `libs/partners/quickjs` | `langchain-quickjs` | [`langchain-quickjs`](https://pypi.org/project/langchain-quickjs/) |
| `langchain-repl` | `libs/repl` | `langchain-repl` | [`langchain-repl`](https://pypi.org/project/langchain-repl/) |

## Overview

Releases are managed via release-please, which:

1. Analyzes commits made to `main`
2. Creates/updates a release PR [(example)](https://github.com/langchain-ai/deepagents/pull/1956) with automated changelog and version bumps
3. When said release PR is merged, creates both a GitHub and PyPI release

## How It Works

### Automatic Release PRs

When commits land on `main`, release-please analyzes them and, **per package**, either:

- Creates a new release PR
- Updates an existing release PR (with additional changes)
- Does nothing — commit types that don't trigger a version bump (e.g., `chore`, `refactor`, `ci`, `docs`, `style`, `test`, `hotfix`) won't create a release PR on their own. However, if a release PR already exists, release-please may still rebase/update it. See [Version Bumping](#version-bumping) for which types trigger bumps.

Each package gets its own **draft** release PR on a branch named `release-please--branches--main--components--<package>`. Mark the PR as ready for review before merging.

### Triggering a Release

To release a package:

1. Merge qualifying conventional commits to `main` (see [Commit Format](#commit-format))
2. Wait for the release-please action to create/update the release PR (can take a minute or two)
3. Review the generated changelog in the PR and make any edits as needed
4. Merge the release PR — this triggers the pre-release checks, PyPI publish, and GitHub release

> [!IMPORTANT]
> **(CLI only)** The CLI pins an exact `deepagents==` version in `libs/cli/pyproject.toml`. Bump this pin as part of any PR that depends on new SDK functionality — don't defer it to release time. The pin should always reflect the minimum SDK version the CLI actually requires. See [Release Failed: CLI SDK Pin Mismatch](#release-failed-cli-sdk-pin-mismatch) for recovery if a mismatch slips through.

### Version Bumping

Version bumps are determined by commit types. All packages are currently pre-1.0, so the effective bumps are shifted down one level:

| Commit Type                    | Standard (≥ 1.0) | Pre-1.0 (current) | Example                                  |
| ------------------------------ | ----------------- | ------------------ | ---------------------------------------- |
| `fix:`                         | Patch (0.0.x)     | Patch (0.0.x)      | `fix(cli): resolve config loading issue` |
| `perf:`                        | Patch (0.0.x)     | Patch (0.0.x)      | `perf(sdk): reduce graph compile time`   |
| `revert:`                      | Patch (0.0.x)     | Patch (0.0.x)      | `revert(cli): undo config change`        |
| `feat:`                        | Minor (0.x.0)     | Patch (0.0.x)      | `feat(cli): add new export command`      |
| `feat!:`                       | Major (x.0.0)     | Minor (0.x.0)      | `feat(cli)!: redesign config format`     |

### Changelog Inclusion

Not every commit type lands in the generated changelog. The set is configured in [`release-please-config.json`](https://github.com/langchain-ai/deepagents/blob/main/release-please-config.json) under `changelog-sections`:

| Commit Type | In Changelog? | Section                  |
| ----------- | ------------- | ------------------------ |
| `feat`      | Yes           | Features                 |
| `fix`       | Yes           | Bug Fixes                |
| `perf`      | Yes           | Performance Improvements |
| `revert`    | Yes           | Reverted Changes         |
| `docs`      | No (hidden)   | —                        |
| `style`     | No (hidden)   | —                        |
| `chore`     | No (hidden)   | —                        |
| `refactor`  | No (hidden)   | —                        |
| `test`      | No (hidden)   | —                        |
| `ci`        | No (hidden)   | —                        |
| `hotfix`    | No (hidden)   | —                        |

Breaking changes are additionally surfaced under a `⚠ BREAKING CHANGES` section at the top of the release notes — see [Breaking Changes](#breaking-changes).

A few rules of thumb for picking a type that respects what *should* end up in user-facing notes:

- A change is **release-note-worthy** if a downstream user could observe it: new API, changed behavior, fixed bug, perceptible perf delta. Use `feat`, `fix`, or `perf`.
- Internal-only work (refactors, test-only changes, CI tweaks, dependency bumps with no behavior change, comment/docstring updates) belongs in a hidden type. These still trigger a release PR rebase if one is open, but never appear in the changelog.
- Don't smuggle user-visible changes into hidden types (e.g., a `chore:` that adds a feature). The change won't appear in release notes and users will be surprised by undocumented behavior.
- You may manually edit the generated `CHANGELOG.md` in the release PR before merging to add, polish, or reorder entries — see [Triggering a Release](#triggering-a-release). Edits made *after* the release PR is merged will be regenerated by release-please on the next run.

## Commit Format

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/) format with types and scopes defined in [`.github/workflows/pr_lint.yml`](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/pr_lint.yml). **Scope is required** — PRs without a scope will fail the title lint check.

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Examples

```bash
fix(cli): resolve type hinting issue
feat(cli): add new chat completion feature
feat(cli)!: redesign configuration format
```

### Breaking Changes

Mark a change as breaking using either form supported by Conventional Commits — both are recognized by release-please:

1. **Bang notation** — append `!` after the scope.

   ```text
   feat(cli)!: redesign configuration format
   ```

2. **`BREAKING CHANGE:` footer** — include a footer (separated from the body by a blank line). The token must be uppercase; lowercase `breaking change:` is ignored. `BREAKING-CHANGE:` (hyphenated) is also accepted as a synonym.

   ```text
   feat(sdk)!: rename `Backend.read` to `Backend.fetch`

   BREAKING CHANGE: `Backend.read` has been removed. Callers must update to
   `Backend.fetch`, which returns a `FetchResult` instead of raw bytes.
   ```

The two forms should be combined, as shown above. The `!` makes the breaking nature obvious in `git log` and PR titles, and the footer body becomes the migration note that release-please surfaces under `⚠ BREAKING CHANGES`.

> [!IMPORTANT]
> All packages are pre-1.0, so a breaking change bumps the **minor** version, not the major (see [Version Bumping](#version-bumping)). The change is still flagged as `⚠ BREAKING CHANGES` at the top of the release notes regardless of the resulting version bump.

PRs containing breaking changes should:

- Use the `!` form in the PR title so the squash commit (whose subject is the PR title) carries the marker. Release-please reads the merged commit message, not the PR body. Put the marker in the title.
- Spell out the migration path in the PR body: what broke, how to update calling code, what the equivalent new API looks like.
- Be reviewed against the [stable public interfaces](https://github.com/langchain-ai/deepagents/blob/main/CLAUDE.md#maintain-stable-public-interfaces) guidance in `CLAUDE.md` — the bar for breaking a public API is high, especially for the SDK.
- Avoid bundling unrelated changes. A breaking commit should isolate the breaking surface so the changelog entry is precise.

## Configuration Files

### `release-please-config.json`

Defines release-please behavior for each package.

### `.release-please-manifest.json`

Tracks the current version of each package. Automatically updated by release-please — **do not edit manually**. Example (versions shown are illustrative; check the actual file for current values):

```json
{
  "libs/cli": "0.0.35",
  "libs/deepagents": "0.5.1",
  "libs/acp": "0.0.5",
  "libs/partners/daytona": "0.0.5",
  "libs/partners/modal": "0.0.3",
  "libs/partners/runloop": "0.0.4",
  "libs/partners/quickjs": "0.0.1",
  "libs/repl": "0.0.1"
}
```

## Release Workflow

### Detection Mechanism

The [release-please workflow (`.github/workflows/release-please.yml`)](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/release-please.yml) detects releases by checking two conditions on the merge commit:

1. The package's `CHANGELOG.md` was modified (e.g., `libs/cli/CHANGELOG.md` for the CLI)
2. The commit message matches the `release(<component>): <version>` pattern

Both must be true. release-please always satisfies both when merging a release PR — a manual `CHANGELOG.md` edit alone will not trigger a release.

### Lockfile Updates

When release-please creates or updates a release PR, the `update-lockfiles` job automatically regenerates `uv.lock` files since release-please updates `pyproject.toml` versions but doesn't regenerate lockfiles.

### Release Pipeline

The [release workflow (`.github/workflows/release.yml`)](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/release.yml) runs when a release PR is merged:

1. **Setup** - Resolves package name to working directory
2. **Build** - Creates distribution package
3. **Release Notes** + **Pre-release Checks** - Run in parallel; release notes extracts changelog and collects contributor shoutouts; pre-release checks run tests against the built package
4. **Test PyPI** - Publishes to test.pypi.org for validation (after pre-release checks pass)
5. **Publish** - Publishes to PyPI (requires Test PyPI to succeed)
6. **Mark Release** - Creates a published GitHub release with the built artifacts; updates PR labels. For the SDK (`libs/deepagents`), we set it as the repository's `latest` (unless it's a pre-release).

### Release PR Labels

Release-please uses labels to track the state of release PRs:

| Label | Meaning |
| ----- | ------- |
| `autorelease: pending` | Release PR has been merged but not yet tagged/released |
| `autorelease: tagged` | Release PR has been successfully tagged and released |

Because `skip-github-release: true` is set in the release-please config (we create releases via our own workflow instead of using the one built into release-please), our `release.yml` workflow must update these labels manually for state management! After successfully creating the GitHub release and tag, the `mark-release` job updates the label from `pending` to `tagged`.

This label transition signals to release-please that the merged PR has been fully processed, allowing it to create new release PRs for subsequent commits to `main`.

## Manual Release

For hotfixes or exceptional cases, you can trigger a release manually. Use the `hotfix` commit type so as to not trigger a further PR update/version bump.

1. Go to **Actions** > `⚠️ Manual Package Release`
2. Click **Run workflow**
3. Select the package to release
4. **Provide `version`**: the version being released (e.g. `0.0.35`). This is required and used for the run name and a sanity-check warning against `pyproject.toml` — it does not control the released version.
5. **Provide `release-sha`**: the SHA of the release-PR merge commit. Look it up with `gh pr view <release-pr-number> --json mergeCommit --jq .mergeCommit.oid`. The workflow validates the commit's subject matches `release(<package>): <version>` and refuses to run otherwise.
6. (Optionally enable `dangerous-nonmain-release` for hotfix branches AKA not `main` — when set, `release-sha` may be left empty and the dispatched HEAD is used instead. Validation is skipped.)

> [!WARNING]
> Manual releases should be rare. Prefer the standard release-please flow for managed packages. Manual dispatch bypasses the changelog detection in `release-please.yml` and skips the lockfile update job. Only use it for recovery scenarios (e.g., the release workflow failed after the release PR was already merged).
>
> **Why `release-sha` is required:** when the auto-triggered release fails (e.g., at `pre-release-checks`) and you re-run via `workflow_dispatch`, `github.sha` resolves to whatever HEAD is at dispatch time — *not* the original release commit. Without `release-sha`, the GitHub release tag would land on an unrelated commit, which breaks release-please's tag-anchored changelog generation on the next run. release-please walks history back from the most recent tag for each package; if the tag sits on the wrong commit, the next run either includes commits that already shipped or treats the package as un-released and (incorrectly) regenerates the full changelog. The `setup` job's `Resolve and validate release SHA` step enforces this and refuses to run when the input is missing or doesn't match a `release(<package>): <version>` commit.

## Alpha / Beta / Pre-release Versions

release-please is SemVer-only internally. Its `prerelease` versioning strategy produces versions like `0.0.35-alpha.1`, which is **not valid [PEP 440](https://peps.python.org/pep-0440/)**. Python/PyPI requires `0.0.35a1` (no hyphen, no dot). The Python file updaters write the SemVer string verbatim and their regexes cannot round-trip PEP 440 versions, so bumping version files on `main` to a PEP 440 pre-release would break subsequent release-please runs.

### How to publish a pre-release

Alpha releases use a **throwaway branch** + [manual release](#manual-release). This keeps `main`, the release-please manifest, and any pending release PR completely untouched.

1. **Create a branch from `main`:**

   ```bash
   git checkout main && git pull
   git checkout -b alpha/<PACKAGE>-<VERSION>
   ```

   Replace `<PACKAGE>` with the PyPI name (e.g., `deepagents-cli`) and `<VERSION>` with the alpha version using hyphens instead of periods (e.g., `0-0-35a1`).

2. **Bump the version** in both files to a [PEP 440 pre-release](https://peps.python.org/pep-0440/#pre-releases) (e.g., `0.0.35a1`):

   - `libs/cli/pyproject.toml` — `version = "0.0.35a1"`
   - `libs/cli/deepagents_cli/_version.py` — `__version__ = "0.0.35a1"`

3. **Commit and push:**

   ```bash
   git add <path>/pyproject.toml <path>/<module>/_version.py
   git commit -m "hotfix(<SCOPE>): alpha release <VERSION>"
   git push -u origin alpha/<PACKAGE>-<VERSION>
   ```

4. **Trigger the release workflow:**

   - Go to **Actions** > `⚠️ Manual Package Release` > **Run workflow**
   - Branch: `alpha/<PACKAGE>-<VERSION>`
   - Package: `<PACKAGE>`
   - Version: `<VERSION>` (e.g. `0.0.35a1`) — required input; surfaces in the run name
   - Enable `dangerous-nonmain-release` ✓
   - (CLI only): leave `dangerous-skip-sdk-pin-check` unchecked (unless the SDK pin is intentionally behind)

5. **Verify the GitHub release** — the workflow automatically detects PEP 440 pre-release versions (`a`, `b`, `rc`, `.dev`) and marks the GitHub release as a **pre-release**. Pre-releases are never set as the repository's "Latest" release. The release body will contain a warning banner and contributor shoutouts (no changelog or git log).

6. **Clean up** — delete the branch after the workflow succeeds:

   ```bash
   git checkout main
   git branch -D alpha/<PACKAGE>-<VERSION>
   git push origin --delete alpha/<PACKAGE>-<VERSION>
   ```

### Promoting a pre-release to GA

After validating the alpha, merge the pending release PR (e.g., `release(deepagents-cli): 0.0.35`) as normal from `main` — release-please handles the GA version, changelog, and tag. No extra steps needed.

If no release PR exists yet (e.g., no releasable commits since the last GA, which is extremely rare), you can force one with a `Release-As` commit footer:

```bash
git commit --allow-empty -m "feat(cli): release 0.0.35" -m "Release-As: 0.0.35"
```

### Multiple pre-release iterations

Increment the PEP 440 pre-release number on each iteration: `0.0.35a1`, `0.0.35a2`, `0.0.35a3`, etc. Each iteration follows the same branch + manual dispatch flow above.

For beta or release candidate stages, use `b` or `rc`: `0.0.35b1`, `0.0.35rc1`.

## Troubleshooting

### Empty commit fan-out

> [!CAUTION]
> Never push an empty commit (`git commit --allow-empty`) to `main`. release-please scopes commits to packages by the file paths they touch. An empty commit has no paths, so it falls back to bumping **every** package — producing a release PR for each managed component, not the one you intended.

This most commonly bites when someone tries to "fix up" a merged PR's changelog entry by pushing an empty commit with a corrected conventional-commit subject (e.g., adding a missing `!` for a breaking change). The corrected subject does land in `git log`, but release-please reads file paths, not commit subjects, when deciding scope.

The `guard-empty-commit` job in [`release-please.yml`](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/release-please.yml) blocks this at CI time: any push to `main` whose head commit changes zero files fails fast with a clear error before the release-please action runs.

**If you need to amend a release note for a commit that already merged**, see [Overriding a Merged Commit's Changelog Entry](#overriding-a-merged-commits-changelog-entry) below. Do not push empty commits to `main`.

**If a fan-out has already happened** (release PRs opened for packages you didn't change), revert the offending commit on `main`. release-please will reconcile the open release PRs on the next push that actually touches package files; PRs for unaffected packages can be closed manually.

### Overriding a Merged Commit's Changelog Entry

Append a `BEGIN_COMMIT_OVERRIDE` block to the **merged PR's body** when release-please needs to use a different message than the actual squash-merge commit. release-please reads merged PR bodies on every run within its lookback window and uses the override in place of the original commit message — no history rewrite, no force-push.

Two situations call for this:

1. **Wrong type/scope inferred** — e.g. a `feat:` that should have been `refactor:` or `chore:`.
2. **Parser cannot read the commit body** — `@conventional-commits/parser` (which release-please uses) is grammar-strict and does not honor markdown code fences. Bodies containing function calls split across lines (`name(` followed by a newline), even inside ` ``` ` blocks, throw a parse error and the commit is silently dropped from the changelog. The pre-merge `release_please_parse_check.yml` check catches this before merge; if a commit slipped through, use the override to recover.

```txt
BEGIN_COMMIT_OVERRIDE
refactor(scope): corrected description
END_COMMIT_OVERRIDE
```

Notes:

- Place the block at the bottom of the PR body, after a horizontal rule.
- To produce multiple changelog entries from one PR, separate corrected messages by a **blank line** with each starting `type(scope):`, or wrap each in `BEGIN_NESTED_COMMIT`/`END_NESTED_COMMIT` markers — release-please's splitter requires one of these forms; a bare newline between messages is parsed as a single commit's body.
- Only effective with **squash merges**. release-please attaches the override to the squash commit by matching it to the PR's `merge_commit_sha`; for plain-merge or rebase-merge strategies the per-branch commits have no PR association and the override is ignored.
- Effect lands when release-please next syncs the open release PR (push to `main` or manual workflow dispatch). Verify the entry moved/disappeared in the corresponding `release(<component>): X.Y.Z` PR.
- Update via `gh pr edit <num> --body-file <file>` to avoid shell-escaping the multi-line body. (`gh api -f body=@<file>` does **not** work — `-f` writes the literal string `@<file>` rather than reading the file.)

### Yanking a Release

If you need to yank (retract) a release:

#### 1. Yank from PyPI

Using the PyPI web interface or a CLI tool.

#### 2. Delete GitHub Release/Tag (optional)

```bash
# Delete the GitHub release (<PACKAGE> = package name from Managed Packages table)
gh release delete "<PACKAGE>==<VERSION>" --yes

# Delete the git tag
git tag -d "<PACKAGE>==<VERSION>"
git push origin --delete "<PACKAGE>==<VERSION>"
```

#### 3. Fix the Manifest

Edit `.release-please-manifest.json` to the last good version for the affected package, and update the corresponding `pyproject.toml` and `_version.py` to match.

### Release PR Stuck with "autorelease: pending" Label

If a release PR shows `autorelease: pending` after the release workflow completed, the label update step may have failed. This can block release-please from creating new release PRs.

**To fix manually:**

```bash
# Find the PR number for the release commit (<PACKAGE> = package name from Managed Packages table)
gh pr list --state merged --search "release(<PACKAGE>)" --limit 5

# Update the label
gh pr edit <PR_NUMBER> --remove-label "autorelease: pending" --add-label "autorelease: tagged"
```

The label update is non-fatal in the workflow (`|| true`), so the release itself succeeded—only the label needs fixing.

### Release Failed: Pre-release Checks

If the `pre-release-checks` job fails (unit tests, integration tests, or import verification), nothing has been published yet — neither Test PyPI nor PyPI have the package. The release PR is already merged, so the normal release-please flow won't re-trigger.

**To fix:**

1. **Inspect the failure** in the workflow run logs. Pre-release checks install the built wheel into a fresh venv (no cache) and run:
   - Package import verification (`python -c "import <pkg>"`)
   - Unit tests (`make test`)
   - Integration tests (`make integration_test`, if the target exists)

2. **Fix the issue on `main`** — open a PR titled `hotfix(<scope>): <description>`. This won't re-trigger the release because the commit doesn't modify the package's `CHANGELOG.md`.

3. **Manually trigger the release:**
   - Go to **Actions** > `⚠️ Manual Package Release`
   - Click **Run workflow**
   - Select `main` branch and the affected package

4. **Verify the `autorelease: pending` label was swapped.** The `mark-release` job swaps this automatically (even on manual dispatch), but check the workflow logs to confirm. If it emitted a warning that the label swap failed, fix it manually — see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label). **If this label isn't swapped, release-please will not create new release PRs.**

> [!TIP]
> Because pre-release checks run against the built wheel (not the editable install), failures here sometimes indicate missing files in the package manifest or undeclared dependencies that happen to be present locally. Check `pyproject.toml` `[tool.setuptools.packages]` and dependency lists if the failure is an import error rather than a test assertion.

### Re-releasing a Version

PyPI does not allow re-uploading the same version. If a release failed partway:

1. If already on PyPI: bump the version and release again
2. If only on test PyPI: the workflow uses `skip-existing: true`, so re-running should work
3. If the GitHub release exists but PyPI publish failed (e.g., from a manual re-run): delete the release/tag and re-run the workflow

> [!NOTE]
> The Test PyPI step uses `skip-existing: true` so that **workflow re-runs** don't fail when the version was already uploaded on a previous attempt. The tradeoff: on re-runs the Test PyPI step is silently skipped rather than re-validated, so it no longer acts as an upload gate.

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
   - Go to **Actions** > `⚠️ Manual Package Release`
   - Click **Run workflow**
   - Select `main` branch and `deepagents-cli` package

3. **Verify the `autorelease: pending` label was swapped.** The `mark-release` job will attempt to find the release PR by label and update it automatically, even on manual dispatch. If the label wasn't swapped (e.g., the job failed), fix it manually — see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label). **If you skip this step, release-please will not create new release PRs.**

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
# Find what commit the tag points to (<PACKAGE> = package name from Managed Packages table)
git ls-remote --tags origin | grep "<PACKAGE>==<VERSION>"

# Find the release PR's merge commit
gh pr view <PR_NUMBER> --json mergeCommit --jq '.mergeCommit.oid'
```

If these differ, release-please is confused.

**To fix**, move the tag and update the GitHub release:

```bash
# 1. Delete the remote tag (<PACKAGE> = package name from Managed Packages table)
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
