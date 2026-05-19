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
> `deepagents-code` pins an exact `deepagents==` version in `libs/code/pyproject.toml`. Bump this pin as part of any PR that depends on new SDK functionality — don't defer it to release time. The pin should always reflect the minimum SDK version `deepagents-code` actually requires. See [Release Failed: Code SDK Pin Mismatch](#release-failed-code-sdk-pin-mismatch) for recovery if a mismatch slips through.

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

The `!` alone is sufficient to trigger the version bump. The `BREAKING CHANGE:` footer is optional — it only changes what text appears under the `⚠ BREAKING CHANGES` heading in the changelog. Without the footer, that entry is just the commit subject; with it, the entry becomes your footer text (use this to spell out the migration). Combine both whenever the migration path isn't obvious from the subject alone — the `!` makes the breaking nature obvious in `git log` and PR titles, and the footer carries the migration instructions.

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
  "libs/partners/quickjs": "0.0.1"
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
4. **Provide `version`**: the version you want to publish (e.g. `0.0.35`). The workflow checks that the code you selected has the same version.
5. **Provide `release-sha`**: the commit to publish. Usually this is the release-please PR's merge commit. Find it with `gh pr view <release-pr-number> --json mergeCommit --jq .mergeCommit.oid`. If a release failed before anything reached PyPI, you can also use the hotfix commit you merged afterward. See [Hotfix Protocol > Case A](#case-a--release-failed-before-pypi-publish) for that recovery flow.
6. (Optionally enable `dangerous-nonmain-release` for hotfix branches that are not `main`. When enabled, `release-sha` may be left empty and the workflow uses the branch's current commit.)

> [!WARNING]
> Manual releases should be rare. Prefer the normal release-please flow whenever possible. Use this workflow mainly for recovery, such as when the release workflow failed after the release PR was already merged!
>
> **Why `release-sha` matters:** it tells the workflow exactly which commit to build, test, publish, and tag. That keeps the PyPI package and the GitHub tag pointing at the same code. The workflow also checks that the selected commit declares the version you are releasing.

## Hotfix Protocol

Something went wrong with a release. This section tells you what to do.

The right answer depends on a single question: **is the broken version already on PyPI?**

- **No** → [Case A](#case-a--release-failed-before-pypi-publish): the release workflow failed partway through. Nothing public, you have options.
- **Yes** → [Case B](#case-b--bug-found-after-pypi-publish): the bad version is out there. You'll ship a new patch version.

> [!IMPORTANT]
> **The rule we have to maintain:** a version should mean one exact thing. If `mypackage==1.2.3` is on PyPI, then the GitHub tag for `mypackage==1.2.3` must point at the same code.
>
> PyPI does its part automatically: once a version is uploaded, you cannot upload different files for that same version. GitHub tags are easier to move by accident, so we have to be careful. Do not move or recreate a tag for a version that is already on PyPI. If a shipped release needs a fix, ship a new version.
>
> Why it matters: if PyPI and GitHub disagree, different users can install different code for the same version without knowing it. See [Why one version = one artifact](#why-one-version--one-artifact) at the end of this section.

### Case A — Release failed before PyPI publish

The release-please PR was merged, but the release workflow failed before publishing anything. PyPI does not have the package yet, and no GitHub release was created.

Because nothing was published, you still get to decide what eventually goes out as this version. The fix:

1. **Figure out why the release failed.** Look at the workflow run logs.
2. **Open a PR with the fix.** Use a `hotfix(<scope>): <description>` title so it doesn't trigger another release PR update. Merge it to `main`.
   - Important: leave `pyproject.toml`'s version exactly as the release-please PR set it. The hotfix should only fix the problem that broke the release.
3. **Manually re-dispatch the release workflow** ([Manual Release](#manual-release)). Pass `release-sha` = `main`'s HEAD (the hotfix commit). The workflow will build, publish, and tag that commit.
4. **Confirm the label swap.** The `mark-release` job swaps the original release-please PR's `autorelease: pending` label to `autorelease: tagged` — it finds the right PR via a fallback label search, even though `release-sha` points at the hotfix commit, not the release-please commit. Double-check the original release-please PR in GitHub after the workflow succeeds. If the label didn't swap, fix it by hand — see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label).

> [!NOTE]
> The git tag for the version ends up on the hotfix commit, not on the earlier release-please commit. That is okay: the hotfix commit is the code that actually shipped.

### Case B — Bug found *after* PyPI publish

The version is published.

**Do not:**

- Re-run the manual release workflow against the same version (PyPI will reject it anyway).
- Delete and re-create the git tag.
- Open a PR titled something like `hotfix(sdk): bump _version.py` and try to push out a "corrected" version with the same number.

**Do this instead:**

1. Open a normal `fix(<scope>): <description>` PR with the fix. Merge it to `main`.
2. release-please will open (or update) the next release PR — something like `release(<package>): <next-patch>`. Merge it. The standard auto-release flow handles PyPI and the GitHub tag.
3. If the broken release is actively harmful (security hole, won't install, corrupts data), also [yank it from PyPI](#yanking-a-release). Yanking hides a version from default installs but keeps it findable for anyone who pinned it explicitly. You don't need to delete the GitHub tag — leaving it preserves an audit trail.

That's it. The new patch version has its own commit, tag, and wheel. The broken version stays exactly as it was when it shipped.

#### Why one version = one artifact

If the GitHub tag for `<version>` points at different code than the PyPI package for `<version>`, users get different software depending on how they install it:

- `pip install <pkg>==<version>` → gets the PyPI wheel.
- `pip install git+https://.../<repo>@<pkg>==<version>` → gets whatever's at the GitHub tag.
- `git checkout <pkg>==<version>` (vendored copies, distro packagers, security tools pinning by SHA) → also gets the GitHub tag's code.

When these disagree, the same version can behave differently for different users. The workflow pinning and the "never re-release a version" rule are there to prevent that.

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
   - For `deepagents-code`: leave `dangerous-skip-sdk-pin-check` unchecked (unless the SDK pin is intentionally behind)

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

### Reverting a Merged-but-Unreleased PR

When a PR has merged to `main` but its `release(<component>): X.Y.Z` PR has **not** yet shipped, the bad commit is sitting in the open release PR's changelog. Pick a path based on whether the change should appear in the eventual release notes. (For commits that already shipped, see [Yanking a Release](#yanking-a-release) instead — and ship a follow-up `revert:` patch via the standard flow.)

#### Path A — Hide and Revert (Quiet)

Use when the original commit is a mistake the changelog should not record (broken feature, accidental merge, scope/type mistake that escaped lint). Net effect: the open release PR rebases without the entry, and the version may be recomputed if no other releasable commits remain.

1. **Override the original PR's commit message to a hidden type (`chore`).** Append at the bottom of the merged PR's body, after a horizontal rule:

   ```txt
   ---

   BEGIN_COMMIT_OVERRIDE
   chore(<scope>): <short description of the original change>
   END_COMMIT_OVERRIDE
   ```

   The `<short description>` should describe the *original change*, not the override or revert — release-please uses this verbatim as the (now-hidden) commit message. Apply with `gh pr edit <num> --body-file body.md` or via the web interface — see the caveats in [Overriding a Merged Commit's Changelog Entry](#overriding-a-merged-commits-changelog-entry).

2. **Open a revert PR off `main`** titled `chore(<scope>): revert <original title>`. The `chore` type keeps the revert itself out of the changelog as well.

   ```bash
   git checkout main && git pull
   git revert <merge_sha>
   ```

   (This repo squash-merges, so `<merge_sha>` is a single-parent commit — no `-m` flag needed.)

3. **Wait for release-please to rebase the open release PR** on the next push to `main` (or dispatch the workflow manually). Verify the entry has disappeared from the corresponding `release(<component>): X.Y.Z` PR's rendered body before merging it.

#### Path B — `revert:` with Audit Trail

Use when something measurable has already happened off `main` (downstream consumers tracking the SHA, internal pre-release builds, public discussion of the change). The release PR will list the same change *twice* — once under its original section (`Features`, `Bug Fixes`, etc.) and once under `Reverted Changes` — because `revert` is configured as a visible section in `release-please-config.json`. Trade-off: honest history at the cost of a duplicated entry in a version that never shipped externally.

1. **Open a revert PR off `main`** titled `revert(<scope>): "<original title>"` (Conventional Commits convention quotes the original subject). Body should reference the merge SHA being reverted.

   ```bash
   git checkout main && git pull
   git revert <merge_sha>
   ```

   As in Path A, no `-m` flag — squash-merged commits are single-parent.

2. **Merge the revert PR.**

3. **Wait for release-please to rebase the open release PR** on the next push to `main` (or dispatch the workflow manually). Verify the corresponding `release(<component>): X.Y.Z` PR's rendered body now contains both the original entry and a `Reverted Changes` entry before merging it.

#### Don'ts

- **No force-push to `main`** — branch protection blocks it and would drop unrelated commits anyway.
- **No empty commits** to "fix up" the changelog — `guard-empty-commit` fails them, and even if it didn't, the empty fan-out would open release PRs for every package (see [Empty commit fan-out](#empty-commit-fan-out)).
- **Don't edit the release PR body to remove the entry directly** — release-please regenerates the body from merged-PR commits on every sync, so manual edits persist only until the next push to `main`. The override on the original PR is the durable mechanism.
- **Don't edit `.release-please-manifest.json`** — manifest edits only matter for [Yanking a Release](#yanking-a-release) (versions that already shipped).

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

The `pre-release-checks` job runs after the package is built but before anything is published. If it fails, nothing reached PyPI or GitHub Releases, but the release PR is already merged. release-please will not retry on its own. This is **Case A** in the [Hotfix Protocol](#case-a--release-failed-before-pypi-publish).

**Steps:**

1. **Look at the workflow logs** to see why it failed. Pre-release checks install the built package in a clean environment and run:
   - `python -c "import <pkg>"` — does the package even import?
   - `make test` — do the unit tests pass against the built wheel?
   - `make integration_test` (if defined) — do the integration tests pass?

2. **Open a `hotfix(<scope>): <description>` PR with the fix.** Merge it to `main` on top of the release-please commit. **Leave `pyproject.toml`'s version exactly as the release-please PR set it.**

3. **Manually re-dispatch the release** ([Manual Release](#manual-release)). Pass:
   - `version` = the same version you were originally trying to release.
   - `release-sha` = `main` HEAD (the hotfix commit you just merged).

   The workflow will build, test, publish, and tag that commit.

4. **Confirm the label swap.** The `mark-release` job should change the original release-please PR from `autorelease: pending` to `autorelease: tagged`. If the swap didn't happen, fix it manually — see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label).

> [!TIP]
> Pre-release checks run against the *built wheel*, not against your editable working copy. That means failures here often point at missing files in the wheel or undeclared dependencies — things that worked locally because they were sitting in your venv but didn't get packaged. If the failure is an import error rather than a test assertion, check the `packages` config in `pyproject.toml` and the declared dependencies first.

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

### Release Failed: Code SDK Pin Mismatch

If the release workflow fails at the "Verify package pins latest SDK version" step with:

```txt
deepagents-code SDK pin does not match SDK version!
SDK version (libs/deepagents/pyproject.toml): 0.4.2
deepagents-code SDK pin (libs/code/pyproject.toml): 0.4.1
```

This means `deepagents-code`'s pinned `deepagents` dependency in `libs/code/pyproject.toml` doesn't match the current SDK version. This can happen when the SDK is released independently and the pin isn't updated before the `deepagents-code` release PR is merged.

**To fix:**

1. **Hotfix the pin on `main`:**

   ```bash
   # Update the pin in libs/code/pyproject.toml
   # e.g., change deepagents==0.4.1 to deepagents==0.4.2
   cd libs/code && uv lock
   git add libs/code/pyproject.toml libs/code/uv.lock
   git commit -m "hotfix(code): bump SDK pin to <VERSION>"
   git push origin main
   ```

2. **Manually trigger the release** (the push to `main` won't re-trigger the release because the commit doesn't modify `libs/code/CHANGELOG.md`):
   - Go to **Actions** > `⚠️ Manual Package Release`
   - Click **Run workflow**
   - Select `main` branch and `deepagents-code` package

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
