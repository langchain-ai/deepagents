# Release Process

This document describes the release process for packages in the Deep Agents monorepo using [release-please](https://github.com/googleapis/release-please).

## Managed Packages

| Package | Path | Component | PyPI |
| ------- | ---- | --------- | ---- |
| `deepagents` (SDK) | `libs/deepagents` | `deepagents` | [`deepagents`](https://pypi.org/project/deepagents/) |
| `deepagents-cli` | `libs/cli` | `deepagents-cli` | [`deepagents-cli`](https://pypi.org/project/deepagents-cli/) |
| `deepagents-acp` | `libs/acp` | `deepagents-acp` | [`deepagents-acp`](https://pypi.org/project/deepagents-acp/) |
| `deepagents-code` | `libs/code` | `deepagents-code` | [`deepagents-code`](https://pypi.org/project/deepagents-code/) |
| `deepagents-talon` | `libs/talon` | `deepagents-talon` | [`deepagents-talon`](https://pypi.org/project/deepagents-talon/) |
| `langchain-daytona` | `libs/partners/daytona` | `langchain-daytona` | [`langchain-daytona`](https://pypi.org/project/langchain-daytona/) |
| `langchain-modal` | `libs/partners/modal` | `langchain-modal` | [`langchain-modal`](https://pypi.org/project/langchain-modal/) |
| `langchain-runloop` | `libs/partners/runloop` | `langchain-runloop` | [`langchain-runloop`](https://pypi.org/project/langchain-runloop/) |
| `langchain-vercel-sandbox` | `libs/partners/vercel` | `langchain-vercel-sandbox` | [`langchain-vercel-sandbox`](https://pypi.org/project/langchain-vercel-sandbox/) |
| `langchain-quickjs` | `libs/partners/quickjs` | `langchain-quickjs` | [`langchain-quickjs`](https://pypi.org/project/langchain-quickjs/) |

## Overview

Releases are managed via release-please, which:

1. Analyzes commits made to `main`
2. Creates/updates a release PR [(example)](https://github.com/langchain-ai/deepagents/pull/1956) with automated changelog and version bumps
3. When said release PR is merged, triggers the release workflow for that merge commit, which creates both a GitHub and PyPI release

## How It Works

### Automatic Release PRs

When commits land on `main`, release-please analyzes them and, **per package**, either:

- Creates a new release PR
- Updates an existing release PR (with additional changes)
- Does nothing — commit types that don't trigger a version bump (e.g., `chore`, `refactor`, `ci`, `docs`, `style`, `test`, `hotfix`) won't create a release PR on their own. However, if a release PR already exists, release-please may still rebase/update it. See [Version Bumping](#version-bumping) for which types trigger bumps.

Each package gets its own **draft** release PR on a branch named `release-please--branches--main--components--<package>`. Mark the PR as ready for review before merging.

### Triggering a Release

To release a package:

1. Merge one or more [releasable conventional commits](#releasable-commit-types-and-version-bumping) to `main`
2. Wait for the release-please action to create/update the release PR (can take a minute or two)
3. Review the generated changelog in the PR and make any edits as needed
4. Merge the release PR — this triggers the pre-release checks, PyPI publish, and GitHub release

> [!IMPORTANT]
> `deepagents-code` pins an exact `deepagents==` version in `libs/code/pyproject.toml`. Bump this pin as part of any PR that depends on new SDK functionality — don't defer it to release time. The pin should always reflect the minimum SDK version `deepagents-code` actually requires. If you intentionally need to ship a release PR with an older SDK pin, add the `release: skip sdk pin check` label before merging. See [Release Failed: Code SDK Pin Is Older Than SDK](#release-failed-code-sdk-pin-is-older-than-sdk) for recovery if a stale pin slips through.

### Releasable Commit Types and Version Bumping

A commit creates or updates a release PR for a package only when release-please assigns it to that package and its type bumps the version. The releasable types are listed below. All packages are currently pre-1.0, so the effective bumps are shifted down one level:

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

Tracks the current version of each package. Automatically updated by release-please — **do not edit manually** except when adding a new release-please-managed package. Example (versions shown are illustrative; check the actual file for current values):

```json
{
  "libs/cli": "0.0.35",
  "libs/deepagents": "0.5.1",
  "libs/acp": "0.0.5",
  "libs/talon": "0.0.1",
  "libs/partners/daytona": "0.0.5",
  "libs/partners/modal": "0.0.3",
  "libs/partners/runloop": "0.0.4",
  "libs/partners/vercel": "0.0.1",
  "libs/partners/quickjs": "0.0.1"
}
```

### Adding a release-please-managed package

When adding a new managed package, add it to both `release-please-config.json` and `.release-please-manifest.json`. The manifest entry is the **latest released version baseline**, not the package's current source version.

User story: you are adding a first-party integration package, such as a new sandbox provider under `libs/partners/<provider>`, and the PR title is release-worthy (`feat(<scope>): ...`). The package source starts at `0.0.1`, and you want the first release PR for that package to publish `0.0.1`, not immediately bump to `0.0.2` before the package has ever shipped. In this case, set the new `.release-please-manifest.json` entry to `0.0.0` while keeping the package's own `pyproject.toml` and `_version.py` at `0.0.1`.

Do **not** add a new managed package to the manifest at `0.0.1` unless `0.0.1` has already been released outside release-please. If the new manifest baseline is `0.0.1`, release-please treats that as already released and opens the initial release PR for `0.0.2`. The `Release-please initial baseline check` workflow blocks PRs that add a new `0.0.1` managed package baseline.

## Release Workflow

### Detection Mechanism

The [release-please workflow (`.github/workflows/release-please.yml`)](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/release-please.yml) detects merged release PRs by checking two conditions on the merge commit:

1. The package's `CHANGELOG.md` was modified (e.g., `libs/cli/CHANGELOG.md` for the CLI)
2. The commit message matches the `release(<component>): <version>` pattern

Both must be true. release-please always satisfies both when merging a release PR — a manual `CHANGELOG.md` edit alone will not trigger a release.

> [!NOTE]
> Merged release PRs dispatch the publish workflow directly and skip the release-please PR-maintenance step for that push. This intentionally keeps publishing from being blocked behind normal release-please updates while another package is publishing. If any next release PR needs to be refreshed after the merge, the next normal push to `main` will handle it.

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

1. Go to **Actions** > `🚀 Package Release`
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

- **No** -> [Case A](#case-a--release-failed-before-pypi-publish): the release workflow failed partway through. Nothing public, you have options.
- **Yes** -> [Case B](#case-b--bug-found-after-pypi-publish): the bad version is out there. You'll ship a new patch version.

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
3. **Manually re-dispatch the release workflow** ([Manual Release](#manual-release)). Pass `release-sha` = the SHA of your hotfix commit — the one that fixed the release *and* still declares the target version. Right after you merge it, that's the tip of `main`, but pin the explicit SHA rather than relying on `HEAD` (e.g. `gh pr view <hotfix-pr-number> --json mergeCommit --jq .mergeCommit.oid`), since `main` can advance if another PR lands first. The workflow checks out, builds, publishes, and tags that exact commit.
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

- `pip install <pkg>==<version>` -> gets the PyPI wheel.
- `pip install git+https://.../<repo>@<pkg>==<version>` -> gets whatever's at the GitHub tag.
- `git checkout <pkg>==<version>` (vendored copies, distro packagers, security tools pinning by SHA) -> also gets the GitHub tag's code.

When these disagree, the same version can behave differently for different users. The workflow pinning and the "never re-release a version" rule are there to prevent that.

## Alpha / Beta / Pre-release Versions

release-please can maintain normal Python versions for us, but it cannot safely maintain Python pre-release versions on long-lived branches.

The problem is the difference between SemVer and Python's [PEP 440](https://peps.python.org/pep-0440/) version syntax. release-please's built-in prerelease strategy produces SemVer versions like `0.0.35-alpha.1`, but PyPI requires the PEP 440 form `0.0.35a1`. If we manually commit that PEP 440 version to `main` or a long-lived `vX.Y` branch, a later release-please PR may not be able to move every version file back to the final GA version. In particular, the `_version.py` updater only matches stable-looking `X.Y.Z` / `X.Y.Z-suffix` values, not values like `0.0.35a1` or `0.0.35rc1`. The next GA release PR could update `pyproject.toml` to `0.0.35` while leaving `_version.py` stuck at `0.0.35a1`.

For that reason, never commit PEP 440 pre-release version bumps to `main` or a long-lived `vX.Y` version branch. Keep those branches in the stable-version shape release-please can maintain, and use the throwaway branch flow below for alpha, beta, RC, or `.dev` artifacts.

### How to publish a pre-release

Alpha releases use a **throwaway branch** + [manual release](#manual-release). This keeps `main`, the release-please manifest, and any pending release PR completely untouched.

1. **Create a branch from the version line you are releasing:**

   ```bash
   git checkout <BASE_BRANCH> && git pull
   git checkout -b alpha/<PACKAGE>-<VERSION>
   ```

   Replace `<BASE_BRANCH>` with `main` for normal pre-releases, or the relevant `vX.Y` branch when staging or maintaining a separate version line. If no `vX.Y` branch exists, use `main` and confirm the next alpha number from existing `<PACKAGE>==*aN` tags/releases. Replace `<PACKAGE>` with the PyPI name (e.g., `deepagents-cli`) and `<VERSION>` with the alpha version using hyphens instead of periods (e.g., `0-0-35a1`).

   For example, when staging `deepagents` `0.7.0` on `v0.7` while `main` still tracks `0.6.x` and you need an installable alpha for validation, branch from `v0.7`, not `main`, so the artifact contains the staged `0.7` work — the PEP 440 version `0.7.0a1` becomes `alpha/deepagents-0-7-0a1` (hyphens instead of periods) as the branch name.

2. **Bump the version** in both files to a [PEP 440 pre-release](https://peps.python.org/pep-0440/#pre-releases) (e.g., `0.0.35a1`):

   - `libs/cli/pyproject.toml` — `version = "0.0.35a1"`
   - `libs/cli/deepagents_cli/_version.py` — `__version__ = "0.0.35a1"`

3. **Regenerate package lockfiles** if the package has a `uv.lock`. The pre-commit lock check compares the local package version in the lockfile, so alpha version bumps need the same lockfile refresh as release-please PRs.

   ```bash
   uv lock --directory <path> --python <PYTHON_VERSION>
   ```

   Use the package's required Python version for `<PYTHON_VERSION>`: `3.14` for `acp`, `3.12` for every other package. This mapping is the same one the lock check enforces — see `python_version` in `libs/Makefile` and `_python_version` in `.github/scripts/check_lockfiles_pre_commit.py`. Locking with the wrong version will fail the pre-commit `lock-check`.

   For example, for the SDK:

   ```bash
   uv lock --directory libs/deepagents --python 3.12
   ```

4. **Commit and push:**

   ```bash
   git add <path>/pyproject.toml <path>/<module>/_version.py <path>/uv.lock
   git commit -m "hotfix(<SCOPE>): alpha release <VERSION>"
   git push -u origin alpha/<PACKAGE>-<VERSION>
   ```

   Omit `<path>/uv.lock` only when the package does not have one.

5. **Trigger the release workflow:**

   - Go to **Actions** > `🚀 Package Release` > **Run workflow**
   - Branch: `alpha/<PACKAGE>-<VERSION>`
   - Package: `<PACKAGE>`
   - Version: `<VERSION>` (e.g. `0.0.35a1`) — required input; surfaces in the run name
   - Enable `dangerous-nonmain-release` ✓
   - For `deepagents-code`: leave `dangerous-skip-sdk-pin-check` unchecked (unless the SDK pin is intentionally older than the workspace SDK)

   Or dispatch it with the GitHub CLI:

   ```bash
   gh workflow run release.yml \
     --repo langchain-ai/deepagents \
     --ref alpha/<PACKAGE>-<VERSION> \
     -f package=<PACKAGE> \
     -f version=<VERSION> \
     -f dangerous-nonmain-release=true
   ```

6. **Verify the GitHub release** — the workflow automatically detects PEP 440 pre-release versions (`a`, `b`, `rc`, `.dev`) and marks the GitHub release as a **pre-release**. Pre-releases are never set as the repository's "Latest" release. The release body will contain a warning banner, contributor shoutouts (no changelog or git log), and — because the branch is not `main` — a "Released from" line linking the originating branch and the release commit.

7. **Clean up** — delete the branch after the workflow succeeds:

   ```bash
   git checkout main
   git branch -D alpha/<PACKAGE>-<VERSION>
   git push origin --delete alpha/<PACKAGE>-<VERSION>
   ```

### Promoting a pre-release to GA

After validating the alpha, merge the pending release PR (e.g., `release(deepagents-code): 0.0.35`) as normal from `main` — release-please handles the GA version, changelog, and tag. No extra steps needed.

If no release PR exists yet (e.g., no releasable commits since the last GA, which is rare), you can force one with a package-scoped `Release-As` override. Do **not** use an empty commit on `main`: release-please assigns commits to packages by the file paths they change, not by the commit scope string. A commit titled `chore(code): ...` is not enough on its own! It must also touch a file under `libs/code` so release-please knows the override belongs to `deepagents-code` (instead of another managed package).

For example, after making a real edit under `libs/code`:

```bash
git add libs/code/<changed-file>
git commit -m "chore(code): release 0.0.35" -m "Release-As: 0.0.35"
```

If there is no meaningful package-file edit to make, use the config-file form instead: temporarily add `"release-as": "0.0.35"` to the `libs/code` package entry in `release-please-config.json`:

```diff
 "libs/code": {
   "release-type": "python",
   "package-name": "deepagents-code",
   "component": "deepagents-code",
+  "release-as": "0.0.35",
   "bump-minor-pre-major": true,
   "bump-patch-for-minor-pre-major": true,
   "extra-files": [
     "pyproject.toml",
     "deepagents_code/_version.py"
   ],
   "changelog-path": "CHANGELOG.md"
 }
```

> [!IMPORTANT]
> This is a temporary override, not permanent configuration. Let release-please open the release PR, then remove the `release-as` line in a follow-up `hotfix(code): remove release-as override` PR (so the next release-please run does not keep forcing `0.0.35`).

### Multiple pre-release iterations

Increment the PEP 440 pre-release number on each iteration: `0.0.35a1`, `0.0.35a2`, `0.0.35a3`, etc. Each iteration follows the same branch + manual dispatch flow above.

For beta or release candidate stages, use `b` or `rc`: `0.0.35b1`, `0.0.35rc1`.

## Developing a new version line

Most version progression needs **no dedicated branches**. Keep developing on `main` and let release-please cut the next version — including minor bumps, since a `feat!:` / `BREAKING CHANGE:` bumps the minor pre-1.0 (see [Version Bumping](#version-bumping)).

Reach for a dedicated branch only when you need to (often temporarily) *decouple* a version line from `main`:

| Scenario | Branch | release-please runs there? | Releases via |
| -------- | ------ | -------------------------- | ------------ |
| Normal progression (incl. minor bumps) | none — use `main` | yes (on `main`) | automatic (on release PR merge) |
| **Staging** the next line before cutover (e.g. work toward `0.7` while `main` stays `0.6.x`) | `vX.Y` integration branch | no | optional pre-release builds ([Alpha/Beta](#how-to-publish-a-pre-release)) |
| **Maintenance** of an old line after cutover (e.g. patch `0.6.x` after `main` moves to `0.7`) | `vX.Y` maintenance branch | no (not wired) | [Manual Release](#manual-release) + `dangerous-nonmain-release` |

> [!IMPORTANT]
> **Name the branch with a `v` prefix** — `v0.7`, `v0.6`, etc. A branch named `0.7` gets **no branch protection**.

Both `main` and `v[0-9].*` require a CI-passing PR (no direct pushes). The only difference is that `v[0-9].*` allows merge commits in order to facilitate syncing `main` -> `vX.Y` ([staging](#staging-branch-main-stays-on-the-current-line) step 2) and the **cutover** (admin bypass — see below). A version-line DRI with branch-rule bypass privileges may occasionally force-push a staging branch after rebasing it onto `main`, **but only when they intentionally own the history rewrite** and have verified the final `main..vX.Y` range contains only the branch's intended commits.

### TL;DR — staging the next line of work (e.g. `v0.7` while `main` stays `v0.6.x`)

1. **Branch:** create `v0.7` from `main`.
2. **Build `0.7`:** land net-new work via **squash PRs into `v0.7`** (same flow as `main`).
3. **Keep `v0.7` current with `main`:** default to a PR with **base `v0.7`, head `main`** and merge it with **"Create a merge commit"** (not squash!). Use a PR title like `chore(repo): sync main into v0.7`; do not use `release` as the scope because PR title lint reserves `release` for the type and disallows it as a scope. CI runs on the merged result; `main`'s commits arrive as shared history, so the cutover stays clean. If the staging branch is being actively maintained by a DRI with branch-rule bypass privileges who can safely rewrite it, rebasing `v0.7` onto `main` and force-pushing with lease is also acceptable; verify `git rev-list --left-right --count main...v0.7` reports `0 N` and that `git log --oneline --no-merges main..v0.7` lists only the intended version-line commits. Cherry-pick instead only if `v0.7` deliberately diverges from `main` (e.g. `v0.7` deleted or rewrote a module that `main` is still bug-fixing, so a full merge would keep dragging the old code back and re-conflict on every sync — cherry-pick just the fixes you still want).
4. **Cutover:** an admin merges `v0.7` onto `main` with `git merge --no-ff` under admin bypass. See [Cutover](#cutover-main-adopts-the-new-line).

### Staging branch (`main` stays on the current line)

1. Create `vX.Y` from `main`. Do feature work via **squash PRs into `vX.Y`** — same flow as `main`, so every change is CI-gated and reviewed. Each PR becomes one clean conventional commit on the branch.
2. **Pulling in `main` fixes:** keep `vX.Y` current with one of these two workflows:

   - **Default for shared branches: merge `main` into `vX.Y`.** Open a **merge PR from `main` -> `vX.Y`** and land it as a **merge commit (not squash)**. Do this periodically. It buys three things:
     - **Still CI-gated.** The PR runs CI on the *merged* result, so you test `vX.Y` against the latest `main` before it lands.
     - **Conflicts stay small.** They surface in each sync PR instead of piling up for the final cutover.
     - **Clean cutover.** The merge brings `main`'s commits in as **shared history** (same SHAs, not copies), so release-please does not see copied `main` commits as new version-line work.

     Use a merge commit **only** for these sync PRs.

   - **Controlled exception: rebase `vX.Y` onto `main`.** If the version-line DRI has branch-rule bypass privileges and intentionally owns rewriting the staging branch, they may rebase and force-push with lease instead of creating sync merge commits. This keeps the GitHub compare view at `0 behind, N ahead` and makes the final cutover easy to audit. If no authorized maintainer can bypass the non-fast-forward rule, use the merge-PR workflow instead. Before pushing, verify:

     ```bash
     git rev-list --left-right --count main...vX.Y  # expect: 0 N
     git log --oneline --no-merges main..vX.Y      # only intended version-line commits
     git log --oneline --merges main..vX.Y         # empty, unless intentional
     ```

     After that verification, merging `vX.Y` into `main` makes release-please parse only the commits in `main..vX.Y`. Do **not** use this workflow if other contributors are basing active work on the staging branch unless they know the branch will be rewritten.

   > [!TIP]
   > If `vX.Y` deliberately *diverges* from `main` (it removed or rewrote code that `main` keeps patching), a full sync re-surfaces the same conflict every time. In that case **cherry-pick only the fixes you want** instead. Avoid cherry-picking commits that already exist on `main`: cherry-picks get new SHAs, so release-please can treat them as new commits at cutover.
3. **Need an installable build?** Cut a pre-release (`0.7.0a1`, …) with the throwaway-branch flow in [How to publish a pre-release](#how-to-publish-a-pre-release). release-please is never involved and `main` is untouched.

### Cutover (`main` adopts the new line)

When the new line is ready to become `main`:

1. Confirm `vX.Y` `HEAD` is green.
2. **Merge `vX.Y` onto `main` preserving individual commits.** The cutover can't be a normal PR (a `vX.Y` -> `main` PR would squash the whole version branch into one commit and gut the changelog!), so an **admin** brings it over with a merge commit under bypass. If you've kept `vX.Y` synced (staging step 2), there's little left to reconcile here:

   ```bash
   git checkout main && git pull
   git merge --no-ff vX.Y
   git push origin main
   ```

   release-please ignores the merge commit itself and itemizes each per-PR squash commit from `vX.Y` into the changelog(s).
3. After the merge, release-please reads the incoming commits and computes the next version. Compare it to the version you intend to cut:

   - **If they match, you're done.** The commits already justify the target (e.g. a `feat!:` / `BREAKING CHANGE:` in the line bumps the minor if pre-1.0).
   - **If release-please picks a lower version, force it.** The commits resolve to less than your target (e.g. a line of only `feat:`/`fix:` stays as a `PATCH` bump pre-1.0). Override release-please's choice in one of two ways:

     - **`Release-As` footer** — put the footer on a commit that touches the package's files. release-please reads the footer and pins that version for the next release PR:

       ```bash
       git commit -m "feat(sdk): release X.Y.Z" -m "Release-As: X.Y.Z"
       ```

     - **`release-as` config key** — set `"release-as": "X.Y.Z"` on the package's entry in [`release-please-config.json`](https://github.com/langchain-ai/deepagents/blob/main/release-please-config.json). Same effect, but it lives in config rather than a commit message. It's a standing override, so **delete the key once the release PR is open!** — otherwise every later run keeps pinning that same version.

   > [!CAUTION]
   > Don't put the `Release-As` footer on an `--allow-empty` commit on `main` — an empty commit touches no package paths and triggers the [empty-commit fan-out](#empty-commit-fan-out) guard, opening a release PR for *every* package. That's why the footer goes on a commit that actually edits the package's files; the `release-as` config key sidesteps this since editing the config file is itself a non-empty change.

### Maintenance branch (patching the old line after cutover)

After `main` adopts the new line, cut a `vX.Y` branch from the **last release commit** of the old line (e.g. branch `v0.6` from the `release(deepagents): 0.6.N` merge commit). Branching from the release commit means the latest `0.6` tag is its ancestor, so version math stays on the `0.6.x` line.

- **Backport** fixes by landing them on `main` first, then cherry-picking onto `vX.Y` with the conventional-commit message intact.
- **Release** from the branch with [Manual Release](#manual-release) + `dangerous-nonmain-release` (its stated purpose is backports): bump the version files on the branch, then dispatch `🚀 Package Release` with that branch, package, version, and `dangerous-nonmain-release` ✓. It is usually rare to need to release old versions so these steps remain manual.

## Troubleshooting

### Why don't I see a release PR?

Check these common causes first:

- **The [release-please workflow](https://github.com/langchain-ai/deepagents/actions/workflows/release-please.yml) has not run yet.** Wait a minute or two after the PR merges to `main`, then check the `release-please` workflow run.
- **The merged commit uses a hidden type.** `chore`, `refactor`, `ci`, `docs`, `style`, `test`, and `hotfix` do not create release PRs on their own. See [Releasable Commit Types and Version Bumping](#releasable-commit-types-and-version-bumping).
- **The commit was not assigned to the package you expected.** release-please scopes commits by **changed file paths**, not just the Conventional Commit scope. For example, a `feat(code): ...` commit must touch files under `libs/code` to create or update the `deepagents-code` release PR.
- **An [existing draft release PR](https://github.com/langchain-ai/deepagents/issues?q=is%3Apr+is%3Aopen+author%3Aapp%2Fgithub-actions) was updated instead.** Each package has at most one active release PR, on a branch named `release-please--branches--main--components--<package>`.
- **A previous merged release PR [is still pending](https://github.com/langchain-ai/deepagents/issues?q=state%3Aopen%20label%3A%22autorelease%3A%20pending%22).** If a release PR still has `autorelease: pending` after the release workflow finished, see [Release PR Stuck with "autorelease: pending" Label](#release-pr-stuck-with-autorelease-pending-label).

### Empty commit fan-out

> [!CAUTION]
> Never push an empty commit (`git commit --allow-empty`) to `main`. release-please scopes commits to packages by the file paths they touch. An empty commit has no paths, so it falls back to bumping **every** package — producing a release PR for each managed component, not the one you intended.

This most commonly bites when someone tries to "fix up" a merged PR's changelog entry by pushing an empty commit with a corrected conventional-commit subject (e.g., adding a missing `!` for a breaking change). The corrected subject does land in `git log`, but release-please reads file paths, not commit subjects, when deciding scope.

The `guard-empty-commit` job in [`release-please.yml`](https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/release-please.yml) blocks this at CI time: any push to `main` whose `HEAD` commit changes zero files fails fast with a clear error before the release-please action runs.

There is one narrow exception for history repair: an empty merge commit titled `hotfix(repo): ...` may pass if each commit introduced by the merged branch touches files. This covers cases where the final file tree is intentionally unchanged, but preserving the individual commits matters. For example, release-please reads commit history to decide package scope, version bumps, and changelog entries, so restoring a lost `feat(sdk)!` commit with a `BREAKING CHANGE:` footer can be necessary even when the files already match `main`.

**If you need to amend a release note for a commit that already merged**, see [Overriding a Merged Commit's Changelog Entry](#overriding-a-merged-commits-changelog-entry) below. Do not push empty commits to `main`.

**If a fan-out has already happened** (release PRs opened for packages you didn't change), revert the offending commit on `main`. release-please will reconcile the open release PRs on the next push that actually touches package files; PRs for unaffected packages can be closed manually.

### Lockfile churn fan-out

A subtler sibling of the empty-commit case. release-please scopes a commit to a package by the file *paths* it touches and has no notion of "this file is just a lockfile." When a bump-worthy commit (a `feat:`/`fix:` in one package) also regenerates the `uv.lock` of every package that depends on it, release-please attributes the bump-worthy commit to those dependents too and opens a release PR for each — even though their only change is a regenerated lockfile.

> [!NOTE]
> Closing such a stray release PR does **not** make it stay closed. release-please decides what to release by comparing each component's last-released SHA in `.release-please-manifest.json` against `main`; the unreleased lockfile commit is still there, so the PR is regenerated on the next run. The only ways to stop it are to release the package (merge the PR) or to remove the unreleased bump from `main` — see [Reverting a Merged-but-Unreleased PR](#reverting-a-merged-but-unreleased-pr).

**Avoid it** by landing lockfile regeneration in a separate `chore(deps):` commit/PR — `chore` is hidden and triggers no release, so only the package with real source changes is released.

The `release_please_scope_check.yml` workflow ([`.github/scripts/check_lockfile_release_scope.py`](https://github.com/langchain-ai/deepagents/blob/main/.github/scripts/check_lockfile_release_scope.py)) catches this at PR time: when a bump-worthy PR changes only a lockfile inside a managed package, it posts a sticky comment naming the affected components and **fails the check**. Resolve it (route the lockfile churn through a `chore(deps):` commit), or — for an intentional lockfile-only release such as a leaf-package security bump — apply the `allow-lockfile-release` label to acknowledge the fan-out and let the PR pass. For the failure to actually gate merges, add the check to the branch's required status checks (repo settings).

### Overriding a Merged Commit's Changelog Entry

Append a `BEGIN_COMMIT_OVERRIDE` block (shown below) to the **merged PR's body** when release-please needs to use a different message than the actual squash-merge commit. release-please reads merged PR bodies on every run within its lookback window and uses the override in place of the original commit message — no history rewrite, no force-push.

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
   - `release-sha` = `main` `HEAD` (the hotfix commit you just merged).

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

### Release Failed: Code SDK Pin Is Older Than SDK

If the release workflow fails at the "Verify package pins SDK at or ahead of workspace version" step with:

```txt
deepagents-code SDK pin is older than the workspace SDK version!
SDK version (libs/deepagents/pyproject.toml): 0.4.2
deepagents-code SDK pin (libs/code/pyproject.toml): 0.4.1
```

This means `deepagents-code`'s pinned `deepagents` dependency in `libs/code/pyproject.toml` is older than the current SDK version. This can happen when the SDK is released independently and the pin isn't updated before the `deepagents-code` release PR is merged. A pin ahead of the workspace SDK is allowed for intentional prerelease coordination.

If the older pin is intentional, add the `release: skip sdk pin check` label to the release PR before merging. The automatic release dispatch will pass `dangerous-skip-sdk-pin-check=true`, preserving the normal auto-release path while recording the bypass decision on the PR. Only use this when `deepagents-code` does not depend on SDK functionality newer than the pinned version.

**To fix after an unlabeled release PR already failed:**

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
   - Go to **Actions** > `🚀 Package Release`
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
