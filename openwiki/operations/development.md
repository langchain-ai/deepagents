---
type: operations guide
title: Development, Dependencies, Lockfiles, and Releases
description: Package-scoped uv development and release operations for the Deep Agents distributions. Explains published dependency contracts, editable local-source lockfile closure, frozen-lock behavior, and release-please publication safeguards.
tags: [development, dependencies, uv, lockfiles, dcode, releases, release-please]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-26T08:11:58.203Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-594b8b7a84e0fb527fbafd52
    resource: repo://.github/scripts/checks/raise_langchain_minimums.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-92e05f1f5d419b3d69cc3a1e
    resource: repo://.github/scripts/release/draft-release-notes.js
  - id: openwiki-source-1bc32c2b33ab6e58128b0319
    resource: repo://.github/scripts/tests/checks/test_raise_langchain_minimums.py
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-bc54cf7ca3addab1b243d7a6
    resource: repo://.github/workflows/raise_langchain_minimums.yml
  - id: openwiki-source-4541a26c837a99dc39a0ee0c
    resource: repo://.github/workflows/release_notes.yml
  - id: openwiki-source-46fa34397e41ebf7491c7359
    resource: repo://.github/workflows/release-please.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-2c81844c6fc58eace9ea6aba
    resource: repo://libs/code/scripts/check_process_cwd.py
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-6985df297a9bf809e10bc5d8
    resource: repo://libs/evals/uv.lock
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-5742b8b792426010995c1325
    resource: repo://libs/partners/quickjs/uv.lock
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-966f1489233b6fc6ee5f2e4c
    resource: repo://libs/talon/uv.lock
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-10-26T08:11:58.203Z" }
---

# Development, Dependencies, Lockfiles, and Releases

This is a monorepo of independently versioned distributions under `libs/`, not a root Python project. Work in the package that owns the change: its `pyproject.toml` is its published compatibility contract, its `Makefile` is the authority for normal commands, and it owns its uv environment and `uv.lock`. See the [architecture overview](../architecture/overview.md) for runtime context and the [Testing Guide](../testing/testing-guide.md) for test design.

## Package-local development

Use `uv` for interpreters, environments, and dependencies; do not substitute `pip`, Poetry, or Conda in the normal workflow. `uv` provisions the interpreter, so there is no repository-wide Python version to install or pin. Run `make help` in the package being changed, sync that package, and use its targets for ordinary edit/test/lint work.

```bash
uv tool install pre-commit
pre-commit install --install-hooks

cd libs/deepagents
uv sync --all-groups
make test
make lint
```

`uv sync --all-groups` installs the package and its declared groups. Use `uv run ...` for an intentional one-off command. Each package's `requires-python` and dependency constraints—not editable workspace resolution—define what downstream users can install.

### Package command boundaries and frozen locks

The package Makefile, rather than this page or an ad-hoc pytest command, defines the supported local workflow. For example, `libs/code` exports `UV_FROZEN=true`: normal Make targets will not silently repair a stale lock. Its unit target uses parallel pytest with non-Unix sockets disabled and Unix sockets allowed; its integration target selects integration tests with a 30-second timeout. `make lint` runs Ruff checking and format verification, `ty`, the slash-command catalog check, and the process-directory gate. `make bootstrap` syncs the test group and installs hooks; `make check` adds import checks, tests, extras synchronization, version equality, and lock freshness. The SDK-pin subcheck is advisory only for exit status `1`; any other failure is fatal.

```bash
cd libs/code
make bootstrap
make check
```

Talon also exports `UV_FROZEN=true`. Its `make test` first runs the Node WhatsApp-bridge tests, then the Python suite with socket restrictions and a 10-second timeout; its lint and format targets run Ruff and `ty`. Evals does not export that frozen setting; use its Makefile for unit tests, linting, type checks, evals, Harbor staging, and dataset operations. In particular, `make evals MODEL=<id>` requires a model, while the Harbor targets stage local Deep Agents, dcode, ACP, and QuickJS checkouts before invoking Harbor.

Deep Agents and QuickJS also export `UV_FROZEN=true`. Their unit targets use parallel pytest with non-Unix sockets disabled and Unix sockets allowed; QuickJS applies the same socket restriction to parallel integration tests. Their lint targets, like Code's, run Ruff checks, Ruff format verification, and `ty`; `TEST_FILE` narrows the usual test target where supported.

Frozen normal targets make a stale lock visible rather than mutating it as a side effect. Repair lock state deliberately with the lockfile procedure below, then return to the owning package's Makefile commands.

### dcode process-directory review gate

`make lint` for `libs/code` also verifies the generated slash-command catalog and runs `scripts/check_process_cwd.py deepagents_code`. This AST-based gate exists because one workspace-server process can serve many bound workspaces: reading the process current directory answers correctly only for the workspace from which that server was launched. Server paths should use the directory recorded for the workspace instead.

The checker recognizes direct `Path.cwd()`, `os.getcwd()`, and calls to `find_project_root` that have no argument. Every recognized call must have a matching `CallSite` allowlist entry with a reason; an allowlist entry without a matching call fails as stale. A call identity includes its package-relative path, enclosing scope, kind, a digest of the call and enclosing statement, and—only for duplicate calls—an occurrence number. Editing the statement therefore creates an unreviewed replacement and a stale old entry, deliberately requiring renewed review.

A clean run is not proof that no process-directory read exists: the scanner intentionally does not detect equivalent indirect spellings. Prefer an explicit workspace directory; if a recognized call is legitimate, add a narrowly scoped allowlist entry explaining why the process directory is correct. Read or parse errors fail closed.

Pre-commit installs commit-message and pre-push hooks. It validates Conventional Commit messages and repository hygiene, runs matching package format/lint targets, and checks affected lockfiles, extras, and version-file pairs. The pre-push hook delegates branch naming to `.githooks/pre-push`; re-run `pre-commit install --install-hooks` in an existing checkout after hook configuration changes.

External contributors must link an approved issue or discussion and be assigned before opening a PR. Use a scoped Conventional Commit title. Keep a bump-worthy change in one releasable component; put cross-package dependency and lockfile churn in a separate `chore(deps):` PR so release-please does not attribute one visible change to several distributions.

## Published requirements versus local sources

Every package's `pyproject.toml` is a public contract. A requirement in `[project.dependencies]` declares what an installer may resolve from PyPI; `[tool.uv.sources]` is checkout-only wiring used by uv. Keep these layers separate and do not loosen a published bound merely to make sibling development convenient.

For example, dcode (`deepagents-code`) publishes the exact `deepagents==0.7.32` requirement and bounded QuickJS/ACP dependencies, but locally resolves Deep Agents, ACP, and partner packages from sibling editable paths. Evals publishes requirements for Deep Agents, dcode, and QuickJS, while its local sources point to `../deepagents`, `../code`, and `../partners/quickjs`. Talon publishes `deepagents>=0.7.32` and `deepagents-code>=0.2.1,<1.0.0`, while resolving both from editable sibling paths. QuickJS follows the same split: it publishes `deepagents>=0.7.32,<0.8.0` while mapping Deep Agents locally. Evals and Talon publish Python 3.14-only compatibility, whereas QuickJS publishes Python 3.11 through 3.x compatibility; their lockfiles encode the corresponding resolution coverage. These source mappings change local resolution; they do not change the packages a downstream user receives.

### The dcode-to-SDK pin invariant

`deepagents-code` must pin the **minimum** exact `deepagents==X.Y.Z` that its code requires. Raise that pin in the same PR that first uses new SDK functionality; do not defer it to release time. The scheduled dependency-floor updater leaves exact pins alone. The local check reports a stale pin as advisory because an editable checkout runs against the workspace SDK, but release validation requires the pin to be at or ahead of the workspace SDK. `ci:dcode-skip-sdk-pin` and `dangerous-skip-sdk-pin-check` are exceptional overrides for intentionally older pins; pins ahead of the workspace SDK are allowed for deliberate prerelease coordination.

## Lockfiles: resolve the reverse editable-consumer closure

An editable consumer lock embeds its local producer's requirements. Therefore a changed `pyproject.toml` can stale both its own `uv.lock` and the locks of every transitive package which consumes it through `[tool.uv.sources]`. Do **not** hand-edit `uv.lock`: regenerate it with uv and commit the resulting lockfiles.

```mermaid
flowchart TD
    Manifest["Changed pyproject.toml"] --> Owner["Changed package lock"]
    Manifest --> Sources["Editable local sources"]
    Sources --> Consumers["Transitive reverse consumers"]
    Owner --> Resolve["uv lock for closure"]
    Consumers --> Resolve
    Resolve --> Verify["uv lock check"]
```

*The lockfile repair set is the changed package plus its transitive reverse consumers declared through local sources.*

For the current dcode source graph, a `libs/code/pyproject.toml` change requires consideration of `libs/code/uv.lock` and the direct editable consumers `libs/evals/uv.lock` and `libs/talon/uv.lock`; the helper computes the actual transitive closure from manifests rather than relying on this example. Conversely, an Evals manifest change has no consumer in its declared local-source graph, while a Talon manifest change affects Talon unless another discovered package locally consumes it.

From the repository root, use the refresh helper for manifest changes:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

It reads every discovered library, partner, and example manifest, resolves each `path` under `[tool.uv.sources]` relative to its manifest, repeatedly adds consumers of the stale set, and runs `uv lock --directory <package> --python <version>` for each resulting directory. The checker is different: with no paths it checks every discovered package, while with paths it checks the package owners of those paths. A failure prints a root-relative `uv lock` repair command.

The script-based checker and refresh helper use Python 3.14 for first-party `acp`, `code`, `evals`, and `talon`, and Python 3.12 for other discovered packages. This is intentionally more specific than `libs/Makefile`, whose aggregate fan-out currently selects 3.14 only for ACP and 3.12 for every other package. The Evals and Talon lock headers require Python 3.14; Evals additionally records its macOS arm64, Linux x86_64, and Linux aarch64 resolution coverage and requires the first two environments. Treat these locks as resolution outputs, not manually maintained manifests.

From `libs/`, aggregate Makefile targets are useful for whole-repository operations and stop at the first failure:

| Command | Purpose |
| --- | --- |
| `make lock [no-cache]` | Resolve library, partner, and example locks; `no-cache` adds `--no-cache`. |
| `make lock-check` | Run `uv lock --check` for every discovered package. |
| `make lock-bump DEP=<pkg>` | Re-resolve every discovered lock with `-P <pkg>`; `DEP` is required. |
| `make lint` / `make format` | Fan out across library packages. |

CI invokes the lock check for pull requests, pushes to `main`, and merge groups. Release-please branches also receive an `update-lockfiles` pass because release-please changes package versions but does not regenerate `uv.lock`.

### Scheduled dependency floors

The daily minimums workflow raises eligible PyPI floors for normalized names beginning `langchain`, `langgraph`, `langsmith`, or `deepagents`, including project dependencies, extras, and dependency groups. Only `>=` and `~=` lower bounds qualify; the selected release must be stable, non-yanked, and within the existing range.

The updater preserves upper bounds, extras, markers, and compatible-release ceilings, never lowers a deliberately higher floor, and leaves exact pins alone. It plans all manifest edits before writing. Metadata failures, unrewriteable requirements, and an ineligible requested dependency fail rather than produce a misleading clean result. A versioned local path source remains a published floor and caps the raised value at the local checkout version so `uv lock` can resolve it. After changes, automation regenerates the affected consumer closure before opening or refreshing its `chore(deps):` PR; `skip-dependent-locks` explicitly defers that repair.

## Release ownership and scope

Release-please manages a package only when it appears in both `release-please-config.json` and `.release-please-manifest.json`. The manifest holds the last released baseline, not a desired working-tree version. A never-published `0.0.1` package needs a `0.0.0` manifest baseline so its first proposal is `0.0.1`, rather than `0.0.2`.

| Distribution | Path | Manifest baseline |
| --- | --- | --- |
| `deepagents` | `libs/deepagents` | `0.7.32` |
| `deepagents-acp` | `libs/acp` | `0.0.13` |
| `deepagents-code` | `libs/code` | `0.2.1` |
| `deepagents-talon` | `libs/talon` | `0.1.0` |
| `langchain-daytona` | `libs/partners/daytona` | `0.0.9` |
| `langchain-modal` | `libs/partners/modal` | `0.0.6` |
| `langchain-runloop` | `libs/partners/runloop` | `0.0.7` |
| `langchain-vercel-sandbox` | `libs/partners/vercel` | `0.0.2` |
| `langchain-quickjs` | `libs/partners/quickjs` | `0.3.9` |

Configuration creates separate draft Python release PRs titled `release(${component}): ${version}`. All managed packages use pre-1.0 behavior: `feat` makes a patch bump and a breaking feature makes a minor bump. Generated changelogs show `feat`, `fix`, `perf`, and `revert`; docs, style, chore, refactor, test, CI, and hotfix entries are hidden.

`skip-github-release` means release-please prepares PRs but `release.yml` publishes, tags, and creates GitHub releases. The detector recognizes a managed release only when the merge commit has a matching release title **and** changes that package's `CHANGELOG.md`. Ordinary empty commits are blocked before release-please can pathlessly fan out to every component.

## Release PR to publication

After releasable commits land on `main`, wait for both the release-please refresh and the release PR's `Update lockfiles` job. Do not manually rebase a release branch, merge `main` into it, or edit it while automation is queued or running: it can replace changes, invalidate reviewed notes, or race note application.

```mermaid
flowchart TD
    Land["Releasable commit on main"] --> Draft["Package release PR"]
    Draft --> Locks["Update lockfiles"]
    Locks --> Ready["Maintainer marks ready"]
    Ready --> Curate["Release bot applies notes"]
    Curate --> Gate["Curated notes gate"]
    Gate --> Merge["Merge release PR"]
    Merge --> Publish["Build and publish"]
    Publish --> Tag["Tag and GitHub release"]
```

*Each package has an independent release PR, while its lock and publication gates run in this order.*

Keep a release PR draft while changes accumulate. When ready, mark it ready for review, review or revise the release-bot draft while retaining its version heading, then comment `@release-bot apply`. The bot writes that package's `CHANGELOG.md` and mirrors the notes into the PR body. Merge only after the `curated release notes` check passes; repeat for every PR in a fan-out. `ci:skip-curated-notes` is the explicit opt-out for intentionally shipping the generated changelog uncurated.

Run `@release-bot draft` if automatic drafting failed or later `main` changes touch that package. A release-branch lockfile-only refresh caused by unrelated work does not invalidate an existing draft. Curation checks out trusted `main` automation and reads release-PR content through the API as untrusted data; it does not execute PR content. Commands serialize per PR without cancelling an in-flight run. The drafting helper constrains provider output to structured user-facing sections, accepts only PR links, and rejects non-normal or malformed model completions.

A trusted changelog-only `@release-bot apply` commit on a release-please branch has a narrow CI optimization. CI requires a conclusive parent `✅ CI Success` result, suppresses selected package lint and test jobs, and derives the head result from that parent. An inconclusive parent falls back to ordinary package jobs.

## Publisher gates, recovery, and escape hatches

Merging a release PR dispatches `release.yml` before release-please refreshes other open release PRs. The publisher maps the distribution to its working directory. For an ordinary release, provide a version and exact 40-character `release-sha`; the selected commit's `pyproject.toml` must declare that version. It builds from that resolved SHA, checks whether the PyPI version already exists, and fails closed when PyPI is unavailable or unverifiable. Build artifacts and pre-release checks precede TestPyPI and trusted PyPI publication; the tag and GitHub release use the same source tree.

Release-note generation is non-blocking: a failure does not stop publication or GitHub release creation. The critical integrity invariant is that a PyPI version and its GitHub tag identify the same code. If failure occurs before PyPI upload, fix the problem without changing the version and manually dispatch from the exact hotfix SHA. Once PyPI has a version, do not republish it or move/recreate its tag; release a new version instead.

Before recomputing open release PRs, the post-merge path waits for every merged PR still marked `auto:release-pending`. This repository-wide wait prevents release-please from seeing a new manifest baseline without its tag and proposing a bootstrap downgrade. Successful publication changes the source release PR label to `auto:release-tagged`.

Manual dispatch is for recovery, backports, and prereleases rather than the normal route. `dangerous-nonmain-release` permits a throwaway non-main prerelease or maintenance branch; keep PEP 440 prerelease version transitions off `main` and long-lived version branches because release-please cannot safely maintain them.

Ripgrep is a release-sensitive artifact-test gate. Without an override, a failed ripgrep installation fails release prechecks. Applying `ci:skip-ripgrep` to the merged release PR causes release-please to dispatch `dangerous-skip-ripgrep-check=true`; an installation failure is then tolerated and `rg`-gated artifact tests skip. Use this only for a transient infrastructure failure: it weakens release validation rather than fixing ripgrep.

For detailed operator procedures—including pending-label stalls, pre-PyPI failures, post-publication fixes, and alpha/beta releases—use [`.github/RELEASING.md`](https://github.com/langchain-ai/deepagents/blob/main/.github/RELEASING.md).
