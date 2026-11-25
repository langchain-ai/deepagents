---
type: operations guide
title: Development and Release Operations
description: Package-local development, generated lockfile propagation, and the automated release-please, curated-notes, and publishing runbook for Deep Agents distributions.
tags: [development, uv, python, dependencies, lockfiles, releases]
sources:
  - id: openwiki-source-c902cb95854c98734bef1bde
    resource: repo://.github/release-please/history.mjs
  - id: openwiki-source-af240e595c2f1524872d109d
    resource: repo://.github/release-please/README.md
  - id: openwiki-source-a8eeaaca51cd690abd309b97
    resource: repo://.github/release-please/run.mjs
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-3943bedb80b27cdf64ea8bff
    resource: repo://.github/scripts/release/build_release_notes.py
  - id: openwiki-source-d0bc66880aeeb0817674e1c3
    resource: repo://.github/scripts/release/check_lockfile_release_scope.py
  - id: openwiki-source-ea29da8749b893917f11666d
    resource: repo://.github/scripts/release/release-notes.js
  - id: openwiki-source-86d8c13f5b625f095cd4c496
    resource: repo://.github/scripts/release/update_changelog_dependencies.py
  - id: openwiki-source-e0d5bd90fac633227fff1f23
    resource: repo://.github/scripts/tests/release/release-notes.test.js
  - id: openwiki-source-5f1576ab75e9b69811ec2f36
    resource: repo://.github/scripts/tests/release/release-please-history.test.mjs
  - id: openwiki-source-74e4ee6ae7c1e3e1008eb655
    resource: repo://.github/scripts/tests/release/test_build_release_notes.py
  - id: openwiki-source-6e115e2c0860618907327d75
    resource: repo://.github/scripts/tests/release/test_release_please_workflow.py
  - id: openwiki-source-4541a26c837a99dc39a0ee0c
    resource: repo://.github/workflows/release_notes.yml
  - id: openwiki-source-7f3822065cb7dc086752e9d5
    resource: repo://.github/workflows/release_please_scope_check.yml
  - id: openwiki-source-46fa34397e41ebf7491c7359
    resource: repo://.github/workflows/release-please.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
verified:
  - by: openwiki/0.4.2
    at: 2026-11-24T08:04:12.461Z
generated: { by: "openwiki/0.4.2", at: "2026-11-24T08:04:12.461Z" }
---

# Development and Release Operations

This monorepo has independently versioned packages under `libs/`, not a root Python project. Develop in the package that owns a change. Its `pyproject.toml` is the published compatibility contract and its `Makefile` is the supported interface. See [Quickstart](../quickstart.md) for initial setup and the [Testing Guide](../testing/testing-guide.md) for package tests.

## Package-local development and dependency contracts

Use `uv` for interpreters, environments, and dependencies, and `make` for normal tasks. `uv` provisions the required interpreter; each package owns its supported range through `requires-python`.

```bash
uv tool install pre-commit
pre-commit install --install-hooks

cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Use `make help` in a package to discover its targets and `uv run ...` for an exceptional one-off command. From `libs/`, the aggregate `Makefile` fans out operations over library, partner, and example packages, but does not replace package-local setup.

| Layer | Meaning | Operational rule |
| --- | --- | --- |
| `[project]` dependencies, extras, `requires-python` | Wheel metadata installed by users | Treat this as the public compatibility contract. |
| `[tool.uv.sources]` | Editable sibling checkout wiring | Never use local resolution as evidence that the published graph resolves. |
| `uv.lock` | Generated package development and CI resolution | Regenerate with `uv`; never hand-edit. |
| `release-please-config.json` | Managed paths, components, changelogs, exclusions, and release behavior | Update as release integration wiring. |
| `.release-please-manifest.json` | Last released baseline at each managed path | Release-please owns ordinary updates. |

### SDK, Code, and Talon compatibility

`deepagents-code` publishes an exact `deepagents==0.7.41` requirement while its workspace maps `deepagents` to the editable `../deepagents` checkout. Thus a successful Code workspace can use unreleased sibling code and is not proof of the PyPI installation a user receives. Raise the exact Code SDK pin in the PR that first requires new SDK behavior; do not postpone it to a release PR. A deliberately older pin requires the documented `ci:dcode-skip-sdk-pin` acknowledgement.

Talon is Python `>=3.14,<3.15`, publishes `deepagents>=0.7.41` and `deepagents-code>=0.2.13,<1.0.0`, and maps those packages locally only for development. Its floor means Talon releases must follow a published Code release that satisfies it.

## Lockfile propagation

A changed package manifest can invalidate its own lock and every transitive consumer that uses it as a local `uv` source. Refresh the **reverse-consumer closure**, then check direct changed package paths:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/code/pyproject.toml
```

```mermaid
flowchart TD
    Change["Changed package manifest"] --> Owner["Select owning package"]
    Owner --> Sources["Read local uv source paths"]
    Sources --> Consumers["Add local-source consumers"]
    Consumers --> Closure["Repeat to fixed point"]
    Closure --> Resolve["Run uv lock with package policy"]
    Resolve --> Commit["Commit generated uv.lock files"]
```

*The refresh operation includes the changed owner and all transitive local-source consumers.*

`refresh_lockfiles.py` resolves that closure and stops on resolver failure. The checker deliberately checks only directly touched packages when paths are supplied; without paths it checks every discovered package. The shared policy uses Python 3.14 for ACP, Code, Evals, and Talon, and Python 3.12 for the other packages. For broad dependency work, run the no-argument check:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

The release branch is also automated: after release-please creates or refreshes a PR, its lockfile job regenerates every tracked lock and dependency-note section. On a rejected branch push it fetches the tip, regenerates, and retries up to three times. Generated locks and their propagation are therefore committed output, not handwritten release changes.

## Release-please ownership and scope

The registry manages nine distributions. Current manifest baselines are Deep Agents `0.7.41`, ACP `0.1.0`, Code `0.2.14`, Talon `0.1.1`, Daytona `0.0.9`, Modal `0.0.8`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.4.0`. Configuration creates separate draft PRs and skips GitHub-release creation; the dispatched `release.yml` workflow publishes after a release PR merges.

Release-please owns version/manifest updates and generated release-PR artifacts. Do not manually rebase a release branch, merge `main` into it, or edit it while an update is queued. Wait for both the `release-please.yml` run for the relevant `main` commit and the release PR's **Update lockfiles** job to finish before review or note curation.

Attribution is path-based, not solely Conventional Commit scope. The configuration excludes each package's tests and `uv.lock`; a bump-worthy PR touching release-relevant files in more than one managed component is blocked unless a maintainer applies `ci:allow-lockfile-release`. That acknowledgement permits the intentional fan-out; it does not stop the additional releases. The gate reads trusted base configuration and fails closed if GitHub's changed-file response is incomplete.

`feat`, `fix`, `perf`, and `revert` appear in generated changelogs. Current packages are pre-1.0: `feat` produces a patch bump and a breaking `feat!` produces a minor bump. Hidden commit types can still ship with an existing release but do not add their own visible entry. Dependency range changes remain visible through the generated dependency section described below.

## Curated release notes

Every managed release PR is eligible because the notes workflow derives its component from the trusted internal release-branch name and confirms it against the registry and `release(<component>): <version>` title. The workflow runs trusted automation from `main`; it reads release-branch content through the API at a validated SHA rather than checking out or executing it. Only maintainers with write-level permission can issue manual commands.

1. Keep a release PR draft while changes accumulate. Mark it ready for review to create a bot-authored draft.
2. Review and edit the marked draft comment, retaining its version heading and generated `### Dependencies` section.
3. Comment `@release-bot apply`. The guarded flow writes the curated section to that package's `CHANGELOG.md` and mirrors it to the PR preview.
4. Merge only after the `curated release notes` gate passes. `ci:skip-curated-notes` is the explicit, exceptional bypass when generated notes are intentionally shipped unchanged.

```mermaid
flowchart TD
    Ready["Release PR ready for review"] --> Draft["Bot validates PR identity and head"]
    Draft --> Model["Draft prose from generated entries"]
    Model --> Comment["Post trusted editable draft"]
    Comment --> Apply["Maintainer runs apply"]
    Apply --> Verify["Verify hashes, head, scope, and dependencies"]
    Verify --> Write["Commit changelog and mirror PR preview"]
    Write --> Gate["Curated-notes check passes"]
```

*Curated prose is a guarded transformation of release-please output, not a direct edit to generated release artifacts.*

The model receives generated changelog entries as untrusted source material, but never the dependency subsection. The workflow restores that subsection verbatim, so dependency ranges cannot be reworded or dropped. It validates exactly one release section, binds drafts to bot identity, component, version, release head, generated-entry fingerprint, and (for new drafts) the `main` baseline. It rejects altered headings, changed generated entries, edited dependencies, stale package changes, untrusted comments, concurrent PR/body changes, and branch-head races. A release-branch rewrite caused solely by unrelated `main` changes can retain a package-scoped draft; unknown, incomplete, or package-relevant change detection requires a fresh draft.

Use `@release-bot draft` to redo a failed draft or after new `main` changes affect that package. It may take up to 500 characters of one-line editorial guidance; a second `@` ends the guidance. `apply` ignores trailing instructions and republishes the stored draft. Re-drafting updates the original bot comment in place and posts a pointer comment; commands are serialized per PR rather than cancelling an in-flight curation. A manual command is acknowledged with 👀 and changed to 🚀 when it succeeds.

The automatic dependency generator diffs the package `pyproject.toml` from the resolved predecessor tag to the release branch head and inserts, replaces, or removes `### Dependencies` in that version section. It warns and leaves the changelog untouched if it cannot compute the list, so a dependency-note problem cannot block lock regeneration. Because the published GitHub release body is extracted from the merged `CHANGELOG.md`, this keeps dependency metadata changes visible from one source of truth.

## Release lifecycle and publication

```mermaid
flowchart TD
    Push["Push to main"] --> Detect["Reject empty commit and detect release"]
    Detect --> Dispatch["Dispatch release.yml for exact SHA"]
    Detect --> Wait["Wait for pending publishes"]
    Dispatch --> Publish["Build and publish selected package"]
    Publish --> Tagged["Create tag and GitHub release"]
    Wait --> Runner["Run patched release-please"]
    Runner --> Locks["Refresh release branch locks and dependency notes"]
```

*Publishing a detected merged release is dispatched before serialized release-PR maintenance.*

Empty commits are rejected because they contain no paths to scope and could fan out a release to every component. A release is detected only when the `release(<component>): <version>` subject and that package's `CHANGELOG.md` change agree. Before recomputing PRs, the workflow waits for every merged PR still marked `auto:release-pending`, since any package between version bump and tag creation makes repository-wide version computation unsafe. An unreadable GitHub state fails closed; a publish that remains in flight is deferred rather than recomputed against partial state.

`release.yml` resolves a named package and exact release SHA, verifies the version in that tree and normal mainline reachability, then builds and publishes the same tree. The operational sequence is build, pre-release checks, TestPyPI, PyPI, and GitHub release. A version that has reached PyPI is immutable: never move or recreate its tag; publish a new version for a fix.

The repository runs a pinned, patched local `release-please` runner rather than the upstream action. It uses full-history blobless `git log` attribution instead of the GraphQL walk and removes the 500-commit cap, while retaining path attribution for merge and rename cases. Its frozen pnpm install applies version-specific patches and validates them before GitHub calls. Upgrade the runner and patch artifacts together, then compare a full-history dry run with open release PRs.

## Focused validation

For dependency or package metadata changes, run package tests, refresh the reverse-consumer closure, and run the all-package lock check. For release automation, use focused tests that protect the actual failure boundaries:

```bash
uv run --no-project --with "pytest" pytest \
  .github/scripts/tests/checks/test_check_lockfiles_pre_commit.py \
  .github/scripts/tests/checks/test_refresh_lockfiles.py \
  .github/scripts/tests/release/test_check_lockfile_release_scope.py \
  .github/scripts/tests/release/test_build_release_notes.py
node --test .github/scripts/tests/release/release-notes.test.js
node --test .github/scripts/tests/release/release-please-history.test.mjs
uv run --no-project --with "pyyaml" pytest .github/scripts/tests/release/test_release_please_workflow.py
```

The release-note tests cover exact PR identity, untrusted/fork rejection, command permission and ambiguity, head-change retries, marker and heading injection resistance, stale-state detection, dependency immutability, and non-force application. The Python release-body tests cover predecessor-tag selection across stable and pre-release versions, SHA-bounded changelog extraction, bounded and escaped package history, and dependency-section updates. These tests matter whenever changing the note protocol, its parser, release tags, or publication body construction.
