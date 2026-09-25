---
type: operations guide
title: Development and Release Operations
description: Package-scoped uv and Makefile workflows, repository lockfile validation, Python-version boundaries, pre-commit expectations, and independent package release management.
tags: [development, monorepo, uv, make, lockfiles, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-25T08:06:00.203Z
sources:
  - id: openwiki-source-baf30c604828cfde90a8ab63
    resource: repo://.githooks/pre-push
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-46fa34397e41ebf7491c7359
    resource: repo://.github/workflows/release-please.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-6c2e9cfaa20096e021221d47
    resource: repo://libs/code/CHANGELOG.md
  - id: openwiki-source-ac769408e1d61a20b9874382
    resource: repo://libs/code/deepagents_code/_version.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-09-25T08:06:00.203Z" }
---

# Development and Release Operations

This repository is a monorepo of independently versioned Python packages under `libs/`, rather than one root Python project. Do ordinary work at the package boundary; use aggregate tooling only when validating shared lockfile or dependency changes. For code structure and test strategy, see [Source Map](../architecture/source-map.md), [Testing Guide](../testing/testing-guide.md), and [Quickstart](../quickstart.md).

## Contribution and package-local loop

Before an external contributor opens a PR, the contributor must be assigned to a maintainer-approved issue or discussion linked from that PR. Each package owns its own `pyproject.toml`, `Makefile`, and README; there is no root `pyproject.toml`. Local sibling dependencies can be editable, so changes in one package can be visible to a dependent sibling during in-tree development.

Use `uv` for interpreters, environments, and dependencies—do not substitute `pip`, Poetry, or Conda. `uv` selects an interpreter compatible with the package's `requires-python`; there is no single repository-wide Python version to install. The package Makefile is the command authority, and `make help` lists the targets that package actually supports.

Install Git hooks once, then enter the package being changed:

```bash
uv tool install pre-commit
pre-commit install --install-hooks

cd libs/deepagents
uv sync --all-groups
make test
make lint
```

`uv sync --all-groups` explicitly installs the package and its dependency groups. Use `uv sync --group <name>` when a narrower environment is appropriate, and use `uv run ...` for one-off commands. Do not create an environment outside its package or mix environments in the same session.

```mermaid
flowchart TD
    Enter["Enter changed package"] --> Sync["Sync required dependency groups"]
    Sync --> Change["Change source and focused tests"]
    Change --> Validate["Run package test and lint targets"]
    Validate --> Pass{"Checks pass"}
    Pass -->|"No"| Change
    Pass -->|"Yes"| Review["Commit and open scoped PR"]
```

Caption: the normal development loop is explicitly synchronized and package-scoped before it reaches review.

Typical targets vary by package, but the core packages provide the following pattern:

| Command | Purpose |
| --- | --- |
| `make test` | Run unit tests; the core package runs socket-disabled pytest in parallel with coverage. |
| `make integration_test` | Run network-capable integration tests when the package supplies the target. |
| `make lint` | Check Ruff formatting/linting and type checking. |
| `make format` | Apply formatting and safe lint fixes. |
| `make type`, `make coverage`, `make test_watch` | Run a focused check when offered by the current package. |

Package targets run tools through `uv run`. In `deepagents`, `UV_FROZEN = true` makes an out-of-date lockfile fail rather than update itself during a Make target. This keeps validation deterministic: regenerate locks intentionally, then commit them.

### Pre-commit is an early guard, not a replacement for package checks

The hook installation above enables `pre-commit`, `commit-msg`, and `pre-push` stages. The configuration checks Conventional Commit message types, prevents direct commits to `main`, validates YAML and TOML, and normalizes whitespace. Its local hooks run package-specific `make ... format lint` commands for changed `deepagents`, Code, evals, and ACP paths; they also check affected lockfiles, dependency extras, and version equality for the SDK and Code packages.

The lock hook only checks package or example directories touched by the supplied paths, but checks every known lock-owning directory when it receives no paths. For each selected directory it runs `uv lock --check` with the repository's chosen lock interpreter. A hook can be bypassed, so retain the package loop and CI validation before merging.

The pre-push branch-name hook is installed through pre-commit. Except for protected or automation/release branch patterns, it requires `<github-username>/<scope>/<short-description>`; the username is resolved from `github.user`, then the GitHub CLI, then the local part of `user.email`. It is a local convenience and can be skipped, while server-side branch checking remains authoritative.

### Code package parity checks

`libs/code` provides `make check` as its local CI-parity entry point. After linting, import checks, and unit tests, it verifies extra synchronization, equality of `pyproject.toml` and `_version.py`, and lock freshness. Its SDK-pin checker treats exit status 1 (a stale pin) as advisory, while other checker failures stop the command.

The current `deepagents-code` source version is `0.1.77` in project metadata and `deepagents_code/_version.py`; its changelog begins with the same release. It declares an exact `deepagents==0.7.19` dependency. Change that pin only when Code needs a newer SDK, regenerate `libs/code/uv.lock`, and validate with `make check`.

## Aggregate locks and Python boundaries

Run repository fan-out operations from `libs/`. Its Makefile discovers library packages with Makefiles, partner packages with Makefiles, and example projects with `pyproject.toml` for lock operations. The loops use `set -e`, so the first failing package ends the command.

| Command | Purpose |
| --- | --- |
| `make lint` / `make format` | Run the corresponding target in every discovered library package. |
| `make lock [no-cache]` | Regenerate library and example locks; `no-cache` passes `--no-cache` to uv. |
| `make lock-check` | Verify all discovered locks are current. |
| `make lock-bump DEP=<pkg>` | Re-resolve each lock with `-P <pkg>`; `DEP` is required. |
| `make bench-all` | Run `bench` for `deepagents` and `code`. |

The aggregate lock command resolves ACP using Python 3.14 and every other package or example using Python 3.12. That is a reproducibility policy for lock creation, not the published support floor: ACP declares `>=3.11`, Talon declares `>=3.12`, core `deepagents` declares `>=3.11,<4.0`, and Code declares `>=3.12,<4.0`. Always consult the package's own `requires-python` for runtime support.

When changing package metadata or resolved dependencies, regenerate the owning lock rather than editing it manually. For a shared dependency upgrade, run `make -C libs lock-bump DEP=<pkg>` and commit all resulting locks. Package-local Makefiles deliberately fail on stale locks, while aggregate `lock-check` finds drift across the repository.

## Independent versions and release manifest

Release-please manages nine independent Python distributions: `deepagents`, `deepagents-acp`, `deepagents-code`, `deepagents-talon`, `langchain-daytona`, `langchain-modal`, `langchain-runloop`, `langchain-vercel-sandbox`, and `langchain-quickjs`. `separate-pull-requests` is enabled. For each managed path, the configuration specifies Python release type, distribution and component names, changelog path, version-bearing extra files, and excluded test paths.

The manifest is the last-released baseline, not necessarily a package's editable source version:

| Manifest path | Current baseline |
| --- | --- |
| `libs/deepagents` | `0.7.19` |
| `libs/acp` | `0.0.12` |
| `libs/code` | `0.1.77` |
| `libs/talon` | `0.0.8` |
| `libs/partners/daytona` | `0.0.8` |
| `libs/partners/modal` | `0.0.6` |
| `libs/partners/runloop` | `0.0.7` |
| `libs/partners/vercel` | `0.0.2` |
| `libs/partners/quickjs` | `0.3.7` |

Do not manually advance those values during normal development. When introducing a new managed package whose source starts at `0.0.1` and has not been released, register it in both the configuration and manifest with a `0.0.0` manifest baseline; otherwise release-please interprets `0.0.1` as already shipped and proposes `0.0.2`. Release-please updates metadata and version markers on its release PR, but does not regenerate `uv.lock`; the release workflow's lock updater does that work.

## Release flow and operating constraints

```mermaid
flowchart TD
    Main["Releasable change lands on main"] --> ReleasePR["Release-please creates or updates draft release PR"]
    ReleasePR --> Lock["Regenerate affected lockfile"]
    Lock --> Merge["Merge release PR"]
    Merge --> Detect["Detect release title and changelog change"]
    Detect --> Build["Build at explicit release SHA"]
    Build --> Verify["Run pre-release validation"]
    Verify --> Test["Publish to Test PyPI"]
    Test --> Publish["Publish to PyPI"]
    Publish --> Tag["Create GitHub release and tag"]
```

Caption: release-please prepares independent package releases, while the package release workflow publishes and tags the validated release tree.

Release-please is configured to draft separate release PRs and skip its own GitHub-release creation. After a release PR is merged, the release workflow detects a matching `release(<component>): <version>` commit that changes the package changelog, then dispatches the publisher. The publisher is a `workflow_dispatch` workflow because PyPI Trusted Publishing does not officially support reusable workflows. Its normal path requires an explicit release SHA whose package metadata declares the requested version, so the build, published artifacts, and tag refer to the same tree.

Keep a bump-worthy change limited to one managed component. Component attribution follows changed paths, not Conventional Commit scope alone; an empty commit has no path and could otherwise fan out to every package, so the release workflow rejects it before release-please proceeds. A release branch lock update occurs after version metadata changes because release-please itself does not maintain uv lockfiles.

The publication stages are build, pre-release validation, TestPyPI, PyPI, and GitHub release/tag. The build refuses an already published package version and fails closed when the PyPI version check cannot be trusted. Publishing and repository-write permissions are kept out of the minimally permitted build job.
