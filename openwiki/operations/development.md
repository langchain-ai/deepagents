---
type: operations guide
title: Development and Release Operations
description: Package-local development contracts, dependency compatibility, lockfile closure, release baselines, and the scheduled process that raises compatible LangChain-ecosystem floors.
tags: [development, dependencies, uv, lockfiles, python, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-13T08:03:55.116Z
sources:
  - id: openwiki-source-3638ff2c170b760df159d81b
    resource: repo://.github/actions/uv_setup/action.yml
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-594b8b7a84e0fb527fbafd52
    resource: repo://.github/scripts/checks/raise_langchain_minimums.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-e7d931fcc9d582768b59c30e
    resource: repo://.github/scripts/release/check_sdk_pin.py
  - id: openwiki-source-1bc32c2b33ab6e58128b0319
    resource: repo://.github/scripts/tests/checks/test_raise_langchain_minimums.py
  - id: openwiki-source-0e2f8f3ef1c6d8452a906e09
    resource: repo://.github/workflows/bump_code_sdk_pin.yml
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-bc54cf7ca3addab1b243d7a6
    resource: repo://.github/workflows/raise_langchain_minimums.yml
  - id: openwiki-source-7f3822065cb7dc086752e9d5
    resource: repo://.github/workflows/release_please_scope_check.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-da577cbe81ec29338f1388b2
    resource: repo://libs/partners/daytona/pyproject.toml
  - id: openwiki-source-936554ac5f0a201f8696be25
    resource: repo://libs/partners/modal/pyproject.toml
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-8d2c8381956c1c023bcdb565
    resource: repo://libs/partners/runloop/pyproject.toml
  - id: openwiki-source-03a39f44d8ccfde2fd47e57a
    resource: repo://libs/partners/vercel/pyproject.toml
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-11-13T08:03:55.116Z" }
---

# Development and Release Operations

This repository is a monorepo of independently versioned Python packages. Work in the package that owns a change: packages under `libs/` have their own `pyproject.toml`, `Makefile`, and README, and there is no root `pyproject.toml`. A manifest is the published dependency contract; local `uv` source mappings make sibling packages editable during checkout development. See the [Quickstart](../quickstart.md) for initial setup and the [Testing Guide](../testing/testing-guide.md) for test selection.

## Package-local environment

Use `uv` to provision interpreters, environments, and dependencies; do not use `pip`, Poetry, Conda, or a shared manually activated environment. Each package's `requires-python` is its compatibility contract, rather than a repository-wide Python pin. Its `Makefile` is the supported task interface, so begin with `make help`.

```bash
uv tool install pre-commit
pre-commit install --install-hooks

cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Use `uv run ...` for a one-off command. The aggregate `libs/Makefile` can fan out some operations, but it does not replace package-specific targets or their requirements.

### Dependency layers

| Layer | Owner and purpose | Change rule |
| --- | --- | --- |
| Published metadata | `[project]` dependencies, extras, and `requires-python` in `pyproject.toml` | This is what downstream users install. Choose bounds and pins deliberately. |
| Workspace wiring | `[tool.uv.sources]` path mappings | These select editable sibling checkouts locally; they do not change published metadata. |
| Resolved environment | Package-local `uv.lock` | Generated development and CI state. Regenerate with `uv`; do not edit it manually. |

Editable workspace sources can conceal an incompatibility in a released dependency: local tests can consume a sibling checkout even when an end user would receive an insufficient published version. Validate a source-free or installed route when changing a public dependency boundary.

Deep Agents and ACP publish support for Python `>=3.11,<4.0`; Code and Talon publish only `>=3.14,<3.15`. These ranges describe what each package supports, not necessarily the interpreter chosen to resolve its checked-in lockfile.

## Lockfile policy and refresh

The shared CI `uv_setup` action pins uv to `0.13.6`, requires a major.minor Python input, and normally enables a pruned dependency cache. Cache invalidation is intentionally limited to `pyproject.toml`, `uv.lock`, and `requirements*.txt` in the caller's working-directory scope.

The repository has two lockfile entrypoints with different interpreter mappings:

- `.github/scripts/checks/check_lockfiles_pre_commit.py` uses Python 3.14 for top-level ACP, Code, Evals, and Talon packages; it uses 3.12 for every other discovered library, partner, or example package.
- `make -C libs lock`, `lock-bump`, and `lock-check` map only ACP to 3.14 and all other packages to 3.12.

For changed Code, Evals, or Talon manifests, use the targeted scripts below rather than treating the aggregate Make target as CI-equivalent. The reusable `Check Lockfiles` workflow disables cache setup because `uv lock --check` does not create the cache directory that setup-uv would attempt to save; it invokes the checker with no paths and therefore checks every discovered package.

### Refresh the reverse-consumer closure

A manifest change can invalidate locks beyond its owning package. `refresh_lockfiles.py` starts from the owners of its supplied manifest paths, reads each discovered package's local `[tool.uv.sources]` paths, and repeatedly includes packages that consume an already selected local package. It then runs `uv lock --directory <directory> --python <selected version>` for every package in that reverse transitive closure and prints the resulting lockfile paths. Commit every printed `uv.lock`.

```mermaid
flowchart TD
    Change["Changed pyproject.toml"] --> Owner["Select manifest owner"]
    Owner --> Sources["Read local uv source paths"]
    Sources --> Consumers["Add reverse consumers"]
    Consumers --> Closure["Repeat to a fixed closure"]
    Closure --> Resolve["Run uv lock with policy Python"]
    Resolve --> Commit["Commit each reported uv.lock"]
```

*The refresh path includes the changed package and its transitive local-source consumers.*

From the repository root:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/code/pyproject.toml
```

The refresh tool computes the closure, but the checker intentionally does not. With paths, the checker tests only package directories directly touched by those paths; with no paths, it tests all discovered packages. If no supplied path belongs to a package, it succeeds as a no-op. At the first stale lock it returns `uv lock --check`'s failure and prints an exact non-check `uv lock` repair command. Run the complete final check with no paths:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

## Code and sibling-package constraints

`deepagents-code` publishes an exact `deepagents==0.7.37` dependency, but its workspace configuration resolves `deepagents` from editable `../deepagents`. Thus local Code development does not exercise the version of the SDK that PyPI users receive. When Code begins to require a newer SDK capability, update the exact published pin and regenerate Code's lockfile with Python 3.14.

The Code manifest also maps ACP and the Daytona, Modal, QuickJS, Runloop, and Vercel integrations to editable sibling sources. Its published dependencies retain their independent constraints: the QuickJS dependency is `>=0.3.12,<0.4.0`, while Code's sandbox extras declare floors for the other partners. Do not infer release compatibility from a checkout's editable source mapping.

Talon is a Python-3.14-only package and requires `deepagents>=0.7.36` plus `deepagents-code>=0.2.7,<1.0.0`. The five partner packages each support Python `>=3.11,<4.0` and publish `deepagents>=0.7.36,<0.8.0`; they use the Deep Agents checkout only for local development. Raising Deep Agents across this boundary must remain within the partners' `<0.8.0` ceiling or be accompanied by an intentional compatibility change.

`make -C libs/code check` includes an SDK-pin comparison after its local checks. An equal or ahead pin succeeds. An older pin produces exit code 1 and is advisory to that Make target; missing, malformed, or unsupported version data produces exit code 2 and fails it. The release workflow is the hard gate against shipping Code with a pin older than the workspace SDK, unless its dangerous pin-check bypass is deliberately selected.

After an SDK release is published, the Code SDK-pin workflow creates or refreshes its marked automation PR only if Code's pin is behind. It preserves an ahead pin, updates Code's manifest and Python 3.14 lockfile together, and uses a `chore(deps):` commit so the maintenance update does not itself create an unintended Code release.

## Scheduled ecosystem floor updates

`raise_langchain_minimums.yml` is a daily 09:00 UTC maintenance workflow and can also be dispatched for one release component or `all`. Its selector is based on the release-package labels, and an optional comma-separated `dependencies` input narrows a run to exact, PEP 503-normalized distribution names. Without that input, the updater considers dependency names beginning with `langchain`, `langgraph`, `langsmith`, or `deepagents` in project dependencies, optional dependencies, and dependency groups.

The updater only raises concrete `>=` or `~=` floors. It selects the newest stable PyPI release with a non-yanked file that remains inside the existing requirement range. It preserves upper bounds, extras, and environment markers, preserves the effective ceiling of compatible-release (`~=`) requirements, never lowers a floor already ahead of PyPI, and leaves exact `==` pins alone. A versioned local path source is still a published floor worth raising, but the candidate is capped at the local checkout's declared version so local `uv` resolution remains possible; unreadable local sources, URL requirements, and the manifest's own package are not candidates.

```mermaid
flowchart TD
    Trigger["Daily schedule or manual dispatch"] --> Scope["Select release manifests and dependency scope"]
    Scope --> Fetch["Fetch stable PyPI versions"]
    Fetch --> Plan["Plan compatible lower-bound edits"]
    Plan --> Resolution["Dry-run resolution for owners and reverse dependents"]
    Resolution -->|"All resolve"| Write["Write accepted manifest edits"]
    Resolution -->|"Conflict"| Select["Keep resolvable dependency groups"]
    Select --> Write
    Write --> Locks["Regenerate affected uv.lock files"]
    Locks --> PR["Create or refresh chore deps PR"]
```

*The scheduled updater preserves dependency contracts, verifies resolution, and regenerates the full affected lock closure before proposing a maintenance PR.*

With `--check-resolution`, all proposed edits are tested together using `uv lock --dry-run` for the changed package directories and, by default, their reverse local-source consumers. If the combined update conflicts, the script first verifies that the pre-edit baseline resolves; it then tries dependency-wide edit groups in name order, keeps compatible groups, and reports skipped incompatible floors. Resolution tool failures and an unresolvable baseline fail the run rather than being treated as an update-free success. PyPI-fetch and manifest-rewrite failures are also surfaced explicitly, so a partial update cannot read as complete.

After accepted edits, the workflow regenerates every returned `dir=python` lock specification using the same interpreter policy as the lock checker, verifies that each claimed manifest actually changed, and stages both manifests and locks. The default is therefore safe closure, not just direct locks. `skip_dependent_locks` is an explicit manual exception: it regenerates only directly changed package locks and warns that reverse-dependent locks remain stale until fixed.

A narrowed request fails closed when a requested name is absent, workspace-only, an exact pin/no-floor requirement, or otherwise not raiseable; a genuinely current raiseable floor is a successful no-op. The workflow uses a `chore(deps):` PR title specifically to avoid turning maintenance into a release-please bump. It creates or refreshes a branch identified by the selected package and canonical dependency set, refuses to overwrite a branch whose tip was not authored by its bot identity, and uses an app installation token so the PR can trigger normal `pull_request` checks. On a workflow failure, it creates or updates one marked tracking issue.

The focused test suite covers range and `~=` ceiling preservation, exact-pin exclusion, requirements from all supported dependency sections, safe quoted-literal replacement, local-source version capping, transitive stale-lock discovery, selector/dispatch synchronization, narrowed-run failures, resolution conflict selection and restoration, workflow-expression validity, and refresh-branch ancestry. Use it when changing the updater or its workflow:

```bash
uv run --no-project --with "packaging>=26.2" pytest .github/scripts/tests/checks/test_raise_langchain_minimums.py
```

This workflow changes dependency constraints and lockfiles only. It does **not** change package versions, release baselines, or publish releases.

## Release boundaries and fan-out

Release Please tracks nine independently released package paths. Its present baselines are Deep Agents `0.7.37`, ACP `0.0.13`, Code `0.2.7`, Talon `0.1.0`, Daytona `0.0.9`, Modal `0.0.7`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.12`. Configuration assigns Python release handling, separate draft release PRs, and pre-1.0 version-bump behavior to every managed package. Release Please does not create the GitHub release; `release.yml` publishes after a release PR is merged.

When adding, removing, renaming, or changing the release-relevant paths of a package, reconcile `.release-please-manifest.json`, `release-please-config.json`, and release workflow routing. The manifest is release automation state: do not hand-edit an established entry except as part of adding a managed package. For a new first release at source version `0.0.1`, use a `0.0.0` manifest baseline unless `0.0.1` was previously released; otherwise Release Please treats `0.0.1` as already shipped and proposes `0.0.2`.

Keep a user-visible `feat`, `fix`, `perf`, or `revert` scoped to one owning component. A bump-worthy commit that touches lockfiles or other real files in another managed path can cause Release Please to open multiple release PRs. Put cross-package dependency and lockfile maintenance in a separate `chore(deps):` change when possible.

The release-please scope guard runs its detector and configuration from the trusted PR-base revision, fails closed if GitHub's changed-file list is incomplete, and requires `ci:allow-lockfile-release` to acknowledge detected bump-worthy lockfile-only or multi-component fan-out. See [CI labeling](ci-labeling.md) for label operations.

## Validation expectations

Run the smallest package-local checks that prove the changed behavior, then expand the validation scope when metadata crosses package boundaries.

- **Package implementation:** use the owning package's `make test` and `make lint`. Package tests distinguish network-free `tests/unit_tests/` from networked `tests/integration_tests/`; unexpected pytest warnings are errors unless explicitly accepted.
- **Manifest or local-source change:** use `refresh_lockfiles.py` on the changed manifest, commit the whole reported closure, and run the targeted checker. Before merging broad dependency work, run the checker without arguments.
- **Code SDK dependency:** update the exact pin if the change requires newer SDK behavior, regenerate the Code Python 3.14 lock, and run `make -C libs/code check`. Do not treat an editable-source test as proof of the published pin.
- **Scheduled floor updater:** retain its range/pin invariants, run its focused tests, and review both any deliberately skipped incompatible floors and the complete generated lock closure. It must not alter release versions.
- **Code command registry change:** run `make -C libs/code commands-catalog`; Code's lint target verifies that `COMMANDS.md` matches the registry.
- **Repository hygiene:** `make -C libs hygiene` applies only its explicit text hooks to the configured changed-file range. Installed pre-commit hooks also run package formatting/linting for matching changed files, lock freshness for manifest or lock changes, and Conventional Commit validation.
- **Release configuration:** verify manifest/configuration/routing correspondence and avoid a bump-worthy multi-package dependency or lockfile commit unless the fan-out is intentional and acknowledged.
