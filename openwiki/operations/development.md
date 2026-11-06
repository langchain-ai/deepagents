---
type: operations guide
title: Development, Dependencies, Lockfiles, and Releases
description: Develop Python packages from their owning directory, maintain package-local dependency contracts and uv locks, and follow the safeguards that connect releases to publication. Explains interpreter selection, Code's exact SDK pin, lockfile closure, and Release Please boundaries.
tags: [development, dependencies, uv, lockfiles, releases, automation]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-06T08:04:19.814Z
sources:
  - id: openwiki-source-3638ff2c170b760df159d81b
    resource: repo://.github/actions/uv_setup/action.yml
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-594b8b7a84e0fb527fbafd52
    resource: repo://.github/scripts/checks/raise_langchain_minimums.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-3943bedb80b27cdf64ea8bff
    resource: repo://.github/scripts/release/build_release_notes.py
  - id: openwiki-source-e7d931fcc9d582768b59c30e
    resource: repo://.github/scripts/release/check_sdk_pin.py
  - id: openwiki-source-1bc32c2b33ab6e58128b0319
    resource: repo://.github/scripts/tests/checks/test_raise_langchain_minimums.py
  - id: openwiki-source-74e4ee6ae7c1e3e1008eb655
    resource: repo://.github/scripts/tests/release/test_build_release_notes.py
  - id: openwiki-source-0e2f8f3ef1c6d8452a906e09
    resource: repo://.github/workflows/bump_code_sdk_pin.yml
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-bc54cf7ca3addab1b243d7a6
    resource: repo://.github/workflows/raise_langchain_minimums.yml
  - id: openwiki-source-7f3822065cb7dc086752e9d5
    resource: repo://.github/workflows/release_please_scope_check.yml
  - id: openwiki-source-46fa34397e41ebf7491c7359
    resource: repo://.github/workflows/release-please.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
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
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-11-06T08:04:19.814Z" }
---

# Development, Dependencies, Lockfiles, and Releases

This is a monorepo of independently versioned Python packages. Work from the package that owns a change: its `pyproject.toml` is the published dependency contract and its `Makefile` is the normal development interface. See the [Quickstart](../quickstart.md) for setup and the [Testing Guide](../testing/testing-guide.md) for test selection.

## Package-local development and dependency layers

`uv` provisions interpreters, environments, synchronization, and resolution. Install the repository hooks, then synchronize and test the package being changed:

```bash
uv tool install pre-commit
pre-commit install --install-hooks

cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` in the owning directory before assuming a target exists. Use `uv run ...` for deliberate one-off commands. CI's shared setup pins uv to `0.13.6`, accepts a major.minor Python input, and normally uses a pruned cache keyed only by manifests, locks, and requirements files in the configured working-directory scope.

The package manifests define distinct interpreter support: Deep Agents and ACP declare `>=3.11,<4.0`; Code and Talon declare `>=3.14,<3.15`. That published support range is separate from the interpreter used to produce a repository lockfile (see [Lockfile closure and validation](#lockfile-closure-and-validation)).

A dependency has three separate layers:

| Layer | Meaning | Safe change rule |
| --- | --- | --- |
| Published metadata | `[project]` dependencies, extras, and `requires-python` | This is the installer-visible compatibility contract. Declare versions and ranges that downstream users can install. |
| Workspace wiring | `[tool.uv.sources]` path entries | These editable sibling mappings are for monorepo development; they are not wheel metadata. |
| Resolution | Package-local `uv.lock` | This is the generated concrete closure for development and CI. Regenerate it with `uv`; do not hand-edit it. |

An editable sibling can make local tests pass while a released dependency floor is inadequate. When changing a public compatibility boundary, validate a source-free or installed-artifact path as well as the workspace path.

## Two ownership models: exact SDK pins and lower-bound floors

The distinction is intentional:

- **Exact pins are release-coupled compatibility decisions.** Code publishes `deepagents==0.7.34`, while its `[tool.uv.sources]` maps `deepagents` to `../deepagents` for local work. Raise the pin in the change that begins to depend on newer SDK behavior; editable local development cannot prove that a PyPI installation receives an adequate SDK.
- **Range floors are ecosystem-maintenance input.** The scheduled updater raises compatible `>=` and `~=` floors for `langchain*`, `langgraph*`, `langsmith*`, and `deepagents*` dependencies. It does not change `==` pins, including Code's SDK pin.

`make -C libs/code check` runs the SDK comparison, but a pin older than the workspace SDK is advisory locally: it returns status 1 and the aggregate check continues. Malformed or unsupported version data returns status 2 and fails. The release workflow is the hard boundary: it rejects a Code release whose pin is older than the workspace SDK unless `dangerous-skip-sdk-pin-check` is explicitly selected. The release-PR check comments on stale pins; prerelease pins require the `ci:ack-release-deps` acknowledgement before that check passes.

After a published `deepagents` release, `release.yml` dispatches `bump_code_sdk_pin.yml`. The workflow acts only when Code's main-branch pin is behind the workspace SDK, creates or refreshes its marked automation PR, changes the exact quoted `deepagents==...` line, and regenerates `libs/code/uv.lock` with Python 3.14. A pin ahead of the workspace version is preserved for intentional prerelease coordination. Review and merge this `chore(deps):` PR before the next Code release.

```mermaid
flowchart TD
    SDK["deepagents version advances"] --> Publish["SDK is published"]
    Publish --> PinBot["bump_code_sdk_pin workflow"]
    PinBot --> Compare["Compare Code pin and SDK"]
    Compare -->|"pin behind"| Update["Edit exact pin and Code lock"]
    Compare -->|"equal or ahead"| Keep["No manifest edit"]
    Update --> Review["Review and merge chore PR"]
    Review --> CodeRelease["Code release can pass pin gate"]
```

*The exact Code pin is a separately reviewed release contract, not input to the broad lower-bound updater.*

## Automated stable lower-bound raises

`.github/workflows/raise_langchain_minimums.yml` runs daily at 09:00 UTC and can be dispatched for one managed release package or `all`. An optional comma-separated `dependencies` input narrows the run to exact, PEP 503-normalized distribution names; it replaces prefix matching rather than augmenting it. The workflow serializes runs at package granularity and uses an App installation token for branch and PR writes so the resulting PR runs normal `pull_request` checks.

The entrypoint is:

```bash
python .github/scripts/checks/raise_langchain_minimums.py --package deepagents-code
```

The script scans `[project.dependencies]`, `[project.optional-dependencies]`, and `[dependency-groups]`. For each in-scope `>=` or `~=` requirement, it obtains stable PyPI releases with a non-yanked file, chooses the newest one still admitted by the existing requirement range, and rewrites only the strongest floor. Upper bounds, extras, and markers survive. For `~=`, it retains the original component width so the compatible-release ceiling is not inadvertently narrowed. A floor already newer than PyPI's stable release is not lowered.

A versioned local path source remains eligible: its published floor still matters to PyPI users, but the candidate is capped at the local checkout's declared version so local `uv lock` can resolve. URL requirements, the package's own name, and local sources without a readable version are excluded. The script plans all manifest rewrites before writing any, avoiding a half-written multi-manifest run.

Automation fails rather than reporting an incomplete result as current: PyPI metadata failures, failed manifest rewrites, an empty broad scope, an unknown narrowed dependency, and a narrowed dependency with no raiseable floor all fail. A genuine no-op—requested raiseable floors already current—succeeds without opening a PR. Its PR body summarizes edits and explicitly identifies dependencies or manifests that could not be raised. The workflow verifies that every reported manifest actually changed before it stages generated files, refreshes an existing bot-authored PR rather than duplicating it, and creates or updates one tracking issue if an unattended run fails.

Use `chore(deps):` for generated bump PRs. A releasable commit type inside managed package paths could otherwise cause Release Please multi-component fan-out.

## Lockfile closure and validation

A manifest update can affect more than its own lock because a consumer lock can resolve a sibling through a local `[tool.uv.sources]` path. `refresh_lockfiles.py` starts with changed manifests, finds every reverse local-source consumer transitively, and locks each selected package with the target checker's interpreter policy.

```mermaid
flowchart TD
    Changed["Changed pyproject.toml"] --> Owner["Manifest owner"]
    Owner --> Scan["Read local uv source paths"]
    Scan --> Reverse["Add direct consumers"]
    Reverse --> Repeat["Repeat to transitive closure"]
    Repeat --> Locks["uv lock per selected package"]
    Locks --> Commit["Commit every generated uv.lock"]
```

*Lock regeneration starts with the manifest owner and includes every reverse transitive local-source consumer.*

From the repository root, refresh after a manifest change and then check the resulting locks:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

The refresh helper runs `uv lock --directory … --python …` for the full closure and prints each generated lock path. The ordinary checker has a different selection rule: supplied paths cause it to check directly touched package directories, while no paths checks every discovered library, partner, and example package. For an out-of-date lock it prints the exact non-check `uv lock` command to repair it. CI invokes this no-argument checker under the `Check Lockfiles` reusable workflow, after setting up Python 3.14 with caching disabled because `uv lock --check` does not create a cache directory.

The targeted checker resolves ACP, Code, Evals, and Talon with Python 3.14, and all other discovered packages with Python 3.12. This is the policy to use for targeted manifest work. In contrast, `libs/Makefile` discovers library, partner, and example packages but maps **only ACP** to Python 3.14; its `lock`, `lock-bump`, and `lock-check` targets therefore are not a CI-equivalent replacement for Code or Talon lock validation. Use the targeted refresher/checker for a changed package and treat `make -C libs lock-check` as a convenient fan-out interface only where its Python mapping is suitable.

The lower-bound workflow writes `lock_specs` as `directory=python` pairs for changed manifests plus reverse dependents, locks each pair, and verifies the expected `uv.lock` exists. Its optional `skip_dependent_locks` mode deliberately limits regeneration to direct owners and leaves reverse-consumer locks stale; use it only when the deferred work is understood and will be repaired.

## Release boundaries

Release Please manages nine independently released components. `.release-please-manifest.json` records their current released baselines: Deep Agents `0.7.34`, ACP `0.0.13`, Code `0.2.5`, Talon `0.1.0`, Daytona `0.0.9`, Modal `0.0.7`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.11`. Its configuration makes separate draft Python release PRs, applies pre-1.0 bump behavior, and delegates publication and GitHub release creation to `release.yml`.

Release Please uses changed paths as well as Conventional Commit semantics. Keep user-facing `feat` or `fix` work contained to one managed component, and isolate cross-package dependency and lockfile maintenance in a `chore(deps):` change. The scope guard evaluates its detector and configuration from the trusted PR base revision, fails closed if GitHub's changed-file list is incomplete, and requires `ci:allow-lockfile-release` to acknowledge detected bump-worthy lockfile-only or multi-component fan-out.

After a release PR merge, publication runs only when the commit is recognized as a release-form commit and changes that component's changelog; ordinary empty commits are blocked because no changed path could fan out to every component. The publisher builds the selected source SHA, validates normal releases are reachable from `main` and package metadata matches the requested version, refuses an already published PyPI version, and carries SHA-256 sums across artifact boundaries before TestPyPI and PyPI publication. Manual recovery uses an exact 40-character source SHA whose metadata declares the requested version. Once published, a version must not be republished and its tag must not move.

Release Please serializes release-branch maintenance against pending publication across all packages: unreadable state blocks maintenance, a pending publish can defer a refresh, and recomputation waits because it covers all components.

## Release notes and focused validation

The release workflow generates GitHub release notes from the changelog at the selected release SHA, finds a reachable predecessor tag, and adds a package-scoped bounded Git log. Missing or unusable history is a warning, not fabricated history. `.github/scripts/release/build_release_notes.py` is also the recovery entrypoint: it can write to a file or GitHub output, supports `--offline` without GitHub attribution lookups, and fails loudly for an invalid package directory or unresolved SHA. Notes generation is intentionally fail-open after artifact prerequisites: publication and tagging proceed, but an empty GitHub release body must be repaired using the emitted recovery command.

Run focused validation for the layer changed:

- Package implementation: run its `make test` and `make lint`. For Code's full local gate, use `make -C libs/code check`.
- Any manifest edit: run `refresh_lockfiles.py` for the changed manifest, commit the entire reported closure, then use `check_lockfiles_pre_commit.py` (with no arguments for repository-wide validation, or affected paths for direct-touch selection).
- Shared Makefile or hygiene changes: run `make -C libs hygiene`; CI passes the PR base and head through `HYGIENE_FROM` and `HYGIENE_TO` and uses this target's explicit hook list.
- Lower-bound updater changes: run `pytest .github/scripts/tests/checks/test_raise_langchain_minimums.py`. These tests cover range preservation, exact-pin exclusion, local-version caps, narrowing failure semantics, closure, branch identity, and workflow package-option parity.
- Release-note changes: run `pytest .github/scripts/tests/release/test_build_release_notes.py`; it checks release-SHA extraction and history behavior, recovery output, and workflow/helper package-map parity.
- Managed package, release mapping, or release-label changes: update the Release Please config and manifest together with relevant workflow mappings. Consult [`.github/RELEASING.md`](https://github.com/langchain-ai/deepagents/blob/main/.github/RELEASING.md) for operational release and incident procedures.
