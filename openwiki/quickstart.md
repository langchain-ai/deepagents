---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning package, choose focused validation, and safely maintain published dependency constraints and package lockfiles. Use linked domain guides for runtime design, release operations, and detailed tests.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-13T08:03:55.116Z
sources:
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-594b8b7a84e0fb527fbafd52
    resource: repo://.github/scripts/checks/raise_langchain_minimums.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-0e2f8f3ef1c6d8452a906e09
    resource: repo://.github/workflows/bump_code_sdk_pin.yml
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-bc54cf7ca3addab1b243d7a6
    resource: repo://.github/workflows/raise_langchain_minimums.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-11-13T08:03:55.116Z" }
---

# Deep Agents Monorepo Quickstart

Start in the package that owns the published contract or runtime behavior, not where a symptom happens to surface. `libs/` is an independently versioned monorepo: each package has its own `pyproject.toml`, `Makefile`, README, source, tests, and `uv.lock`. The release manifest maintains separate baselines for the SDK, ACP, dcode, Talon, and partner distributions.

Use `uv` for interpreters, environments, and dependency synchronization; use `make help` and the package Makefile as the supported command interface. `uv` provisions the interpreter required by that package, so do not establish a global Python pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

For package relationships and published-versus-workspace dependencies, see the [architecture overview](./architecture/overview.md). For runtime ownership, use the [source map](./architecture/source-map.md); for detailed release and lock procedures, use [development operations](./operations/development.md).

## Route the task

| Change concerns… | Start in | Focused validation | Continue with |
| --- | --- | --- | --- |
| SDK graph assembly, middleware, backends, profiles, filesystem tools, or subagents | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| dcode CLI/TUI, local server, workspace policy, model/configuration resolution, headless operation, costs, or traces | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Run a dcode session](./workflows/run-dcode-session.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| ACP protocol translation or dcode ACP launch mode | `libs/acp/`, then the dcode seam when it selects or consumes the protocol | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md), [Run a dcode session](./workflows/run-dcode-session.md) |
| Talon channels, host lifecycle, MCP, history, sandbox selection, scheduling, or background delegation | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon runtime integration](./integrations/talon.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| Evaluation definitions, Harbor execution, datasets, or reports | `libs/evals/` | `make help`, then the closest package target | [Development operations](./operations/development.md), [testing guide](./testing/testing-guide.md) |
| Daytona, Modal, QuickJS, Runloop, or Vercel sandbox behavior | `libs/partners/<provider>/` | `make help`, then the owning package target | [Sandbox and partner integrations](./integrations/sandbox-partners.md), [testing guide](./testing/testing-guide.md) |
| A dependency bound, Python range, extra, or local source mapping | Changed package `pyproject.toml`, then its local consumers | Refresh the lock closure and test a direct consumer | [Architecture overview](./architecture/overview.md), [development operations](./operations/development.md) |

The usual layout reserves deterministic, network-free tests for `tests/unit_tests/` and networked contracts for `tests/integration_tests/`; unaccepted pytest warnings fail the suite. Follow the nearest existing test and local Makefile when a package uses a different layout. Run the narrowest owning suite first, then `make lint`.

## Compatibility checkpoints

- The `deepagents` SDK supports Python `>=3.11,<4.0` and is at release baseline `0.7.37`. dcode (`deepagents-code`) supports Python `>=3.14,<3.15` and publishes the exact consumer pin `deepagents==0.7.37`. An SDK capability required by dcode is therefore a cross-package change: update the published pin and Code lockfile, then validate Code rather than testing the SDK alone.
- `deepagents-talon` is an experimental Python `>=3.14,<3.15` host. It publishes `deepagents>=0.7.36` and `deepagents-code>=0.2.7,<1.0.0`; its editable `../deepagents` and `../code` sources are development wiring, not replacement constraints. It also carries LangChain/LangGraph, channel/MCP, and SQLite checkpoint dependencies. History adapters and MongoDB/PostgreSQL persistence remain opt-in extras.
- ACP and Talon `make test` disable network sockets. Talon runs WhatsApp bridge Node tests before Python tests; the Python route allows Unix sockets, has a 10-second timeout, and collects coverage. Its lint route runs Ruff checks, format verification, and `ty`.

For the full consumer, partner, release, and optional-extra matrix, read [development operations](./operations/development.md) and the [architecture overview](./architecture/overview.md).

## Dependency and lockfile workflow

Keep these three layers separate:

1. **Published metadata:** `[project]` dependencies, extras, and `requires-python` are the downstream installation contract.
2. **Checkout wiring:** `[tool.uv.sources]` may substitute editable sibling source in this repository; it does not strengthen the published constraint.
3. **Generated resolution:** a package's `uv.lock` is the concrete development and CI resolution. Regenerate it with `uv`; do not edit it by hand.

A manifest edit can invalidate the owning package and every transitive reverse consumer that has a local uv path source. From the repository root, refresh that closure and commit every printed lockfile:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/talon/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/talon/pyproject.toml
```

The refresher computes the reverse-consumer closure and locks each selected package with the checker policy interpreter. The checker is deliberately narrower: supplied paths select directly touched package directories; no paths selects every discovered package. On stale resolution it prints the non-check `uv lock` repair command. The reusable CI workflow uses the no-argument route.

Do not treat `make -C libs lock-check` as CI-equivalent for dcode, evals, or Talon: its fan-out mapping selects Python 3.14 only for ACP, while the checker selects 3.14 for ACP, dcode, evals, and Talon.

### Automated ecosystem floor maintenance

`raise_langchain_minimums.yml` runs daily at 09:00 UTC and can be manually dispatched for one release component or all of them. It considers concrete `>=` and `~=` floors for `langchain*`, `langgraph*`, `langsmith*`, and `deepagents*` requirements across normal dependencies, extras, and dependency groups; an optional `dependencies` CSV instead selects exact normalized names.

The updater raises only to the latest stable non-yanked PyPI release allowed by the existing range. It preserves upper bounds, extras, and markers, leaves exact pins (including dcode's SDK pin) alone, and never lowers an intentional ahead-of-PyPI floor. Versioned local-path dependencies still have a published floor, but their candidate is capped at the sibling checkout version so workspace locking remains resolvable.

Before writing, the workflow checks the combined changes and affected reverse dependents. It fails rather than silently treating PyPI, rewrite, or baseline-resolution failures as current; incompatible dependency-wide bumps are reported and left at their old floors. When it changes manifests, it regenerates every affected lockfile with the checker interpreter policy and creates or refreshes a `chore(deps):` PR. Review the changed bounds and package comments before merging; use the [testing guide](./testing/testing-guide.md) for the updater's focused test seam.

## Finish a focused change

1. Select the package that owns the state or published contract; run `uv sync` there and inspect `make help` if targets are unfamiliar.
2. Change the owning layer. For dependency work, review the published requirement and any local source substitution separately.
3. Run the nearest behavioral test and package lint. Test dcode after its SDK pin or compatibility changes; test Talon after SDK or dcode changes because it consumes both.
4. For a manifest edit, refresh the reverse-consumer lock closure, commit every printed `uv.lock`, and run the targeted checker. Run it without paths for a repository-wide final check.
5. Use the linked [architecture](./architecture/overview.md), [operations](./operations/development.md), [workflows](./workflows/build-a-deep-agent.md), [integrations](./integrations/talon.md), and [testing](./testing/testing-guide.md) pages for system-specific guidance.
