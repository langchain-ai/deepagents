---
type: task routing guide
title: Repository Quickstart
description: Route a Deep Agents repository change to its owning package, related architecture, integration, operations, and testing guidance. Includes package-local development entry points, compatibility boundaries, and release units.
tags: [deepagents, monorepo, development, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-27T08:05:28.881Z
sources:
  - id: openwiki-source-248c9119a9fc632bf11e2c4a
    resource: repo://.github/workflows/check_partner_bounds.yml
  - id: openwiki-source-477b456c1269748d01a9f090
    resource: repo://.github/workflows/check_release_deps.yml
  - id: openwiki-source-d70f26033a54319a6c391236
    resource: repo://.github/workflows/check_sdk_pin.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
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
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
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
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-80976c09402c30d4f886a2fa
    resource: repo://libs/talon/deepagents_talon/commands.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-75458be2d378c9102e37d6c4
    resource: repo://libs/talon/tests/cron/test_expression.py
  - id: openwiki-source-058eda257c62daed009e3f78
    resource: repo://libs/talon/tests/cron/test_jobs.py
  - id: openwiki-source-376016a439d0559796a191a0
    resource: repo://libs/talon/tests/cron/test_scheduler.py
  - id: openwiki-source-3c0ee8cc5cf93b1411287e26
    resource: repo://libs/talon/tests/cron/test_until.py
  - id: openwiki-source-18959cdb729a1a796d950993
    resource: repo://libs/talon/tests/unit_tests/test_commands.py
  - id: openwiki-source-1e472b2d67bd29dc25c17e85
    resource: repo://libs/talon/tests/unit_tests/test_context_doctor.py
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-09-27T08:05:28.881Z" }
---

# Repository Quickstart

Start in the package that owns the behavior rather than at the repository root. Deep Agents is the opinionated harness over LangChain's `create_agent()` and the LangGraph runtime; Code, ACP, Talon, evaluations, and provider adapters are separate consumers or integration boundaries. This is a task-routing map; follow the linked pages for detailed contracts.

## Route the change

| Change type | Owner and read next | First focused validation |
| --- | --- | --- |
| Reusable graph assembly, middleware, backends, skills, memory, filesystem, permissions, or SDK subagents | `libs/deepagents/`; [architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md) | Run the closest test, then `make test TEST_FILE=tests/unit_tests/<file>.py` and `make lint`. |
| dcode CLI or TUI, headless operation, sessions, workspace policy, approvals, costs, context offload, MCP loading, or product sandbox selection | `libs/code/`; [Code architecture](./architecture/code-agent.md), [run a dcode session](./workflows/run-dcode-session.md), and [testing guide](./testing/testing-guide.md) | `make test TEST_FILE=tests/unit_tests/test_<area>.py`; use `make integration_test TEST_FILE=...` only when the changed contract crosses an external boundary. |
| Editor protocol, stdio, ACP sessions, options, stream conversion, or replay | `libs/acp/`; inspect Code too if `dcode --acp` assembly changes; [ACP integration](./integrations/acp.md) | `make test TEST_FILE=tests/test_<area>.py` in ACP; test the dcode ACP path separately if its launcher or graph factory changed. |
| Talon host process, channel adapters, inbound-turn lifecycle, runtime graph assembly, tool/approval wiring, background delegation, or MCP loading | `libs/talon/`; [long-running runtime behavior](./architecture/runtime-behavior.md) for lifecycle and graph behavior, then [Talon runtime integration](./integrations/talon.md) for operator-facing channel/configuration behavior | Start with the nearest `tests/test_host.py`, `tests/test_runtime.py`, or channel test, then run `make test TEST_FILE=tests/<focused-path>.py` and `make lint`. |
| Talon persistent scheduled work: agent-facing cron tools, job storage, schedule parsing, timezone/DST calculation, `until` bounds, claiming, execution, or result delivery | `libs/talon/deepagents_talon/cron/`; [Talon scheduled work and cron semantics](./concepts/talon-scheduling.md) and [long-running runtime behavior](./architecture/runtime-behavior.md) | Run the narrow cron suite: `make test TEST_FILE=tests/cron/test_expression.py`, `tests/cron/test_until.py`, `tests/cron/test_jobs.py`, or `tests/cron/test_scheduler.py`, as applicable. |
| Talon chat command registry, `/context-doctor`, help text, host dispatch, or Discord slash-command registration | `libs/talon/deepagents_talon/commands.py` plus the host/channel code; [Talon runtime integration](./integrations/talon.md) | Run `make test TEST_FILE=tests/unit_tests/test_commands.py`, `tests/unit_tests/test_context_doctor.py`, or `tests/channels/test_discord.py` for the changed boundary. Preserve the diagnostic's read-only, counts-not-content behavior. |
| Evaluation scenario, report, model group, or Harbor execution | `libs/evals/`; [testing guide](./testing/testing-guide.md) | Run the owning eval test and the product regression test; reserve real-model evaluation for a changed trajectory or external evaluation contract. |
| Provider sandbox adapter or QuickJS behavior | `libs/partners/<provider>/`; [sandbox and partner backends](./integrations/sandbox-partners.md) | Run the adapter's package-local tests and the SDK or Code contract test that consumes it. |
| Dependency metadata, lockfile, version, release PR, or publishing automation | The changed release unit; [development, CI, and releases](./operations/development.md) | Run package checks and `make -C libs lock-check` when a lock can change. |

Talon is an experimental local host: its process owns channel adapters, cron schedulers, and an agent runtime in one event loop. Keep ordinary interactive-runtime changes separate from scheduler changes: scheduled runs persist job state and have different timing, claim, delivery, and unattended-execution invariants. Keep command work separate as well: the shared registry supplies both the host's text commands and platform registrations, while `/context-doctor` inspects configured context and the current thread without invoking the model or exposing source contents.

## Package boundaries and compatibility

`libs/` is a monorepo of independently versioned packages. Each package owns its `pyproject.toml`, `Makefile`, and README; there is no root `pyproject.toml`. Local first-party dependencies are editable, so a sibling consumer sees an in-tree SDK change during development. Use `uv` for interpreters, environments, and dependencies, and treat the current package's Makefile as the command authority. Select Python from that package's `requires-python`; there is no repository-wide interpreter pin.

| Package or group | Current release baseline or source version | Responsibility | Python requirement |
| --- | ---: | --- | --- |
| `deepagents` | `0.7.19` | SDK: `create_deep_agent`, middleware, and backends | `>=3.11,<4.0` |
| `deepagents-code` | release baseline and source: `0.1.77` | Prebuilt terminal coding agent invoked as `dcode` | `>=3.12,<4.0` |
| `deepagents-acp` | `0.0.12` | Agent Client Protocol editor integration | `>=3.11` |
| `deepagents-evals` | source version `0.0.1` | Evaluation suite and Harbor integration | `>=3.12,<3.14` |
| `deepagents-talon` | `0.0.8` | Experimental local long-running host | `>=3.12` |
| Partners | Daytona `0.0.8`; Modal `0.0.6`; Runloop `0.0.7`; Vercel `0.0.2`; QuickJS `0.3.7` | Provider and sandbox integrations | `>=3.11,<4.0` |

`deepagents-code` declares `0.1.77` in both its project metadata and its release-managed `deepagents_code/_version.py`; the Release Please manifest has the same `libs/code` baseline. Keep these source and release values aligned through the package release process rather than editing a consumer's version in isolation.

```mermaid
flowchart TD
    Code["deepagents-code and dcode"] --> SDK["deepagents SDK"]
    ACP["deepagents-acp"] --> SDK
    Evals["deepagents-evals"] --> SDK
    Evals --> Harbor["Harbor"]
    Evals --> Code
    Talon["deepagents-talon"] --> SDK
    Talon --> Code
    Partners["Partner packages"] --> SDK
```
*Published dependencies flow from consumers and adapters to the SDK, dcode, or Harbor capability they use.*

The important release-facing dependency is exact: `deepagents-code` pins `deepagents==0.7.19`. ACP has an unpinned SDK dependency; evals depends on the SDK, Code, and Harbor; Talon depends on the SDK and Code. An SDK change that Code consumes therefore requires reviewing the Code pin, its lockfile, and its release unit—not merely testing editable local sources. The five partner packages each depend on the SDK but are package-local integration boundaries, not prerequisites for ordinary SDK or Code development.

## Focused edit–test loop

Install dependencies explicitly, work in the changed package, and use its documented targets. Before preparing a pull request, install the repository's pre-commit hooks once; they run formatting, linting, lockfile, and Conventional Commit checks for changed packages.

```bash
uv tool install pre-commit
pre-commit install --install-hooks
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_server_graph.py
make lint
```

The SDK and Code `test` targets accept `TEST_FILE`; their unit runs disable network sockets except Unix sockets, run pytest in parallel, and report missing-line coverage. Code lint also validates the generated command catalog and process current-working-directory rule. ACP defaults `TEST_FILE` to `tests/` and applies a 10-second pytest timeout. Talon's `test` target runs its WhatsApp bridge Node tests before Python tests; its Python invocation disables sockets except Unix sockets, applies a 10-second timeout, and reports missing-line coverage. Use `TEST_FILE` for the focused Talon route above rather than treating the bridge tests as coverage for unrelated runtime, cron, or command changes.

For metadata or dependency work, update the affected package lock deliberately, then apply the appropriate wider check:

```bash
cd libs/code
uv lock
make check
make -C ../ lock-check
```

Code's `make check` runs lint, import checks, unit tests, extras/version consistency checks, a lockfile check, and an advisory SDK-pin check. The SDK-pin workflow warns about a stale Code pin, but the publication path enforces the pin unless an intentional bypass is acknowledged. Release dependency validation strips editable local sources and resolves changed release manifests against PyPI, so a local sibling installation does not prove that the published dependency graph resolves. The partner-bounds workflow is advisory and identifies partner SDK upper bounds that would exclude an SDK release version.

## Release-unit checklist

Release Please manages nine independent units: `deepagents`, `deepagents-acp`, `deepagents-code`, `deepagents-talon`, and the Daytona, Modal, Runloop, Vercel, and QuickJS partner distributions. `libs/evals` is a monorepo package but not a manifest release unit. The manifest baselines are `deepagents` `0.7.19`, `deepagents-acp` `0.0.12`, `deepagents-code` `0.1.77`, `deepagents-talon` `0.0.8`, Daytona `0.0.8`, Modal `0.0.6`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.7`.

Before changing release-facing metadata:

1. Confirm the changed package and its distribution/component in the [source map](./architecture/source-map.md).
2. Keep the Code exact SDK pin at the SDK version it requires; if it changes, regenerate `libs/code/uv.lock` and run `make check`.
3. Regenerate and check locks at the package or aggregate scope required by the dependency change.
4. For a release PR, treat public-index resolution and any advisory partner-bound or SDK-pin warning as release work to resolve, not as proof supplied by editable development.

## Continue in the relevant domain

- [Repository architecture overview](./architecture/overview.md) — ownership boundaries, SDK construction, and consumer roles.
- [Repository source map](./architecture/source-map.md) — public surfaces, entrypoints, focused tests, and release units.
- [Long-running runtime behavior](./architecture/runtime-behavior.md) — Talon host lifecycle, graph assembly, channels, background work, and scheduler integration.
- [Talon scheduled work and cron semantics](./concepts/talon-scheduling.md) — persistent job lifecycle, calendar/timezone behavior, expiry, and delivery.
- [Talon runtime integration](./integrations/talon.md) — channel configuration, chat commands, diagnostics, and operational behavior.
- [Build and customize a deep agent](./workflows/build-a-deep-agent.md) — model, backend, tools, profiles, middleware, state, and subagent decisions.
- [MCP servers, trust, and OAuth](./integrations/mcp.md) and [sandbox and partner backends](./integrations/sandbox-partners.md) — external tool and execution boundaries.
- [Development and release operations](./operations/development.md) and [security boundaries and runbook](./operations/security.md) — setup, locks, CI gates, publishing, and safe operation.
- [Testing guide](./testing/testing-guide.md) — deterministic seams, Talon-focused suites, and confidence runs.
- [Run a dcode session](./workflows/run-dcode-session.md) — interactive, headless, ACP, and diagnostic operations.
