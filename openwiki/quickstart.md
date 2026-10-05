---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Route Deep Agents changes to the package, dcode session owner, durable cost state, slash-command registry, and focused Textual regression seam. Use package-local development for deterministic checks and route real-model behavior changes to evals.
tags: [deepagents, dcode, development, testing, evaluation]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-05T08:14:03.003Z
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-fcc71dc507b62bee0432e12e
    resource: repo://libs/code/deepagents_code/command_registry.py
  - id: openwiki-source-f2ac9d5fb6c7c6a21f241281
    resource: repo://libs/code/deepagents_code/cost_tracking.py
  - id: openwiki-source-5f08fb59ac37d796df875608
    resource: repo://libs/code/deepagents_code/tui/modals/_cost_breakdown.py
  - id: openwiki-source-f8c8eb69e25f569e0f8a5adb
    resource: repo://libs/code/deepagents_code/tui/modals/cost_breakdown.py
  - id: openwiki-source-2c41bc0b19795204a48854ee
    resource: repo://libs/code/deepagents_code/tui/widgets/status.py
  - id: openwiki-source-11d6c59d85493653aee76558
    resource: repo://libs/code/tests/unit_tests/test_app.py
  - id: openwiki-source-8574be7f7f29e3e1dd328837
    resource: repo://libs/code/tests/unit_tests/test_js_cost_tracking.py
  - id: openwiki-source-bfb9f0ea03fdda310b93ef72
    resource: repo://libs/code/tests/unit_tests/tui/widgets/test_status.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-05T08:14:03.003Z" }
---

# Repository Quickstart and Change Routing

Start with the package that owns the observed behavior. This repository is a monorepo of independently versioned packages under `libs/`; each package has its own environment, manifest, Makefile, and tests. The core `deepagents` SDK owns reusable graph behavior, while `deepagents-code` (`dcode`) owns the terminal product and its interactive session surfaces.

## Route the change

| Change affects… | Primary owner / starting point | Read next | Focused validation |
| --- | --- | --- | --- |
| Agent construction, middleware, tools, backends, memory, subagents, or approvals shared by products | `libs/deepagents` | [Code agent architecture](./architecture/code-agent.md) when the change reaches dcode | The closest SDK unit test, then the consuming package when its contract changes |
| dcode startup, connection/recovery state, transcript, status bar, prompts, session switching, or Textual modal behavior | `libs/code/deepagents_code/app.py` — `DeepAgentsApp` is the client/session coordinator | [Run and change a dcode session](./workflows/run-dcode-session.md) · [dcode client and agent server](./architecture/code-agent.md) | `libs/code/tests/unit_tests/test_app.py` or the affected widget/modal test |
| dcode graph/server construction, model execution, or checkpointed agent behavior | `libs/code/deepagents_code/agent.py` and server-facing client code | [dcode client and agent server](./architecture/code-agent.md) | A graph/cost/client unit test plus the relevant app-level seam |
| Thread-wide estimated cost, token/cost categories, subagent transfers, or pricing persistence | `libs/code/deepagents_code/cost_tracking.py`; presentation is in `app.py` and `tui/modals/` | [Cost and session operations](./operations/cost-and-sessions.md) | `test_cost_tracking.py`, `test_js_cost_tracking.py`, and modal/app tests as applicable |
| A slash command’s description, aliases, autocomplete metadata, queue behavior, or failed-startup recovery eligibility | `libs/code/deepagents_code/command_registry.py` | [Run and change a dcode session](./workflows/run-dcode-session.md) | `test_command_registry.py`, plus `test_app.py` if routing behavior changes |
| ACP/editor protocol translation or editor session behavior | `libs/acp` | [System source map](./architecture/source-map.md) | The nearest ACP test before an editor smoke test |
| Real-model trajectory, tool-use quality, benchmark score, or Harbor sandbox benchmark behavior | `libs/evals` | [Run evals](./workflows/run-evals.md) | A targeted eval or Harbor run, while retaining deterministic coverage for the changed contract |
| A provider or sandbox integration | `libs/partners/<provider>` | [System source map](./architecture/source-map.md) | Partner test and consuming factory/configuration coverage |

The [System Source Map](./architecture/source-map.md) is the next stop when the product boundary is clear but the actual state owner is not.

## dcode interactive-session map

`DeepAgentsApp` is the client-side coordinator, not the durable accounting owner. A successful server startup posts `ServerReady`: the app settles connection state, installs the agent/server information, refreshes MCP and the status-bar model, and schedules the ordered session-start sequence. That sequence restores resumed history, handles startup work, and only then dispatches initial or queued user input. Reconnects must not rehydrate an already populated transcript.

When changing a recovery path, keep the split explicit:

- **Server/client readiness:** change the `ServerReady` or startup-failure path in `app.py`; verify model/status recovery and queue draining in `test_app.py`.
- **Slash-command policy:** declare normal commands once in `COMMANDS`. The registry derives aliases and queue-bypass sets, and autocomplete derives its entries from that registry. Do not add a second hard-coded command-metadata list.
- **Failed-startup escape hatches:** `/install`, `/reload`, and `/update` remain normally queue-bound but may bypass a failed-startup queue only when no agent, shell, or modal command is running. This lets a user repair the configuration that prevented the server from starting without allowing a repair operation to replace a running session.

### Cost state and the clickable breakdown

The graph owns durable main-thread cost: private checkpoint channels hold the cumulative `_session_cost_usd` and structured `_session_cost_breakdown`. `CostTrackingMiddleware` records durable model cost, and completed nested-agent cost is checkpointed locally then transferred to the owning parent graph. The client reads streamed or restored totals; it may show a keyed provisional estimate during a turn, but an authoritative server total supersedes it rather than becoming a second persistent ledger.

The remote client reconciles checkpointed graph cost with separately persisted side-question cost for presentation. The status bar renders the displayed total; only the marked cost span responds to a single left click and opens `action_open_cost_breakdown`. The modal is intentionally read-only: it rebuilds from the current client-held authoritative total/detail, refreshes while open, and copies plain text. An entire-thread table is available only for a versioned, historically complete breakdown; otherwise the user keeps the headline total and receives no empty detail modal. In a valid table, unavailable category detail and unpriceable requests are explicitly marked as partial rather than treated as zero.

For implementation and operational semantics, use [Cost and Session Operations](./operations/cost-and-sessions.md). For the broader client/server boundary, use [dcode Client and Agent Server](./architecture/code-agent.md).

## Test the behavior at the boundary

Use mounted Textual tests for interaction and focus behavior, not private call-order assertions. The narrowest seams for an interactive-session change are:

| Contract changed | Start with |
| --- | --- |
| Server-ready transition, status-model refresh, resumed-history idempotence, or queue recovery | `libs/code/tests/unit_tests/test_app.py` |
| Footer click hit testing and hidden-cost behavior | `libs/code/tests/unit_tests/tui/widgets/test_status.py` |
| Opening, live refresh, duplicate-modal prevention, focus restoration, or unavailable cost details | `libs/code/tests/unit_tests/test_app.py` (`TestFooterCostBreakdown`) |
| Durable graph cost, historical-completeness rules, or parent/subagent transfer | `libs/code/tests/unit_tests/test_cost_tracking.py` |
| QuickJS JavaScript subagent cost surviving checkpoint/replay and appearing in the entire-thread breakdown | `libs/code/tests/unit_tests/test_js_cost_tracking.py` |
| Registry classification, recovery-command membership, aliases, or autocomplete derivation | `libs/code/tests/unit_tests/test_command_registry.py` |

Repository tests should be network-free and deterministic under `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. Unaccepted pytest warnings fail the suite.

## Develop package-locally

Use `uv` for interpreters, environments, and dependencies, and treat each package’s Makefile as the command authority. `uv` provisions the required interpreter, so do not globally pin Python or replace the package workflow with `pip`, Poetry, or Conda. Install dependencies explicitly in the changed package; local sibling dependencies are editable, so validate affected consumers when a shared interface changes.

For a dcode UI/session change:

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_app.py
make lint
```

Use a narrower test file when the table above identifies one. The Code Makefile runs unit tests with network sockets disabled; its `lint` target also checks generated command-catalog drift, so run it after editing `command_registry.py`. From `libs/`, `make lint`, `make lock`, and `make lock-check` fan out across packages and are appropriate for cross-package work.

## Route evaluation work separately

`libs/evals` is an end-to-end behavioral evaluation suite: it runs an agent against a real LLM, captures its trajectory, and scores correctness and efficiency. It also contains Harbor integration for sandboxed benchmarks. Its package manifest uses local editable sources for `deepagents`, `deepagents-code`, and QuickJS during development, so an eval can exercise a checkout change without publishing it.

Use deterministic unit tests to establish the contract first, then select an eval when the question is model behavior rather than deterministic UI or state behavior:

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`make evals` requires `MODEL` and runs `tests/evals` with a LangSmith test-suite name. Harbor targets stage local SDK, Code, ACP, and QuickJS sources into its sandbox project before invoking the selected backend; use the target matching the intended environment rather than treating a local unit test as a benchmark result. See [Run Evals](./workflows/run-evals.md) for selection and operational setup.

## Continue by question

- **Where are package boundaries and major entrypoints?** [System Source Map](./architecture/source-map.md)
- **How do the dcode client and server divide responsibilities?** [dcode Client and Agent Server](./architecture/code-agent.md)
- **How do session cost, durable accounting, and the breakdown work?** [Cost and Session Operations](./operations/cost-and-sessions.md)
- **How do I run or change an interactive terminal session?** [Run and Change a dcode Session](./workflows/run-dcode-session.md)
- **Which focused tests protect the behavior?** [Testing Guide](./testing/testing-guide.md)
- **How do I run real-model evals or Harbor benchmarks?** [Run Evals](./workflows/run-evals.md)
