---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Task-routing entry point for Deep Agents maintainers. Find the package that owns a contract, run its narrowest deterministic regression seam, and then use the linked system guide for cross-package work.
tags: [deepagents, maintenance, backends, talon, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-19T08:10:20.273Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-c972622237a22631e36f3625
    resource: repo://libs/deepagents/deepagents/backends/utils.py
  - id: openwiki-source-fb022ddbcc554eaabedfa8cd
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_state_backend.py
  - id: openwiki-source-e11958a6a68a079be57552fc
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_store_backend.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-5b9a69640ff3d94216c614ce
    resource: repo://libs/talon/deepagents_talon/messaging.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-4d6726e17c8a0c78539a7d33
    resource: repo://libs/talon/tests/test_runtime.py
  - id: openwiki-source-1698129adea358c8813da5a5
    resource: repo://libs/talon/tests/unit_tests/test_messaging.py
  - id: openwiki-source-242a21b2da46507f58415265
    resource: repo://libs/talon/tests/unit_tests/test_tool_approval_runtime.py
  - id: openwiki-source-1e7998c9f5a94d22f3819721
    resource: repo://libs/talon/tests/unit_tests/test_verbosity.py
generated: { by: "openwiki/0.4.2", at: "2026-10-19T08:10:20.273Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract—not necessarily the product where its behavior is visible. `libs/` contains independently versioned packages for the reusable `deepagents` SDK, dcode, ACP, evals, Talon, and provider integrations. Package-local source, tests, `pyproject.toml`, and `Makefile` are authoritative; this page is a router, not a replacement for them.

For package boundaries, see [Repository Architecture Overview](./architecture/overview.md). For an entry point or a more granular test seam, use the [Source Map](./architecture/source-map.md).

## Route the task

| If the change concerns… | Contract owner and entry seam | Start with this deterministic regression seam | Then read |
| --- | --- | --- | --- |
| SDK file-data compatibility, including legacy `content: list[str]` | `libs/deepagents/`; shared normalization begins in `deepagents/backends/utils.py`, while `StateBackend` and `StoreBackend` own their storage adapters. Reads, listing/search sizing, and download normalize without persisting a conversion; successful writes/edits migrate to string content. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/backends/test_state_backend.py`; repeat with `tests/unit_tests/backends/test_store_backend.py`. | [Backends and Durable File State](./concepts/backends.md), [Testing Guide](./testing/testing-guide.md) |
| Reusable graph assembly, middleware, filesystem policy, skills, state, or delegation beyond that compatibility seam | `libs/deepagents/` owns the SDK harness. It composes its graph through LangChain `create_agent`; do not put product transport behavior here. | Nearest SDK `tests/unit_tests/` test, then `make lint`. | [SDK construction and execution](./architecture/sdk-construction-execution.md), [Source Map](./architecture/source-map.md) |
| Terminal CLI, Textual UI, session/workspace policy, commands, interactive approvals, retries, or sandbox choice | `libs/code/` owns dcode, the terminal client and agent-server surface. | Nearest `libs/code/tests/unit_tests/` seam, then `make lint` or `make check`. | [Run a dcode Session](./workflows/run-dcode-session.md), [Code Agent Architecture](./architecture/code-agent.md) |
| Editor-facing Agent Client Protocol behavior or `dcode --acp` | `libs/acp/` owns generic protocol/session translation; dcode owns its terminal-specific ACP specialization. | `cd libs/acp && make test TEST_FILE=tests/<affected_test>.py`; validate dcode too when its ACP mode changes. | [Repository Architecture Overview](./architecture/overview.md) |
| Talon verbosity, interim updates, `send_message`, or narration before tools | `libs/talon/`; configuration is in `deepagents_talon/config.py`, and invocation-scoped progress delivery is in `messaging.py`. Verbosity `1` omits progress updates, `2` shares a two-update budget per turn, and `3` is unrestricted. | `cd libs/talon && make test TEST_FILE=tests/unit_tests/test_verbosity.py`; use `tests/unit_tests/test_messaging.py` for delivery, isolation, and cleanup. | [Talon Runtime Integration](./integrations/talon.md), [Talon Channel Admission](./concepts/talon-channel-admission.md) |
| Talon graph refresh/rebuild, MCP/runtime tools, approval policy, or operator authority | `libs/talon/`; `DeepAgentRuntime` owns graph construction and invocation-scoped context, while the approval store supplies the snapshot used to build a graph. | `cd libs/talon && make test TEST_FILE=tests/test_runtime.py`; use `tests/unit_tests/test_tool_approval_runtime.py` for snapshot concurrency, fail-closed reload, and unattended-request behavior. | [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md), [Talon Runtime Integration](./integrations/talon.md) |
| Long-running channel lifecycle, scheduling, persistence, or channel delivery | `libs/talon/` owns the experimental local host and channel adapters, downstream of SDK and dcode. | Nearest Talon test, then `make lint`. | [Talon Runtime Integration](./integrations/talon.md), [Talon Channel Admission](./concepts/talon-channel-admission.md) |
| A sandbox provider or provider-specific integration | The relevant `libs/partners/<provider>/` package owns the provider; dcode and Talon consume it at their product boundary. | Provider tests, then affected dcode, Talon, or SDK consumer coverage. | [Sandbox Partners](./integrations/sandbox-partners.md) |
| Real-model quality, trajectories, scoring, or a sandbox benchmark | `libs/evals/` owns behavioral evaluation and Harbor integration; it is not Release Please-managed. | Add/run deterministic regression coverage first; then use `make evals MODEL=<id>` or the applicable Harbor job. | [Testing Guide](./testing/testing-guide.md) |
| Dependency bounds, editable siblings, locks, CI, or publication | The changed manifest owns its compatibility contract; CI and release automation determine consumer and publication work. | Refresh affected locks, run package checks, and use `cd libs && make lock-check` when lock scope is uncertain. | [Development, Dependencies, and Releases](./operations/development.md) |

## Guardrails for the two current high-change areas

- **Backend compatibility is a shared observable contract.** Keep legacy normalization centralized; test both state and store adapters because their persistence and graph-context mechanics differ. `StateBackend` is thread-local graph state and must run in graph execution; `StoreBackend` is cross-thread storage scoped by its namespace factory.
- **Talon state is scoped to an invocation.** `DeepAgentRuntime.invoke()` refreshes tools, snapshots approvals, rebuilds when the persisted approval snapshot changes, then binds the chosen graph and request contexts. It resets approval, authorization, progress, graph, and pending-result context even when invocation fails. A reload must construct a replacement graph before replacing the usable one, so a failed rebuild leaves the old graph active.
- **Progress is delivery, not authorization.** `send_message` has a host-fixed destination and returns a generic failure to the model when delivery cannot occur. It enters the exact-name Talon approval policy when protected; neither that approval nor verbosity is sandboxing or channel admission.

## Work and test in the owning package

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the command authority. `uv` provisions the appropriate interpreter, so there is no repository-wide Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/backends/test_state_backend.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. ACP and Talon's normal `make test` targets disable network sockets; Talon's target runs WhatsApp bridge Node tests before Python tests.

Local sibling dependencies are editable; consequently, a shared-interface change must be validated in affected consumers, and CI path filters schedule dependent coverage when the SDK changes. Package manifests establish the local development dependency direction: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode through editable sibling sources.

## Release handoff

`deepagents-code` pins `deepagents==0.7.28` exactly. A change requiring new SDK behavior must update that pin in the same pull request. The release manifest versions `deepagents` at `0.7.28`, `deepagents-acp` at `0.0.13`, `deepagents-code` at `0.1.86`, and `deepagents-talon` at `0.0.9`; it also versions Daytona `0.0.8`, Modal `0.0.6`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.9`.

Release Please creates a separate draft release PR for each managed package after releasable commits reach `main`. Review the generated changelog, draft/review/edit curated notes, and use `@release-bot apply` unless intentionally skipping curation; merging the reviewed release PR triggers its GitHub and PyPI publication workflow.

## Follow the detailed guide

- **Find an owner, entry point, or focused test:** [Source Map](./architecture/source-map.md)
- **Change shared backend representation or persistence scope:** [Backends and Durable File State](./concepts/backends.md)
- **Change Talon runtime, progress, configuration, or graph lifecycle:** [Talon Runtime Integration](./integrations/talon.md)
- **Change Talon approvals or operator authority:** [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md)
- **Choose deterministic, integration, or real-model coverage:** [Testing Guide](./testing/testing-guide.md)
- **Change dependencies, locks, CI, or releases:** [Development, Dependencies, and Releases](./operations/development.md)
