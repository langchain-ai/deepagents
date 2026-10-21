---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Task-routing entry point for Deep Agents maintainers. Find the package that owns a contract, run its narrowest deterministic regression seam, and then use the linked system guide for cross-package work.
tags: [deepagents, maintenance, backends, talon, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-21T08:05:42.350Z
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
generated: { by: "openwiki/0.4.2", at: "2026-10-21T08:05:42.350Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not necessarily the product where its behavior appears. `libs/` contains independently versioned packages for the reusable `deepagents` SDK, the dcode terminal product, ACP, evals, experimental Talon, and provider integrations. Each package's source, tests, `pyproject.toml`, and `Makefile` are authoritative; this page is a routing map, not a substitute for them.

For package relationships, see [Repository Architecture Overview](./architecture/overview.md). To locate a symbol or a more specific test, use the [System Source Map](./architecture/source-map.md).

## Route a change request

| Change request | Owning boundary and first place to inspect | Narrow validation first | Continue with |
| --- | --- | --- | --- |
| Build or customize an agent; change graph assembly, middleware, profiles, permissions, skills, subagents, filesystem behavior, state, or a backend | `libs/deepagents/` owns the reusable harness. It composes the graph through LangChain's `create_agent`; product-specific CLI, protocol, and channel behavior belongs outside the SDK. | Run the nearest `tests/unit_tests/` test with `cd libs/deepagents && make test TEST_FILE=<path>`, then `make lint`. | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md), [SDK Construction and Execution](./architecture/sdk-construction-execution.md), [Backends and Durable File State](./concepts/backends.md) |
| Change dcode CLI startup, terminal UI, session/workspace behavior, commands, MCP, sandbox selection, interactive approvals, or headless execution | `libs/code/` owns `deepagents-code`, launched as `dcode`, and its terminal client/server boundary. | Run the nearest `libs/code/tests/unit_tests/` file with `make test TEST_FILE=<path>`; use `make check` for the local CI gauntlet. | [Run a dcode Session](./workflows/run-dcode-session.md), [Deep Agents Code Architecture](./architecture/code-agent.md) |
| Change generic editor-facing Agent Client Protocol session translation | `libs/acp/` owns the ACP integration. dcode owns its ACP specialization, so a change to `dcode --acp` crosses both boundaries. | `cd libs/acp && make test TEST_FILE=tests/<affected_test>.py`; validate dcode when its ACP mode changed. | [Agent Client Protocol Integration](./integrations/acp.md) |
| Change channel adapters, per-channel turn admission, scheduling, runtime graph invocation, history, delivery, or approvals in the local host | `libs/talon/` owns the experimental Talon host, downstream of SDK and dcode. | Run the nearest Talon test with `make test TEST_FILE=<path>`, then `make lint`. | [Talon Runtime Integration](./integrations/talon.md), [Talon Channel Admission and Turn Ownership](./concepts/talon-channel-admission.md), [Talon Scheduling and Background Work](./concepts/talon-scheduling.md) |
| Change Talon interim updates or `send_message` | `libs/talon/deepagents_talon/messaging.py` owns invocation-scoped progress delivery; configuration is in `config.py`. | `cd libs/talon && make test TEST_FILE=tests/unit_tests/test_verbosity.py`; use `tests/unit_tests/test_messaging.py` for delivery, isolation, and cleanup. | [Talon Runtime Integration](./integrations/talon.md) |
| Change Talon runtime tools, MCP reload, approval snapshots, or operator authority | `DeepAgentRuntime` in `libs/talon/` owns tool/graph replacement and request contexts; the approval store supplies the snapshot used to build the graph. | `cd libs/talon && make test TEST_FILE=tests/test_runtime.py`; add `tests/unit_tests/test_tool_approval_runtime.py` for policy reload and concurrent-turn behavior. | [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md), [Talon Runtime Integration](./integrations/talon.md) |
| Change a provider or sandbox adapter | The corresponding `libs/partners/<provider>/` package owns the adapter. dcode and Talon consume it at their product boundary. | Run provider coverage, then the affected SDK, dcode, or Talon consumer seam. | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Change behavioral scoring, model trajectories, Harbor integration, or benchmarks | `libs/evals/` owns evaluations and Harbor integration; it is not a Release Please-managed package. | Add/run deterministic regression coverage first, then use the applicable eval or Harbor workflow. | [Run Evaluations and Benchmarks](./workflows/run-evals.md), [Testing Guide](./testing/testing-guide.md) |
| Change dependencies, locks, CI scope, versions, or publication | The changed package manifest owns the compatibility declaration; CI and release automation determine consumer coverage and release work. | Refresh affected locks and run package checks; from `libs/`, use `make lock-check` if lock scope is uncertain. | [Development, Dependencies, and Releases](./operations/development.md) |

## Two high-value regression seams

### Backend file-data compatibility

`StateBackend` and `StoreBackend` accept legacy `content: list[str]` data and normalize it for reads, listings, searches, sizes, and downloads. A read must not silently rewrite the legacy representation; a successful write or edit persists current string content. Treat both adapters as one observable compatibility contract:

```bash
cd libs/deepagents
make test TEST_FILE=tests/unit_tests/backends/test_state_backend.py
make test TEST_FILE=tests/unit_tests/backends/test_store_backend.py
```

Use the backend guide when deciding persistence scope: graph-backed state is thread-scoped, while store-backed data is scoped by its namespace factory.

### Talon request lifecycle and progress

For every Talon request, `DeepAgentRuntime` refreshes runtime tools, reads the approval snapshot, and rebuilds the graph when that snapshot changed. It pins the selected graph and request contexts for the invocation, then resets them in a `finally` block. A replacement graph is constructed before it becomes active, so a failed replacement leaves the previous usable graph in place.

`DEEPAGENTS_TALON_VERBOSITY` accepts `1`, `2`, or `3`: level `1` excludes `send_message`, level `2` shares a two-message progress budget within one turn, and level `3` has no budget. Progress is tied to the originating channel handler, rejects blank or unavailable delivery, hides transport details from the model, and is cleaned up after each invocation. This is a delivery feature, not channel admission, authorization, or sandboxing.

## Work and test in the owning package

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the command authority. `uv` provisions the appropriate interpreter, so there is no repository-wide Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/backends/test_state_backend.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The SDK and dcode `make test` targets disable network sockets; ACP does as well, and Talon's normal target first runs WhatsApp bridge Node tests before its socket-disabled Python tests.

Local sibling dependencies are editable. Validate shared-interface changes in affected consumers: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode through editable sibling sources. CI path filters likewise schedule consumer coverage for an SDK change, including dcode, ACP, Talon, and evals where applicable.

## Release handoff

`deepagents-code` pins `deepagents==0.7.29` exactly. If dcode needs new SDK behavior, update that pin in the same pull request. The current Release Please manifest versions `deepagents` at `0.7.29`, `deepagents-acp` at `0.0.13`, `deepagents-code` at `0.1.87`, and `deepagents-talon` at `0.0.9`; it also manages Daytona `0.0.8`, Modal `0.0.6`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.9`.

After releasable commits reach `main`, Release Please creates or updates a separate draft release PR per managed package. Wait for its refresh and lockfile update, review the generated changelog, then mark it ready, review/edit the curated-notes draft, and use `@release-bot apply` unless deliberately using the skip label. Merging the reviewed release PR triggers the GitHub and PyPI release workflow.

## Pick the next detailed guide

- **Find an owner, entry point, or focused test:** [System Source Map](./architecture/source-map.md)
- **Understand package boundaries and dependency direction:** [Repository Architecture Overview](./architecture/overview.md)
- **Change SDK construction:** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **Change terminal behavior:** [Run a dcode Session](./workflows/run-dcode-session.md)
- **Change Talon runtime behavior:** [Talon Runtime Integration](./integrations/talon.md)
- **Choose deterministic, integration, or behavioral coverage:** [Testing Guide](./testing/testing-guide.md)
- **Change dependencies, locks, CI, or publication:** [Development, Dependencies, and Releases](./operations/development.md)
