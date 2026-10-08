---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: Route Deep Agents maintainers from repository orientation to the owning SDK, dcode, Talon, integration, workflow, operations, and testing guidance. Use this page to choose a package, lifecycle boundary, and focused validation path before changing code.
tags: [deepagents, dcode, maintenance, development, testing, evaluation, integrations]
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-fcc71dc507b62bee0432e12e
    resource: repo://libs/code/deepagents_code/command_registry.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-d1add1f969d9ef0a3687cc02
    resource: repo://libs/code/tests/unit_tests/test_textual_patches.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
verified:
  - by: openwiki/0.4.2
    at: 2026-10-08T08:07:53.482Z
generated: { by: "openwiki/0.4.2", at: "2026-10-08T08:07:53.482Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the observable behavior, then follow its runtime or persistence boundary rather than repairing a downstream symptom. This is a monorepo of independently versioned packages under `libs/`: the `deepagents` SDK is the reusable harness; `deepagents-code` (`dcode`) is the terminal product; ACP, evals, the experimental Talon host, and provider integrations have their own packages and delivery surfaces.

## Route the change

| If the change concerns… | Begin in | Then use |
| --- | --- | --- |
| SDK graph construction, built-in tools, backends, middleware, permissions, skills, or subagents | `libs/deepagents/` | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md) for a safe change path; [SDK construction and execution](./architecture/sdk-construction-execution.md) and [middleware stack assembly](./architecture/middleware-stack.md) for lifecycle and ordering. |
| dcode startup, terminal UI, model/configuration, sessions, workspace, graph execution, or commands | `libs/code/` | [Run a Deep Agents Code Session](./workflows/run-dcode-session.md); [Deep Agents Code Architecture](./architecture/code-agent.md); then the configuration, persistence, tools, or context concept pages as appropriate. |
| Editor-facing Agent Client Protocol or dcode ACP mode | `libs/acp/` or the ACP branch in `libs/code/` | [Agent Client Protocol](./integrations/acp.md). |
| Persistent channels, pairing/admission, schedules, or background execution | `libs/talon/` | [Talon Runtime Host](./integrations/talon.md), [Talon Channels and Admission](./concepts/talon-channel-admission.md), and [Talon Scheduling and Background Work](./concepts/talon-scheduling.md). |
| MCP, a GitHub Action, or a provider/sandbox boundary | the owning integration package or `.github/` action files | [MCP Integration](./integrations/mcp.md), [GitHub Action Integration](./integrations/github-action.md), or [Sandbox and Partner Integrations](./integrations/sandbox-partners.md). |
| Real-model behavior, trajectories, Harbor jobs, or benchmark scoring | `libs/evals/` | [Run and Extend Evaluations](./workflows/run-evals.md). |
| CI selection, release wiring, package dependencies, or lockfiles | `.github/`, package `pyproject.toml`, and package `Makefile` | [Development, Dependencies, and Releases](./operations/development.md). |
| The package is known but the owner, runtime domain, or narrowest regression test is not | the owning package's public entrypoint | [Source Map and Ownership Boundaries](./architecture/source-map.md), then [Testing Guide](./testing/testing-guide.md). |

For the package roles and dependency direction, read [Repository Architecture Overview](./architecture/overview.md). The source map is the practical next stop when a symptom crosses package, process, state, or protocol boundaries.

## High-value ownership boundaries

### SDK: preserve graph assembly invariants

`create_deep_agent` in `libs/deepagents/deepagents/graph.py` is the public construction seam. It assembles the model/profile, backend, middleware, tools, subagents, skills, memory, approval interrupts, and compiled LangGraph graph; products should configure that seam rather than duplicate its assembly.

Middleware is an ordering contract. A caller-supplied middleware with a name already in the base stack replaces that entry in place. A new entry is placed after the core stack and before profile/prompt/memory tail behavior. Filesystem and synchronous-subagent middleware are protected: excluding either fails because they provide built-in filesystem/permission enforcement and the `task` delegation path. For stack changes, start with [Middleware Stack Assembly](./architecture/middleware-stack.md), prove the compiled-stack behavior in `libs/deepagents/tests/unit_tests/test_graph.py`, then add the subsystem regression test.

Use the focused concepts page for the affected extension seam:

- [Backends and Storage Routing](./concepts/backends.md) for filesystem, shell, state, sandbox, and store routing.
- [Tools, Filesystem, and Permissions](./concepts/tools-filesystem.md) and [Permissions and Human Approval](./concepts/permissions-hitl.md) for executable versus model-visible capability and approval policy.
- [Subagents and Skills](./concepts/subagents-skills.md) for delegation and pinned skill discovery.
- [Models, Profiles, and Provider Configuration](./concepts/profiles-models.md), [Context Management](./concepts/context-management.md), and [State, Checkpoints, and Sessions](./concepts/state-persistence.md) for their respective state and lifecycle boundaries.

### dcode: find the client or server owner

Both `dcode` and `deepagents-code` console scripts invoke `deepagents_code:cli_main`. dcode runs a terminal client and loopback agent server as separate processes: the client owns terminal input and presentation, while the server owns graph execution, model/tool work, checkpoints, workspace runtime selection, and server-side hooks. Keep a change on the side that owns the state; use [Deep Agents Code Architecture](./architecture/code-agent.md) before moving behavior across that boundary.

- **Startup and modes:** begin at `deepagents_code/main.py`; the session workflow distinguishes Textual, headless, and ACP execution.
- **Presentation:** begin at `deepagents_code/app.py` and the relevant `tui/` component. `_textual_patches` is applied at import time before any Textual `App` is created, so framework compatibility belongs at that boundary and needs behavioral patch coverage.
- **Configuration and models:** begin with `model_*.py` and configuration resolution, not a screen. `DEEPAGENTS_CODE_<NAME>` wins over an unprefixed environment variable; a present empty prefixed value deliberately suppresses the canonical variable. See [Deep Agents Code Configuration Layering](./concepts/config-layering.md) and [Models, Profiles, and Provider Configuration](./concepts/profiles-models.md).
- **Sessions and durable state:** trace from `sessions.py`, thread ownership, and graph checkpoints. A UI selector is not the source of truth for a durable thread; consult [State, Checkpoints, and Sessions](./concepts/state-persistence.md) and [Cost Tracking and Session Operations](./operations/cost-and-sessions.md).
- **Slash commands:** declare static commands in `command_registry.py`. `COMMANDS` is the single registry: bypass sets include aliases and autocomplete entries derive from it. Regenerate the catalog with `make commands-catalog`; `make lint` rejects generated-catalog drift.

## Validate from the owning package

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the source of truth for supported commands. `uv` provisions the appropriate Python interpreter automatically, so there is no global Python version to pin. Synchronize the package you changed, run `make help`, and use the narrowest test that observes the behavior.

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_sessions.py
make lint
```

Repository tests must be deterministic and network-free under `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. Unaccepted pytest warnings fail the suite. The Code unit target disables network sockets, while its integration target is the intentional network boundary. For UI work, test mounted behavior—rendering, focus, bindings, screen stack, or persisted state—rather than private handler order. The [Testing Guide](./testing/testing-guide.md) selects the relevant SDK, UI, async/remote, snapshot, integration, and eval layers.

Sibling dependencies are local editable development sources where configured, so an SDK interface change needs validation in its consumers as well as its own package. CI detects changed packages on pull requests; its Code and Talon filters include `libs/deepagents/**` because those products consume the SDK from editable paths.

## Evals, partners, and automation

`deepagents-evals` is an end-to-end real-LLM behavioral suite: it captures trajectories and scores correctness and efficiency, and Harbor runs sandboxed benchmarks. Its development environment resolves `deepagents`, `deepagents-code`, and QuickJS from local source paths; Harbor staging copies the checked-out packages into the sandbox project. First establish a deterministic unit-level contract, then use an evaluation to answer model-behavior questions.

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`make evals` requires `MODEL` and runs `tests/evals`. Follow [Run and Extend Evaluations](./workflows/run-evals.md) to select a model group, extend trajectory/scoring behavior, or run Harbor rather than treating a local unit test as benchmark evidence.

Each partner owns its versioning, environment, manifest, Makefile, and tests. Adding a partner also requires repository wiring for CI, labels, release, and applicable Harbor or integration surfaces; begin with `libs/partners/AGENTS.md` and continue with [Sandbox and Partner Integrations](./integrations/sandbox-partners.md). For release, dependency, or CI work, use [Development, Dependencies, and Releases](./operations/development.md) rather than attaching repository automation to a product package.

## Continue by question

- **How do the packages and layers fit together?** [Repository Architecture Overview](./architecture/overview.md)
- **Where is the implementation owner and nearest test?** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **How do I change the SDK safely?** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **How do I operate or change a dcode session?** [Run a Deep Agents Code Session](./workflows/run-dcode-session.md)
- **Which test type and command should protect this?** [Testing Guide](./testing/testing-guide.md)
- **How do package dependencies, locks, CI, and releases work?** [Development, Dependencies, and Releases](./operations/development.md)
