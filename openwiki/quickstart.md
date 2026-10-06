---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Find the owning package, lifecycle boundary, focused regression neighborhood, and companion guide for Deep Agents SDK, dcode, ACP, Talon, evals, partners, and repository automation changes.
tags: [deepagents, dcode, development, testing, evaluation, automation]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-06T08:06:27.683Z
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
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-06T08:06:27.683Z" }
---

# Repository Quickstart and Change Routing

Start with the package that owns the observable behavior, then follow the state or protocol boundary rather than patching a downstream UI symptom. This is a monorepo of independently versioned packages under `libs/`; each package owns its manifest, environment, Makefile, and tests. `deepagents` is the reusable harness, while `deepagents-code` (`dcode`) is the terminal product.

## Choose the owning domain

| Change affects… | Start here | Read next | First focused validation |
| --- | --- | --- | --- |
| Shared agent construction, middleware, tools, backends, memory, subagents, or approvals | `libs/deepagents/` | [Architecture overview](./architecture/overview.md) · [System Source Map](./architecture/source-map.md) | The closest SDK unit test, then each affected editable consumer |
| dcode command-line parsing, startup policy, headless mode, or process launch | `libs/code/deepagents_code/main.py` | [Run and Change a dcode Session](./workflows/run-dcode-session.md) | `libs/code/tests/unit_tests/test_main.py` |
| dcode client rendering, input, approvals, queueing, Textual screens, or status presentation | `libs/code/deepagents_code/app.py`, `client/`, or `tui/` | [dcode Client and Agent Server](./architecture/code-agent.md) · [Testing Guide](./testing/testing-guide.md) | The matching mounted widget/app test |
| dcode graph construction, tools, checkpoints, workspace runtime, MCP resources, or server-side offload | `libs/code/deepagents_code/agent.py` and `server_graph.py` | [dcode Client and Agent Server](./architecture/code-agent.md) | `test_server_graph.py` plus the changed feature’s test |
| dcode model catalog, metadata, provider credentials, selection, retry, or cache identity | `libs/code/deepagents_code/model_*.py` and server model routes | [Models and Harness Profiles](./concepts/profiles-models.md) · [Configuration Layering](./concepts/config-layering.md) | `test_model_metadata.py`, `test_model_retry.py`, and the owning config test |
| Thread resume/switching, SQLite records, writer leases, pending work, or workspace binding | `libs/code/deepagents_code/sessions.py`, `thread_ownership.py`, and workspace modules | [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md) · [Run and Change a dcode Session](./workflows/run-dcode-session.md) | `test_thread_ownership.py` and the affected session/client test |
| Hook events, hook responses, trust, or cache-expiry notifications | `libs/code/deepagents_code/hooks/` and the graph/client event boundary | [dcode Client and Agent Server](./architecture/code-agent.md) · [Run and Change a dcode Session](./workflows/run-dcode-session.md) | The focused hooks/cache test; use `test_cache_expiry.py` for interactive cache handoff |
| Slash-command metadata, aliases, queue tier, or generated command catalog | `libs/code/deepagents_code/command_registry.py` | [Run and Change a dcode Session](./workflows/run-dcode-session.md) | `test_command_registry.py`, then `make commands-catalog` and `make lint` |
| Cost totals, breakdown detail, side-question cost, or cost UI | `libs/code/deepagents_code/cost_tracking.py`, `client/`, and `tui/modals/` | [dcode Cost, Sessions, and Context Operations](./operations/cost-and-sessions.md) | The matching cost, app, and status-widget tests |
| ACP protocol translation or editor-facing session behavior | `libs/acp/` | [Agent Client Protocol Integration](./integrations/acp.md) | `libs/acp/tests/test_agent.py` |
| Long-running channels, scheduling, checkpoints, or host lifecycle | `libs/talon/` | [System Source Map](./architecture/source-map.md) | `libs/talon/tests/test_host.py` and the focused resource test |
| Provider or sandbox integration | `libs/partners/<provider>/` | [Sandbox Provider Integrations](./integrations/sandbox-partners.md) | That partner’s package tests and affected factory/configuration tests |
| Real-model trajectory, benchmark score, or Harbor sandbox benchmark | `libs/evals/` | [Run Evals](./workflows/run-evals.md) | Deterministic contract test first, then a targeted eval or Harbor run |
| CI path routing, workflow behavior, release packaging, or automation helper | `.github/workflows/`, `.github/actions/`, or `.github/scripts/` | [System Source Map](./architecture/source-map.md) · [Development, Packaging, and Releases](./operations/development.md) | The nearest helper/workflow contract test |

Use the [System Source Map](./architecture/source-map.md) when the package is clear but the state owner is not.

## dcode: locate the side of the boundary first

`dcode` and `deepagents-code` both invoke `deepagents_code:cli_main`. That CLI establishes the launch and policy path, while the loopback server owns graph execution, model/tool work, checkpoints, workspace runtime selection, and server-side hooks. The Textual client owns terminal input, rendering, approvals, queued input, and provisional presentation state.

This separation guides safe changes:

- **Do not repair server state in a widget.** Workspace fencing, model resolution, checkpoint mutation, and offload belong on the server side. Start with [dcode Client and Agent Server](./architecture/code-agent.md).
- **Keep model work server-owned.** Catalog and metadata resolution must use the bound workspace where provider credentials and environment exist. Use [Models and Harness Profiles](./concepts/profiles-models.md) for selection and retry semantics, and [Configuration Layering](./concepts/config-layering.md) for what crosses the client/server handoff.
- **Treat a thread as durable state, not just a displayed transcript.** A session change can affect the SQLite record, checkpoint binding, workspace binding, and writer lease. Follow [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md) before changing resume, switching, or deletion behavior.
- **Treat hooks as an integration boundary.** Server/graph lifecycle remains authoritative; hook payloads are validated and projected across the boundary. Follow the session workflow for recovery, headless behavior, and hook completion.
- **Keep command metadata centralized.** `COMMANDS` is the declaration point for static slash commands; aliases, queue-bypass sets, and autocomplete entries derive from it. Regenerate `COMMANDS.md` rather than editing that generated catalog.

For the runtime lifecycle, startup failure recovery, queue ordering, model retries, and cache-expiry handoff, use [Run and Change a dcode Session](./workflows/run-dcode-session.md). For the detailed client/server ownership and focused seams, use [dcode Client and Agent Server](./architecture/code-agent.md).

## Validate the narrow boundary

Repository unit tests are network-free and deterministic under `tests/unit_tests/`; `tests/integration_tests/` is for networked integration contracts. Unaccepted pytest warnings fail the suite. For dcode UI changes, mount the real Textual component and assert visible behavior—rendering, focus, screen stack, or persisted widget state—rather than private handler order.

Run from the changed package. For example:

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_server_graph.py
make lint
```

The Code Makefile disables network sockets for its unit-test target and makes `lint` check command-catalog drift. Use the test named in the routing table, then expand only when the changed contract crosses a process, package, or network boundary. `make check` is the Code package’s broader local CI target.

## Work package-locally

Use `uv` for interpreters, environments, and dependencies, and treat each package Makefile as the command authority. `uv` provisions the required interpreter automatically, so there is no global Python version to pin. Install dependencies explicitly in the package you changed; do not substitute `pip`, Poetry, or Conda.

Sibling dependencies are editable. A shared SDK interface change therefore requires consumer validation: Code maps the SDK, ACP, and relevant partners to local source paths, and CI schedules dependent package coverage for SDK changes. Consult [Development, Packaging, and Releases](./operations/development.md) before changing manifests, lockfiles, version constraints, or releases.

From `libs/`, `make lint`, `make lock`, and `make lock-check` fan out across packages. A lockfile belongs to its package or example: regenerate it after changing its manifest or resolved dependencies; do not edit it by hand.

## Evals, partners, and automation are separate change types

`libs/evals` is an end-to-end real-LLM behavioral suite: it captures trajectories and scores correctness and efficiency, with Harbor support for sandboxed benchmarks. Its development setup resolves `deepagents`, `deepagents-code`, and QuickJS from local editable paths; Harbor staging copies those checked-out packages into the sandbox project. Establish deterministic behavior first, then run an eval when the question is model behavior.

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`make evals` requires `MODEL` and runs `tests/evals`. Use the Harbor target matching the intended environment rather than treating a local unit test as a benchmark result.

Partner packages are independently versioned and own their own environment, Makefile, and tests. Adding a partner also requires repository wiring—labels, change detection, CI, release configuration, and, where applicable, Harbor and integration-test setup—so begin with `libs/partners/AGENTS.md`.

For automation, reuse existing workflow and composite-action conventions. CI detects affected package paths on pull requests; SDK changes deliberately fan out to dependent Code, Talon, ACP, eval, and partner coverage. Place a helper in its existing `.github/scripts/` domain and mirror it under `.github/scripts/tests/`.

## Continue by question

- **What is the package and dependency direction?** [Architecture Overview](./architecture/overview.md)
- **Where is a behavior’s owner and nearest regression neighborhood?** [System Source Map](./architecture/source-map.md)
- **How do dcode client and server divide work?** [dcode Client and Agent Server](./architecture/code-agent.md)
- **How do I change or recover an interactive, headless, or ACP dcode session?** [Run and Change a dcode Session](./workflows/run-dcode-session.md)
- **How do configuration and model selection cross the boundary?** [Configuration Layering](./concepts/config-layering.md) · [Models and Harness Profiles](./concepts/profiles-models.md)
- **How are threads and checkpoints made durable and fenced?** [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md)
- **Which tests should protect this change?** [Testing Guide](./testing/testing-guide.md)
- **How do I package, release, or run evals?** [Development, Packaging, and Releases](./operations/development.md) · [Run Evals](./workflows/run-evals.md)
