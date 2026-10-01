---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Task-oriented map for locating the Deep Agents SDK, dcode terminal client, ACP bridge, Talon host, integrations, tests, and release-managed packages. Start with the owner of the observable behavior and validate at its narrowest boundary.
tags: [deepagents, navigation, development, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-01T08:06:30.386Z
sources:
  - id: openwiki-source-37e02a57730563a4b4de1690
    resource: repo://.github/LAYOUT.md
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-1d73b3e2b56b5f0d27273379
    resource: repo://libs/code/README.md
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-01T08:06:30.386Z" }
---

# Repository Quickstart and Change Routing

This is a monorepo of independently versioned packages under `libs/`, rather than one root Python project. Start with the package that owns the behavior a user can observe; then use its README, `pyproject.toml`, `Makefile`, source, and nearby tests. Reusable agent-harness behavior belongs in the SDK, terminal-product behavior in dcode, editor-protocol translation in ACP, and local channels, schedules, and host policy in Talon.

## Choose the owning domain

| If the task changes… | Owner and responsibility | Read next |
| --- | --- | --- |
| Agent construction, middleware, backends, filesystem tools, memory, skills, subagents, or approvals | **`libs/deepagents`** — the model-agnostic, LangGraph-based SDK harness. It provides bundled subagents, pluggable filesystem backends, context management, persistent memory, human intervention, skills, and tools/MCP support. | [Architecture overview](./architecture/overview.md) · [Build or modify an agent](./workflows/build-a-deep-agent.md) · [Source map](./architecture/source-map.md) |
| Terminal CLI/TUI, coding-agent configuration, sessions, MCP, sandbox selection, cost, or tool approval UX | **`libs/code`** — `deepagents-code` / `dcode`, the pre-built terminal coding agent layered on the SDK, with an interactive TUI, conversation resume, remote sandboxes, persistent memory, skills, headless mode, and tool-call approvals. | [dcode architecture](./architecture/code-agent.md) · [Run a dcode session](./workflows/run-dcode-session.md) |
| Editor session events, ACP modes, cancellation, session loading, or model switching | **`libs/acp`** — the ACP adapter between a Python Deep Agent and an ACP-capable editor. It can also expose dcode as a stdio ACP server. | [ACP integration](./integrations/acp.md) |
| Local host startup, channels, pairing, turns, interruption, background delivery, cron, history, MCP configuration, or sandbox lifecycle | **`libs/talon`** — an experimental local host whose one event loop owns channel adapters, cron schedulers, and the agent runtime. | [Talon runtime behavior](./architecture/runtime-behavior.md) · [Talon integration](./integrations/talon.md) |
| Provider-specific sandbox or execution integration | **`libs/partners/*`** — independently released provider integrations for Daytona, Modal, Runloop, Vercel, and QuickJS. | [Sandbox partner integrations](./integrations/sandbox-partners.md) |
| Model trajectory, tool-use quality, or benchmark score | **`libs/evals`** — real-LLM evaluation that captures tool calls, file mutations, and final responses, scores correctness and efficiency, and includes Harbor-backed benchmarks. | [Run evaluations](./workflows/run-evals.md) |
| CI workflows, labels, issue forms, release automation, or workflow helper scripts | **`.github/`** — repository automation and release wiring. | [Development, packaging, and releases](./operations/development.md) |

The [source map](./architecture/source-map.md) is the next stop when the public owner is clear but the source entrypoint or focused test seam is not.

## Route the change to a focused regression

| Change class | Validate first | Escalate only when needed |
| --- | --- | --- |
| SDK graph, middleware, backend, or filesystem behavior | A deterministic SDK unit test at the middleware, backend, tool, or compiled-agent boundary. | Use the fake-model end-to-end suite when the behavior crosses the compiled graph. |
| dcode CLI, configuration, persistence, TUI, approvals, MCP, or sandbox routing | A Code package test that exposes the command, server, configuration, or persistence contract. | Add a real-provider test only for the integration contract. |
| ACP protocol behavior | ACP fake-client or session tests, because the output contract is the ACP session stream. | Run an editor-facing smoke test when changing process or stdio integration. |
| Talon channel, host, cancellation, history, scheduler, OAuth/MCP, or sandbox behavior | A Talon test with fake channels, runners, clocks, or durable stores. Keep adapter parsing separate from host composition. | Use a live channel/provider test only when its external contract changed. |
| Sandbox partner behavior | The owning partner-package test plus the consuming factory or configuration wiring test. | Run provider integration coverage only when credentials and the external service are intentionally in scope. |
| Model quality or benchmark behavior | A focused real-LLM evaluation that captures trajectory and score. | Keep deterministic owner-boundary regression coverage alongside the eval. |
| CI, labels, releases, or scripts | The affected workflow/helper contract test and package or lock check. | Run the applicable workflow only after its local contract is covered. |

Repository tests belong in `tests/unit_tests/` when they are network-free and deterministic; `tests/integration_tests/` is for networked integration contracts. Test observable behavior rather than private call order, and treat an unaccepted pytest warning as a failure. See the [Testing Guide](./testing/testing-guide.md) for package-specific commands and high-risk filesystem and Talon seams.

## Local development loop

`uv` manages interpreters, environments, and dependencies; do not substitute `pip`, Poetry, or Conda. Each package's `Makefile` is its command authority, and `uv` provisions the required interpreter, so do not establish a global Python pin. The target package’s `requires-python` remains authoritative: Deep Agents and ACP support `>=3.11`, while Code and Talon require `>=3.12`.

```bash
cd libs/deepagents
uv sync --all-groups
make help
make test TEST_FILE=tests/unit_tests/test_specific.py
make lint
```

Run the equivalent commands from the package being changed. Use `uv run ...` for a one-off direct command. From `libs/`, use fan-out targets such as `make lint` or `make lock-check` only for cross-package work. Package-local sibling dependencies are editable, so a shared-interface change must be checked in its consumers as well.

## Operational and security routing

- **dcode trust boundary:** dcode trusts the directory from which it runs by default. Approval prompts apply to model-requested tool calls, but project artifacts can be read before approval. Use a remote sandbox for an untrusted repository.
- **Talon trust boundary:** Talon is alpha software, not a production or multi-tenant security boundary. Channel access is effectively access to the operator's agent, credentials, MCP tools, and host resources. Read the [security guide](./operations/security.md) before changing access, credential, MCP, or sandbox behavior.
- **Talon execution choice:** host shell and file tools run locally unless `DEEPAGENTS_TALON_SANDBOX` selects a sandbox. A sandbox startup failure exits rather than silently falling back to host execution; MCP tools and channel media still run on the host. See [Talon integration](./integrations/talon.md).
- **State and scheduled work:** changes to checkpoints, archives, conversations, or cron jobs cross durable lifecycle boundaries. Start with [state and persistence](./concepts/state-persistence.md) and [Talon scheduling](./concepts/talon-scheduling.md), then use fixed-clock and temporary-store tests.

## Releases, dependencies, and repository automation

The release manifest currently manages the SDK, ACP, Code, Talon, and each partner distribution independently. A package manifest declares its distribution name, runtime range, dependencies, extras, and entrypoints; update its lockfile through the package workflow rather than editing it by hand. For compatibility and editable dependency relationships, use [Development, Packaging, and Releases](./operations/development.md).

A new partner package needs more than its own environment, manifest, Makefile, and tests: wire its issue forms, dependency automation, labels, CI/change detection, release management, and applicable integration and secret matrices. Repository automation is organized under `.github/` into workflows, composite actions, scripts, issue templates, and release/labeling documentation. Its labeling workflows apply package, integration, priority, and PR metadata labels; helper-script tests mirror their production domain layout.

## Continue by question

- **Where is the public entrypoint, state owner, or source/test seam?** [System Source Map](./architecture/source-map.md)
- **How do the SDK, dcode, ACP, and Talon relate at runtime?** [System Architecture Overview](./architecture/overview.md)
- **How do I build or change an agent safely?** [Build and Modify a Deep Agent](./workflows/build-a-deep-agent.md)
- **How do I configure and operate the local host?** [Talon Runtime Integration](./integrations/talon.md)
- **How do I test the changed behavior?** [Testing Guide](./testing/testing-guide.md)
- **How do packaging, locks, and releases work?** [Development, Packaging, and Releases](./operations/development.md)
