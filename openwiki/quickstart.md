---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Task-oriented map for locating the independently released Deep Agents packages, their current Code and Talon versions, and the focused architecture, operations, integration, workflow, concept, and test references. Start with the owner of observable behavior and validate its narrowest boundary.
tags: [deepagents, navigation, development, testing, releases]
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
  - id: openwiki-source-6c2e9cfaa20096e021221d47
    resource: repo://libs/code/CHANGELOG.md
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
  - id: openwiki-source-e2a176528c4d510dcc417820
    resource: repo://libs/talon/CHANGELOG.md
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-02T08:06:05.669Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-10-02T08:06:05.669Z
---

# Repository Quickstart and Change Routing

This is a monorepo of independently versioned packages under `libs/`, not one root Python project. Start with the package that owns the behavior a user observes, then read its README, `pyproject.toml`, `Makefile`, source, and nearby tests. Shared agent-harness behavior belongs in the SDK; terminal-product behavior in dcode; editor-protocol translation in ACP; and local channels, schedules, and host policy in Talon.

## Start with the owning domain

| If the task changes… | Owner and responsibility | Go to |
| --- | --- | --- |
| Agent construction, middleware, backends, filesystem tools, memory, skills, subagents, or approvals | **`libs/deepagents`** — the model-agnostic, LangGraph-based SDK harness. It provides bundled subagents, pluggable filesystem backends, context management, persistent memory, human intervention, skills, and tools/MCP support. | [Architecture overview](./architecture/overview.md) · [Build or modify an agent](./workflows/build-a-deep-agent.md) · [Source map](./architecture/source-map.md) |
| Terminal CLI/TUI, coding-agent configuration, sessions, MCP, sandbox selection, cost, or tool-approval UX | **`libs/code`** — `deepagents-code` / `dcode`, the pre-built terminal coding agent layered on the SDK, with an interactive TUI, conversation resume, remote sandboxes, persistent memory, skills, headless mode, and tool-call approvals. | [dcode client and agent server](./architecture/code-agent.md) · [Run a dcode session](./workflows/run-dcode-session.md) · [Cost, session, and context operations](./operations/cost-and-sessions.md) |
| Editor session events, ACP modes, cancellation, session loading, or model switching | **`libs/acp`** — the ACP adapter between a Python Deep Agent and an ACP-capable editor; it can also expose dcode as a stdio ACP server. | [ACP integration](./integrations/acp.md) · [Source map](./architecture/source-map.md) |
| Local host startup, channels, pairing, turns, interruption, background delivery, cron, history, MCP configuration, or sandbox lifecycle | **`libs/talon`** — an experimental local host whose single event loop owns channel adapters, cron schedulers, and the agent runtime. | [Talon runtime behavior](./architecture/runtime-behavior.md) · [Talon runtime integration](./integrations/talon.md) |
| Talon checkpoint/history records or durable scheduled work | **`libs/talon`** — persistence and scheduler lifecycle are host-owned rather than dcode-client state. | [State, checkpoints, and persistent records](./concepts/state-persistence.md) · [Talon scheduling and cron records](./concepts/talon-scheduling.md) |
| Talon sender admission, pairing, Slack thread context, or revocation | **`libs/talon`** — channel admission is host policy, not a model prompt. | [Talon channel admission and conversation identity](./concepts/talon-channel-admission.md) · [Talon runtime integration](./integrations/talon.md) |
| Provider-specific sandbox or execution integration | **`libs/partners/*`** — independently released integrations for Daytona, Modal, Runloop, Vercel, and QuickJS. | [Sandbox provider integrations](./integrations/sandbox-partners.md) |
| Model trajectory, tool-use quality, or benchmark score | **`libs/evals`** — real-LLM evaluation captures tool calls, file mutations, and final responses, scores correctness and efficiency, and includes Harbor-backed benchmarks. | [Run evaluations](./workflows/run-evals.md) |
| CI workflows, labels, issue forms, release automation, or workflow helper scripts | **`.github/`** — repository automation and release wiring. | [Development, packaging, and releases](./operations/development.md) |

When the public owner is clear but the entrypoint or first regression seam is not, use the [System Source Map](./architecture/source-map.md).

## Validate at the narrowest boundary

| Change class | Validate first | Escalate only when needed |
| --- | --- | --- |
| SDK graph, middleware, backend, or filesystem behavior | A deterministic SDK test at the middleware, backend, tool, or compiled-agent boundary. | Use the fake-model end-to-end suite when behavior crosses the compiled graph. |
| dcode CLI, configuration, persistence, TUI, approvals, MCP, or sandbox routing | A Code package test for the command, server, configuration, persistence, prompt, or rendered UI contract. | Add a real-provider test only for the external integration contract. |
| ACP protocol behavior | ACP fake-client or session tests, because the output contract is the ACP session stream. | Run an editor-facing smoke test when changing process or stdio integration. |
| Talon checkpoint, channel, pairing, host, history, scheduler, OAuth/MCP, or sandbox behavior | A Talon test with fake channels, runners, clocks, temporary durable stores, or a fake driver. | Use a live channel/provider test only when its external contract changed. |
| Sandbox partner behavior | The owning partner-package test plus the consuming factory or configuration-wiring test. | Run provider integration coverage only when credentials and the external service are intentionally in scope. |
| Model quality or benchmark behavior | A focused real-LLM evaluation that captures trajectory and score. | Keep deterministic owner-boundary regression coverage alongside the eval. |
| CI, labels, releases, or scripts | The affected workflow/helper contract test and package or lock check. | Run the applicable workflow only after its local contract is covered. |

Network-free, deterministic tests belong in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. Test observable behavior rather than private call order, and treat an unaccepted pytest warning as a failure. The [Testing Guide](./testing/testing-guide.md) identifies focused Talon checkpoint, cron concurrency, Slack/pairing, dcode prompt, and Textual seams.

## Package-local development

Use `uv` for interpreters, environments, and dependencies; do not substitute `pip`, Poetry, or Conda. Each package's `Makefile` is the command authority, and `uv` provisions the required interpreter, so there is no global Python version to pin. The target package's `requires-python` is authoritative: Deep Agents and ACP support `>=3.11`, while Code and Talon require `>=3.12`.

```bash
cd libs/deepagents
uv sync --all-groups
make help
make test TEST_FILE=tests/unit_tests/test_specific.py
make lint
```

Run equivalent commands from the changed package. Use `uv run ...` for a one-off direct command. From `libs/`, use fan-out targets such as `make lint` or `make lock-check` only for cross-package work. Package-local sibling dependencies are editable, so validate a shared-interface change in its consumers too. For commands, locks, compatibility ranges, and release mechanics, read [Development, Packaging, and Releases](./operations/development.md).

## Current release routing

| Distribution | Current version | Use this when… |
| --- | --- | --- |
| `deepagents-code` / `dcode` | `0.1.80` | A terminal product change needs its manifest, changelog, SDK pin, lockfile, and package-local validation checked together. |
| `deepagents-talon` | `0.0.9` | A host, channel, checkpoint, pairing, cron, MCP, or sandbox change needs its manifest, changelog, configuration, and focused lifecycle coverage checked together. |

The release manifest independently tracks Deep Agents, ACP, Code, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partner packages. A package manifest declares its distribution name, runtime range, dependencies, extras, and entrypoints; regenerate the owning lock through its package workflow rather than editing it by hand.

A new partner package needs more than its own environment, manifest, Makefile, and tests: wire its issue forms, dependency automation, labels, CI/change detection, release management, and applicable integration and secret matrices. Repository automation is organized under `.github/` into workflows, composite actions, scripts, issue templates, and release/labeling documentation. Its labeling workflows apply package, integration, priority, and PR metadata labels; helper-script tests mirror their production domain layout.

## Operational boundaries

- **dcode trust boundary:** dcode trusts the directory from which it runs by default. Approval prompts gate model-requested tool calls, but project artifacts can be read before approval. Use a remote sandbox for an untrusted repository.
- **Talon trust boundary:** Talon is alpha software, not a production or multi-tenant security boundary. Channel access is effectively access to the operator's agent, credentials, MCP tools, and host resources. Start with the [Talon integration guide](./integrations/talon.md) before changing access, credentials, MCP, or sandbox behavior.
- **Talon execution choice:** host shell and file tools run locally unless `DEEPAGENTS_TALON_SANDBOX` selects a sandbox. A sandbox startup failure exits rather than silently falling back to host execution; MCP tools and channel media remain host-resident.
- **Durable state:** checkpoint, archive, conversation, and cron changes cross persistence and lifecycle boundaries. Start with [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md) or [Talon Scheduling and Cron Records](./concepts/talon-scheduling.md), then use fixed-clock and temporary-store tests.

## Continue by question

- **Where is the public entrypoint, state owner, or source/test seam?** [System Source Map](./architecture/source-map.md)
- **How do the SDK, dcode, ACP, and Talon relate at runtime?** [System Architecture Overview](./architecture/overview.md)
- **How do I build or change an agent safely?** [Build and Modify a Deep Agent](./workflows/build-a-deep-agent.md)
- **How do I run a dcode session?** [Run a dcode Session](./workflows/run-dcode-session.md)
- **How do I configure and operate the local host?** [Talon Runtime Integration](./integrations/talon.md)
- **How do I test the changed behavior?** [Testing Guide](./testing/testing-guide.md)
- **How do packaging, locks, and releases work?** [Development, Packaging, and Releases](./operations/development.md)
