---
type: task routing guide
title: Engineer Navigation Guide
description: Navigation for maintainers changing the independently released Deep Agents SDK, dcode, ACP, Talon, partner integrations, evaluations, or repository automation. Routes each task to its owner, operational guide, and narrowest observable test boundary.
tags: [deepagents, navigation, architecture, development, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-30T08:06:28.871Z
sources:
  - id: openwiki-source-37e02a57730563a4b4de1690
    resource: repo://.github/LAYOUT.md
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-1d73b3e2b56b5f0d27273379
    resource: repo://libs/code/README.md
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.4.2", at: "2026-09-30T08:06:28.871Z" }
---

# Engineer Navigation Guide

Start at the package that owns the observable behavior. `libs/` contains independently versioned packages, not one root Python project: use `uv` for interpreters and environments, then treat the owning package's `Makefile` as the command authority. The SDK is the reusable agent harness; dcode, ACP, Talon, and evals are distinct products or integration boundaries rather than interchangeable runtimes.

## Choose an owner

| Domain | Owns | Start with |
| --- | --- | --- |
| **Deep Agents SDK** (`libs/deepagents`) | The model-agnostic LangGraph-based harness: agent construction, middleware, backends, tools, filesystem behavior, memory, skills, human intervention, and subagents. | [Repository runtime architecture](./architecture/overview.md) · [System ownership map](./architecture/source-map.md) |
| **dcode** (`libs/code`) | The pre-built terminal coding agent layered on the SDK: interactive Textual UI, conversation resume, remote sandboxes, persistent memory, skills, headless mode, and tool-call approvals. | [dcode client and agent server](./architecture/code-agent.md) · [Run and resume a dcode session](./workflows/run-dcode-session.md) |
| **ACP** (`libs/acp`) | The Agent Client Protocol bridge between a Python Deep Agent and an ACP-capable editor; it can also expose dcode's prebuilt coding agent as an ACP server over standard input/output. | [ACP integration](./integrations/acp.md) |
| **Talon** (`libs/talon`) | An experimental local host whose single event loop owns channel adapters, cron schedulers, and the agent runtime; it also owns local runtime lifecycle, channel sessions, and pairing. | [Talon host runtime behavior](./architecture/runtime-behavior.md) · [Talon integrations](./integrations/talon.md) |
| **Partner packages** (`libs/partners/*`) | Provider-specific execution and sandbox integrations: Daytona, Modal, Runloop, Vercel, and QuickJS. | [Sandbox providers, QuickJS, and execution boundaries](./integrations/sandbox-partners.md) |
| **Evaluations** (`libs/evals`) | Real-model behavioral evaluation: it captures tool calls, file mutations, and final responses, scores correctness and efficiency, and includes Harbor-backed sandboxed benchmarks. | [Run evaluations](./workflows/run-evals.md) |
| **Repository automation** (`.github/`) | CI workflows, release wiring, PR/issue labels, and their helper scripts and tests. | [Development, dependency, release, and labeling operations](./operations/development.md) |

Use the [system ownership map](./architecture/source-map.md) for public entrypoints and source/test seams. Use the [repository runtime architecture](./architecture/overview.md) before a change crosses packages or runtime boundaries.

## Route the change and its test

| Change | Read first | Validate at the narrowest observable boundary |
| --- | --- | --- |
| SDK graph assembly, middleware, backend/filesystem behavior, permissions, skills, memory, or subagents | [Repository runtime architecture](./architecture/overview.md) | The SDK graph, middleware, backend, or tool test that exposes the behavior; see the [testing guide](./testing/testing-guide.md). |
| dcode CLI, configuration, model/reasoning selection, graph, session, cost, MCP, approval, or Textual behavior | [dcode client and agent server](./architecture/code-agent.md) · [Run and resume a dcode session](./workflows/run-dcode-session.md) | The dcode agent/server/config/persistence/UI test; use a fake model or service rather than a live provider unless the integration itself is under test. |
| dcode trust, tool authority, MCP, sandbox, or local project state | [MCP servers, trust, OAuth, and tool execution](./integrations/mcp.md) · [Sandbox providers, QuickJS, and execution boundaries](./integrations/sandbox-partners.md) | The owning policy and server/agent boundary test, not only a UI test. dcode trusts its launch directory by default. |
| ACP editor sessions, event translation, cancellation, model switching, or reload | [ACP integration](./integrations/acp.md) | ACP fake-client/session tests, because the public result is an ACP session stream. |
| Talon startup, graph lifecycle, turns, interruption, delivery, background work, or shutdown | [Talon host runtime behavior](./architecture/runtime-behavior.md) | Talon host/runtime tests with fake channels, clocks, runners, or stores. |
| Talon channel access, pairing/revocation, media, or Slack/other adapter behavior | [Talon channel admission and pairing](./concepts/talon-channel-admission.md) · [Talon integrations](./integrations/talon.md) | The adapter test plus a host-composition test only when routing or lifecycle is affected. |
| Talon durable state, archives, scheduled-history isolation, cron expressions, or delivery | [State, sessions, and archives](./concepts/state-persistence.md) · [Talon scheduled work and cron semantics](./concepts/talon-scheduling.md) | Focused store, scheduler, or cron test using temporary durable state and a fixed clock. |
| Sandbox lifecycle, partner API behavior, or QuickJS subagent replay | [Sandbox providers, QuickJS, and execution boundaries](./integrations/sandbox-partners.md) | The partner-package test and consuming factory/wiring test; for QuickJS restart behavior, test the checkpoint/replay contract. |
| Model quality, tool trajectory, or benchmark score | [Run evaluations](./workflows/run-evals.md) | A focused real-LLM eval in addition to deterministic owner-boundary tests. |
| Lockfiles, package pins, release metadata, CI labels, issue routing, or workflow helpers | [Development, dependency, release, and labeling operations](./operations/development.md) | The affected workflow/helper-script contract test and the relevant package or lock check. |

## Safety and operating rules

1. **Keep responsibility at the boundary.** Reusable harness behavior belongs in the SDK; terminal-product policy in dcode; editor-protocol translation in ACP; and local channel, scheduling, and host policy in Talon.
2. **Treat authority as part of the change.** dcode approvals do not make an untrusted launch directory safe: project artifacts are read before a prompt, so use a sandbox for untrusted repositories. Talon is alpha software, not a production or multi-tenant security boundary; channel access is effectively access to the operator's agent, credentials, MCP tools, and host resources.
3. **Test the behavior, not a private call sequence.** Put deterministic, network-free tests under `tests/unit_tests/`; reserve `tests/integration_tests/` for a real integration contract. Warnings are errors unless deliberately accepted.
4. **Work package-locally.** Run `uv sync` explicitly in the owning package, find supported commands with `make help`, run a focused `make test TEST_FILE=...`, then the package's lint/check target. Use `libs/` fan-out Make targets only for cross-package work.
5. **Wire a new partner completely.** A partner package owns its environment, manifest, Makefile, and tests, but it must also be registered in release, CI/change detection, labels, issue forms, and applicable integration/secret matrices.

## Continue by question

- **Which public surface or source owner should I inspect?** [System ownership map](./architecture/source-map.md)
- **How do the packages and runtimes relate?** [Repository runtime architecture](./architecture/overview.md)
- **How does a dcode request reach, run, and resume through its graph?** [Run and resume a dcode session](./workflows/run-dcode-session.md)
- **Where do state, checkpoints, archives, and session data live?** [State, sessions, and archives](./concepts/state-persistence.md)
- **How do MCP, sandboxes, and partner integrations execute?** [MCP servers, trust, OAuth, and tool execution](./integrations/mcp.md) · [Sandbox providers, QuickJS, and execution boundaries](./integrations/sandbox-partners.md)
- **How is Talon admitted to channels and how does scheduled work run?** [Talon channel admission and pairing](./concepts/talon-channel-admission.md) · [Talon scheduled work and cron semantics](./concepts/talon-scheduling.md)
- **What command, release, or label automation applies?** [Development, dependency, release, and labeling operations](./operations/development.md)
- **What regression should I add or run?** [Testing by runtime boundary](./testing/testing-guide.md)
