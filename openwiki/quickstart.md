---
type: task routing guide
title: Engineer Navigation Guide
description: Concise navigation for engineers changing the Deep Agents SDK, dcode, ACP, Talon, partner integrations, or evaluation suite. Routes common work to the architecture, concepts, workflows, operations, and focused testing guidance.
tags: [deepagents, navigation, architecture, development, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-29T08:06:56.235Z
sources:
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-1d73b3e2b56b5f0d27273379
    resource: repo://libs/code/README.md
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.4.2", at: "2026-09-29T08:06:56.235Z" }
---

# Engineer Navigation Guide

Start with the package that owns the behavior. `libs/` is a monorepo of independently versioned packages: use `uv` for the package environment and that package's `Makefile` as the command reference. The reusable **Deep Agents SDK** is the base layer; dcode, ACP, Talon, and evals consume or assess it rather than being interchangeable runtimes.

## Find the owning domain

| Domain | What it owns | Start here |
| --- | --- | --- |
| **SDK** (`libs/deepagents`) | `create_deep_agent`, graph assembly, middleware, backends, tools, filesystems, skills, memory, and subagents. | [SDK construction and execution](./architecture/sdk-construction-execution.md); [build a Deep Agent](./workflows/build-a-deep-agent.md) |
| **dcode** (`libs/code`) | The terminal coding-agent product: Textual/headless clients, server-hosted graph, workspace policy, sessions, cost/offload, MCP, and sandbox selection. | [dcode client and agent server](./architecture/code-agent.md); [run and resume a dcode session](./workflows/run-dcode-session.md) |
| **ACP** (`libs/acp`) | The editor-facing Agent Client Protocol adapter around a Deep Agent or dcode's prebuilt agent. | [ACP integration](./integrations/acp.md) |
| **Talon** (`libs/talon`) | Experimental local long-running host: channels, turns, local state, cron, and channel-mediated agent execution. | [Talon runtime behavior](./architecture/runtime-behavior.md); [Talon integrations](./integrations/talon.md) |
| **Partners** (`libs/partners`) | Provider-specific sandbox/execution integrations, including Daytona, Modal, Runloop, Vercel, and QuickJS. | [Sandbox providers and execution boundaries](./integrations/sandbox-partners.md) |
| **Evaluations** (`libs/evals`) | Real-model behavioral evaluations and Harbor benchmark integration. | [Run evaluations](./workflows/run-evals.md) |

For public entrypoints, ownership boundaries, and representative test seams across all packages, use the [system ownership map](./architecture/source-map.md). For the dependency and runtime picture, use the [repository architecture overview](./architecture/overview.md).

## Route a common change

| If you are changing… | Read first | Then validate at… |
| --- | --- | --- |
| SDK graph construction, profiles, middleware ordering, tools, permissions, filesystem/backend behavior, skills, memory, or subagents | [SDK construction and execution](./architecture/sdk-construction-execution.md), then the relevant [concept](./concepts/index.md) | SDK graph/tool tests; see [testing by runtime boundary](./testing/testing-guide.md) |
| dcode command/UI behavior, server graph, workspace policy, configuration, sessions, offload, cost, or cancellation | [dcode client and agent server](./architecture/code-agent.md) and [run a dcode session](./workflows/run-dcode-session.md) | dcode client/server or persistence tests, selected in the [testing guide](./testing/testing-guide.md) |
| dcode trust, approvals, MCP/extensions, sandbox use, or sensitive local state | [security boundaries and trust decisions](./operations/security.md) and [approvals and human intervention](./concepts/permissions-hitl.md) | The owner’s policy and server-boundary tests—not only UI tests |
| Editor protocol sessions, content conversion, cancellation, model switching, or session reload | [ACP integration](./integrations/acp.md) | ACP fake-client/session tests |
| Talon startup, turns, interruption, delivery, background work, runtime graph assembly, or shutdown | [Talon host runtime behavior](./architecture/runtime-behavior.md) | Talon host or runtime tests |
| Talon sender exposure, pairing/revocation, channel media, or provider transport | [Talon channel admission and pairing](./concepts/talon-channel-admission.md) and [Talon integrations](./integrations/talon.md) | Channel-adapter tests, plus host composition when routing changes |
| Talon checkpoints, archives, history search, assistant state, or dcode session persistence | [state, sessions, and archives](./concepts/state-persistence.md) | Persistence tests with a temporary durable store where durability is the contract |
| Talon cron expressions, durable jobs, timezone/DST, execution, or delivery | [Talon persistent scheduling](./concepts/talon-scheduling.md) | Focused cron expression, job-store, or scheduler tests |
| Sandbox-provider lifecycle or host-versus-sandbox execution | [sandbox providers and execution boundaries](./integrations/sandbox-partners.md) | Partner-package tests and the consuming runtime’s factory/wiring tests |
| Model behavior, prompt quality, trajectory efficiency, or a benchmark | [run evaluations](./workflows/run-evals.md) | A focused real-LLM eval **in addition to** deterministic boundary tests |
| Package setup, lockfiles, releases, or normal test/lint commands | [development, dependency, and release operations](./operations/development.md) | The changed package’s `make` targets |

## Working rules

1. **Follow the ownership boundary.** Put reusable agent behavior in the SDK; put dcode product policy in dcode; put ACP protocol translation in ACP; and put Talon transport, scheduling, and local-host policy in Talon.
2. **Treat execution authority explicitly.** dcode trusts its launch directory by default, while Talon is explicitly experimental and not a production or multi-tenant security boundary. Read the applicable security page before widening tool, channel, MCP, or sandbox authority.
3. **Test at the narrowest observable boundary.** Use fakes for models, transports, clocks, and external clients in unit tests. Use integration targets only for the integration contract, and use real-LLM evals only for behavior that cannot be made deterministic.
4. **Run commands from the owning package.** Install dependencies explicitly with `uv sync`, start with a focused `make test TEST_FILE=...`, then run that package’s lint/check target. Use `make help` when a package’s targets differ.

## Continue by question

- **How does the SDK compile and run an agent?** [SDK construction and execution](./architecture/sdk-construction-execution.md) · [build a Deep Agent](./workflows/build-a-deep-agent.md)
- **How does a dcode request reach the graph and resume safely?** [run and resume a dcode session](./workflows/run-dcode-session.md) · [cost, session, and context operations](./operations/cost-and-sessions.md)
- **Where do state and approval decisions live?** [state, sessions, and archives](./concepts/state-persistence.md) · [approvals and human intervention](./concepts/permissions-hitl.md)
- **How do MCP and provider sandboxes fit in?** [MCP integration](./integrations/mcp.md) · [sandbox providers](./integrations/sandbox-partners.md)
- **How do I operate an assistant on channels?** [Talon integrations](./integrations/talon.md) · [Talon admission](./concepts/talon-channel-admission.md) · [Talon scheduling](./concepts/talon-scheduling.md)
- **What test should I add or run?** [testing by runtime boundary](./testing/testing-guide.md)
