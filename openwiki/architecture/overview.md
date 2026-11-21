---
type: architecture overview
title: Repository Architecture Overview
description: Package responsibilities, published Python compatibility contracts, development-only uv checkout wiring, lockfile propagation, and release baselines for the Deep Agents monorepo.
tags: [architecture, monorepo, python, dependencies, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-21T08:04:00.255Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-ffc41789c892ca61e2829a4c
    resource: repo://libs/acp/deepagents_acp/server.py
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-4d4186e9d62fb4abe495cdd0
    resource: repo://libs/code/deepagents_code/acp.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
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
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-11-21T08:04:00.255Z" }
---

# Repository Architecture Overview

This repository is an independently versioned Python monorepo rather than a single deployable service. The `deepagents` SDK is the reusable harness; ACP adapts it to the Agent Client Protocol; dcode is a terminal coding-agent product; Talon hosts long-running local agents; evals measures behavior; and partner distributions add optional sandbox or JavaScript capabilities. A package's `pyproject.toml`, rather than a workspace checkout, defines the contract an installer sees.

Related: [runtime behavior](./runtime-behavior.md), [source map](./source-map.md), [ACP](../integrations/acp.md), [Talon](../integrations/talon.md), [development](../operations/development.md), and [quickstart](../quickstart.md).

## Package roles and published contracts

| Distribution | Source version | Python range | Architectural role and entry point |
| --- | --- | --- | --- |
| `deepagents` | `0.7.41` | `>=3.11,<4.0` | Reusable agent harness. `create_deep_agent()` assembles the graph. |
| `deepagents-acp` | `0.1.0` | `>=3.11,<4.0` | Agent Client Protocol bridge. |
| `deepagents-code` / `dcode` | `0.2.13` | `>=3.14,<3.15` | Terminal coding-agent product; both console scripts call `deepagents_code:cli_main`. |
| `deepagents-talon` | `0.1.1` | `>=3.14,<3.15` | Experimental local host for channels, schedules, and agent lifecycle; `deepagents-talon` starts it. |
| `deepagents-evals` | `0.0.1` | `>=3.14,<3.15` | Evaluation suite and Harbor integration; `deepagents-evals` runs its CLI. |
| `langchain-daytona`, `langchain-modal`, `langchain-runloop` | `0.0.9`, `0.0.8`, `0.0.7` | `>=3.11,<4.0` | Optional sandbox integrations. |
| `langchain-vercel-sandbox`, `langchain-quickjs` | `0.0.2`, `0.4.0` | `>=3.11,<4.0` | Optional Vercel sandbox and JavaScript-REPL middleware integrations. |

Deep Agents is an opinionated layer over LangChain `create_agent()`, whose agent runtime is LangGraph. `create_deep_agent()` is the SDK composition boundary for middleware, backends, skills, memory, profiles, and delegation. Synchronous task-tool subagents and non-blocking deployed `AsyncSubAgent` specifications are separate delegation mechanisms; consumers build on this SDK rather than introducing a competing runtime.

`[project.dependencies]`, optional dependencies, and scripts are published wheel metadata. In contrast, `[tool.uv.sources]` replaces a named dependency with a local path only while `uv` resolves this checkout. A source is editable only where its mapping says `editable = true`; it is not a downstream installation guarantee.

```mermaid
flowchart TD
  SDK["deepagents SDK"]
  ACP["deepagents-acp"]
  Dcode["deepagents-code and dcode"]
  Talon["deepagents-talon"]
  Evals["deepagents-evals"]
  QuickJS["langchain-quickjs"]
  Partners["sandbox partner packages"]
  LangChain["LangChain create_agent"]
  LangGraph["LangGraph runtime"]

  ACP -->|"published"| SDK
  Dcode -->|"published exact pin"| SDK
  Dcode -->|"published"| ACP
  Dcode -->|"published"| QuickJS
  Dcode -->|"optional extras"| Partners
  Talon -->|"published"| SDK
  Talon -->|"published"| Dcode
  Evals -->|"published"| SDK
  Evals -->|"published"| Dcode
  Evals -->|"published"| QuickJS
  QuickJS -->|"published compatibility range"| SDK
  Partners -->|"published compatibility range"| SDK
  SDK --> LangChain
  LangChain --> LangGraph

  ACP -. "uv editable checkout" .-> SDK
  Dcode -. "uv editable checkout" .-> SDK
  Dcode -. "uv editable checkout" .-> ACP
  Dcode -. "uv editable partner checkouts" .-> Partners
  Talon -. "uv editable checkout" .-> SDK
  Talon -. "uv editable checkout" .-> Dcode
  Evals -. "uv checkout substitution" .-> SDK
  Evals -. "uv checkout substitution" .-> Dcode
  Evals -. "uv checkout substitution" .-> QuickJS
```

*Solid edges are installer-visible published requirements. Dotted edges are `uv` checkout substitutions and are not published dependency edges.*

### Compatibility floors, extras, and workspace mappings

Dcode publishes `deepagents==0.7.41`, ACP `>=0.0.13,<1.0.0`, and QuickJS `>=0.4.0,<0.5.0`. The exact SDK pin is an explicit compatibility promise: raise it in the same change that starts depending on newer SDK behavior, because an editable checkout can otherwise mask an inadequate PyPI requirement. ACP declares `deepagents` without a version constraint.

Talon publishes `deepagents>=0.7.41` and `deepagents-code>=0.2.12,<1.0.0`. Evals uses separately selected floors—`deepagents>=0.7.32`, `deepagents-code>=0.2.1`, and `langchain-quickjs>=0.3.3`. All five partner distributions require `deepagents>=0.7.41,<0.8.0`; their local SDK mappings are editable.

The checkout graph deliberately differs from the wheel graph:

- ACP maps only the SDK sibling, editable.
- Dcode maps SDK, ACP, and all five partner siblings editable, even though its sandbox providers are optional extras.
- Evals maps SDK and QuickJS editable, but its dcode path mapping is deliberately non-editable.
- Talon maps its direct SDK and dcode siblings editable; it does not map ACP or QuickJS directly.

Extras choose implementations; they neither provide credentials nor create an authorization boundary. The SDK has `aws`, `quickjs`, and `video` extras. Dcode includes QuickJS and `langsmith[sandbox]` as normal dependencies, exposes model-provider and sandbox-provider extras, and retains an empty `quickjs` extra for compatible install commands. Talon's `history-local`, `history-voyage`, and `history-openai` extras select semantic-history implementations; `mongodb` and `postgres` select remote persistence; and `openshell` pulls the corresponding dcode provider. Its `media` and `media-cpu` extras conflict; on Linux, the CPU variant directs `torch` and `torchaudio` to the explicit `pytorch-cpu` index.

## Runtime ownership boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| SDK | Harness composition, middleware, backends, profiles, and subagent integration. | ACP transport, terminal UI, channels, or schedules. |
| ACP | Protocol translation and session-local state. | Dcode terminal policy. |
| dcode | Terminal product behavior and its ACP specialization. | Generic ACP semantics or provider implementations. |
| Talon | One local process's channels, cron, lifecycle, checkpoints, history, and approval integration. | Production multi-tenant containment. |
| Evals | Repeatable evaluation and Harbor benchmarks. | Online serving. |
| Partners | Optional sandbox or middleware implementations. | A required SDK installation. |

`AgentServerACP` accepts either a compiled graph or a factory receiving `AgentSessionContext` with working directory, mode, and optional model. It tracks modes, models, plans, working directories, MCP configuration, cancellation, and command allowances by protocol session. Dcode's subclass belongs above this generic bridge: for each prompt it supplies trusted Auto-mode approval state, annotates the last user message with prompt metadata, and passes CLI context to the graph.

Talon is experimental and has one local event loop for channel adapters, cron schedulers, and agent lifecycle. By default its shell and filesystem tools run on the host. Selecting a sandbox moves that work to a provider; failure to start the sandbox exits instead of falling back to host execution. MCP tools, web tools, channel-media handling, skills, and memory remain host-side, so sandboxing is not complete containment or a multi-tenant boundary. Talon deletes a sandbox it created at shutdown, but never deletes an attached existing sandbox.

Talon selects SQLite, PostgreSQL, MongoDB, or a trusted entry-point checkpoint backend by URI. Checkpoint and conversation-history storage are independent; remote checkpoint thread IDs are not automatically assistant-namespaced. History backends have their own trusted entry-point extension point. Assistant-local `tools.json` maps exact tool names to approval booleans: updates are atomic compare-and-swap, active turns retain a snapshot, and an invalid next policy fails closed. This prompt-approval control is not authorization or sandbox isolation.

## Release ownership and baselines

Release Please creates separate draft release pull requests for the SDK, ACP, dcode, Talon, and five partner distributions; evals has no managed package configuration. The configuration disables Release Please's own GitHub-release action. When a managed release PR is merged, the repository workflow publishes PyPI artifacts and creates the GitHub release.

The release manifest is a released-version baseline rather than ordinary dependency metadata. It currently records SDK `0.7.41`, ACP `0.1.0`, dcode `0.2.13`, Talon `0.1.1`, Daytona `0.0.9`, Modal `0.0.8`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.4.0`. Release Please updates this shared baseline; package manifests describe the prospective source distribution. Adding a managed package requires both configuration and a manifest baseline, while evals remains intentionally outside that system.

## Lockfile propagation

Each package-local `uv.lock` is a generated concrete development and CI resolution, not a second manifest. The lock helper parses every package's `[tool.uv.sources]` path, then finds reverse consumers to a fixed point. It does not filter on `editable`, so non-editable local dcode consumed by evals still propagates changes.

```mermaid
flowchart TD
  Changed["Changed pyproject.toml"] --> Owner["Owning package"]
  Owner --> Scan["Read uv source paths"]
  Scan --> Consumers["Add direct reverse consumers"]
  Consumers --> Closure["Repeat until closure"]
  Closure --> Lock["Run uv lock for each package"]
  Lock --> Commit["Commit every changed uv.lock"]
```

*Refresh starts with a changed manifest and follows every local-source reverse consumer, including non-editable substitutions.*

An SDK manifest change reaches ACP, dcode, evals, Talon, and all five partners, then continues through any transitive consumers. A dcode manifest change reaches dcode plus its Talon and evals consumers. Run the helper from the repository root rather than selecting lockfiles manually:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/deepagents/pyproject.toml
```

Use `libs/code/pyproject.toml` as the argument for a dcode-only change. The helper uses Python 3.14 for ACP, dcode, evals, and Talon locks, and Python 3.12 for other discovered package directories. Validate a package under its declared Python range, review published requirements separately from workspace substitutions, and commit every resulting lockfile.

## Safe change checklist

- Start SDK construction, middleware, profile, backend, or delegation changes at `create_deep_agent()` and focused SDK tests. Keep task-tool subagents distinct from deployed async subagents.
- Treat a dependency floor as an installer-facing promise. Update dcode's exact SDK pin whenever new SDK behavior is required; never infer that floor from an editable source mapping.
- For a package-manifest change, validate its Python contract, refresh the complete reverse-consumer closure, and commit every changed `uv.lock`.
- For ACP work, preserve the compiled-graph versus session-context factory distinction. Keep dcode-specific approval and CLI policy in its wrapper, not generic ACP translation.
- For Talon work, test the affected lifecycle, persistence, channel, cron, MCP, or policy boundary. Do not describe sandboxing or approval prompts as complete security controls.
- For managed releases, review the package-specific draft release PR and changelog, and ensure dcode's exact SDK pin remains the minimum version its code needs.
