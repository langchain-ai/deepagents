---
type: architecture overview
title: Repository Architecture Overview
description: Package roles, published Python compatibility contracts, development-only uv source substitutions, lockfile propagation, and release baselines in the Deep Agents monorepo.
tags: [architecture, monorepo, python, dependencies, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-26T08:03:59.911Z
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
generated: { by: "openwiki/0.4.2", at: "2026-11-26T08:03:59.911Z" }
---

# Repository Architecture Overview

This is an independently versioned Python monorepo, not one deployable service. The reusable `deepagents` SDK is the center; ACP adapts it to Agent Client Protocol, dcode is the terminal coding-agent product, Talon is a local long-running host, evals measures behavior, and partner packages add optional integrations. A distribution's `pyproject.toml` defines its installer-facing contract; a checkout's `uv` source mapping does not.

Related: [runtime behavior](./runtime-behavior.md), [source map](./source-map.md), [ACP](../integrations/acp.md), [Talon](../integrations/talon.md), [development](../operations/development.md), and [quickstart](../quickstart.md).

## Package roles and public contracts

| Distribution | Source version | Python range | Role and entry point |
| --- | --- | --- | --- |
| `deepagents` | `0.7.42` | `>=3.11,<4.0` | Reusable agent harness; `create_deep_agent()` is its assembly boundary. |
| `deepagents-acp` | `0.1.0` | `>=3.11,<4.0` | Agent Client Protocol bridge. |
| `deepagents-code` / `dcode` | `0.2.15` | `>=3.14,<3.15` | Terminal coding-agent product; both console scripts call `deepagents_code:cli_main`. |
| `deepagents-talon` | `0.1.2` | `>=3.14,<3.15` | Experimental local host for channels, schedules, and agent lifecycle; `deepagents-talon` starts it. |
| `deepagents-evals` | `0.0.1` | `>=3.14,<3.15` | Evaluation suite and Harbor integration; `deepagents-evals` runs its CLI. |
| `langchain-daytona`, `langchain-modal`, `langchain-runloop` | `0.0.9`, `0.0.8`, `0.0.7` | `>=3.11,<4.0` | Optional sandbox integrations. |
| `langchain-vercel-sandbox`, `langchain-quickjs` | `0.0.2`, `0.4.0` | `>=3.11,<4.0` | Optional Vercel sandbox and JavaScript REPL middleware integrations. |

Deep Agents is an opinionated layer over LangChain `create_agent()`, whose graph runtime is LangGraph. `create_deep_agent()` composes middleware, backends, skills, memory, profiles, and delegation. Synchronous task-tool subagents and non-blocking deployed `AsyncSubAgent` specifications are distinct delegation mechanisms; products should build on the SDK rather than introduce another runtime.

`[project.dependencies]`, optional dependencies, and scripts are wheel metadata. Conversely, `[tool.uv.sources]` replaces a named dependency with a local path only while `uv` resolves this checkout. `editable = true` is a development-resolution choice, not a promise made to downstream installers.

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

*Solid edges are installer-visible requirements. Dotted edges are checkout-only `uv` substitutions.*

### Compatibility floors, extras, and checkout mappings

Dcode publishes `deepagents==0.7.42`, ACP `>=0.1.0,<1.0.0`, and QuickJS `>=0.4.0,<0.5.0`. Its exact SDK pin is an explicit compatibility promise: raise it in the same change that requires newer SDK behavior, because an editable checkout can hide an inadequate PyPI requirement. ACP declares `deepagents` without a version constraint.

Talon publishes `deepagents>=0.7.41` and `deepagents-code>=0.2.14,<1.0.0`. Evals instead has floors of `deepagents>=0.7.32`, `deepagents-code>=0.2.1`, and `langchain-quickjs>=0.3.3`. Each of the five partner distributions supports Python `>=3.11,<4.0`, requires `deepagents>=0.7.41,<0.8.0`, and maps the sibling SDK editable during checkout development.

The development graph intentionally differs from the wheel graph:

- ACP maps only the SDK sibling, editable.
- Dcode maps SDK, ACP, and all five partner siblings editable, even though sandbox providers are extras.
- Evals maps SDK and QuickJS editable, but its dcode mapping is deliberately non-editable.
- Talon maps its direct SDK and dcode siblings editable; it does not map ACP or QuickJS directly.

Extras select implementations rather than providing credentials or authorization. The SDK offers `aws`, `quickjs`, and `video`. Dcode includes QuickJS and `langsmith[sandbox]` as normal dependencies, offers model-provider and sandbox-provider extras, and preserves an empty `quickjs` extra for compatible install commands. Talon's history, remote-persistence, OpenShell, and media extras select optional implementations; `media` and `media-cpu` conflict, and Linux `media-cpu` directs `torch` and `torchaudio` to the explicit `pytorch-cpu` index.

## Runtime ownership boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| SDK | Harness composition, middleware, backends, profiles, and subagent integration. | ACP transport, terminal UI, channels, or schedules. |
| ACP | Protocol translation and session-local state. | Dcode terminal policy. |
| dcode | Terminal product behavior and ACP specialization. | Generic ACP semantics or provider implementations. |
| Talon | One local process's channels, cron, lifecycle, checkpoints, history, and approval integration. | Production multi-tenant containment. |
| Evals | Repeatable evaluation and Harbor benchmarks. | Online serving. |
| Partners | Optional sandbox or middleware implementations. | A required SDK installation. |

`AgentServerACP` accepts either a compiled graph or a factory receiving `AgentSessionContext` with working directory, mode, and optional model. It keeps modes, models, plans, working directories, MCP configuration, cancellation, and command allowances local to the protocol session. Dcode's wrapper remains above this generic translation: it writes trusted Auto-mode approval state, annotates the last user message with prompt metadata, and passes CLI context to the graph.

Talon is experimental and operates channel adapters, cron schedulers, and agent lifecycle on one local event loop. Its default shell and filesystem tools execute on the host. Selecting a sandbox moves that work to a provider; startup failure exits rather than falling back to host execution. MCP and web tools, channel-media handling, skills, and memory remain host-side, so sandboxing is neither complete containment nor a multi-tenant boundary. Talon removes a sandbox it created at shutdown but never deletes an attached existing sandbox.

Talon selects SQLite, PostgreSQL, MongoDB, or a trusted entry-point checkpoint backend by URI. Checkpoint and conversation-history storage are separate; remote checkpoint thread IDs are not automatically assistant-namespaced. History has its own trusted entry-point extension boundary. An assistant-local `tools.json` maps exact tool names to approval booleans: updates are atomic compare-and-swap, an active invocation retains its policy snapshot, and an invalid policy file fails closed on the next invocation. Prompt approval is not authorization or sandbox isolation.

## Release ownership and baselines

Release Please creates distinct draft release pull requests for the SDK, ACP, dcode, Talon, and five partner distributions; evals has no managed-package configuration. The configuration disables Release Please's own GitHub-release action. Merging a managed release PR triggers the repository workflow that publishes PyPI artifacts and creates the GitHub release.

The release manifest is a released-version baseline, not ordinary dependency metadata. It currently records SDK `0.7.42`, ACP `0.1.0`, dcode `0.2.15`, Talon `0.1.2`, Daytona `0.0.9`, Modal `0.0.8`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.4.0`. Source manifests describe prospective distributions, so adding a managed package requires both release configuration and a manifest baseline; evals intentionally remains outside this process.

## Lockfile propagation

Each package-local `uv.lock` is a generated concrete development and CI resolution, not another manifest. The lock helper parses every package's `[tool.uv.sources]` path and finds reverse consumers to a fixed point. It does not filter on `editable`, so evals' non-editable dcode substitution still propagates a dcode manifest change.

```mermaid
flowchart TD
  Changed["Changed pyproject.toml"] --> Owner["Owning package"]
  Owner --> Scan["Read uv source paths"]
  Scan --> Consumers["Add direct reverse consumers"]
  Consumers --> Closure["Repeat until closure"]
  Closure --> Lock["Run uv lock for each package"]
  Lock --> Commit["Commit every changed uv.lock"]
```

*Refresh starts at the changed manifest and follows all local-source reverse consumers, including non-editable substitutions.*

An SDK manifest change reaches ACP, dcode, evals, Talon, and all five partners, then continues to their consumers. A dcode manifest change reaches dcode plus Talon and evals. Run the helper from the repository root instead of selecting lockfiles manually:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/deepagents/pyproject.toml
```

Use `libs/code/pyproject.toml` for a dcode-only change. The helper locks ACP, dcode, evals, and Talon with Python 3.14 and other discovered package directories with Python 3.12. Validate under each package's declared Python contract, review published requirements separately from source substitutions, and commit every changed lockfile.

## Safe change checklist

- Start SDK construction, middleware, profile, backend, or delegation changes at `create_deep_agent()` and focused SDK tests. Keep task-tool subagents distinct from deployed async subagents.
- Treat dependency floors as installer-facing promises. Update dcode's exact SDK pin whenever it relies on newer SDK behavior; never infer a floor from an editable mapping.
- For a manifest change, validate its Python contract, refresh the full reverse-consumer closure, and commit every resulting `uv.lock`.
- For ACP changes, retain the compiled-graph versus session-context factory distinction. Keep dcode-specific approval and CLI policy in its wrapper, not generic ACP translation.
- For Talon, test the affected lifecycle, persistence, channel, cron, MCP, or policy boundary. Do not represent sandboxing or approval prompts as complete security controls.
- For a managed release, review the package-specific draft release PR and changelog, and confirm dcode's exact SDK pin remains its true minimum.
