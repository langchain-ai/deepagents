---
type: integration guide
title: Sandbox Provider Integrations
description: Install and operate optional dcode sandbox providers, including ownership and lifetime rules. Distinguishes provider-backed remote execution from Talon routing and the local QuickJS middleware.
tags: [sandbox, providers, dcode, talon, quickjs]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-07T08:06:51.789Z
sources:
  - id: openwiki-source-9f207ab48c42b84dcfd05f43
    resource: repo://libs/code/deepagents_code/integrations/sandbox_config.py
  - id: openwiki-source-bcf1f68e7989964d2fcec7aa
    resource: repo://libs/code/deepagents_code/integrations/sandbox_factory.py
  - id: openwiki-source-03e3942e51522a3aa485168d
    resource: repo://libs/code/deepagents_code/integrations/sandbox_provider.py
  - id: openwiki-source-668d65d09330d04370b47300
    resource: repo://libs/code/deepagents_code/integrations/sandbox_registry.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-e3efb5f3e4a9e8517eb6d8f5
    resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-d4463137befa776cd47750d4
    resource: repo://libs/deepagents/deepagents/backends/sandbox.py
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
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-580d91c607e0a09e0659e565
    resource: repo://libs/talon/deepagents_talon/sandbox.py
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-57a0613315e23277d358df76
    resource: repo://libs/talon/tests/unit_tests/test_sandbox.py
generated: { by: "openwiki/0.4.2", at: "2026-10-07T08:06:51.789Z" }
---

# Sandbox Provider Integrations

A **sandbox provider** acquires or attaches to an environment that implements filesystem and shell execution. It is not the same thing as `langchain-quickjs`: QuickJS is JavaScript REPL middleware running in the agent process, not a selectable remote-sandbox provider. Provider containment, network policy, credentials, and retention are properties of the provider and its deployment; protocol conformance alone is not an isolation guarantee.

## The backend and provider boundary

`SandboxBackendProtocol` extends the generic backend contract with an `id` plus `execute()`/`aexecute()` for shell commands. `BaseSandbox` supplies the standard filesystem behavior on top of four provider-adapter primitives: `id`, `execute()`, `upload_files()`, and `download_files()`. Implement transfers as ordered, per-file responses: a failed file is reported in that response while unrelated transfers may succeed.

The base class is a convenience layer, not a security boundary. In particular, its recursive-delete quoting makes the supplied path one shell argument; it does not restrict deletion to a workspace root. Capture-at-source offload is also deliberately opt-in. When an adapter enables it, oversized output is captured in the sandbox and represented by a head/tail preview; capture can be capped without killing the command, preserving its exit code. With the default disabled setting, the command is unwrapped and full output is returned.

```mermaid
sequenceDiagram
    participant Agent as Agent tools
    participant Base as BaseSandbox
    participant Adapter as Provider adapter
    participant Environment as Provider environment

    Agent->>Base: filesystem operation or execute
    Base->>Base: generate command or transfer
    Base->>Adapter: execute or transfer request
    Adapter->>Environment: provider SDK request
    Environment-->>Adapter: provider result
    Adapter-->>Base: protocol response
    Base-->>Agent: structured result
```

*Provider-specific execution and transfer primitives support the shared sandbox filesystem interface.*

## dcode installation and selection

The `deepagents-code` distribution is version `0.1.81`, requires Python `>=3.12,<4.0`, and pins `deepagents==0.7.22`. It bundles `langsmith[sandbox]>=0.14.4` and `langchain-quickjs>=0.3.4,<0.4.0`. Thus LangSmith is available without a sandbox extra, and QuickJS is installed with dcode but remains middleware rather than a provider. The backward-compatible `quickjs` extra is empty.

The curated provider adapters are optional extras. Install only the needed adapter, or all five:

| Provider name | dcode extra and adapter constraint | Current partner package | Provider SDK dependency |
| --- | --- | --- | --- |
| `agentcore` | `agentcore`: `langchain-agentcore-codeinterpreter>=0.0.5,<1.0.0` | — | — |
| `daytona` | `daytona`: `langchain-daytona>=0.0.7` | `langchain-daytona` `0.0.8` | `daytona` |
| `modal` | `modal`: `langchain-modal>=0.0.5` | `langchain-modal` `0.0.6` | `modal` |
| `runloop` | `runloop`: `langchain-runloop>=0.0.6` | `langchain-runloop` `0.0.7` | `runloop-api-client` |
| `vercel` | `vercel`: `langchain-vercel-sandbox>=0.0.1` | `langchain-vercel-sandbox` `0.0.2` | `vercel>=0.5.9,<0.6` |

```bash
pip install 'deepagents-code[daytona]'
pip install 'deepagents-code[agentcore,daytona,modal,runloop,vercel]'
pip install 'deepagents-code[all-sandboxes]'
```

All listed partner packages require Python `>=3.11,<4.0` and `deepagents>=0.7.0,<0.8.0`; dcode's Python and Deep Agents constraints are therefore the effective constraints when it owns the installation. Package installation is separate from configuring provider credentials and any provider-specific operational setup.

`SandboxProviderMetadata` lets the CLI show install hints, working directory, and capability flags without creating credential-dependent clients. A provider then implements synchronous `get_or_create()` and `delete()`; the base interface exposes asynchronous wrappers using `asyncio.to_thread`.

The registry combines curated providers, third-party entry points in `deepagents_code.sandbox_providers`, and `[sandboxes.providers]` declarations. On a duplicate name, configuration wins over an entry point, which wins over a built-in. A configured `class_path` imports executable Python and is operator-trusted configuration. `[sandboxes].default` is considered only after the user enables sandbox mode; setting it does not silently enable remote execution.

Built-in working directories are provider metadata: AgentCore `/tmp`, Daytona `/home/daytona`, LangSmith `/root`, Modal `/workspace`, Runloop `/home/user`, and Vercel `/vercel/sandbox`. Snapshot names are supported by LangSmith and Runloop; AgentCore does not support attachment by sandbox ID. Capability metadata is the authority for validation, so extensions should advertise accurate flags rather than relying on provider-name special cases.

## Factory lifecycle and server ownership

`create_sandbox()` resolves metadata before constructing the provider. It rejects a snapshot unsupported by that provider and rejects a snapshot combined with `sandbox_id`, because snapshots apply only to newly created sandboxes. Configured provider parameters are loaded first and explicit call parameters override them. After acquisition, an optional host setup script is expanded using the workspace environment and executed with `bash -c` inside the sandbox.

```mermaid
flowchart TD
    Request["Provider request"] --> Validate["Resolve metadata and validate options"]
    Validate --> Acquire["Create or attach backend"]
    Acquire --> Setup{"Setup file supplied"}
    Setup -->|"yes"| Run["Expand workspace variables and run bash"]
    Setup -->|"no"| Use["Yield backend"]
    Run --> Use
    Use --> Exit{"Context exits"}
    Exit -->|"new environment"| Delete["Delete provider resource"]
    Exit -->|"attached ID"| Retain["Leave resource running"]
```

*The factory deletes an environment it created but never takes ownership of an attached ID.*

Setup failure still enters cleanup for an owned sandbox. Cleanup exceptions are reported but do not mask the setup or caller failure. This ownership rule is important for scripts and tests: attaching an ID is a deliberate retention decision.

A dcode server opens the factory context for the server-process lifetime. Its runtime factory is cached because repeated construction would recreate sandboxes and duplicate cleanup registration. The first sandboxed workspace reserves that process-wide backend; a request for another sandboxed workspace is rejected rather than sharing the first workspace's remote filesystem and credentials. Start a distinct server process for an independent sandboxed workspace.

## Talon reuse and routing

Talon does not implement separate provider adapters. `DEEPAGENTS_TALON_SANDBOX` opts into the dcode registry and factory; when it is unset, Talon yields no sandbox and uses its normal host backend. `DEEPAGENTS_TALON_SANDBOX_ID` attaches an existing environment, while `DEEPAGENTS_TALON_SANDBOX_SNAPSHOT` and `DEEPAGENTS_TALON_SANDBOX_SETUP` select a snapshot and host-side setup script.

For a new LangSmith sandbox, Talon proposes `talon-<assistant_id>` by default. An explicit Talon snapshot takes precedence. If `LANGSMITH_SANDBOX_SNAPSHOT_NAME`, including its dcode-prefixed resolution path, is set, Talon defers to that provider-level setting instead. Talon holds the factory context for its host lifetime, deletes owned resources, and retains an attached one. Configured startup failures become `SandboxStartupError`; Talon does not fall back to host execution. Its thread handoff also closes an owned resource if startup completes after the awaiting task was cancelled.

Talon constructs a `CompositeBackend` whose default target, including every `execute` call, is the remote sandbox. Only the assistant `skills/` and `memory/` directories route to host virtual filesystem backends; `tools.json` and other assistant state remain off those host routes. This is a mixed-plane routing decision, not complete host containment: Talon's MCP and web tools, channel media handling, and provider credentials remain in the Talon process.

## QuickJS is not a provider

`langchain-quickjs` version `0.3.8` is a JavaScript REPL middleware package, requiring Python `>=3.11,<4.0`. Its direct dependencies are `deepagents>=0.7.0,<0.8.0`, `quickjs-rs>=0.2.5,<0.3.0`, `langchain>=1.4.3,<2.0.0`, `langchain-core>=1.6.6,<2.0.0`, `langgraph>=1.2.13,<2.0.0`, and `bsdiff4>=1.2.6,<2.0.0`; it does not depend on a sandbox-provider SDK. It is installed by dcode's base dependency range, but it neither provisions an environment nor appears in the sandbox registry or `DEEPAGENTS_TALON_SANDBOX` choices. Use a provider for remote filesystem and shell execution; use QuickJS when the intended capability is local JavaScript evaluation and explicitly configured middleware/tool bridges.

## Extension and verification guidance

- New provider adapters should implement the four `BaseSandbox` primitives and supply `SandboxProvider` lifecycle methods plus truthful metadata. Publish external providers through `deepagents_code.sandbox_providers`, or use a trusted config `class_path` for local deployment.
- Verify an adapter's create, attach, setup-failure cleanup, snapshot, and transfer partial-success behavior. dcode coverage is concentrated in `libs/code/tests/unit_tests/test_sandbox_factory.py`, `test_sandbox_provider.py`, `test_sandbox_registry.py`, and `test_sandbox_config.py`; provider integration coverage lives in `libs/code/integration_tests/test_sandbox_factory.py` and `test_sandbox_operations.py`.
- After changing Talon selection, ownership, cancellation, or routes, run `libs/talon/tests/unit_tests/test_sandbox.py`.
- Before selecting any provider for sensitive workloads, evaluate its image, filesystem persistence, network controls, credential exposure, and cleanup policy. Neither `BaseSandbox` nor Talon's routing changes those provider-level controls.
