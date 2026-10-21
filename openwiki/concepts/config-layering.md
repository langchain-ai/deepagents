---
type: configuration architecture
title: dcode Configuration Layering
description: How dcode resolves typed configuration across managed policy, CLI values, environment, user TOML, and defaults. Covers generation and reload semantics, project dotenv and trust boundaries, and client-to-server workspace policy.
tags: [configuration, deepagents-code, precedence, managed-policy, environment, reload, workspace, server]
sources:
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-1728494bdd59604ce9b5f65b
    resource: repo://libs/code/deepagents_code/_server_config.py
  - id: openwiki-source-2fb89d2b59c886d0cb3ee3ea
    resource: repo://libs/code/deepagents_code/config_manifest.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-dfdee0a6f0ea427a4490f98a
    resource: repo://libs/code/deepagents_code/configuration/providers.py
  - id: openwiki-source-52d96f61bc4737f02a18cf79
    resource: repo://libs/code/deepagents_code/configuration/resolver.py
  - id: openwiki-source-80ad1e0223472d67f28c7919
    resource: repo://libs/code/deepagents_code/configuration/writer.py
  - id: openwiki-source-216ca680d81dc35eb4d3e76e
    resource: repo://libs/code/deepagents_code/mcp_config.py
  - id: openwiki-source-20b5bbd05beabea1df7e2b53
    resource: repo://libs/code/deepagents_code/mcp_disabled.py
  - id: openwiki-source-f6d553e7afdf54acac36e7d3
    resource: repo://libs/code/deepagents_code/mcp_tools.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-17253964e859bb0abf2094e8
    resource: repo://libs/code/deepagents_code/workspace_diagnostics.py
  - id: openwiki-source-030d8bd153a9c3ea2a99cb7d
    resource: repo://libs/code/deepagents_code/workspace.py
  - id: openwiki-source-149abfd7a1ab6a5a2d1a0e71
    resource: repo://libs/code/tests/unit_tests/test_configuration_resolver.py
  - id: openwiki-source-5a5147d4654f226b03e92ab9
    resource: repo://libs/code/tests/unit_tests/test_workspace_diagnostics.py
  - id: openwiki-source-877b53371bf970f1b38a1809
    resource: repo://libs/code/tests/unit_tests/test_workspace.py
verified:
  - by: openwiki/0.4.2
    at: 2026-10-21T08:05:42.350Z
generated: { by: "openwiki/0.4.2", at: "2026-10-21T08:05:42.350Z" }
---

# dcode Configuration Layering

`dcode` uses a typed, ranked resolver rather than a single configuration file. That design gives managed policy stronger precedence than user-controlled settings while retaining useful per-invocation and per-project inputs. It also draws an explicit boundary between user-authorized configuration and values carried by a checkout.

For the client/server split, see [Code agent](/openwiki/architecture/code-agent.md). For MCP operation, see [MCP](/openwiki/integrations/mcp.md); for security posture, see [Security](/openwiki/operations/security.md).

## Typed resolution and precedence

The manifest is the configuration surface: it declares scalar option types, typed defaults, TOML locations, environment names, merge strategies, and redaction metadata. Structured configuration is deliberately handled by dedicated typed loaders rather than flattened into scalar options. Invalid environment or TOML values become an invalid provider result, so resolution can use a weaker valid value without discarding valid siblings. Credential-bearing structured tables are presence-only in configuration introspection.

Providers return `Found`, `Unset`, or `Invalid`; the resolver owns rank, health, provenance, and merge semantics. Lower numeric rank is stronger.

```mermaid
flowchart TD
    M["Managed policy rank 200"] --> C["CLI rank 300"]
    C --> R["Reload override rank 350 when installed"]
    R --> E["Environment rank 400"]
    E --> U["User config.toml rank 500"]
    U --> D["Manifest defaults rank 1000"]
```

The standard ordering of configuration tiers. Reload retention is a caller-installed tier, not part of every resolver.

For ordinary replacement options, the strongest usable result wins; a higher-precedence durable result masks a weaker non-durable one. `union` and `deep_merge` options instead retain valid values from every tier; deep merges also preserve leaf-level provenance. Defaults are a fallback, not an accumulated contribution. This is essential for deny lists, where allowing a user replacement to erase a managed denial would be fail-open.

`resolver_from_snapshots()` requires keyword-only `managed=` and `user=` snapshots, preventing same-typed inputs from being swapped and accidentally granting user content managed rank. `get_config_resolver()` caches one resolver generation for normal process readers, and rejects a different CLI provider for the same process. A resolver built ad hoc from snapshots has no CLI tier unless the caller supplies it.

## Generations, reloads, and writes

Configuration files are not watched. A cached resolver generation stays in force until `/reload` or a committed write to the default user configuration path advances it; this avoids a read observing a partially applied edit. The environment tier is the notable live exception: `EnvProvider` reads the active environment at resolution time. Outside scoped construction it is `os.environ`; inside `use_environment()` it is an immutable, context-local mapping.

```mermaid
flowchart TD
    A["Reload or default-path write"] --> B["Fetch managed candidate"]
    B --> C{"Candidate enforceable"}
    C -->|"no"| K["Retain active managed tier"]
    C -->|"yes"| I["Install candidate"]
    I --> J["Reload other providers"]
    J --> P["Publish next generation"]
```

Managed policy is fetched and validated before replacing the resolver tier, avoiding network I/O while the resolver lock is held and preventing a bad refresh from dropping restrictions.

A TOML provider retains its last usable snapshot when a later candidate is unreadable or malformed; an initial failed read falls through. A managed candidate must be usable and enforceable before it can replace the active policy. Source diagnostics are reset per generation, so a continuing error is reportable again after another reload.

`update_user_config()` is the supported user-file transaction. It serializes read-modify-write operations, refuses the managed path and malformed existing TOML, writes beside the destination, and atomically replaces the target. After a successful default-path write it best-effort refreshes the shared resolver; refresh failure leaves the landed write intact and is reported as stale in-process state, not as a failed write.

## Environment, dotenv, and sensitive values

For prefix-aware credentials and provider settings, `DEEPAGENTS_CODE_{NAME}` takes precedence over `{NAME}` even when the prefixed variable is empty. That presence rule permits an explicit empty prefixed variable to suppress a canonical credential or stored value.

Project dotenv loading derives an explicit environment snapshot. Inherited environment values win; the nearest project `.env` and global profile dotenv only fill absent keys. The trusted global dotenv controls whether the project file is read; if that file cannot be read, dcode skips the project dotenv rather than assuming that loading an untrusted project file is safe.

A project `.env` cannot inject project-MCP trust lists, Auto classifier model or timeout, MCP tool timeout, forked-subagent behavior, the upstream recursion fallback, or `TERM_PROGRAM`. These security-sensitive settings remain available through the shell and, where applicable, the trusted global dotenv or user configuration. The boundary stops a cloned checkout from choosing its own authorization conditions or trace metadata.

At workspace construction, dcode derives a dotenv mapping and credential snapshot, freezes it, and uses it to construct the runtime. Later process-environment changes therefore do not alter an already-built workspace. LangSmith tracing uses process-wide SDK state, so workspaces on one server must agree on its selectors and secret-redaction setting. If an active upload target requests redaction but the redacting client cannot be installed, tracing is disabled rather than uploading unredacted data.

## MCP controls

MCP configuration interpolates only `${VAR}` and `${VAR:-default}` in `command`, `url`, `args`, `env`, `headers`, and `oauth`. The default form applies when a variable is unset or empty. Unsupported fields are copied verbatim. Malformed braced references, an unset required variable, or a wrong supported-field shape fail before the resolved configuration is used.

Disabled MCP servers are an accumulating policy: user and managed disabled-server names union and are filtered before validation, connection, or tool exposure. A managed deny-list read or validation failure fails closed by disabling every candidate rather than treating the policy as empty.

## Client/server handoff and project trust

The interactive client and agent server are separate processes and do not share the resolver cache. `ServerConfig` is the typed handoff: the client serializes it to `DEEPAGENTS_CODE_SERVER_*` variables, with `None` clearing a variable; the server reconstructs and validates it. In particular, malformed or unsafe filesystem-tool allowlists are rejected instead of becoming unrestricted access.

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Binding as SQLite binding
    participant Runtime
    Client->>Server: ServerConfig environment values
    Server->>Binding: Bind resolved workspace policy
    Binding-->>Server: Durable binding and fingerprints
    Server->>Server: Resolve current workspace policy
    Server->>Runtime: Build or reuse compatible runtime
```

The process handoff, durable binding, and per-request validation are distinct safeguards.

A workspace binding canonicalizes an existing workspace directory and persists server-resolved non-secret policy plus separate policy and runtime fingerprints in SQLite. It atomically refuses a different workspace or incompatible policy. A compatible runtime-only identity change updates the runtime fingerprint and rebuilds the runtime without discarding the binding or its checkpoints.

Project-scoped MCP, sandbox setup, extension paths, and trust are valid only for the launch project. Resolving another or unresolvable project drops MCP and sandbox grants and re-evaluates extension trust. On each request, the server compares current project and durable policy with the binding: policy drift or revoked extension trust refuses execution, while runtime-only drift causes a rebuild.

## Diagnostics and change guidance

Bindings retain a bounded, allowlisted snapshot for conflict diagnostics. Only reportable policy scalars and lists are persisted. Paths, credentials, environment values, model data, and prompts are excluded from persistence and logs; a refusal either shows safe changed values or marks them unavailable.

When changing configuration, define the option in the manifest or a dedicated structured loader; decide its merge behavior and invalid-value fallback; keep project dotenv untrusted for authorization-sensitive settings; and classify server-facing fields as policy or runtime identity. Tests should target rank and merge behavior, generation retention, failed managed refresh, handoff validation, and policy/runtime drift—not only parsing.

Focused tests in `test_config_manifest.py` verify manifest/runtime alignment, bounded resolution, invalid-value fallback, and secret redaction. Resolver, workspace, and server-graph tests cover precedence, generation and refresh behavior, durable binding, scoped project policy, safe diagnostics, refusal, and rebuild paths.
