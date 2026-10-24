---
type: configuration architecture
title: dcode Configuration Layering
description: How dcode resolves typed settings across managed policy, CLI values, environment, user TOML, and defaults. Covers provenance, reload generations, dotenv trust boundaries, MCP preflight validation, and the client-to-server policy handoff.
tags: [configuration, deepagents-code, precedence, managed-policy, environment, reload, workspace, server]
sources:
  - resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - resource: repo://libs/code/tests/unit_tests/test_server_manager.py
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-1728494bdd59604ce9b5f65b
    resource: repo://libs/code/deepagents_code/_server_config.py
  - id: openwiki-source-2fb89d2b59c886d0cb3ee3ea
    resource: repo://libs/code/deepagents_code/config_manifest.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-a9ed482415c6566921385305
    resource: repo://libs/code/deepagents_code/configuration/paths.py
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
  - id: openwiki-source-2843d86bbaef4173c77ca3e1
    resource: repo://libs/code/tests/unit_tests/test_config.py
  - id: openwiki-source-149abfd7a1ab6a5a2d1a0e71
    resource: repo://libs/code/tests/unit_tests/test_configuration_resolver.py
  - id: openwiki-source-5a5147d4654f226b03e92ab9
    resource: repo://libs/code/tests/unit_tests/test_workspace_diagnostics.py
  - id: openwiki-source-877b53371bf970f1b38a1809
    resource: repo://libs/code/tests/unit_tests/test_workspace.py
verified:
  - by: openwiki/0.4.2
    at: 2026-10-24T08:05:08.866Z
generated: { by: "openwiki/0.4.2", at: "2026-10-24T08:05:08.866Z" }
---

# dcode Configuration Layering

`dcode` resolves configuration through a typed, ranked provider chain rather than treating one file as the entire configuration surface. This preserves useful session, shell, and user settings while ensuring that organization-managed policy has the strongest authority. The same design makes both the winning value and its contributing source tiers inspectable.

For the client/server boundary, see [Code agent](/openwiki/architecture/code-agent.md). Related concerns include [permissions and HITL](/openwiki/concepts/permissions-hitl.md), [MCP](/openwiki/integrations/mcp.md), and [running a dcode session](/openwiki/workflows/run-dcode-session.md).

## Configuration surface and providers

`config_manifest.py` is the canonical description of scalar settings. A `ConfigOption` identifies its key, type/coercion kind, typed default, TOML path, environment variables, optional CLI binding, merge strategy, and whether display must redact its raw value. Structured, user-defined tables are not flattened into scalar environment variables: they are marked `STRUCTURED` for discovery and read by dedicated typed loaders. Credential-capable structured tables are presence-only in `dcode config` output.

Each provider normalizes its own input into `Found`, `Unset`, or `Invalid` before the resolver considers precedence. Environment and TOML coercion rejects wrong types and invalid tokens without discarding the rest of that source: an invalid setting falls through to a weaker valid tier, while valid siblings remain usable. `ResolvedValue` retains the selected ranks, provider health, diagnostics, masked ranks, and rank/path provenance.

## Precedence, durable policy, and composition

Lower numeric rank is stronger. The standard chain has a managed TOML tier, an optional parsed-CLI tier, live environment values, user `config.toml`, and typed defaults. A caller may also install a retained reload override tier.

```mermaid
flowchart TD
    M["Managed policy rank 200"] --> C["CLI rank 300"]
    C --> R["Reload override rank 350 when installed"]
    R --> E["Environment rank 400"]
    E --> U["User config.toml rank 500"]
    U --> D["Manifest defaults rank 1000"]
```

The standard configuration tier order. The reload tier is caller-installed rather than universal.

For ordinary `replace` options, the strongest usable `Found` value wins. A durable result at a stronger rank masks weaker non-durable values, preventing an environment value from bypassing a managed or durable policy result; the masking is directional, so a weaker user-file result cannot hide a stronger environment value.

`union` and `deep_merge` mean a setting is intentionally cumulative. They retain every valid tier contribution rather than selecting a winner: unions preserve ordered, deduplicated list entries, and deep merges apply stronger leaves over weaker tables while retaining leaf-level provenance. Defaults are excluded from accumulated contributions and used only as a fallback. This matters for restrictions such as MCP deny lists, where allowing a user replacement to erase an administrator denial would be fail-open. If an accumulating tier has an incompatible shape, resolution uses a copied strongest value rather than allowing a weaker value to displace policy.

`resolver_from_snapshots()` takes keyword-only `managed=` and `user=` snapshots. Since both arguments have the same type, this blocks an accidental positional swap that would grant a writable user configuration managed precedence. `ConfigResolver` also rejects duplicate ranks.

## Managed policy boundary, generations, and writes

Managed configuration comes from an operating-system-controlled fixed path: `/etc/dcode/managed_config.toml` on Linux-like platforms, `/Library/Application Support/dcode/managed_config.toml` on macOS, and `ProgramData/dcode/managed_config.toml` on Windows. Production Windows lookup uses the HKLM registry value for ProgramData rather than a user-controlled process environment variable. If that registry lookup fails, the fallback path is explicitly marked indeterminate rather than reported as confidently absent policy.

Managed policy is durable throughout reload. A provider that previously loaded a usable TOML snapshot keeps serving it when a later candidate is unreadable or malformed; an initial unreadable source instead contributes no values and lets resolution continue. A managed refresh is fetched and policy-validated before installation. An unusable or unenforceable candidate cannot replace the active managed tier, so a broken refresh does not silently discard restrictions and reveal weaker values.

The shared resolver caches one generation for normal process reads. Files are not watched: an on-disk edit is not visible until reload, resolver invalidation, or a committed write to the default user configuration path advances the generation. The environment provider deliberately differs: it reads the active environment at resolution time. `get_config_resolver()` owns the process cache and prevents installation of a second, different CLI provider in the same process.

`update_user_config()` is the supported user-`config.toml` transaction. It serializes read-modify-write operations, rejects the managed path and malformed existing TOML, writes a replacement beside the destination, and atomically replaces the destination. Once committed, it best-effort refreshes the shared resolver for the default path; a refresh error does not misreport the completed write as failed.

## Environment and dotenv trust boundary

For prefix-aware credentials and provider settings, `DEEPAGENTS_CODE_{NAME}` takes precedence over `{NAME}` by *presence*, including an empty prefixed variable. An explicit empty override can therefore suppress a canonical credential or setting.

Workspace construction derives an explicit dotenv environment rather than mutating the process environment as part of every read. The inherited shell environment wins; the nearest project `.env` fills absent keys, followed by the trusted global profile dotenv for keys still absent. Each dotenv file interpolates against the original environment baseline, so project values cannot steer interpolation in the global file. `use_environment()` exposes such a mapping as an immutable, context-local snapshot during construction; outside that scope reads use live `os.environ`.

A project `.env` is untrusted checkout content. It cannot set project-MCP trust lists, Auto classifier model or timeout, MCP tool timeout, forked-subagent behavior, the upstream recursion fallback, or `TERM_PROGRAM`. The trusted global dotenv controls whether the project file may be read; if that global file cannot be read, dcode fails closed by skipping the project dotenv unless another trusted source explicitly opts in.

Each server runtime is built from a frozen workspace dotenv and credential snapshot, so later process-environment changes do not change an existing runtime. LangSmith selectors and the redaction setting must agree across workspaces on a server because the SDK uses process-wide state. If an active upload target requests secret redaction but dcode cannot install the redacting client, it disables tracing instead of uploading unredacted data.

## MCP configuration and launch preflight

MCP configuration interpolates only `${VAR}` and `${VAR:-default}` in `command`, `url`, `args`, `env`, `headers`, and `oauth`. The default form is used when the variable is absent or empty. Unsupported fields are copied unchanged. Malformed references, an unset required variable, or an invalid supported-field shape are rejected before the resolved configuration is used.

Disabled MCP servers are cumulative: user and managed names are unioned, then filtered before validation, connection, or tool exposure. If the managed deny list cannot be read or validated, dcode fails closed by disabling every candidate server rather than interpreting a broken restriction as an empty list.

An explicit `--mcp-config` receives an additional parent-process preflight before server construction. `start_server_and_get_agent()` skips it when `--no-mcp` is active or no explicit path was supplied; otherwise it calls `load_mcp_config()`, preserves `MCPConfigError`, translates a missing file and ordinary value/type validation failures into a path-bearing `MCPConfigError`, and deliberately lets unrelated `RuntimeError` failures propagate. This produces an actionable startup error instead of a server-side health-check failure with truncated logs. Discovered project and user MCP configurations are intentionally not part of this fatal preflight: their loader errors remain viewer-visible entries.

## Client/server handoff and workspace continuity

The interactive client and agent server are separate processes and do not share a resolver cache. `ServerConfig` is their typed handoff: the client resolves CLI paths against the launch project, serializes fields to `DEEPAGENTS_CODE_SERVER_*` environment variables, and the server reconstructs the schema. `None` clears a previous variable rather than becoming an empty string. This avoids a server launched in a temporary runtime directory interpreting a relative MCP, sandbox-setup, or extension path against the wrong directory.

```mermaid
flowchart TD
    A["CLI options and project context"] --> B["Validate explicit MCP config"]
    B --> C["Build ServerConfig"]
    C --> D["Serialize server environment"]
    D --> E["Spawn langgraph server"]
    E --> F["ServerConfig.from_env"]
    F --> G["Bind server-resolved workspace policy"]
```

The launch process validates the explicit MCP file before serialization, then the server reconstructs and binds configuration independently.

The filesystem-tool allowlist is security-sensitive serialization. An absent `ALLOW_FS_TOOLS` means unrestricted SDK-default filesystem tools, but a present value must parse as a non-empty JSON list of recognized tool names. `ServerConfig` also requires an explicit list to contain `read_file`; malformed JSON, a wrong shape, unknown names, an empty list, or omission of `read_file` raises rather than falling through to unrestricted access.

A binding canonicalizes an existing workspace directory and records server-resolved, non-secret policy plus separate policy and runtime fingerprints in SQLite. It atomically refuses a different workspace or incompatible policy. A compatible runtime-only identity change rebuilds the runtime without deleting the binding or checkpoints.

Project-scoped MCP, sandbox setup, extension paths, and extension trust apply only to the launch project. Resolving another or unresolvable project removes project MCP and sandbox grants and re-evaluates extension trust. On execution, policy drift or revoked extension trust refuses the request; runtime-only identity drift is a rebuild condition.

## Diagnostics and safe change practice

Bindings retain a bounded allowlisted snapshot for refusal diagnostics. Only reportable policy scalars and lists are persisted. Paths, credentials, environment values, model data, and prompts are excluded from both persistence and logs; diagnostics either identify safe changed values or state that the old value is unavailable.

When adding or changing configuration, define scalar options in the manifest or provide a dedicated structured loader; choose whether the setting replaces, unions, or deep-merges; preserve invalid-value fallback; and do not let project dotenv define authorization-sensitive controls. Server-facing changes must be classified as durable policy identity or rebuildable runtime identity. Explicit launch-only files need parent-side validation when a server-process error would otherwise be opaque.

Focused tests cover rank ordering, durable masking, union/deep-merge provenance, malformed-value fallback, source retention after failed reload, managed-refresh safety, default-path writes, dotenv precedence and trust boundaries, workspace snapshot isolation, server handoff validation, MCP preflight behavior, and policy/runtime drift.
