---
type: concept
title: Tools and Filesystem Access
description: How Deep Agents and dcode compose model-visible tools, route filesystem operations through backends, and keep capability, path policy, approval, and shell execution separate.
tags: [tools, filesystem, execution, middleware, backends, permissions, mcp]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-16T08:05:50.355Z
sources:
  - resource: repo://libs/deepagents/deepagents/middleware/patch_tool_calls.py
  - id: openwiki-source-f84c83d6fab6028c94be90bc
    resource: repo://libs/deepagents/deepagents/backends/local_shell.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
generated: { by: "openwiki/0.4.2", at: "2026-09-16T08:05:50.355Z" }
---

# Tools and Filesystem Access

A tool's presence is not one authorization decision. The system separates **tool composition and model visibility** (schemas bound for the model), **backend capability** (operations the resolved backend can perform), **filesystem path policy and approval** (whether a particular file-tool call may proceed), and **shell execution** (an independent, potentially unrestricted capability). A visible tool can still return a capability or permission error, or pause for review.

```mermaid
flowchart TD
  Builtins["Middleware built-ins"] --> Candidate["Candidate tool registry"]
  Caller["Caller and dcode tools"] --> Candidate
  MCP["Loaded MCP tools"] --> Candidate
  Candidate --> Capability["Backend capability filtering"]
  Capability --> Exclusion["Profile exclusions"]
  Exclusion --> Visible["Model-visible request tools"]
  Visible --> Call["Model tool call"]
  Call --> Guard["Approval and path-policy checks"]
  Guard --> Backend["Backend operation"]
```

This shows composition-time visibility separately from per-call enforcement and backend execution.

## Tool assembly, visibility, and call repair

`create_deep_agent` composes a middleware stack that contributes filesystem tools and, when configured, subagent delegation tools; caller `tools=` join that construction. In dcode, extension ownership is resolved before `create_deep_agent`: an extension replaces a colliding supplied tool or middleware, its units are added, and extension runtime middleware is installed. A name should therefore have one intended owner in a dcode configuration.

Profiles can rewrite descriptions and suppress tool names. Description overrides copy and rewrite dict tools and `BaseTool` instances, but leave plain callables alone rather than mutating or wrapping caller-owned values. With non-empty `excluded_tools`, graph construction appends `_ToolExclusionMiddleware` after custom middleware. It removes excluded names from sync and async model requests and rejects an emitted excluded call with `Error: <name> is not available.` The second check is necessary because the tool executor still registers and dispatches tools by name. Exclusion keeps advertised and callable surfaces consistent; it is not an authorization boundary.

The base stack places `PatchToolCallsMiddleware` after summarization. Before an agent run, it scans all prior `AIMessage` tool calls, including invalid calls, and appends an error `ToolMessage` for every call ID that has no recorded answer. Thus a resumed, interrupted, cancelled, or malformed call does not leave a dangling protocol obligation in message history. It replaces the message collection atomically only when a repair is needed.

### dcode catalog and MCP tools

The `dcode tools list` command and interactive `/tools` command enumerate real bound tools instead of maintaining a second catalog. They compile a CLI agent with an offline placeholder model, then inspect its bound tool node; no credentials or model network call are needed. The catalog forwards the filesystem allowlist. Since filesystem middleware does not instantiate disallowed tool factories, enumeration should already be limited; a defensive leak check logs an error and deliberately returns the unfiltered list if that invariant fails.

MCP discovery is a separate source of tools. dcode wraps each remote tool as an asynchronous `StructuredTool` backed by `MCPSessionManager`; it normalizes arguments, calls the original remote name through that server session, and retains provenance metadata. Optional prefixing sanitizes server and tool names for providers, enforces a 64-character maximum, and gives changed or overlong names a deterministic SHA-256-derived suffix. Discovery is bounded and concurrent, while its server information remains in configuration order and its combined tools are name-sorted. Unavailable servers report status rather than silently contributing no tools.

## Filesystem tools and backend contract

`FilesystemMiddleware` owns the model-facing filesystem tools. It accepts an initialized `BackendProtocol` instance and defaults to ephemeral `StateBackend`; a callable backend factory is rejected. The backend owns storage and filesystem operations, while middleware validates inputs, applies tool-level path policy, formats `ToolMessage` output, and manages context eviction.

The fixed filesystem vocabulary is `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`.

| Tool | Role |
| --- | --- |
| `ls` | List directory entries. |
| `read_file` | Read a paginated file window. |
| `write_file` | Create or replace a file. |
| `edit_file` | Perform exact string replacements in an existing file. |
| `delete` | Recursively delete a file or directory when supported. |
| `glob` | Find matching regular files. |
| `grep` | Search literal text. |
| `execute` | Run a command only when the backend provides shell execution. |

Backends return structured results rather than preformatted text. `ReadResult` validates pagination at construction: window fields must occur together, bounds must be forward and within `total_lines`, and `next_offset` must be the line immediately after the returned window. Middleware, not the backend, applies line-number formatting. `GrepResult` and `GlobResult` can carry valid but incomplete matches with `truncated=True`; truncation is not a hard failure or proof that no additional matches exist.

### Visibility allowlist and capability gating

`FilesystemMiddleware(tools=...)` is a model-visibility allowlist, not a path-permission policy. `None` and `"all"` opt into all names; a list constructs only listed factories, so omitted tools never reach the dispatchable node. An explicit list must include `read_file`, otherwise construction raises `ValueError`.

Before both sync and async model calls, the middleware filters tools that the resolved backend cannot serve. `execute` requires `SandboxBackendProtocol`; `delete` requires a backend implementation rather than the protocol's default `NotImplementedError`. If `execute` somehow reaches its implementation without support, it returns an execution-not-available error. This request pass also rewrites `grep` and `execute` descriptions for the active search/execution tools and, when execution is active, adds composite-backend shell-path routing guidance.

`grep` is literal substring search, not regex. Its default total match cap is `grep_max_count=1000`; a call can override it with `max_count`, and `None` disables the default. The asynchronous protocol wrapper applies a wait timeout and enforces the requested cap even if an older concrete backend does not accept `max_count`. For actual regex, the `grep` description recommends `rg` through `execute` only if execution is available.

Large results from tools outside the filesystem set can be evicted beneath the backend artifacts root so the model sees a preview and file reference. `ls`, `glob`, `grep`, `read_file`, `edit_file`, `write_file`, and `delete` are excluded because they truncate themselves, have awkward reread behavior, or provide compact confirmations. Large human messages have a related lifecycle: full content remains in state while the model request receives a tagged preview and filesystem reference.

## Filesystem path policy and HITL

`FilesystemPermission` rules are enforced inside filesystem tool implementations, not by removing schemas. Rules use wcmatch operation-and-path matching and return the first matching `allow`, `deny`, or `interrupt` decision. Denied operations return an error, and denied paths are filtered from list and search results.

Exact-path tools (`read_file`, `write_file`, `edit_file`) test their one target. Bulk tools (`ls`, `glob`, `grep`, `delete`) must interrupt when their search subtree may overlap an anchored protected prefix. A pathless bulk call such as `grep(path=None)` conservatively fires for any relevant interrupt rule; `glob` also considers an absolute pattern that can redirect its search outside the supplied path. Graph assembly converts interrupt-mode permission rules into predicates for `HumanInTheLoopMiddleware`, which owns pausing and approval. A preceding deny wins for exact-path calls, so a denial does not become an approval request.

Permission patterns must start with `/` and cannot contain `..` or `~`. Permissions combined with an execution-capable backend are rejected unless every rule path is scoped to routes, because arbitrary shell commands cannot be governed by tool-level filesystem permissions. This is a deliberate boundary: filesystem path policy does not sandbox shell execution.

## Shell execution and route paths

`LocalShellBackend` is execution-capable because it extends `FilesystemBackend` and implements `SandboxBackendProtocol`, but its name must not be read as a security guarantee. It runs `subprocess.run(..., shell=True)` directly on the local host with the user's permissions. `virtual_mode` affects filesystem-tool path mapping only; it does not confine shell commands. Treat it as appropriate for trusted local development, not untrusted or multi-tenant production workloads, and use HITL when it is enabled.

Its default command timeout is 120 seconds and its output is capped at 100,000 bytes by default. It runs commands with `root_dir` as the working directory, combines stdout and stderr, reports nonzero exit codes, and marks capped output as truncated. Middleware also rejects a requested command timeout above its positive `max_execute_timeout`, which defaults to one hour.

In a `CompositeBackend`, file-tool paths can be virtual routes while `execute` runs only on the default backend's shell. The middleware does not rewrite a command: if the default is `LocalShellBackend`, it supplies the model prefix substitutions for local `FilesystemBackend` routes. Routes on remote/sandbox defaults or store-backed routes have no shell mapping and must be accessed through file tools.

## Focused tests and operations

The state-backend integration tests demonstrate that two parallel `write_file` calls merge file updates correctly, while invalid paths become `ToolMessage` errors. Concurrent `edit_file` calls to the same file are deliberately marked `xfail`: reducers and backends can race, so prompt or application logic should avoid them until explicit rejection or serialization exists.

When troubleshooting, identify the layer first:

1. **Absent from model choices:** inspect the filesystem `tools=` list, installed middleware, profile exclusions, extension collisions, and MCP server status.
2. **`execute` or `delete` absent:** inspect backend capability. An allowlist cannot manufacture a missing implementation.
3. **Visible but rejected or paused:** distinguish exclusion, backend error, filesystem denial, and HITL interruption; they have different owners.
4. **A resumed conversation has unmatched calls:** inspect `PatchToolCallsMiddleware` output; its synthetic error explains whether the call was malformed or produced no recorded result.
5. **Search seems incomplete:** inspect `truncated`, then narrow the path or pattern.
6. **Shell cannot see a file-tool path:** inspect composite route guidance; use file tools for mounts without a host mapping.

## Related pages

- [Middleware stack](../architecture/middleware-stack.md) — middleware responsibilities and ordering.
- [Backends](backends.md) — backend implementations, routing, and sandbox capability.
- [Context management](context-management.md) — eviction and context-window behavior.
- [Permissions & HITL](permissions-hitl.md) — approval policy and interrupts.
- [Security](../operations/security.md) — operational security considerations.
