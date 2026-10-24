---
type: architecture concept
title: Filesystem Tools and Backends
description: Deep Agents and dcode expose filesystem and execution tools through capability-aware middleware backed by pluggable storage and sandbox implementations. This page explains tool allowlists, routing, permission and approval controls, content lifecycle, and the containment boundaries verified by focused tests.
tags: [tools, filesystem, execution, middleware, backends, persistence, multimodal, permissions]
sources:
  - id: openwiki-source-1728494bdd59604ce9b5f65b
    resource: repo://libs/code/deepagents_code/_server_config.py
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-a12868876b60484607797187
    resource: repo://libs/code/deepagents_code/file_ops.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-752b52e8fedb86515652254d
    resource: repo://libs/code/deepagents_code/managed_tools.py
  - id: openwiki-source-f6d553e7afdf54acac36e7d3
    resource: repo://libs/code/deepagents_code/mcp_tools.py
  - id: openwiki-source-3300d75e0c132882e2e3b4ce
    resource: repo://libs/code/deepagents_code/tool_catalog.py
  - id: openwiki-source-8ccbc389e8d6198afea76baa
    resource: repo://libs/code/tests/unit_tests/test_file_ops.py
  - id: openwiki-source-f598809da8d8fbff2d7ae090
    resource: repo://libs/code/tests/unit_tests/test_server_manager.py
  - id: openwiki-source-a1549ea98d425efea270be93
    resource: repo://libs/deepagents/deepagents/backends/composite.py
  - id: openwiki-source-e483ff4cfd25918c8107d575
    resource: repo://libs/deepagents/deepagents/backends/filesystem.py
  - id: openwiki-source-f84c83d6fab6028c94be90bc
    resource: repo://libs/deepagents/deepagents/backends/local_shell.py
  - id: openwiki-source-e3efb5f3e4a9e8517eb6d8f5
    resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-d4463137befa776cd47750d4
    resource: repo://libs/deepagents/deepagents/backends/sandbox.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-303a7196a0e1a36cc078621b
    resource: repo://libs/deepagents/deepagents/middleware/_blob_offload.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-903e05891b2ddf4f958276fd
    resource: repo://libs/deepagents/tests/unit_tests/test_local_sandbox_operations.py
generated: { by: "openwiki/0.4.2", at: "2026-10-24T08:05:08.866Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-10-24T08:05:08.866Z
---

# Filesystem Tools and Backends

`FilesystemMiddleware` is the model-facing adapter over an initialized `BackendProtocol`: it creates tools, validates and formats their results, applies middleware-owned filesystem policy, and manages large or binary message content. The backend owns storage and operation implementation. A visible tool is not a guarantee that the backend implements it. SDK profile exclusions are model-surface controls rather than authorization, while dcode treats its explicitly configured filesystem allowlist as a security control and carries it into configurable subagents. See [Backends](backends.md) for storage and routing, [Permissions & HITL](permissions-hitl.md) for approval configuration, and [MCP](../integrations/mcp.md) for remote tools.

## Tool surface and request boundary

The built-in vocabulary is `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`. `tools=None` and `tools="all"` build all of them; an explicit list must include `read_file`, and tools absent from that list are not registered with the dispatchable tool node. At model-request time, filesystem middleware removes `delete` when the backend does not implement the optional operation and removes `execute` unless the backend is execution capable. It also adjusts the `grep` and `execute` descriptions to the remaining visible tools.

A harness profile's `excluded_tools` is a later, independent model-surface control. Graph construction appends `_ToolExclusionMiddleware` after profile and caller middleware so it removes named tools after all injection has happened; it also rejects a model call that still names an excluded tool. This makes the advertised and executable surfaces agree for that agent/profile, but it is explicitly not a security control. It is distinct from an absent construction-time tool, capability filtering, a filesystem permission denial, and a HITL interruption.

```mermaid
flowchart TD
    Config["Middleware configuration"] --> Factories["Create allowed tool factories"]
    Factories --> Capability["Remove unsupported delete and execute"]
    Capability --> Exclusion["Profile excludes named tools last"]
    Exclusion --> Model["Model-visible tools"]
    Model --> Call["Tool call"]
    Call --> Check{"Excluded name"}
    Check --> Reject["Reject excluded name"]
    Check --> Guard["Validate path and enforce policy"]
    Guard --> Backend["Backend file operation or execution"]
    Backend --> Result["Structured result and ToolMessage"]
```

This flow separates construction-time registration, request-time capability and profile filtering, tool-call exclusion rejection, per-call policy, and backend behavior.

`BackendProtocol` returns structured results rather than presentation strings. `ReadResult` rejects incoherent pagination metadata; the middleware owns line-number formatting and long-line continuation rows. `grep` and `glob` can be successful yet incomplete, so `truncated=True` means absence from returned matches is not proof of absence from storage. When a backend can say why a glob is incomplete, `GlobResult.truncation_reason` distinguishes a time/match **budget**, an **unreadable** subtree, and clipped sandbox **transport**; only the budget case is plausibly addressed by narrowing the search. The agent-facing `grep` is literal, not regex; its configured default total cap is 1,000 matches, and a positive per-call `max_count` can override it. Set `grep_max_count=None` when constructing the middleware to disable the default cap.

## dcode composition, enumeration, and trust

At runtime, dcode supplies caller tools (for example, `fetch_url`, `get_current_thread_id`, and credential-gated `web_search`) to its agent construction, and separately loads MCP tools. The filesystem tools remain SDK middleware tools rather than a parallel dcode implementation. An MCP-enabled agent wraps those remote calls with dcode's MCP timeout middleware; MCP loading can be disabled entirely with `no_mcp`.

MCP availability is deliberately not inferred from configuration alone. Project MCP servers, including remote HTTP/SSE servers, are trust-gated because discovery may reach a server or interpolate headers. Whole-project trust, scoped user approvals keyed to project root and server fingerprint, and explicit user denial are applied before loading; plugin-provided servers are treated as install-trusted but still obey user denial and fail closed if that policy cannot be read.

```mermaid
flowchart TD
    Config["dcode configuration"] --> Builtin["SDK filesystem middleware"]
    Config --> Caller["Caller supplied tools"]
    Config --> McpConfig["MCP configuration and trust gate"]
    McpConfig --> Mcp["Loaded MCP tools"]
    Builtin --> Agent["Bound agent tool node"]
    Caller --> Agent
    Mcp --> Agent
    Agent --> Catalog["tools list or slash command"]
```

This shows composition, not equivalent trust: caller tools are supplied by the application, while MCP tools must pass configuration and trust resolution before they enter the bound set.

`dcode tools list` and interactive `/tools` derive the built-in listing from the actual tools bound on an agent compiled with an offline placeholder model, rather than maintaining a display catalog. It forwards the session filesystem allowlist into that compilation and deliberately reports an allowlist leak rather than cosmetically filtering it. MCP tools are grouped by server; non-`ok` servers with no tools are surfaced as unavailable, and a discovery-wide failure leaves built-ins visible with a generic notice rather than leaking exception details.

`--allow-fs-tools` accepts `all` or a case-insensitive comma-separated subset of the eight SDK filesystem names. Omission and `all` leave the SDK's unrestricted default middleware in place; an explicit empty list, unknown name, mixed `all`, or subset without `read_file` is rejected. dcode installs a restricted replacement middleware for an explicit subset and appends equivalent middleware to raw subagent specifications. It rejects compiled subagents instead of silently letting delegation bypass that restriction.

### Server-launched sessions

`start_server_and_get_agent` is the process boundary used by dcode's server-backed client. It first preflights an explicit MCP configuration, builds a `ServerConfig` from the launch options—including `allow_fs_tools`, sandbox selection, and shell settings—and serializes that configuration into the server environment before spawning a temporary LangGraph server. Once the `agent` graph is ready, it creates a `RemoteAgent` and binds the resolved workspace plus a configuration claim and fingerprint. The claim intentionally contains session policy such as the filesystem allowlist, but not the MCP configuration path or project-MCP trust decision.

If startup fails or the await is cancelled before ownership is transferred to the caller, the manager reaps the server in its `finally` block and runs the blocking stop off the event loop. `server_session` then provides the normal owner-facing lifecycle: it delegates startup and stops the returned process on context exit. This protects against an orphaned local server; it does not itself make shell execution or filesystem access safe.

## Managed helper and local presentation lifecycle

The `grep` filesystem tool is backend functionality; dcode optionally provisions its `rg` helper. `ensure_ripgrep` first honors a `system` installer choice, accepts a current checksum-verified managed binary, can use a system binary when no managed copy exists, and otherwise downloads a pinned platform asset, verifies archive SHA-256, extracts it, then atomically installs and exposes it on `PATH`. Offline mode returns no managed binary; unsupported platforms may use a system `rg` or raise a remediation-oriented availability error. A profile fallback binary directory is never put directly on `PATH`: dcode copies a verified `rg` into a process-private shim so repository-controlled sibling executables cannot be shadowed in.

The terminal and non-interactive dcode surfaces also track `read_file`, `write_file`, `edit_file`, and `delete` calls for diffs, metrics, and approval presentation. A tracker captures a pre-image through the configured backend when available and reads a post-image after a successful write or edit. It distinguishes a trustworthy diff from an unavailable pre-image, unavailable post-image, and a line-terminator-only change; it therefore suppresses invented diff counts instead of representing unknown content as an empty file. Sensitive filename heuristics redact display of `.env` variants, known credential names, and private-key/keystore suffixes, failing closed when the path cannot be parsed. This is a UI/transcript safeguard, not filesystem authorization.

Focused unit tests cover the failure semantics: a missing pre-image is normal only when creating with `write_file`; malformed backend responses and failed reads are marked untrustworthy; a successful write with failed read-back retains request-derived metrics and its reason; delete previews distinguish text, directories, and unreadable targets; and credential-like paths are classified for redaction.

## Composite routes are a path namespace, not a shell namespace

`CompositeBackend` chooses the longest matching route prefix for a file path. It strips that prefix before delegating to the routed backend and restores it on path-bearing results, while unmatched paths go to the default backend. A listing of `/` combines the default root with synthetic route directories. This lets one model-visible tree span state, persistent, local, or remote backends without exposing each backend's internal root.

Search has deliberate cross-route semantics. With `path` omitted or `/`, `grep` searches the default and routes under one global match budget, remapping route result paths; if an earlier backend fills the budget, later routes are skipped and the result is conservatively marked truncated. `glob` also merges and remaps results, but rewrites a route-targeting anchored pattern relative to that route, forwards a bare pattern into every route, and skips routes for a root-anchored pattern that cannot match beneath their prefix. An error from any searched backend is surfaced rather than represented as a successful partial glob. These details matter when interpreting an empty or truncated search result.

File operations route individually, but `execute` does not: a composite always delegates execution to its default backend. Consequently, a route prefix is a file-tool addressing mechanism, not evidence that the default shell can access the same path.

## Direct filesystem access and permission policy

`FilesystemBackend` reads and writes the host filesystem directly. Its default `virtual_mode=True` treats `root_dir` (or the process working directory) as a virtual root: `/x` maps below that root, relative inputs are normalized into that virtual namespace, and traversal through `..` or `~` is refused. Resolved paths must remain under the root; agent-visible paths use virtual absolute form and do not reveal the host root. `virtual_mode` provides stable path semantics—especially when `CompositeBackend` strips a route prefix before forwarding a path—but is not a sandbox or process-isolation boundary. With `virtual_mode=False`, absolute paths are used as-is and relative paths can escape `root_dir`.

`FilesystemPermission` rules classify operations as `read` or `write` and have `allow`, `deny`, or `interrupt` mode. Patterns must be absolute and may not contain `..` or `~`. For ordinary exact operations, declaration order is significant: the first matching rule decides. Middleware checks both the requested normalized path and, for `FilesystemBackend` including routed composites, its symlink-resolved target; an unresolved target is refused when rules are configured. List and search results filter denied entries, including aliases that resolve to denied targets.

`interrupt` is not enforced inside a filesystem tool. During `create_deep_agent` assembly, `_fs_interrupt` turns interrupt rules into `HumanInTheLoopMiddleware` predicates. Exact tools (`read_file`, `write_file`, `edit_file`) prompt only for matching paths unless a matching deny wins. Bulk tools (`ls`, `glob`, `grep`, and `delete`) prompt if their target subtree could overlap an interrupt rule; an omitted path or current-directory alias is treated as root. The glob predicate also examines absolute patterns and conservatively prompts for relative patterns containing `..`, preventing a benign `path` argument from bypassing a protected target.

```mermaid
flowchart TD
    Call["Filesystem tool call"] --> Validate["Normalize and validate path"]
    Validate --> Scope{"Exact or bulk scope"}
    Scope -->|Exact| Match["Check requested and resolved paths"]
    Scope -->|Bulk| Overlap["Check possible subtree overlap"]
    Overlap --> Interrupt["HITL predicate may pause"]
    Match --> Deny{"Deny rule"}
    Deny -->|Yes| Refuse["Permission error"]
    Deny -->|No| Operation["Backend operation"]
    Interrupt --> Operation
    Operation --> Filter["Filter denied bulk results"]
```

This policy flow shows the boundary: graph assembly owns approval predicates, while filesystem middleware denies calls and filters results. A recursive or ambiguous delete is stricter: it fails closed if a deny-write pattern could overlap its subtree, while a confirmed leaf uses the normal first-match rule.

## Execution is a separate trust boundary

`execute` requires `SandboxBackendProtocol`; for a composite, only the default backend supplies execution. The middleware validates a supplied timeout against its positive `max_execute_timeout` setting (default 3,600 seconds) and whether the backend accepts per-command overrides. A command invocation that completes gets a successful tool message and an exit-code artifact even if the command exits nonzero; callers must inspect that exit code. Capture-capable sandbox execution can write oversized command output at the source; otherwise generic message eviction applies.

Filesystem permission patterns do not parse or constrain command syntax. Therefore, middleware rejects permissions combined with an execution-capable backend unless every permission pattern is scoped beneath a `CompositeBackend` route; that exception separates governed file-tool routes from the default executor but does not authorize or confine shell commands. A `LocalShellBackend` runs `shell=True` on the host as the current user, with no isolation, and its virtual paths restrict only inherited file operations—not commands. Use an isolated sandbox and HITL for untrusted work.

`SandboxBackendProtocol` is the extension contract that adds a sandbox identifier and synchronous/asynchronous command execution; it does not by itself guarantee a particular isolation mechanism. `BaseSandbox` builds `ls`, paginated `read`, `grep`, `glob`, `write`, and `edit` on that execution primitive and upload/download transport. Consequently, those helpers inherit the trust boundary of the subclass's `execute()` implementation rather than creating a second containment boundary. Capture-at-source output offload is opt-in on `BaseSandbox`; when enabled and its artifact path resolves to the executing backend, large combined command output can remain in the sandbox with a model-facing preview and `read_file` pointer.

The local sandbox operation suite is a behavioral contract, not evidence of host isolation. It runs only with `RUN_SANDBOX_TESTS=true` and supplies a `LocalSubprocessSandbox` test double whose `execute` uses local `subprocess.run(..., shell=True)`. The double maps a virtual test prefix to a temporary real directory so tests can verify structured file results, pagination, errors, edits, search, and uploads without exposing that directory in returned paths. In particular, CRLF files are read as LF for the model, while edit accepts the LF view and preserves the matched file section's line-ending style; mixed-ending `replace_all` affects the first matching style only. Production sandbox containment must instead be established by the selected backend and is covered in [Sandbox partners](../integrations/sandbox-partners.md).

When execution is active, middleware adds route-to-host mapping guidance only for composite routes that share a local `LocalShellBackend` filesystem. Remote, sandbox, and store-backed routes have no shell mapping and must be accessed through file tools. It does not rewrite commands.

## Read media and content lifecycle

For text, `read_file` produces a paginated line-numbered result. For backend-declared base64, it emits a multimodal block selected from the extension (`image`, `audio`, `video`, or generic `file`) rather than line-numbered text, retaining the validated source path and MIME type in metadata. Video decoding, when optional dependencies are available, samples frames into a synthetic `HumanMessage`; middleware moves that message after the entire preceding `ToolMessage` batch to satisfy provider ordering.

`UnsupportedContentMiddleware`, placed after the core, memory, and HITL stack before later profile/caller stack shaping, copies model-request messages and replaces only blocks the active model explicitly cannot accept with a `read_file` path placeholder. Persisted message history remains unchanged. As a fallback, filesystem middleware retries once after a `ModelInvalidRequestError` only when the latest tool turn contains multimodal `read_file` material; it replaces that material with an unsupported-content notice and re-raises unrelated request errors.

Large text and binary payloads have separate lifecycles:

- **Text eviction:** oversized human text and ordinary tool text are written under backend artifact paths with a preview for the model request. Filesystem tools are excluded from generic tool-result eviction because their output is already paginated, bounded, or compact. An artifact-write failure leaves the original content intact.
- **Binary offload:** with `offload_binary_content=True`, valid inline base64 from `read_file` results and eligible human media are SHA-256-addressed under `blobs/`; history stores a reference instead of bytes, and the model request receives rehydrated media. Offload is disabled with a warning if the blob path routes to `StateBackend`, because that would preserve bytes in checkpoint state.

```mermaid
sequenceDiagram
    participant Tool as read_file tool
    participant MW as FilesystemMiddleware
    participant Store as Backend blob storage
    participant History as Message history
    participant Model as Model request
    Tool->>MW: Base64 media result
    MW->>Store: Upload digest-addressed bytes
    MW->>History: Store blob reference
    MW->>Store: Download referenced bytes
    MW->>Model: Send verified Base64 media
```

The binary lifecycle stores references in history but restores verified payloads only for a model request. Binary offload is best effort: failed uploads keep inline content. Rehydration uses an in-run untracked cache when possible and verifies every downloaded payload against its digest because blobs are agent-writable. Missing, malformed, failed, or digest-mismatched references become a text notice rather than untrusted media.

## Operational guidance

Keep `read_file` in explicit tool lists, and treat missing `delete` or `execute` as a backend capability issue rather than an allowlist issue. Use a harness profile exclusion to suppress a tool for a particular agent surface, not to authorize storage or isolate execution. Treat composite search truncation as incomplete output and do not assume file-tool route paths are shell paths. Configure binary offload only when the blob route supports upload/download outside checkpoint state. For direct host backends, keep virtual mode enabled unless trusted local workflows explicitly need host paths; do not mistake it for sandboxing.

When changing this area, exercise the focused behavior rather than only tool registration: virtual-root containment and symlink resolution; composite routing and merged search errors/truncation; request-time capability filtering and excluded-call rejection; permission first-match, recursive-delete, and bulk HITL cases; execution timeout/exit-code behavior; and offload upload, digest verification, and state-backed-route disablement. For server-backed dcode sessions, also test the environment round trip and fail-closed `allow_fs_tools` validation, workspace claim contents, cancellation-safe startup cleanup, and context-manager shutdown. Run the `BaseSandbox` local operation suite with `RUN_SANDBOX_TESTS=true` when modifying its portable file-operation behavior, but do not treat its local subprocess fixture as a sandbox security test.

## Related pages

- [Backends](backends.md) — storage, routing, and execution-capable implementations.
- [Middleware catalog](middleware-catalog.md) — composition and request/tool wrappers.
- [Permissions & HITL](permissions-hitl.md) — policy and approval configuration.
- [Sandbox partners](../integrations/sandbox-partners.md) — isolated execution integrations.
- [Security](../operations/security.md) — operational threat model and safeguards.
