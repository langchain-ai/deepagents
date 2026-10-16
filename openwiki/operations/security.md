---
type: security operations guide
title: Security Boundaries and Operational Safety
description: Operational security boundaries for Deep Agents, dcode, and Talon, covering tool authority, approval limits, OAuth transport and token handling, and safe terminal rendering.
tags: [security, operations, deepagents, filesystem, mcp, oauth, approvals, terminal-ui]
sources:
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
  - id: openwiki-source-4e1eab6b8c54443429c68b30
    resource: repo://libs/code/deepagents_code/_markdown.py
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-a9143c1c174362216a1cfa2c
    resource: repo://libs/code/deepagents_code/approval_mode.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-a97cce048cd7efd394ae7dca
    resource: repo://libs/code/deepagents_code/mcp_auth.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-52062c280ae38e9e9acab191
    resource: repo://libs/code/deepagents_code/thread_titles.py
  - id: openwiki-source-885087faefbbd5773a2d43ab
    resource: repo://libs/code/deepagents_code/tui/widgets/_links.py
  - id: openwiki-source-f820645728a7602f14195650
    resource: repo://libs/code/deepagents_code/tui/widgets/auto_mode_notice.py
  - id: openwiki-source-030d8bd153a9c3ea2a99cb7d
    resource: repo://libs/code/deepagents_code/workspace.py
  - id: openwiki-source-11d6c59d85493653aee76558
    resource: repo://libs/code/tests/unit_tests/test_app.py
  - id: openwiki-source-53083c05d51a08d395327737
    resource: repo://libs/code/tests/unit_tests/test_thread_titles.py
  - id: openwiki-source-877b53371bf970f1b38a1809
    resource: repo://libs/code/tests/unit_tests/test_workspace.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-851e3a9c96663d8db5ca3dec
    resource: repo://libs/deepagents/tests/unit_tests/test_permissions.py
  - id: openwiki-source-31e40ff79779f51cafd03f01
    resource: repo://libs/talon/deepagents_talon/mcp_auth.py
  - id: openwiki-source-111101dcd1462ff54277b1fc
    resource: repo://libs/talon/deepagents_talon/mcp_config.py
  - id: openwiki-source-caa081bf22565f9bef5bc7d1
    resource: repo://libs/talon/tests/unit_tests/test_oauth_transport.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
verified:
  - by: openwiki/0.4.2
    at: 2026-10-16T08:05:35.701Z
generated: { by: "openwiki/0.4.2", at: "2026-10-16T08:05:35.701Z" }
---

# Security Boundaries and Operational Safety

## Trust model: tools and the runtime are the boundary

Deep Agents follows a **trust-the-LLM** model: an agent can do what its configured tools and backend allow. Instructions in a prompt, a repository, tool output, an MCP response, or a generated title are data for a model, not an access-control mechanism. Put confidentiality, integrity, and egress boundaries in the backend, sandbox, operating-system identity, network policy, and explicit tool configuration—not in a system prompt or in the expectation that the model will decline unsafe instructions.

This distinction matters for all controls below:

- `FilesystemPermission` constrains Deep Agents' built-in filesystem **tools**; it does not mediate direct backend use.
- Human approval pauses a configured tool call; it is not a sandbox or a credential boundary.
- A workspace binding prevents a client or a stale thread from silently changing the server-selected workspace and resource policy; it does not make a workspace safe to execute.
- Project configuration, `.env`, MCP definitions, and repository contents should be treated as untrusted input until an operator or trusted policy admits them.

```mermaid
flowchart TD
    Input["Prompt repository MCP response or tool output"] --> Agent["Agent model and graph"]
    Agent --> Fs["Built-in filesystem tools"]
    Agent --> Mcp["MCP tools"]
    Agent --> Exec["Execution-capable backend"]
    Fs --> Rules{"Filesystem rules"}
    Rules -->|"deny"| Refuse["Permission error"]
    Rules -->|"interrupt"| Review["Human decision"]
    Rules -->|"allow"| Backend["Configured backend"]
    Review --> Backend
    Mcp --> Trust{"Project MCP trust"}
    Trust --> Backend
    Exec --> Backend
```

*The graph's policy mechanisms gate mediated tool calls; the selected backend, MCP configuration, process identity, and network controls ultimately define authority.*

## Filesystem permissions and approval flow

`FilesystemPermission` is a declarative rule containing `read` and/or `write` operations, absolute POSIX glob paths, and an `allow`, `deny`, or `interrupt` mode. Paths with `..` are rejected and `~` is unsupported. Rules are checked in declaration order, with the first matching rule taking effect; when no rule matches, the built-in filesystem tool is allowed. Design allowlists with a later catch-all deny when the intended policy is “only this subtree,” for example an allow under `/workspace/**` followed by a deny for `/**`.

### Denial applies to the mediated file operation—not to the backend

`deny` makes the filesystem tool return a permission error before it performs the operation. The middleware applies this to its built-in `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, and `delete` tools; it is not a backend-wide access-control layer, so direct backend calls do not acquire these rules.

For a real `FilesystemBackend`, including one reached through a `CompositeBackend` route, the middleware evaluates both the requested virtual path and its symlink-resolved target. A denial on either blocks the call; a target that cannot be resolved is an error rather than an allowed access. The same resolved-target check removes denied aliases from `ls`, `glob`, and `grep` results. This closes the obvious alias route—for example, `/work/link/key.txt` resolving into a denied `/secrets` tree—but it is a safeguard around these tools, not general symlink containment for arbitrary backend or shell use.

Deletion has deliberately different semantics depending on what the backend can establish. A confirmed plain-file delete follows normal first-match ordering. A directory or any target that may have descendants instead uses an all-or-nothing check: every deny-write pattern that could overlap the target subtree blocks removal, irrespective of an earlier allow. The overlap logic is conservative for recursive or directory-wildcard patterns, but permits a sibling that cannot match, such as deleting `/work/notes.txt` with a deny on `/work/*.log`. Thus a later catch-all deny can block a recursive delete even where a narrower earlier allow permits a leaf; do not model an allowlist as a permission to recursively erase its parent.

### Interrupt routing is scope-aware

`interrupt` is not a denial. During graph assembly it is translated into `HumanInTheLoopMiddleware` `interrupt_on` entries, with a per-call predicate and the choices `approve`, `edit`, `reject`, and `respond`. An approved or edited call returns to the filesystem tool, so its deny check still runs; `reject` and `respond` do not execute the operation. A caller-supplied `interrupt_on` entry wins for the same tool name, replacing the generated review routing but not the filesystem tool's deny enforcement.

Exact tools (`read_file`, `write_file`, and `edit_file`) route only when the requested path—or its resolved target when a backend is available—has an interrupt decision and neither form is denied. `ls`, `glob`, `grep`, and `delete` are bulk tools: review is requested when the potential search or deletion subtree overlaps the literal anchor of an applicable interrupt pattern. A missing bulk path and current-directory aliases are treated as the root; malformed or invalid path arguments do not synthesize an interrupt. Prefer literal leading anchors: an unanchored rule such as `/**/secrets` has root anchor `/` and deliberately over-interrupts bulk calls.

`glob` has an additional bypass guard. An absolute `pattern` may redirect the search independently of `path`, and a relative pattern containing `..` cannot be localized reliably. The predicate therefore separately checks an absolute pattern's anchor and routes `..` patterns to review, so `glob(pattern="/secrets/**", path="/workspace")` cannot evade an interrupt rule on `/secrets/**`.

### Execution and delegation limits

Permissions are a **tool-level** control. `create_deep_agent` documents that direct backend usage does not incorporate them. More importantly, `FilesystemMiddleware` rejects a nonempty permission set when its backend supports command execution: it has no policy for `execute`. The narrow exception is a `CompositeBackend` whose permission patterns are all scoped to configured route prefixes; this can protect those routed file tools without pretending to restrict the execution-capable default backend. Treat that exception as routing hygiene, not shell containment.

Use an execution backend with filesystem, process, and network isolation if commands must not inherit the application process's authority. Subagents inherit the parent rules only when they do not specify their own `permissions`; a subagent-supplied list replaces the inherited policy. Remote async subagents also require their own approval configuration.

## dcode Auto approval: a notice and review layer, not a sandbox

`dcode` has Manual, Auto, and YOLO approval modes. Auto is only selectable where the live runtime says it is eligible; it is unavailable with a sandbox. On its first enable for a notice version, the terminal UI displays `AutoModeNoticeScreen` before activating the mode. The notice says that routine gated actions—for example file edits and read-only Git commands—may proceed without asking, uncertain actions are reviewed by the classifier model, and the agent can still change files, run commands, and use tools. If a distinct classifier is configured, the notice identifies it as separate from the coding model; the model label is markdown-escaped before interpolation.

Enter explicitly confirms and persists the install-local notice before the live approval-mode write; Escape, programmatic dismissal, a failed modal push, or a failed live-mode write leaves Auto inactive. The app prevents competing entry points from stacking notices by sharing a pending confirmation future. Once recorded, the notice is not shown again until its notice version changes. This flow makes the mode transition visible and avoids accidental activation, but neither the classifier nor the acknowledgement reduces the authority of an allowed tool or process.

## Approval is deliberate interaction, not authorization

Use interrupt rules or `interrupt_on` for actions an operator needs to inspect in context, such as writes outside a routine work area, destructive operations, or sensitive MCP calls. Approval is meaningful only if the serving application presents the paused state to an authenticated human and resumes with that decision. It does not retroactively constrain already-issued credentials, direct backend access, an unmediated tool, or a remote service's authority.

For operational safety, make the default backend and tool set least-privileged first, then use approval as an additional decision point. Test both approval and rejection paths, including an edited request whose destination moves into a denied path.

## Configuration and dotenv are trust-sensitive inputs

Deep Agents Code recognizes that a checked-out project can carry hostile configuration. Project dotenv loading is separately configurable. The loader denies high-risk variables from every dotenv layer, including process/profile relocation, dynamic-loader settings, interpreter and shell startup hooks, `PATH`/`PYTHONPATH`, askpass helpers, and Git execution/configuration injection. It also blocks project `.env` from setting trust decisions such as the project MCP allow/deny lists, the Auto approval classifier model and timeout, forked-subagent behavior, and the graph recursion default.

The dotenv precedence is shell, then nearest project `.env`, then global profile `.env` for otherwise-unset values. Before loading a project `.env`, the loader reads the trusted global `startup.read_project_dotenv` toggle. If that global file cannot be inspected, it skips the project file rather than assuming project dotenv is allowed. This is intentionally fail-closed against a repository-controlled `.env`; deployers who do not need it should disable project dotenv loading.

Do not place secrets, endpoint authority, or approval-policy choices in a cloned repository. Supply them through the launch environment, a managed/user-level configuration surface, or platform secret injection, and run the service under an OS identity that has access only to the needed material.

## MCP: project definitions require an out-of-repository grant

A project `.mcp.json` may describe a local command or a remote endpoint that performs discovery with environment interpolation. Deep Agents Code therefore loads project-MCP trust lists only from user-level or managed configuration and selected process environment variables, never from repository configuration. A project cannot self-approve a malicious MCP server by committing both `.mcp.json` and an in-repo config or `.env`.

Project-server approval is scoped rather than merely name-based: persisted approval includes a server-definition fingerprint and binds fixed remote URLs to a validated local Git repository, while local commands and interpolated remote URLs are bound to the exact worktree. A changed definition or different clone asks again. Managed policy has highest precedence; a managed explicit approval list can replace lower permissive grants. Denies accumulate across sources and rejection wins. If trust configuration is unreadable or malformed, callers receive a read error so they can treat project configuration as untrusted rather than silently treating the deny list as empty.

The `DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS` setting is an explicit process-level escape hatch, not a repository setting. Review each approved MCP server as code plus egress policy: stdio commands execute locally, and remote servers can receive requests and any headers or environment-derived values that the configuration permits.

## MCP OAuth credentials and endpoint discovery

### dcode: protected files, not a vault

Deep Agents Code `FileTokenStorage` persists MCP OAuth state under the selected profile's state directory in `mcp-tokens`. Token-file names require safe server names and include a hash of the server URL, separating identically named servers at different endpoints. The JSON envelope can contain bearer tokens, client registration data, OAuth metadata, and expiry state. Treat it as credential material.

Writes serialize a read-modify-write operation with an in-process path lock, create the token directory with owner-only permissions where supported, write a temporary file using mode `0600`, atomically replace the destination, and apply `0600` again. The implementation warns when it cannot set directory mode `0700` or file mode `0600`. Atomic replacement ensures a reader sees an old or new complete file rather than a partial JSON document; it does not protect against another process that can read the account's files.

Token values must never be logged: the OAuth token model's normal representation includes access and refresh tokens. Corrupt, non-object, or unsupported-version token files fail with recovery instructions to delete the file and log in again. Back up or migrate this state only through a protected secret-handling process; do not commit it, expose it through shared volumes, or assume file modes defend against the same user, an administrator, or a compromised host.

### Talon: credential persistence is paired with guarded OAuth requests

Talon stores tokens, expiry, and client registration in `~/.deepagents/mcp-tokens`. The server URL—and non-default OAuth configuration when present—contributes to the file digest, so credentials for incompatible server or client identities do not share a file. Updates take a per-path lock, harden the directory to `0700`, write and `fsync` a `0600` temporary file, then replace the destination. A refresh response that omits `refresh_token` retains the previously stored refresh token; a fresh authorization deliberately does not. This protects update integrity and availability, not confidentiality from the agent process or any principal that can read the account's home directory. Talon warns when the token location is inside its agent workspace because its default shell backend can still read arbitrary absolute paths; moving a secret out of the workspace is not a security boundary.

Talon treats the OAuth discovery and token endpoints as untrusted network input. OAuth HTTP clients use an HTTPS-only SSRF-safe transport, ignore proxy-related environment configuration, disable redirects, apply request and DNS-resolution timeouts, and bound JSON response bodies to 64 KiB. Discovery requires a validated public HTTPS issuer, validates device-flow endpoint origin against that issuer, and rejects malformed metadata; provider-driven requests also validate every metadata endpoint, reject userinfo URLs and redirects, and re-check issuer equality. The transport preserves the hostname for HTTP `Host` and TLS SNI while connecting to the safe resolved address, and rejects a hostname with mixed public and loopback DNS answers. These measures constrain where the OAuth helper connects and how it processes discovery. They do **not** make an approved MCP server trustworthy, constrain the permissions granted by an OAuth token, or replace egress and service authorization policy.

For interactive login, Talon presents a browser URL and accepts a callback only when its scheme, authority, and path exactly match the configured redirect URI and the callback supplies `code` and `state`. Channel-based login binds the authorization prompt and callback to the current invocation; runs without such a channel must use an interactive login instead. Device authorization similarly expires on its server-provided deadline, observes `authorization_pending` and `slow_down`, and stores the result only after a valid bearer-token response.

## Server-authoritative workspace sharing

In server mode, a thread's workspace is not a client-owned capability. `bind_thread_workspace` validates an untrusted `cwd` as an existing absolute canonical directory without traversal, resolves its project root, and persists an immutable thread binding in SQLite. The binding records workspace identity, a server-resolved policy payload, and policy/runtime fingerprints. The public payload deliberately excludes the persisted policy and fingerprints; a client can echo workspace identity but cannot assert the server policy.

On each run, `require_thread_workspace` requires a matching payload and, when supplied, matching server configuration fingerprint. It rejects unbound threads, substituted/stale context, configuration mismatch, unsupported schemas, and a workspace whose resolved identity changed. Binding is transactionally serialized with `BEGIN IMMEDIATE`; concurrent first claims cannot mix different workspaces. A compatible runtime/model change can refresh the runtime fingerprint, but policy drift is refused. Persisted workspace policy is intentionally non-secret: tests verify that model API keys and system prompts are excluded.

Use this boundary when multiple clients can reconnect to a server, but keep tenant isolation outside it. The database, workspace directories, session/checkpoint storage, model provider, and tool backend still need deployment-specific authentication, authorization, storage permissions, and network segmentation.

## Untrusted model inputs and secondary model calls

Repository text, shell output, MCP responses, and conversation content can contain prompt injection. Preserve provenance where possible, minimize what a secondary model receives, and do not reuse user-controlled instructions as privileged system content. Deep Agents Code's thread-title generator illustrates this approach: it labels the conversation untrusted in its system prompt, sends only human and AI text, excludes system/internal/control and user-shell messages, caps input at 8,000 characters, disables callbacks, runs a tool-free model, applies a 10-second timeout, and normalizes the resulting title to the same safe shape as a manual name.

This containment is scoped to title generation; it does not sanitize the primary agent conversation or turn malicious text into trusted content. For any new summarizer, classifier, extractor, or automation model call, define an explicit data allowlist, size limit, tool/callback policy, timeout, output validation, and a no-secret/no-privilege assumption.

## Rendering external content in the terminal UI

Terminal presentation is also an input boundary. In dcode's Textual UI, strings from users, tools, files, API responses, paths, and exceptions must not be interpolated into Rich markup. Use `Content.from_markup()` with `$var` substitution for external values, or compose `Content` fragments; use `markup=False` for notifications containing dynamic values. Raw Markdown is a different parser path: normalize line breaks and escape markdown-significant characters with `escape_markdown`, and build tables through the escaping table helper. This prevents external text from creating markup, links, HTML-like content, additional blocks, or table columns in the UI.

The Auto notice illustrates the split: its static body is rendered as Markdown, but an external classifier-model label is escaped before insertion. It disables Markdown's automatic link handling and sends clicked URLs through the shared URL opener, which checks URL safety before invoking the browser; notifications from that path disable markup and strip dangerous Unicode for display. These are rendering and user-interaction safeguards, not sanitization of content for model consumption or authorization of a clicked destination.

## Deployment checklist and focused tests

1. **Choose the enforcement layer.** Use a sandboxed execution backend, separate service account, read-only mounts, restricted egress, and narrowly scoped credentials for hard boundaries. Do not rely on prompting, filesystem middleware, or approval alone.
2. **Make filesystem policy explicit.** Use absolute patterns, test first-match ordering, include a catch-all deny for isolation, and test symlink aliases, bulk-result filtering, plain-leaf versus recursive deletion, and overlapping wildcard deny patterns.
3. **Protect the approval surface.** Ensure only authenticated operators can decide interrupts, record decisions as appropriate for the deployment, and test malformed, pathless, current-directory, absolute-pattern, `..`-pattern, and edited tool requests.
4. **Do not combine a permission claim with shell authority.** Expect nonempty permissions to be rejected on an execution-capable backend unless every pattern is scoped to a composite route; in either case, use a sandbox for `execute`.
5. **Treat repositories as hostile until trusted.** Disable project dotenv where practical; keep trust controls in launch, user, or managed configuration; review project MCP definitions before granting an out-of-repository approval.
6. **Operate OAuth state as a secret.** Protect the selected profile state directory, watch permission-hardening warnings, avoid token logging, and verify corrupt-file recovery and endpoint-separated storage. For Talon, test unsafe, mixed DNS, userinfo, redirect, oversized-response, and timeout failures as well as normal OAuth flows.
7. **Keep workspace identity server-side.** Require bindings on every server run, reject policy drift, protect the SQLite database, and do not persist credentials or prompts in workspace-policy records.
8. **Make Auto’s limits explicit.** Confirm the first-enable notice before persisting Auto, test Escape and modal-failure paths, disclose a separate classifier model, and never describe the mode as sandboxing or least privilege.
9. **Render external terminal content as data.** Test markup and Markdown metacharacters, line breaks, URLs, and invisible Unicode in tool output, exception messages, model labels, and links.
10. **Test injection boundaries.** Include adversarial repository text, tool output, shell output, and MCP responses in tests for secondary model calls and approval flows.

See [permissions and HITL](/openwiki/concepts/permissions-hitl.md), [filesystem tools](/openwiki/concepts/tools-filesystem.md), [MCP](/openwiki/integrations/mcp.md), [Talon](/openwiki/integrations/talon.md), and [development operations](/openwiki/operations/development.md) for adjacent component guidance.
