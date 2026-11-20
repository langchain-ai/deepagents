---
type: security operations guide
title: Security and Trust Boundaries
description: Operational guidance for host-shell authority and cleanup, filesystem policy, dcode trust, MCP/OAuth safety, and Talon hooks, subagents, channel admission, and unattended work.
tags: [security, operations, trust-boundaries, sandboxing, approvals, credentials, mcp]
sources:
  - id: openwiki-source-cd45145a8c3a51b52eab3c2b
    resource: repo://libs/talon/deepagents_talon/background.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-82dab853903c3a574614fd1e
    resource: repo://libs/talon/tests/unit_tests/test_background.py
  - id: openwiki-source-8de0ff38635f214c7268d8e7
    resource: repo://libs/talon/tests/unit_tests/test_tool_approval_authorization.py
  - id: openwiki-source-242a21b2da46507f58415265
    resource: repo://libs/talon/tests/unit_tests/test_tool_approval_runtime.py
generated: { by: "openwiki/0.4.2", at: "2026-11-20T08:04:05.944Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-11-20T08:04:05.944Z
---

# Security and Trust Boundaries

Deep Agents uses a **trust-the-LLM** model: an agent can exercise the authority that its configured tools and backend expose. Treat prompts, repository content, tool output, MCP responses, and channel messages as untrusted inputs, not as an authorization boundary. Put limits at the tool, backend or sandbox, OS identity, network, and credential layers.

## Execution and filesystem authority

`LocalShellBackend` executes commands on the host without sandboxing or isolation. `virtual_mode` restricts filesystem-tool path handling, not shell commands, and inherited environment is opt-in. Use a genuinely isolated backend for untrusted input, shared hosts, or production: approval can authorize a call, but cannot confine a host shell.

```mermaid
flowchart TD
    Input["Untrusted input"] --> Agent["Agent and tool call"]
    Agent --> Policy["Tool policy and HITL"]
    Policy --> Local["LocalShellBackend"]
    Policy --> Sandbox["Isolated sandbox backend"]
    Local --> Host["Host user authority"]
    Timeout["Timeout or cancellation"] --> Cleanup["Best effort process cleanup"]
    Local --> Cleanup
    Cleanup --> Host
    Sandbox --> Limits["Sandbox identity mounts and egress"]
```

*Tool approval decides whether a mediated action proceeds; the selected backend defines its execution boundary, while timeout and cancellation trigger cleanup rather than containment.*

### Local-shell lifecycle and cleanup

A command is launched through `subprocess.Popen(..., shell=True)` with stdin disconnected, the backend working directory, its constructed environment, and captured stdout/stderr. The default timeout is 120 seconds; output is bounded after collection. These operational limits make a failed command observable, but do not restrict what a running host command can access or do.

On POSIX, each command starts in a new session, so timeout, interruption, and default `aexecute()` cancellation try to send `SIGKILL` to that command's process group, drain and close captured pipes, and briefly reap the shell. A child that creates its own session or process group—for example via `setsid`—is outside that group; `nohup` alone is not. On Windows there is no new group: cleanup kills only the direct shell, and descendants may survive or keep inherited output pipes open. In either case a cleanup failure is reported as a timeout that warns the command may still be running; do not blindly retry an operation with external effects.

Async cancellation signals a worker thread and waits only a short cleanup grace period. If cleanup outlives it, the future is retained and its eventual failure is logged without command arguments; cancellation still returns to the caller. Overriding `execute()` takes the legacy thread-delegation path and does not receive this built-in cancellation cleanup. Treat cancellation, process killing, output limits, and timeout responses as availability and hygiene mechanisms—not as a security boundary or proof that a command, descendant, or side effect stopped.

### Filesystem middleware

Filesystem rules classify the built-in `ls`, `read_file`, `glob`, and `grep` tools as reads and `write_file`, `edit_file`, and `delete` as writes. Rules are evaluated in order with **first-match** semantics and an allow default. Patterns must be absolute and cannot contain `..` or `~`; rule effects are allow, deny, or approval interrupt.

For a local `FilesystemBackend`, including one selected by a `CompositeBackend`, the middleware evaluates both the requested path and its resolved target. A deny on either blocks the tool call; with a nonempty policy, a resolution failure is denied. Recursive deletion additionally checks whether the target overlaps a denied pattern. This applies to mediated filesystem tools, not arbitrary shell commands or direct backend use.

`RepositoryBounds` is a distinct read-only inspection boundary for coding subagents. It fails closed to `ls`, `read_file`, `glob`, and `grep`; applies lexical and canonical containment checks against traversal and symlink escape where possible; bounds result volume; and turns backend failures into bounded unavailable errors. It is not a general shell authorization mechanism.

## dcode server and workspace trust

### Sanitize the child server environment

The dcode launcher creates the `langgraph dev` child environment from the current process, removes project-dotenv-loaded values, then strips dynamic-loader, interpreter/path, shell-startup, askpass, and Node startup variables. It also removes LangGraph cloud/auth and LangSmith tenant/control-plane credentials, sets `LANGGRAPH_AUTH_TYPE=noop`, and disables the control-plane key. The default bind is loopback and the internal server uses an ephemeral port.

`PYTHONPATH` deserves separate treatment: the launcher removes it before the server interpreter starts so an inherited untrusted-project path cannot shadow imports. It carries the launch value in a private bootstrap variable that the approval-gated agent shell can later restore; pre-existing carrier variables are discarded. The launcher similarly replaces its LangSmith carrier from captured bootstrap state. Persistent or scoped restart overrides cannot alter the selected profile, profile marker, Python-path carrier, or LangSmith carrier.

Treat a production deployment's network and route authentication as deployment responsibilities. The generated built-in dcode configuration opts custom HTTP routes into LangGraph's route-auth layer when an auth backend is configured, but its ordinary production launcher uses noop auth and relies on its loopback bind.

### A checkout is not permission to execute its configuration

Project MCP configuration, sandbox setup, extension paths, and project extensions are directory-scoped execution grants which the server—not a client—must resolve. A bound workspace policy is re-resolved per request; if a recorded extension-trust grant disappears, including because the trust store cannot be read, dcode refuses the thread. Switching outside the launch project drops launch-project MCP configuration, sandbox setup, explicit extension paths, and project MCP trust, then reads extension trust for the target project.

```mermaid
sequenceDiagram
    participant Client as dcode client
    participant Server as dcode server
    participant Workspace as Canonical workspace
    Client->>Server: request with workspace context
    Server->>Workspace: resolve identity and project policy
    Server->>Server: bind or validate policy
    alt policy changed or trust revoked
        Server-->>Client: reject request
    else policy remains valid
        Server->>Server: use graph
    end
```

*Project-scoped authority is resolved by the server for the canonical workspace, rather than inherited from a client path or previous checkout.*

Project dotenv files are untrusted. dcode denies values that can relocate the trust root or cause code execution through loaders, interpreter/path settings, shell startup, askpass, or Git configuration; `startup.read_project_dotenv` cannot itself be set by dotenv. Keep secrets in operator-controlled configuration.

Hooks use canonical workspace-root trust, atomically and restrictively persisted under locks; their session grants are tied to the current hook-file fingerprint. Resolution or fingerprint failures fail closed. Python extensions require experimental extensions, extension loading, and applicable project trust before trusted project sources can execute factories and register capabilities. Installed plugin Python declarations must remain existing `.py` files under the plugin root and have a nonempty version, but installation is still an operator trust decision because manifests can supply MCP and hooks.

## MCP and OAuth boundaries

An MCP server is an authority boundary: stdio starts a local command, while remote MCP can receive sensitive data and return hostile tool descriptions or content. Review command, arguments, URL, headers, OAuth configuration, scopes, allowed tools, and expected egress before enabling one.

Talon loads only well-formed named server entries; it expands `${VAR}` references from runtime or process environment and validates string fields. It supports `stdio`, `sse`, and streamable HTTP transports. Stdio environment values cannot set startup-hijacking variables such as `LD_PRELOAD`, `PYTHONPATH`, or shell initialization variables. Remote OAuth cannot be combined with an explicit `Authorization` header. Tool loading is bounded to 30 seconds, failed servers are reported as unavailable rather than contributing tools, and an optional allowlist or denylist filters a server's exposed tools.

OAuth login is intentionally interactive. Without a terminal or current Talon conversation authorization handler, an OAuth-required scheduled job or background subagent fails with instructions to authorize from chat or run `deepagents-talon mcp login <server>`; it does not create a prompt with no accountable recipient. Channel device-code authorization binds the server name, tool invocation, and expiry to its flow.

Talon's OAuth transport validates public HTTPS endpoints with SSRF protection, ignores proxy environment settings, does not follow redirects, bounds DNS resolution, requests, discovery, and response size, and revalidates metadata endpoints before using them. These controls prevent a discovered endpoint from becoming an arbitrary internal-network request; they do not establish trust in a public remote service.

dcode token storage derives safe paths from server identity and warns that token representations must not be logged. Talon token storage uses owner-only directories/files, locking, and atomic replacement, preserves an existing refresh token when a refresh response omits one, and warns that filesystem hardening is not confidentiality isolation from its default shell backend. Store tokens outside agent workspaces and run Talon under an account appropriate to the authority those tokens convey.

## Talon runtime, hooks, subagents, admission, and unattended work

### Assistant-owned command hooks

Talon loads `hooks.json` from the resolved assistant home once at the start of each turn, not from the workspace. It opens that exact file with `O_NOFOLLOW`, requires a regular file, and builds a snapshot with `workspace_trusted=False` and no project hooks. A missing file simply disables hooks; unreadable or invalid content produces diagnostics and an empty effective configuration rather than treating workspace configuration as trusted. Configuration changes therefore apply on a later turn, not midway through an existing turn.

The shared hook runtime supports Talon's `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, and `Stop` server events. It validates that each hook interrupt belongs to the active snapshot, thread, and prompt and caches each invocation response. A `PreToolUse` denial prevents that call and suppresses an unnecessary approval interrupt, but a hook `allow` does **not** bypass Talon's operator-controlled tool approval. Hook commands are executable authority from the assistant home: protect that directory and review `hooks.json` as carefully as any other local command configuration. Hook denial and tool approval remain mediation decisions, not host containment.

### Fresh subagents and inherited control plane

Talon compiles local subagents with a fresh context: no parent history or skills are inherited, and `fork` mode is unsupported. A fresh graph receives only its configured tools plus an explicit per-task addition selected from the parent catalog; duplicate or unknown requested tool names are rejected. It gets Talon's MCP middleware, cooperative subagent-control middleware, and only the enabled approval entries for tools actually attached. Talon refuses to run a precompiled local subagent whenever an approval policy exists because it cannot enforce that policy inside the opaque graph; use a fresh agent with explicit tools instead.

When parent hooks are active, each fresh child gets a child hook adapter that retains the turn's snapshot but creates independent thread and prompt identities, agent identity, invocation responses, and transcript scope. Its graph handles hook interrupts with a bounded resume loop, emits no child `Stop` event, and returns unresolved approval interrupts to the caller. Subagent findings and embedded instructions are untrusted evidence; they cannot prove approval or expand the main agent's authority.

Talon's default backend is still `LocalShellBackend`, but its child environment is deliberately reconstructed rather than inherited. It admits only a small runtime-oriented allowlist plus `LC_` keys, forces a fixed safe `PATH`, and removes loader/shell/Python hijack variables, LangSmith/LangChain settings, and keys whose names indicate credentials or tokens. This reduces incidental secret and startup-state exposure; it does not sandbox the host filesystem, process, or network.

Unknown direct-message senders require a time-limited pairing code bound to provider, sender, and conversation. Only operator surfaces can approve it. A paired sender receives the operator's configured credentials and host access but cannot administer pairing or tool-approval policy; verify identity out of band and revoke promptly.

Talon tool approval policy is an exact-name, revisioned immutable invocation snapshot. Enabled entries create approval interrupts; updates require operator context and a compare-and-swap revision, and take effect only for the next invocation. An absent or `false` entry does not create an approval gate, so policy edits can reduce oversight and need the same operational review as any authorization change.

### Unattended delegation and result exposure

A cron fire and a background-result delivery turn are **unattended**. For delivery, `TalonHost` supplies no tool-approval or authorization handler and disables the question handler; the runtime independently treats `trigger: "cron"` and `background_delivery` as non-operator contexts. It auto-rejects a cron approval interrupt and auto-rejects a background-delivery interrupt when a caller injects an approval handler. Thus metadata or a handler passed through a lower-level invocation cannot turn an unattended turn into an operator-authorized one.

A detached attended-chat worker is different from a delivery turn, but has the same no-prompt rule. It runs in a copied context so it can retain required history and cron-origin scope, then explicitly clears the authorization handler and `APPROVAL_OPERATOR`, uses a new task thread ID, and forbids nested delegation. An approval interrupt is converted to `Subagent needs tool approval; the protected action has not run.` rather than being retained for a conversation that may have ended. Workers are in-memory and conversation-owned: their completed text is later injected as data only into the owning main conversation, where the host may start the unattended delivery turn.

Cron delegation does **not** create such a worker. While `_SCHEDULED_TURN` is set, `task` and `start_async_task` run inline, return their result to the same graph turn, and leave no job to inspect, cancel, or deliver later. Talon hides those meaningless worker-management tools and changes the delegation guidance accordingly. Inline calls use a separate four-slot semaphore, queue rather than consume the attended-worker capacity, and default to a ten-minute per-delegation limit; an outer scheduled run is bounded to 30 minutes. Nested delegation remains refused. These are scheduling and availability controls, not a containment boundary for a capability already granted to the subagent backend or tools.

Both paths turn timeout and ordinary failures into bounded safe result text instead of allowing an exception to escape and retry the graph. The generic failure result does not expose tool arguments or provider exception payloads; recognized HTTP failures provide only a status-class hint. Detached and inline results expose answer text only—excluding non-text reasoning blocks—and cap it at 64,000 characters. Preserve this failure conversion: an escaping inline deadline could cause graph retry to relaunch sibling delegation calls. Operators should pre-authorize only the exact unattended capabilities required, design effects to be safe under cancellation and retry, and never regard prompt suppression, approval denial, result filtering, or interpreter isolation as a substitute for backend, tool, OS, network, or credential authority boundaries.

Scheduled work is repaired after its outer timeout; revoking the paired sender that created a job stops that sender's scheduled run. Time bounds and denial prevent an unattended call from proceeding, but do not contain an action that has already been granted authority.

## ACP and operating checklist

ACP persistent execution approval is scoped to canonical full tool-argument JSON and the current session context. Changed commands, arguments, contexts, or sessions require review; invalid requests cannot receive reusable permission. This narrow reuse rule applies to `execute`; verify the graph's `interrupt_on` configuration for every other sensitive tool.

1. **Choose the backend first.** Do not expose untrusted repository or channel input to a host shell, privileged OS account, inherited secrets, or unrestricted egress.
2. **Operate host-shell cleanup as best effort.** On POSIX avoid programs that detach into their own session; on Windows assume descendants can survive. Treat a timeout warning as an incident signal and make effects idempotent before retrying.
3. **Use path policy as tool mediation, not confinement.** Test requested paths, symlink targets, and recursive deletion; keep shell authority separately constrained.
4. **Grant execution configuration explicitly.** Review dcode checkout hooks, extensions, sandbox setup, and MCP per canonical workspace. Separately protect Talon's assistant-home `hooks.json`; Talon intentionally does not load workspace hooks.
5. **Treat MCP configuration and OAuth as deployment authority.** Restrict stdio commands and remote scopes, protect token state, and never log token objects.
6. **Require a human for human decisions.** Pair senders out of band, keep approval policy operator-owned, use fresh subagents with explicit tools where policy must apply, and ensure cron/background tasks auto-deny gated actions rather than prompting an unrelated channel.
7. **Exercise focused tests after boundary changes.** Relevant coverage includes `libs/deepagents/tests/unit_tests/backends/test_local_shell_backend.py`, `libs/talon/tests/unit_tests/test_hooks.py`, `libs/code/tests/unit_tests/test_server_helpers.py`, `libs/acp/tests/unit_tests/test_execute_permissions.py`, and Talon's `test_tool_approval_batch.py` and `test_questions.py`.

See [Code agent architecture](/openwiki/architecture/code-agent.md), [permissions and HITL](/openwiki/concepts/permissions-hitl.md), [Talon channel admission](/openwiki/concepts/talon-channel-admission.md), and [MCP integration](/openwiki/integrations/mcp.md).
