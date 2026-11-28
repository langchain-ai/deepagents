---
type: runtime integration
title: Talon Runtime Integration
description: Talon is an experimental local host that composes channel adapters, schedules, a Deep Agents runtime, media preparation, scoped tools, and model-specific reasoning controls. This page explains request ownership, lifecycle, local-host limitations, graph reload, delegation, and host-mediated file delivery.
tags: [talon, runtime, host, subagents, hooks, mcp, scheduling, file-delivery]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-28T08:03:41.605Z
sources:
  - id: openwiki-source-2e73fb99e1d9ba2b68646e83
    resource: repo://libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js
  - id: openwiki-source-effb906ade4efbc4dcee12c9
    resource: repo://libs/talon/deepagents_talon/channels/whatsapp_bridge/package-lock.json
  - id: openwiki-source-d4d7551c16a10652b8470897
    resource: repo://libs/talon/deepagents_talon/channels/whatsapp_bridge/package.json
  - id: openwiki-source-3d157a5857f325aceaade7f1
    resource: repo://libs/talon/deepagents_talon/channels/whatsapp.py
  - id: openwiki-source-fd3cc27f279ed4fb8aaca360
    resource: repo://libs/talon/deepagents_talon/file_sending.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-cebe4ea270e21dce4de9b074
    resource: repo://libs/talon/deepagents_talon/interfaces.py
  - id: openwiki-source-82cac27adeecff8a900a40fa
    resource: repo://libs/talon/deepagents_talon/mcp.py
  - id: openwiki-source-ccdb88feadbc25be2fc7b63b
    resource: repo://libs/talon/deepagents_talon/media.py
  - id: openwiki-source-f04ce33d1db21a61b1e6e8b3
    resource: repo://libs/talon/deepagents_talon/model_selection.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-e0db085f967450f48196d999
    resource: repo://libs/talon/tests/test_media.py
  - id: openwiki-source-412987bb51db9ba8f655d511
    resource: repo://libs/talon/tests/unit_tests/test_file_sending.py
  - id: openwiki-source-966f1489233b6fc6ee5f2e4c
    resource: repo://libs/talon/uv.lock
generated: { by: "openwiki/0.4.2", at: "2026-11-28T08:03:41.605Z" }
---

# Talon Runtime Integration

Talon is an experimental local host for long-running Deep Agents channels and schedules. Its boundary is deliberate: channel adapters translate provider events, `TalonHost` owns routing, interactive callbacks, delivery, and shutdown, while `DeepAgentRuntime` owns graph construction, checkpointed execution, and process-local background jobs. Adapters do not invoke the runtime directly.

See [Runtime behavior](../architecture/runtime-behavior.md), [Profiles and models](../concepts/profiles-models.md), [Talon channel admission](../concepts/talon-channel-admission.md), [Talon scheduling](../concepts/talon-scheduling.md), [MCP integration](mcp.md), [Security](../operations/security.md), and the [Testing guide](../testing/testing-guide.md).

> **Experimental local host, not a security boundary.** Talon is alpha software intended for a single operator, not production or enterprise multi-tenant deployment. It does not provide complete production-grade HITL policy, channel-administrator controls, or tenant isolation. Treat channel access—including paired senders—as direct use of the operator's configured agent, credentials, MCP tools, and local resources. Channel admission and approval prompts are useful controls, not containment; incoming text, transcripts, attachment metadata and files, tool output, and cross-session replies remain untrusted data.

## Composition and ownership

The `deepagents-talon` CLI is the production composition root. With a configured model, it loads MCP tools, opens checkpoint and history resources, and constructs `DeepAgentRuntime` with the assistant directory, async-subagent loader, cron store, sandbox backend when configured, and `talon_mcp_middleware()`. It gives the runtime and adapters to `TalonHost`. A `PersistentCronScheduler` exists only when channels exist, and invokes and delivers jobs through the host.

`AgentRequest` is the host-to-runtime contract: it contains the host-resolved conversation ID, text, trusted metadata, selected model/effort, and optional approval, authorization, progress, breakout, and question handlers. `AgentResult` carries output and acknowledged background IDs. The runtime can acknowledge that the model consumed a result, but the host alone knows whether its channel reply was delivered and requeues IDs when it was not.

```mermaid
flowchart TD
    CLI["CLI composition root"] --> MCP["MCP tool provider"]
    CLI --> Host["TalonHost"]
    CLI --> Runtime["DeepAgentRuntime"]
    MCP --> Runtime
    Host --> Channel["Channel adapters"]
    Host --> Scheduler["Persistent cron scheduler"]
    Channel --> Host
    Scheduler --> Host
    Host --> Request["AgentRequest with handlers"]
    Request --> Runtime
    Runtime --> Graph["Deep Agents graph"]
    Runtime --> Bridge["Claude and Codex bridges"]
    Bridge --> Delivery["Background delivery queue"]
    Runtime --> Dcode["dcode bridge"]
    Graph --> FileTool["send_file tool"]
    FileTool --> Host
    Graph --> Result["AgentResult"]
    Result --> Host
    Delivery --> Host
    Host --> Channel
```

*Channels and the scheduler enter through the host. The runtime executes a request but does not own channel delivery or the lifetime of host callbacks. `send_file` returns bytes to the per-turn host callback; the host stages and uploads them through the originating channel rather than granting the runtime an adapter shortcut. Claude and Codex replies enter the host's existing background-delivery path; dcode confirms a direct prompt delivery and has no reply path.*

### Lifecycle and failure containment

`TalonHost.start()` ensures the home directory, starts the runtime, binds and starts channels, then starts the scheduler. A failed partial start is unwound in reverse order. Shutdown cancels the background dispatcher and active conversation work before stopping channels in reverse order, scheduler, and runtime; a component-stop failure is logged without preventing remaining cleanup.

`DeepAgentRuntime.stop()` cancels background workers before closing its graph/checkpointer. If a worker outlives cancellation, it raises and leaves those resources open rather than allow a worker to write through a closed checkpointer. This is process-local worker ownership, not an obligation for a channel adapter or scheduler.

## Graph assembly, reload, and diagnostics

At start, the runtime resolves subagents, reads a `ToolApprovalStore` snapshot, and builds a graph. It creates a catalog from filesystem and ordinary runtime tools, removes MCP tools marked `skillOnly` from the main tool list, attaches approval-management and `get_agent_tools`, resolves local frontmatter tool names exactly against that catalog, then configures background delegation against the resolved agents.

The graph includes filesystem support, selected-model summarization and selection, prompt caching, progress, human approval, server hooks, `TaskTools`, background delegation, optional context-size tools, memory, skills, and caller-supplied middleware. `get_agent_tools` reports configured attachments separately from names selectable per task; opaque compiled or remote agents report unknown tools. Saved MCP/subagent changes require reload, and running turns/tasks retain their original graph and capabilities.

Before an invocation, the runtime refreshes supplied tools, then holds its tools lock while reading approval policy and rebuilding only if that snapshot changed. It publishes a replacement graph only after construction succeeds and places the selected graph and approval snapshot in context variables. Consequently, failed reloads preserve the prior graph and an in-flight turn cannot gain or lose capabilities mid-execution. `reload_subagent_configuration()` follows the same build-before-publish approach for later turns.

Selected model choice is turn-scoped. An unavailable saved selection falls back to the default, and summarization caches a selected-model-specific summarizer while subagents use their startup summarizer.

### Model-specific reasoning effort

`/effort` is separate from `/model`: it does not discover, construct, or admit a model. The host uses the currently selected model (or the runtime default) to list supported controls; only an operator may change a nonempty value. It persists a `provider:model` to effort mapping in `effort.json`, so switching models restores that model's setting, and `/effort default` removes only the current model's entry. Invalid controls and a failed state write leave the prior mapping intact. The host captures the model and effort together when it creates a turn, so a change applies across chats from the next turn and cannot alter work already in flight.

Supported values are an allowlist rather than a generic provider promise. Talon normalizes gateway-style specifications, recognizes documented DeepSeek, Qwen, and GLM families where necessary, otherwise uses SDK model metadata or documented fallbacks. After validation it translates a choice into provider settings: Qwen uses a thinking budget, DeepSeek and GLM use thinking payloads, and other supported models use `reasoning_effort`. During invocation the runtime binds the selected client and settings in context variables, merges nested `extra_body.thinking` settings without mutating the graph request, and resets both afterward. A stale selected model falls back to the default **without** applying its effort override; delegated subagents retain their startup model and do not inherit main-turn effort.

### Channel media and model content

Channels pass normalized local media paths and MIME metadata to the host; they do not make attachment bytes trusted instructions. Before invoking a normal turn, the host optionally transcribes eligible voice/audio/video, lets a sandbox-aware runtime stage documents, augments the text for attachments, and places photo content in request metadata. Documents are represented by a JSON-escaped full-path notice that tells the agent to treat both the attachment and its path as untrusted data; their bytes are not inlined. Video and other binary media similarly become path notices. A preparation failure preserves any user text and adds only a generic unavailable-attachment notice.

Only an image-classified file at or below 5 MiB is read into a provider-neutral base64 image block; inaccessible, oversized, or non-image files are omitted from those blocks. With a remote execution sandbox, documents are copied to a generated `.talon/attachments` sandbox path only after containment under an inbound root is rechecked after symlink resolution, the source is opened without following symlinks, and it is confirmed to be a bounded regular file. This staging changes document path aliases, not the trust level of the file.

The same path discipline protects outbound media. A media path must resolve beneath its configured outbound root and be a regular file; URL paths, home expansion, traversal, and symlink escapes are rejected. Channel adapters also verify the requested image/video/document type and apply their configured size cap before uploading. Markdown image syntax in ordinary model text is not an implicit upload request.

### Host-mediated file delivery

`send_file` is a normal main-agent tool assembled over the same filesystem backend as the agent's file tools, so its path works for both the local workspace and a remote sandbox. It reads one backend file subject to the configured media-byte limit, derives a display filename, and invokes only the `AgentRequest.file_handler` bound for that invocation. There is no handler for a non-chat invocation, and subagents are refused: they must report a path rather than upload a file.

The host supplies that handler only while an active turn can still reach its originating conversation. It passes bytes, sanitized display name, and optional caption to `send_file_to_channel`, which creates a unique `0700` staging directory under the configured outbound-media root, writes a `0600` no-follow file, calls the channel's `send_media` with retry, then removes the staging directory even after failure. Thus the model/backend never calls a channel adapter directly and a sandbox-visible source file is not exposed as a persistent host path. Channel transport error details are logged but replaced with a generic tool response.

Display names retain ordinary Unicode and punctuation but replace separators and control/format characters, turn dot-only names into `file`, and cap the UTF-8 name at 200 bytes while preserving a reasonable extension. The byte limit is applied before the callback; unreadable or oversized backend files are not staged. This delivery path is distinct from outbound media references in ordinary model text, which are validated under a configured root before upload.

### Memory and context doctor

`TalonMemoryMiddleware` replaces the SDK middleware in the same stack position and removes checkpointed `memory_contents` before each synchronous or asynchronous turn. It therefore reloads configured memory sources rather than preserving stale injected memory after a file edit. This behavior applies to checkpointed conversations without mutating the incoming checkpoint state.

`DeepAgentRuntime.context_doctor(conversation_id)` pairs the active compiled graph with its configuration snapshot and returns a bounded, content-free context-cost report. It reads the named checkpoint, recomputes configured memory and skills, enumerates graph tools, and estimates effective conversation tokens after any summarization event. The report intentionally excludes source text and paths, caps row detail, and does not change checkpoint state; provider input tokens are reported only when the latest AI message supplied them. The host exposes this as `/context-doctor` without interrupting active work and returns a safe generic failure when diagnostics are unavailable or fail.

## Hooks, approvals, and MCP are invocation capabilities

The runtime loads `hooks.json` from the trusted assistant home at the beginning of every invocation. It opens it without following symlinks and accepts only a regular file. Missing configuration means no hooks; malformed content becomes diagnostics and an empty hook snapshot so the turn can continue. Workspace/project hooks are deliberately excluded because the workspace is untrusted. Legacy configuration is migrated where safe and emits deprecation diagnostics.

A `TalonHooks` session owns that turn's snapshot, transcript runtime, prompt identity, and completed hook responses. It accepts only `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, and `Stop` server events. Resume validates interrupt ID uniqueness and verifies snapshot, thread, and prompt identities before fulfilling an invocation; duplicate fulfillment is cached. The runtime resets `ACTIVE_HOOKS` in `finally`, so a concurrent or later turn does not inherit this session.

A hook denial blocks its tool call before ordinary approval. A hook allow does **not** bypass Talon approval: `hook_approval_required` still asks for approval unless the hook denied that specific call. `APPROVAL_OPERATOR` is enabled only for a host-designated operator request that is neither cron nor background delivery; authorization, question, breakout, progress, history, hook, pending-result, model, and cron state are installed only for the invocation and reset afterward.

Talon wires one `talon_mcp_middleware()` into the main `DeepAgentRuntime` and installs the same middleware in freshly compiled local subagents. It preserves marked MCP schemas for OpenAI Responses calls, normalizes arguments, scopes authorization to the tool-call ID, and returns a correlated `ToolMessage` for `MCPError`. Talon has no elicitation UI, so MCP elicitation interrupts are resumed as cancellation.

`MCPToolProvider` adds management capabilities around loaded tools: status always describes configured servers, OAuth-capable configurations additionally expose `authenticate_mcp_server`, and configuration/reload tools schedule refresh for a subsequent turn. The provider serializes reloads and rejects a tool-name collision with Talon's management tools. A runtime builds a replacement graph before publishing refreshed tools; a failed refresh therefore leaves the currently usable graph active.

### Operator session bridges

Optional, disabled-by-default session bridges expose list and message tools for Claude Code, Codex, or dcode. They are operator-only and unavailable to subagents. A session send is approval-protected by default when its tool has no saved policy entry; an explicit `false` policy disables that prompt, while operator admission remains required. Their purpose is controlled cross-session collaboration, not a general remote-agent substrate.

- **Claude Code:** Talon discovers only sockets in private, owned runtime directories, validates peer identity, creates its own private inbox, and accepts only correlated protocol frames from the selected session. Replies and optional idle notices are delivered as bounded, escaped external results to the originating conversation; unsolicited or malformed frames are dropped.
- **Codex:** Talon lists valid saved thread identifiers through the Codex app-server protocol, queues a marker-bearing message, and reads only a completed turn that carries the matching request marker. It bounds outbound message size, escapes returned content, and drops a late reply after the owning conversation is cancelled.
- **dcode:** Talon discovers live, non-symlink Unix event sockets only in private directories and sends one framed `prompt` event with a correlation ID. It accepts delivery only when the peer acknowledges that same ID; dcode provides a delivery receipt, not a conversational reply stream.

Claude Code and Codex use `BackgroundSubagents` expectations and delivery: correlated replies are offered only to the owner conversation through the existing retry, acknowledgement, and requeue path. This keeps untrusted external text distinct from a model tool result and prevents a reply from being injected into another conversation. dcode has no `BackgroundSubagents` dependency: its direct, correlated acknowledgement only confirms that the target accepted the prompt.

## Fresh local subagents and task replacement

Talon replaces the SDK `SubAgentMiddleware` with `TaskTools` under the same middleware name, preventing duplicate `task` tools. The replacement accepts a `tools` list only for a named local subagent. Every name must be unique and present in the current selectable catalog; selected tools augment, rather than replace, its configured tools for that task. The subagent is freshly compiled with fresh message state: it does not inherit parent history, memory, skills, or delegation tools.

Fresh local agents receive `talon_mcp_middleware()`, control middleware, only the approval rules applicable to their resolved tools, and `HumanInTheLoopMiddleware` only when such rules exist. For a hook-enabled parent turn, each local execution gets a child hook session with the same immutable snapshot but new thread/prompt/agent identities and a separate transcript. It does not emit `Stop`; it can resume its own hook interrupts and preserves the parent's working directory.

A precompiled runnable is opaque. If any approval policy is active, Talon substitutes a refusal rather than execute it; otherwise it may run with task-only state. Remote `AsyncSubAgent` definitions are likewise opaque and stream the configured remote graph with the task description as user input, requesting cancellation on disconnect. `fork` mode is rejected for every subagent source.

```mermaid
sequenceDiagram
    participant Main as Main agent
    participant Tasker as TaskTools
    participant Worker as BackgroundSubagents
    participant Local as Fresh local agent
    participant Remote as Remote graph
    participant Host as TalonHost

    Main->>Tasker: task with optional exact tools
    Tasker->>Worker: delegated task
    alt attended turn
        Worker-->>Main: task id
        Worker->>Local: fresh task
        Worker->>Remote: remote stream when selected
        Worker-->>Host: owner result ready
        Host->>Main: unattended delivery turn
    else cron turn
        Worker->>Local: inline with timeout
        Worker->>Remote: inline remote stream
        Worker-->>Main: result or safe error
    end
```

*Attended tasks detach into process-local jobs for later host-mediated delivery. Scheduled tasks wait inline because no follow-up conversation is available.*

## Delegation and delivery behavior

For attended turns, `BackgroundSubagents` creates in-memory, conversation-owned jobs and enforces a process-shared running limit. Workers clear approval-operator and authorization-handler authority, block nested delegation, bound result text, and convert timeout, failure, remote error, and approval interrupt into safe results. Listing, inspection, cancellation, and control are limited to the owning conversation; queued guidance is available only to Talon-built local agents. Jobs and pending results disappear on process restart.

The host dispatches eligible completed results using an unattended `background_delivery` request, which has no approval or authorization handler. It avoids active, blocked, mismatched, or history-locked conversations and applies exponential retry backoff. Results are acknowledged only after a successful runtime turn; retry exhaustion marks the job dropped. If the host cannot deliver the reply after acknowledgement, it requeues the IDs for a later turn. For cron requests, delegation runs inline under a separate semaphore and timeout; nested delegation remains refused and failure becomes a safe tool error rather than an escaping graph retry.

## WhatsApp Node bridge boundary

WhatsApp is an optional channel selected with `--whatsapp` or `DEEPAGENTS_TALON_WHATSAPP_ENABLED`. The Python adapter is the host-side channel implementation: it launches the packaged `bridge.js` only when `DEEPAGENTS_TALON_WHATSAPP_START_BRIDGE` is truthy (or uses `DEEPAGENTS_TALON_WHATSAPP_BRIDGE_COMMAND`), waits for its health endpoint before polling, and owns polling, admission, host dispatch, and process shutdown. Without either start setting, it expects an externally run bridge.

```mermaid
sequenceDiagram
    participant CLI as Talon CLI
    participant Adapter as WhatsApp adapter
    participant Bridge as Node bridge
    participant WA as WhatsApp Web
    participant Host as Talon host

    CLI->>Adapter: enable channel
    Adapter->>Bridge: start with private paths and token
    Adapter->>Bridge: GET health
    Bridge->>WA: initialize client
    WA-->>Bridge: events
    Bridge->>Bridge: queue normalized messages
    Adapter->>Bridge: GET messages
    Adapter->>Host: dispatch admitted message
    Host->>Adapter: reply or media
    Adapter->>Bridge: authenticated POST send
    Bridge->>WA: send message
```

*The Python adapter mediates the host-to-bridge lifecycle and HTTP calls; the Node process handles WhatsApp Web events and transport.*

The adapter accepts only an HTTP loopback bridge URL (`127.0.0.1`, `localhost`, or `::1`) and sends the configured per-process bearer token on every request. The bridge refuses requests without that token, exposes health and queue-drain endpoints, and accepts text, media, typing, and edit operations only after its WhatsApp client is connected. This authenticated loopback protocol is an integration boundary, not a claim of a production security boundary.

For media, the adapter validates outbound content against Talon's outbound root, copies it into the bridge media directory when necessary, and removes its copy after the send attempt. The Node bridge independently resolves the supplied path and rejects anything outside its media directory; inbound downloads are written there with private file mode and are capped at WhatsApp's 64 MiB maximum. The Python adapter also applies the configured cap while polling, so a late or oversized local path is not forwarded as an attachment.

### Packaged bridge and Node dependencies

The wheel explicitly packages `bridge.js`, its compatibility and media helpers, and both `package.json` and `package-lock.json`; it does **not** bundle a `node_modules` tree. Install the bridge dependencies from `deepagents_talon/channels/whatsapp_bridge` with `npm install` before starting the packaged bridge. The manifest requires Node `>=22.12.0`, declares `whatsapp-web.js` `^1.34.7` and `qrcode-terminal` `^0.12.0`, and constrains selected transitive packages through npm overrides. The committed lockfile is the resolved installation input and repeats the same Node engine and direct dependency declarations.

The bridge uses `whatsapp-web.js` with a local Chromium profile and prints a QR code during pairing. It installs an ID-compatibility wrapper before upstream event listeners, resolves paired-account phone and LID aliases, and only treats outbound messages addressed to the paired account as self-chat. These compatibility behaviors make provider events usable by the Python adapter; they do not expand channel admission or alter Talon's experimental single-operator posture.

## Compatibility and operations

Talon is an alpha package named `deepagents-talon` (version `0.1.2`), requires Python `>=3.14,<3.15`, and exposes `deepagents-talon`. Its published SDK floors are `deepagents>=0.7.42` and `deepagents-code>=0.2.15,<1.0.0`; the same `deepagents-code` floor applies to the `openshell`, `media`, and `media-cpu` extras. It also depends on LangChain/LangGraph 1.x-compatible packages, MCP, and a SQLite checkpointer by default. In this repository the `deepagents` and `deepagents-code` dependencies are editable sibling sources, so Talon intentionally integrates against their current APIs rather than a bundled copy. dcode remains outside Talon's channel and worker lifecycle; when enabled, the dcode session bridge is a narrowly scoped local Unix-socket messaging integration.

Relevant configuration includes the assistant ID/home (`DEEPAGENTS_TALON_ASSISTANT_ID` or `AGENT_ASSISTANT_ID`, `DEEPAGENTS_TALON_HOME`), model (`DEEPAGENTS_TALON_MODEL` or `AGENT_MODEL`), model/context settings, optional sandbox, checkpoint/history URIs, MCP configuration (`DEEPAGENTS_TALON_MCP_CONFIG` or `~/.deepagents/.mcp.json`), `DEEPAGENTS_TALON_MAX_BACKGROUND_TASKS`, `DEEPAGENTS_TALON_VERBOSITY`, and channel media roots and caps. The per-turn `send_file` limit uses the configured media-byte cap; its temporary upload staging root is the assistant home's `media/outbound/`. The `media` and `media-cpu` extras provide optional local media dependencies; on Linux `media-cpu` uses the PyTorch CPU index and conflicts with `media`.

Sandboxing is opt-in and applies to shell/file execution; it does not contain MCP tools, web tools, or channel media handling, which run in the host process. Skills and memory also stay on the host. Run Talon under an OS account and with credentials appropriate to the most privileged enabled integration, and keep state and MCP configuration outside an agent-editable workspace. Validate a normal attended turn and a cron turn separately: their authority and delegation lifecycle intentionally differ.

When changing this integration:

- Keep the host as the sole owner of provider delivery and interactive callback lifetime.
- Treat hooks, approval authority, authorization, questions, and progress handlers as per-invocation capabilities; never leak them to a detached worker or delivery turn.
- Build replacement graphs before publishing them and preserve per-invocation graph/policy snapshots.
- Keep task attachment catalog-based, exact-name, and fresh-context; do not give a role parent history, all tools, or an approval bypass.
- Treat compiled and remote agents as opaque policy boundaries.
- Preserve trusted-home-only hook loading, child-session isolation, and memory reload semantics.

## Focused validation

`tests/test_runtime.py` proves transactional MCP replacement: a graph is built before the tool tuple and active graph are replaced, and a failed build leaves both prior values active. `tests/unit_tests/test_subagent_reload.py` similarly proves that an active turn stays on its original graph and that a failed subagent compilation preserves the preceding configuration. `tests/test_host.py` covers lifecycle and host interaction boundaries, while `tests/unit_tests/test_background.py` exercises detached delivery, ownership, capacity, cancellation, failure containment, retry/requeue behavior, and the distinct inline scheduled path.

`tests/unit_tests/test_memory_reload.py` is the regression guard for fresh sources: after a checkpointed turn, changes to every configured memory file—including a file edited through the agent's filesystem tools—appear on the next turn without duplicate memory blocks or stale text. `tests/unit_tests/test_context_doctor.py` verifies content-free, non-mutating diagnostics and command behavior. Hook and research-subagent tests cover hook wire payloads, denial versus approval, trusted file loading, fresh context, explicit attachments, MCP protections, and opaque-agent approval enforcement.

`tests/test_media.py` verifies that documents are path notices rather than inlined instruction text, image content is bounded and MIME-gated, unavailable attachments do not disclose their error detail, and unsafe or symlink-escaping paths are rejected. `tests/unit_tests/test_file_sending.py` verifies backend reads, per-turn binding, subagent refusal, byte limits, safe recipient names, generic transport errors, private staging, and cleanup after both successful and failed uploads. `tests/test_host.py` verifies host-side photo blocks, video path notices, lifecycle unwind, replacement/cancellation, and delivery behavior. `tests/unit_tests/test_model_selection.py` covers operator-only model switching, persistent next-turn selection, unavailable-model fallback, credentialed model discovery, and selected-model context budgets. `tests/unit_tests/test_effort.py` covers operator and allowlist enforcement, secure per-model persistence, next-turn capture, failed-write preservation, provider wire settings, nested model-setting merge, and concurrent context-variable cleanup.

`tests/test_mcp.py` covers server status contracts, standard or overridden configuration paths, schema loading and server-prefixed tool names, OAuth capability exposure, deferred refresh, serialized concurrent refresh, cancellation retry, failed-refresh behavior, and redacted connection/authentication failures.

`tests/channels/test_whatsapp.py` covers loopback-token transport, start readiness, polling and exposure filtering, self-chat restrictions, media limits and staging cleanup, health restart, and the presence of the packaged bridge resources. The bridge's Node suites cover WhatsApp identifier compatibility, self-chat and reaction normalization, acknowledgement tracking, quoted-context handling, and media download resolution and failures. Run them after installing the bridge dependencies:

```bash
cd libs/talon/deepagents_talon/channels/whatsapp_bridge
npm install
npm test
```

The focused session-bridge suites are security and routing regression tests. `test_claude_sessions.py` covers private socket discovery, peer/frame correlation, escape framing, idle notifications, owner routing, operator/approval gates, and shutdown. `test_codex_sessions.py` covers app-server handshake and turn correlation, outbound bounds, escaping, cancellation races, and approval-policy behavior. `test_dcode_sessions.py` covers private live-socket discovery, correlation acknowledgements, and the same operator, subagent, and approval-policy boundaries.

```bash
cd libs/talon
uv run pytest tests/test_runtime.py tests/test_host.py tests/test_mcp.py tests/test_media.py tests/unit_tests/test_file_sending.py tests/unit_tests/test_model_selection.py tests/unit_tests/test_effort.py tests/unit_tests/test_context_doctor.py tests/unit_tests/test_memory_reload.py tests/unit_tests/test_hooks.py tests/unit_tests/test_research_subagents.py tests/unit_tests/test_claude_sessions.py tests/unit_tests/test_codex_sessions.py tests/unit_tests/test_dcode_sessions.py
```
