---
type: Engineering Workflow
title: Deep Agents Code runtime, approvals, and MCP trust
description: Maintainer guide to dcode’s Textual transcript client and LangGraph server, user-message rendering and selection, approval modes, experimental Auto policy, sandboxes, and MCP configuration trust.
tags: [dcode, security, approvals, mcp, workflow, tui, transcript, tracing]
openwiki:
  roles: [workflow, integration]
  change_kinds: [ui, transcript, client-server, trace-metadata, configuration, managed-policy]
  source_paths: [libs/code/deepagents_code/config.py, libs/code/deepagents_code/config_manifest.py, libs/code/deepagents_code/configuration/providers.py, libs/code/deepagents_code/configuration/service.py, libs/code/deepagents_code/configuration/types.py, libs/code/deepagents_code/configuration/resolver.py, libs/code/deepagents_code/_ask_user_types.py, libs/code/deepagents_code/tui/widgets/messages.py, libs/code/deepagents_code/tui/textual_adapter.py, libs/code/deepagents_code/app.py, libs/code/deepagents_code/server_graph.py]
  symbols: [build_stream_config, resolve_ranked, RemoteTomlProvider, get_managed_snapshot, require_healthy_managed_config, encode_multi_select_answer, ask_user_answer_is_empty, UserMessage, QueuedUserMessage, AssistantMessage, append_content, _flush_pending_append, _stop_assistant_streams, create_cli_agent, make_graph]
  test_paths: [libs/code/tests/unit_tests/test_coding_agent_metadata.py, libs/code/tests/unit_tests/test_configuration.py, libs/code/tests/unit_tests/test_configuration_resolver.py, libs/code/tests/unit_tests/test_server_graph.py, libs/code/tests/unit_tests/test_ask_user_types.py, libs/code/tests/unit_tests/tui/test_textual_adapter.py, libs/code/tests/unit_tests/tui/widgets/test_messages.py, libs/code/tests/unit_tests/test_app.py]
  invariants: ["A valid managed policy masks lower-precedence environment values for replacement options.", "A remote managed descriptor contains only its HTTPS source, and a failed refresh does not evict the last enforceable policy.", "An empty or malformed multi-select answer never becomes Auto consent evidence.", "Sent-prompt continuation lines align under the message body, not the prefix glyph.", "Full-message selection returns submitted text rather than display-truncated content.", "The first assistant-text fragment renders immediately and later fragments are batched without loss at stream shutdown.", "Trace-wide editable metadata is always a boolean and agrees with the dcode lc_versions value."]
  validation_commands: ["cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_coding_agent_metadata.py tests/unit_tests/tui/test_textual_adapter.py -k 'ContractCompliance or versions_contains_cli_version or versions_marks_editable_cli_version'", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_configuration.py tests/unit_tests/test_configuration_resolver.py tests/unit_tests/test_server_graph.py -k 'remote_managed or failed_remote_refresh_keeps_policy_resolving_in_the_resolver or managed_health_gate_runs_off_event_loop'", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_ask_user_types.py -k 'MultiSelectAnswerEncoding or AskUserAnswerIsEmpty'", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k UserMessageAppearance", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k TestAssistantMessageStreamCoalescing"]
---
# Deep Agents Code: runtime, approvals, and MCP trust

`libs/code` packages the prebuilt terminal coding agent (`dcode` / `deepagents-code`). It is the coding-specific consumer of the SDK described in [Runtime and package architecture](../architecture/overview.md), not a standalone agent runtime.

## Process and graph flow

Deep Agents Code intentionally separates UI from graph execution:

```text
CLI parsing (`main.py`)
  -> Textual client/app (`app.py`, UI widgets)
  -> `langgraph dev` server subprocess
  -> cached `server_graph.make_graph()`
  -> `create_cli_agent()` middleware/tool/subagent assembly
  -> core `create_deep_agent()` / LangGraph execution
```

- `libs/code/deepagents_code/main.py` validates CLI/configuration, prevents autonomous flags in ACP or headless modes, constructs server arguments, and starts the Textual app.
- `server_graph.py` reads `DEEPAGENTS_CODE_SERVER_*` config, resolves models off the event loop, loads MCP/plugins, optionally builds a persistent sandbox, and caches the graph for the server process lifetime behind a lock.
- `agent.py` configures the SDK with model selection, goal/resume state, ask-user, memory/skills/plugins, local context, shell/interpreter support, compaction, rubric grading, approval middleware, and main/general/async subagents.
- Local execution uses `LocalShellBackend` rooted at the working directory; remote execution delegates filesystem and shell operations to the selected sandbox.

`libs/code/ARCHITECTURE.md` and `DEVELOPMENT.md` are the first primary docs to read when changing this path. Changes to the server-side graph construction should also account for the core assembly rules in [Runtime and package architecture](../architecture/overview.md).

## Trace metadata and editable-install attribution

Consult this section when changing LangSmith/LangGraph trace fields, per-turn attribution, or editable-install detection. `config.py::build_stream_config()` is the single assembly point for the `RunnableConfig` passed to graph execution. Both interactive `tui/textual_adapter.py` and headless `client/non_interactive.py` call it, so changing a metadata key affects both user-facing execution paths.

```mermaid
sequenceDiagram
    participant TUI as Textual client
    participant Headless as Non-interactive client
    participant Config as build_stream_config
    participant Graph as LangGraph execution
    TUI->>Config: build config once per submitted prompt
    Headless->>Config: build config once per process turn
    Config->>Config: read cached PEP 610 editable state
    Config->>Graph: configurable thread id and metadata
    Graph-->>Graph: propagate metadata to descendant runs
```

This shows the shared configuration boundary: the metadata block is trace-wide, not a root-run-only payload.

`_resolve_editable_info()` reads `deepagents-code` PEP 610 `direct_url.json` once per process and caches `(is_editable, source_path)`. `build_stream_config()` writes `metadata["editable"]` on **every** invocation, including `False` for ordinary installations. The same cached boolean controls the `+editable` local-version marker in `metadata["lc_versions"]["deepagents-code"]`; trace consumers should filter on the boolean instead of parsing that string. This is diagnostic attribution, not a security or approval-policy signal.

The interactive adapter advances its per-thread turn markers before calling the builder off the Textual event loop. The non-interactive client creates one UUID turn ID and uses turn number `1` for its one-process run. `build_stream_config()` deliberately omits contract keys that apply only to selected run types (`approval_policy`, `ls_subagent_id`, and `ls_subagent_type`), because LangGraph propagates this metadata to root, LLM, tool, subagent, and interrupted runs. Adding a scope-limited key here would leak it into invalid run types.

When extending trace metadata, add it at `build_stream_config()` only if it is valid on every propagated run; otherwise locate a genuinely scoped runtime seam. Preserve the shared editable lookup rather than performing another PEP 610 read or deriving a potentially divergent value. Validate both value states and propagation safety with the focused quiet check:

```bash
cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_coding_agent_metadata.py tests/unit_tests/tui/test_textual_adapter.py -k 'ContractCompliance or versions_contains_cli_version or versions_marks_editable_cli_version'
```

`TestBuildStreamConfig` in `tests/unit_tests/tui/test_textual_adapter.py` exercises editable and non-editable values and their version representation. `TestContractCompliance` in `tests/unit_tests/test_coding_agent_metadata.py` checks the shared metadata against the vendored `coding-agent-v1` validator for every propagated run type. A live trace validation is conditional on changing the external contract or its validator; the unit tests explicitly describe that external check as end-to-end acceptance rather than a default local check.

## Configuration and managed policy

Consult this section when adding a dcode configuration option, changing precedence, or enforcing deployment policy. `config_manifest.py` declares the typed option surface; `configuration/providers.py` coerces each source; and `configuration/resolver.py::resolve_ranked()` resolves them. The normal precedence is managed policy (rank 200), a reserved but currently unwired CLI seam (300), environment (400), user `~/.deepagents/config.toml` (500), then manifest defaults (1000). Lower numeric rank wins.

```mermaid
flowchart TD
    Anchor["Fixed managed config file"] --> Descriptor{"Remote descriptor"}
    Descriptor -->|No| Managed["Managed TOML policy"]
    Descriptor -->|Yes| Remote["Validated HTTPS TOML policy"]
    Remote --> Managed
    Managed --> Resolve["Ranked configuration resolver"]
    Environment["Environment values"] --> Resolve
    UserConfig["User config TOML"] --> Resolve
    Defaults["Manifest defaults"] --> Resolve
    Resolve --> Effective["Effective dcode configuration"]
    Managed --> Gate["Startup health and policy gate"]
    Gate --> Effective
```

This flow shows that the fixed file either supplies policy directly or anchors one validated remote policy; the resulting managed source participates in normal resolution and the launch-time enforcement gate.

`managed_config.toml` is an administrator-owned OS file: `/etc/dcode/managed_config.toml` on Linux, `/Library/Application Support/dcode/managed_config.toml` on macOS, and the registry-derived ProgramData location on Windows. The Windows production lookup intentionally ignores a caller-controlled `ProgramData` environment variable. `configuration/service.py::require_healthy_managed_config()` gates startup: corrupt, unreadable, indeterminate, or unenforceable managed policy raises an error instead of becoming an empty policy. MCP disabled-server checks also fail closed when policy cannot be read.

### Remote managed-policy descriptors

The fixed OS file can now remain a local trust anchor while the complete policy is published remotely. Its remote form is **exclusive**—it contains only this table and one non-empty source string:

```toml
[managed_config]
source = "https://config.example.com/policy.toml"
```

`service.py::_remote_managed_snapshot()` rejects descriptor keys other than `source` and rejects any local policy keys beside `[managed_config]` before making a network request. The downloaded TOML is the managed tier at rank 200, so it has the same precedence and enforcement semantics as a local managed policy; it cannot itself contain `[managed_config]`, preventing policy-source chaining.

`providers.py::RemoteTomlProvider` accepts only normalized, credential-free absolute HTTPS URLs with no query or fragment, uses system TLS validation, bypasses environment proxies, refuses redirects, and applies one five-second end-to-end fetch deadline with a 1 MiB response limit. It accepts only a complete HTTP 200 policy body; empty, malformed, partial, compressed, oversized, or nested-descriptor responses are unhealthy. These are availability and policy-integrity guards, not a substitute for controlling the publisher or TLS trust roots.

The local descriptor remains the repair point only when it is malformed. Once its URL is validated, `ProviderStatus.remote_source` permits `doctor` and startup errors to identify the remote document safely; rejected URLs are not echoed, avoiding credential/query leakage. For a remote outage, repair the published source rather than deleting the anchor—removing it would drop the managed tier.

`get_managed_snapshot()` fetches outside its snapshot lock, and server startup plus Textual `/restart` invoke their health/reload paths off the event loop. A failed first fetch fails startup. After a valid policy is in use, a failed or unenforceable refresh is reported but does not replace the cached last enforceable generation, so resolver reads cannot fall through to a user value during an outage. Preserve this ordering when changing caches, reloads, or diagnostics.

For replacement options, a `Found` value from a durable managed source masks lower-precedence **environment** values; a lower-precedence durable user value cannot reverse an environment value that already wins. Union and deep-merge options deliberately retain valid contributions, including deny-list restrictions. Do not add a resolver bypass or treat a failed managed load as absent policy: that can turn an administrator restriction into a user-controlled configuration.

When extending this seam, register the option in `config_manifest.py`, choose its typed coercion and merge strategy, route it through the ranked providers, and make the user config writer leave the managed path untouched. For remote descriptors or fetch/reload changes, preserve descriptor exclusivity, validated-only diagnostics, bounded no-proxy/no-redirect HTTPS I/O, off-loop execution, and last-known-good resolution. Validate those behavior boundaries quietly before UI polish:

```bash
cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_configuration.py tests/unit_tests/test_configuration_resolver.py tests/unit_tests/test_server_graph.py -k 'remote_managed or failed_remote_refresh_keeps_policy_resolving_in_the_resolver or managed_health_gate_runs_off_event_loop'
```

`test_remote_managed_descriptor_must_be_exclusive`, `test_failed_remote_reload_keeps_previous_policy`, and `test_failed_remote_refresh_keeps_policy_resolving_in_the_resolver` cover anchor shape, refresh retention, and resolver-level authority retention. `test_managed_health_gate_runs_off_event_loop` verifies server scheduling does not wait for remote policy I/O. Use `TestRestartCommand::test_remote_config_refresh_keeps_chat_input_responsive` in `test_app.py` when changing the interactive restart path; it proves a slow refresh leaves the Textual message pump usable. Add `test_configuration_resolution.py` or the specific consumer suite when changing a concrete option. `DEEPAGENTS_CODE_SHOW_USAGE_STATS` is a narrow teardown-output option: falsy values suppress only the session usage table for both TUI and `-x`/`--execute`, not all headless output.

## Ask-user wire contract

`ask_user` is interactive middleware and also feeds the Auto policy described below. `_ask_user_types.py` is the shared wire-format module used by the tool, TUI adapter, and `auto_mode`, avoiding a dependency from those consumers onto one another. `QuestionType` supports `text`, `multiple_choice`, and `multi_select`; choice types require non-empty choices.

A multi-select answer remains one `str` in the positional `answers: list[str]` wire shape, but `encode_multi_select_answer()` serializes selected values as a JSON array. This preserves commas, quotes, and newlines in a choice and makes an unselected question `[]`. Consumers must use `ask_user_answer_is_empty()` rather than `strip()`: `[]` is truthy but is empty for both required-answer validation and Auto consent evidence; malformed multi-select JSON also fails closed. Never restore comma-splitting as a fallback.

Use the focused contract test when changing question types, encoding/decoding, transcript display, or authorization evidence:

```bash
cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_ask_user_types.py -k 'MultiSelectAnswerEncoding or AskUserAnswerIsEmpty'
```

Broaden to `test_ask_user_middleware.py`, `tui/test_textual_adapter.py`, and `test_auto_mode.py` only if the change crosses the tool, client interrupt, or Auto authorization boundary.

## Transcript presentation and selection

Consult this section for interactive dcode transcript changes, not for agent execution semantics. `UserMessage` in `libs/code/deepagents_code/tui/widgets/messages.py` is the sent-prompt widget mounted by `app.py`; it represents client-side input after submission and does not change what the server graph receives. `QueuedUserMessage` is a dimmed, temporary pre-send representation and deliberately retains its separate border/opacity treatment.

```mermaid
flowchart TD
    Submit["Client submits prompt"] --> Widget["UserMessage stores original content"]
    Widget --> Prefix["Render prefix and body"]
    Prefix --> Long{"Body exceeds display threshold"}
    Long -->|No| Full["Render full body"]
    Long -->|Yes| Collapsed["Render head tail and expand hint"]
    Collapsed --> Toggle["Click or Ctrl+O toggles expanded state"]
    Toggle --> Expanded["Render full body and collapse hint"]
    Widget --> Select["Full selection uses original content"]
```

This flow is local to the Textual client: submitted text is retained for copy/selection even when the transcript render is collapsed.

### Rendering invariants and extension seam

- Sent prompts use a primary-tinted surface with one-cell top/bottom padding, no left padding, one-cell right padding, and one row of external separation. The visual boundary makes user input scannable without adding padding to high-frequency assistant/tool rows.
- The prompt/mode prefix is exactly two cells (`> `, `$ `, or `/ `). `_UserMessageContent` shifts wrapped lines by that gutter, so soft-wrap and explicit continuation text begins under the body, not under the glyph. Preserve this when changing prefix text, padding, or custom rendering.
- Long bodies use head-and-tail collapse with a clickable `@click` hint; `Ctrl+O` and click toggle `_expanded`. `get_selection()` must return the original full text for select-all/end selections, while partial selections stay aligned to the displayed render. Mode detection can strip `!`, `!!`, or `/` only when enabled; literal `-m`/`--message` input with a leading path slash remains plain text.
- `set_cancelled()` only dims an interrupted prompt. It is a client transcript state and must not be mistaken for a server cancellation mechanism.

The focused behavioral suite is `libs/code/tests/unit_tests/tui/widgets/test_messages.py::TestUserMessageAppearance`: it asserts the 15%-alpha background, four padding edges, and the continuation gutter for ordinary, shell, and slash prompts. Run the quiet narrow check from `libs/code`:

```bash
uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k UserMessageAppearance
```

Broaden to the surrounding message-widget tests when changing collapse, selection, pointer handling, mode parsing, or queued-message behavior. Do not run server, approval, or integration tests for a CSS/layout-only change unless the edit also crosses the client/server submission boundary.

## Assistant response streaming

Consult this section when changing dcode assistant-text latency, markdown streaming, batching, or turn-exit cleanup. This is a **Textual transcript** concern, not a graph-execution change: the adapter that receives streamed graph events dispatches output to `AssistantMessage` in `libs/code/deepagents_code/tui/widgets/messages.py`.

```mermaid
flowchart TD
    Fragment["Assistant text fragment"] --> Append["append_content stores source text"]
    Append --> First{"Flush timer exists"}
    First -->|No| Immediate["Write first fragment immediately"]
    Immediate --> Timer["Start 100 ms flush timer"]
    First -->|Yes| Pending["Buffer later fragment"]
    Timer --> Flush["Flush pending text to MarkdownStream"]
    Pending --> Flush
    Finish["Completion or adapter exit"] --> Stop["Stop timer and flush pending text"]
    Stop --> Final["Stop stream and re-render full markdown"]
```

This lifecycle makes the first visible assistant text prompt while preserving timer coalescing for later fragments, which avoids a markdown write per token on the UI event loop.

`AssistantMessage.append_content()` appends every non-empty fragment to `_content_parts` and `_pending_append`. With no timer, it awaits `_flush_pending_append()` immediately and then creates one interval timer at `_STREAM_FLUSH_INTERVAL` (0.1 seconds). With a timer already running, it only buffers the text; the timer drains it. `_flush_pending_append()` restores text to the front of the buffer after a write failure, logs the rendering error, and leaves a later tick able to retry rather than dropping content.

`stop_stream()` is the ordinary terminal boundary: it stops the timer, drains pending text, stops `MarkdownStream`, and fully re-renders `_content` to preserve the existing fenced-code correctness workaround. `set_content()` instead stops the timer, clears pending text, stops an active stream, and performs one replacement render. `execute_task_textual()`’s `finally` calls `_stop_assistant_streams()` as an error-path backstop, so a non-cancellation mid-stream error also drains buffered content; that cleanup must not mask the original exception. This adapter-to-widget relationship prevents a silent truncated reply when a stream exits unexpectedly.

When modifying this seam, preserve all of these observable boundaries: the first fragment is immediate, later writes are coalesced by a single timer, full source content remains available for final rendering, stream completion drains pending text, and replacement content cannot be overwritten by a stale timer. Do not move batching into graph assembly or apply it to sent `UserMessage` rendering. Changes that alter event ownership or when a stream is finalized should additionally inspect `tui/textual_adapter.py` and its stream-loop tests; ordinary widget batching does not require server, approval, or integration checks.

The narrow behavioral suite is `libs/code/tests/unit_tests/tui/widgets/test_messages.py::TestAssistantMessageStreamCoalescing`. Its `test_first_append_flushes_immediately`, `test_timer_flushes_later_text`, `test_stop_stream_flushes_and_cancels_timer`, and `test_set_content_drains_and_cancels_active_timer` cover first-write latency, later batching, completion drain, and stale-timer isolation. Run it quietly from `libs/code`:

```bash
uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k TestAssistantMessageStreamCoalescing
```

## Approval modes are safety policy, not containment

The README says that starting in a directory trusts its artifacts before approval. Remote sandboxes are the recommended boundary for untrusted repositories. Human approval complements that boundary but does not turn local execution into a sandbox.

| Mode | Behavior | Important constraint |
| --- | --- | --- |
| `manual` | Interrupts gated operations for user approval. | Default/fail-closed mode. |
| `auto` | Experimental deterministic policy plus classifier review may approve eligible operations. | Limited to local interactive, unsandboxed use with `DEEPAGENTS_CODE_EXPERIMENTAL`; otherwise it downgrades to manual. |
| `yolo` | Bypasses HITL. | Requires a versioned local acknowledgement stored with restrictive permissions. |

Approval state is a hashed per-thread record in LangGraph Store, read and validated by the server against the active thread. Missing, malformed, or unreadable state falls back to Manual. This server/client synchronization exists so a user can change modes during an active conversation; failure to synchronize a return to Manual must not leave an action running under a stale permissive policy.

The gated inventory includes writes/edits/deletes, execute, web search/fetch, subagent/task operations, optional compaction, and non-read-only MCP tools. Keep that inventory synchronized with the middleware’s interrupt configuration when adding a tool.

### Auto-mode authority boundary

The recent classifier-backed Auto feature is deliberately narrow:

- Fast-path writes must stay inside the trusted root and exclude sensitive paths such as CI/hooks, shell scripts, and dependency/config locations.
- Fast-path shell approval permits a small read-only Git set or narrow configured commands; shell control operators and broad/wildcard commands are rejected.
- Classifier input may be authorized only by **literal, pre-expansion user text attached by the client**. File content, tool output, and assistant prose cannot expand authority. A same-turn `ask_user` response is included only after server validation; its empty/malformed multi-select representation is withheld according to the [ask-user wire contract](#ask-user-wire-contract).
- The implementation redacts/sanitizes persisted reasons and validates tool-call identities/batches exactly.
- `readOnlyHint` only bypasses gating when it is literal, coherent boolean metadata with no destructive hint. Ambiguous metadata fails closed.

Auto is neither an OS boundary nor a guarantee that delegated work is classifier-reviewed. Parent Auto review must not be assumed to cover all subagent internals, and PTC/interpreter host-bridge calls have their own policy boundary. Security-sensitive changes here need both a code review focused on authority propagation and explicit top-level/delegated-path tests.

## MCP sources and project trust

MCP configuration is resolved low-to-high from user `~/.deepagents/.mcp.json`, project `.deepagents/.mcp.json`, project `.mcp.json`, then explicit configuration. Plugin configurations are also composed server-side. Supported transports are stdio, HTTP, and SSE; config validation covers server shape, headers/auth, and mutually exclusive tool filters.

Project-declared MCP configuration is a trust boundary: it can spawn a local command, cause SSRF, or exfiltrate interpolated headers. Thus project stdio **and remote** servers are gated. Whole-config `--trust-project-mcp` is possible, but scoped user-owned approvals/environment allowlists can authorize individual servers; explicit denial wins. `${VAR}` values are interpolated only at activation, and the loader isolates individual server errors while redacting resolved values when interpolation was used.

Runtime discovery uses throwaway sessions; tool wrappers use a lazy process-wide session manager with retry/invalidation for transient/dead/reauth sessions. Loading is bounded-concurrent while output ordering remains deterministic.

### Cached MCP tool failure boundary

`_build_cached_mcp_tool()` in `libs/code/deepagents_code/mcp_tools.py` turns each discovered tool into a LangChain `StructuredTool`. Its coroutine obtains a cached session, retries a transient session failure once after invalidation, and raises a `ToolException` for failures the model must see. `_handle_cached_mcp_tool_error()` is the sole `WARNING`/traceback logging boundary for those recoverable `ToolException`s and returns the tool-local error text. Do not add a second warning in the coroutine: one failed tool call must yield one failure warning, not duplicate diagnostics.

Cleanup warnings are separate: invalidating a failed retry session or closing a session can warn independently because they describe a resource-cleanup problem rather than a duplicate tool failure. Preserve that distinction when changing retries or error handling. Re-raise cancellation, keyboard interrupt, system exit, and existing `ToolException` values unchanged; the wrapper must not turn control flow or actionable MCP errors into a generic retry result.

## Tests and safe modification sequence

Run from `libs/code`:

```bash
uv sync --all-groups
make check                 # package full local suite
make test                  # unit/no-network
make integration_test      # network-enabled tests
```

The pytest defaults enforce a 30-second timeout and strict markers/configuration. Relevant anchors:

- `tests/unit_tests/test_approval_mode.py`: store failures/malformed state fail closed; YOLO acknowledgement behavior.
- `tests/unit_tests/test_auto_mode.py`: provenance, annotation coherence, path/Git policies, classifier failures, replay/escalation, denials, and headless MCP guards.
- `tests/unit_tests/test_server_graph.py`: graph cache, startup error handling, MCP discovery, off-loop construction, and no-MCP/read-only conditions.
- `tests/unit_tests/test_mcp_tools.py::TestCachedSessionProxy::test_repeated_transient_error_surfaces_tool_message`: a second transient failure becomes a model-visible error after one retry and logs exactly one tool-failure warning with traceback.
- `tests/unit_tests/test_mcp_tools.py::TestCachedSessionProxy::test_generic_oserror_is_not_retried`: a non-transient `OSError` is model-visible without session retry and has the same single-warning contract.
- `tests/integration_tests/test_auto_approve_remote.py`: actual approved/rejected remote writes, including subagent behavior.

For the cached MCP error seam, use the quiet focused check before broader package checks:

```bash
cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_mcp_tools.py -k 'repeated_transient_error_surfaces_tool_message or generic_oserror_is_not_retried'
```

Before changing dcode: identify whether the behavior is client UI, persisted approval state, graph construction, middleware, backend/sandbox, or MCP session lifecycle; make the change at that boundary; then test both failure-to-manual and success paths. For repository-wide CI/release context, see [Evaluation and release](evaluation-and-release.md) and [Operations and testing](../engineering/operations-and-testing.md).
