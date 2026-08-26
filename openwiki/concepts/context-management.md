---
type: concept
title: Context Management (Summarization, Eviction, Offload)
description: How deepagents keeps model context bounded by summarizing long message histories, evicting or clipping large tool outputs to disk, and offloading full conversation history to per-session archives.
tags: [context-management, summarization, compaction, eviction, offload, middleware, tool-results, conversation-history]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T21:35:57.774Z
sources:
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-2fb89d2b59c886d0cb3ee3ea
    resource: repo://libs/code/deepagents_code/config_manifest.py
  - id: openwiki-source-c100a7d2ff8c43af8ad1b816
    resource: repo://libs/code/deepagents_code/offload_middleware.py
  - id: openwiki-source-9b6cab59e92c8914079f0f53
    resource: repo://libs/code/deepagents_code/offload.py
  - id: openwiki-source-9841bc6daf811e4615c54a88
    resource: repo://libs/deepagents/deepagents/middleware/_message_eviction.py
  - id: openwiki-source-64b92f60456305edc143f48a
    resource: repo://libs/deepagents/deepagents/middleware/_overflow_clip.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T21:35:57.774Z"}
---

# Context Management (Summarization, Eviction, Offload)

Long agent threads and large tool outputs both threaten the model's context
window. deepagents keeps context bounded with three cooperating mechanisms, all
implemented as middleware in the [middleware stack](/openwiki/architecture/middleware-stack.md):

1. **Per-tool-call eviction** — `FilesystemMiddleware` writes an over-large tool
   result to the backend and replaces the message with a short head+tail preview
   that points at the saved file.
2. **Conversation summarization / compaction** — `SummarizationMiddleware`
   summarizes older messages into a single summary once token usage crosses a
   threshold (or when the provider reports an overflow), offloading the full
   pre-summary history to a per-session markdown archive.
3. **Overflow tail clipping** — a reactive fallback that shrinks the trailing
   `ToolMessage` batch after a `ContextOverflowError`.

The `deepagents_code` CLI package layers hook-aware and server-owned compaction
on top of the SDK, and owns where offloaded conversation history physically
lives on the host.

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    Tool["Tool returns result"] -->|"awrap_tool_call"| EvictCheck{"result text over<br/>tool token limit"}
    EvictCheck -->|yes| Evict["write to large_tool_results and<br/>replace with preview pointer"]
    EvictCheck -->|no| Keep["keep result"]
    Model["Before model call"] -->|"wrap_model_call"| SumCheck{"_should_summarize"}
    SumCheck -->|yes| Summarize["offload history archive,<br/>summarize older messages"]
    SumCheck -->|no| CallModel["call model"]
    CallModel -->|ContextOverflowError| Fallback["summarize plus clip overflow tail"]
```

Caption: The two proactive entry points (tool-result eviction and threshold
summarization) plus the overflow fallback.

## Per-tool-call eviction of large tool results

`FilesystemMiddleware` intercepts every tool result through `wrap_tool_call` /
`awrap_tool_call`. When eviction is configured and the tool is not in
`TOOLS_EXCLUDED_FROM_EVICTION`, the result is passed to `_intercept_large_tool_result`.

`_process_large_message` (and its async twin `_aprocess_large_message`) extract
the text from a `ToolMessage`'s text content blocks and compare the length
against a char budget of `NUM_CHARS_PER_TOKEN * _tool_token_limit_before_evict`.
`NUM_CHARS_PER_TOKEN` is `4`, so the threshold is a chars-per-token approximation
of a token budget. Results at or below the threshold pass through unchanged.

Over-threshold results are handed to the shared helper
`_offload_tool_message_content`, which writes the full text to
`{large_tool_results_prefix}/{sanitized_tool_call_id}` on the backend and returns
a replacement `ToolMessage`. The replacement carries the `TOO_LARGE_TOOL_MSG`
body: a head+tail preview (5 lines each, produced by `_create_content_preview`)
and instructions to recover the full content with `read_file` using `offset`/`limit`.
Non-text content blocks (images, audio) are preserved on the replacement so
multimodal context is not lost. If the backend write fails, the original message
is kept intact rather than replaced.

The offload prefix is derived from the backend's `artifacts_root`
(`{root}/large_tool_results`). With the default root `/` this is
`/large_tool_results`; in the CLI's local mode it routes to a real per-session
host directory. See [Backends](/openwiki/concepts/backends.md) for how
`artifacts_root` and composite routes resolve these virtual paths to storage.

## Conversation summarization / compaction

`SummarizationMiddleware` wraps the model call and compacts the message history
when it grows too large. Its `wrap_model_call` / `awrap_model_call`:

1. Optionally truncates over-long tool-call arguments (`_truncate_args`).
2. Recomputes the token count and asks `_should_summarize` whether the
   configured `trigger` threshold has been reached.
3. If summarization is not needed, forwards the (possibly truncated) request to
   the handler — but catches `ContextOverflowError` and falls through to
   summarization as a reactive fallback.
4. Chooses a cutoff via `_determine_cutoff_index`, partitions messages into
   "to summarize" and "to preserve", offloads the pre-summary history to the
   backend archive, and generates an LLM summary that replaces the summarized
   messages.

`trigger` and `keep` are configurable (`ContextSize` tuples such as
`("fraction", 0.85)` or `("tokens", 100000)`); `keep` defaults to
`("messages", 20)` and `trim_tokens_to_summarize` defaults to 4000. The result is
a `HumanMessage` summary (tagged `lc_source='summarization'`) followed by the
preserved tail, plus a `Command` state update recording a `SummarizationEvent`
(`cutoff_index`, `summary_message`, offload `file_path`) and the session id.

A `compact_conversation` tool (from `SummarizationToolMiddleware`) lets the model
or a human-in-the-loop flow trigger the same compaction on demand instead of
waiting for the threshold.

### Where offloaded conversation history lives

`_offload_to_backend` / `_aoffload_to_backend` persist the full pre-summary
messages to a single markdown file per session at
`{artifacts_root}/conversation_history/{session_id}.md`. Each summarization event
**appends** a new `## Summarized at {timestamp}` section (the messages rendered as
XML via `get_buffer_string(..., format='xml')`), building a running log rather
than overwriting. Previous summary messages are filtered out to avoid redundant
storage across chained summarizations.

The session id (`session_<uuid4 hex>`) is resolved by `_get_session_id`: it
reuses a persisted `_summarization_session_id` so history appends to one file
across turns, and is scoped per graph invocation so each invocation (including
each sub-agent) gets its own history file. Base64 media in evicted messages is
written separately under `{conversation_history}/media/` and referenced by path,
keeping the archive text-only; the `DEEPAGENTS_DEFAULT_SUMMARY_PROMPT` teaches the
summarizing model to preserve those media reference tags.

Offload failure is non-fatal: `_offload_to_backend` returns `None`,
summarization still proceeds with `file_path=None`, and a warning is logged
noting older messages will not be recoverable.

### Overflow tail clipping fallback

When summarization runs specifically because a `ContextOverflowError` was raised,
`_clip_overflow_tail` (async `_aclip_overflow_tail`) additionally shrinks the
trailing `ToolMessage` batch of the preserved suffix. It engages only when the
preserved messages end with consecutive `ToolMessage`s whose combined token count
reaches a threshold derived from `keep` by
`_derive_overflow_clip_threshold_tokens` (the keep token budget, or `5_000`
tokens when `keep` is message-based). Two per-message paths:

- A `read_file` result is head-sliced to ~4k chars with a notice pointing back to
  the original `file_path` argument — no new write is needed because the full file
  already exists at that path.
- Any other result is fully offloaded to `/large_tool_results/{tool_call_id}` via
  the same eviction helper and replaced with a `TOO_LARGE_TOOL_MSG` stub.

Replacements reuse the originals' message ids so the `add_messages` reducer
overwrites them in state when propagated via the `Command` update; any message
whose backend write failed keeps its original in both lists.

## CLI compaction: hooks, locking, and server-owned offload

`CLICompactionMiddleware` (in `deepagents_code`) extends the SDK's
`SummarizationToolMiddleware` to add dcode-specific behavior while keeping the
SDK's model-initiated behavior unchanged:

- **Hook gating.** Before automatic (threshold) summarization and before a
  provider-overflow fallback, it runs the `PreCompact` hook. A hook that denies
  continuation skips compaction (and, on overflow, re-raises the original
  overflow via `_AutoCompactionBlockedError`).
- **Archive locking.** Automatic and model-initiated archive appends are
  serialized per session id with a process-local `asyncio.Lock`
  (`_archive_lock`), so concurrent compactions do not corrupt one archive's
  read-append-rewrite cycle.
- **Server-owned `/offload`.** `OffloadOperation` compacts checkpoint state
  behind an HTTP boundary. Its planning path writes only summary/session/cost
  channels (`OffloadStateUpdate`) and never `messages`. Archive appends are
  staged as a `_PendingArchive` and only committed once the checkpoint summary is
  reserved; `_ArchiveAppend.rollback` restores the exact prior archive snapshot
  if the operation aborts. An `_ArchiveReadGuard` makes the SDK's
  "missing file → truncating write" fallback fail closed, refusing to overwrite
  existing history when a prerequisite read failed. Outcomes are reported as a
  typed `OffloadResult` with a `status` of `compacted`, `empty`, `noop`,
  `denied`, or `failed`.

## Offload storage location and lifecycle (local mode)

`offload.py` owns the host storage for conversation-history archives in local
mode. `_offload_fallback_root` prefers the persistent profile root
(`DEEPAGENTS_HOME`, default `~/.deepagents`) so archives survive across sessions,
and archives always live in its `conversation_history` subdirectory
(`CONVERSATION_HISTORY_DIRNAME`). If that root cannot be created or written, it
falls back to a private temporary directory and records
`_EPHEMERAL_OFFLOAD_STORAGE = True`; `offload_storage_is_ephemeral()` exposes
whether history is on non-persistent storage. The archive subdirectory is
hardened to `0o700` and ownership-checked, but the shared profile root's
permissions are left untouched.

Large-tool-result artifacts have their own root resolver (`_artifacts_root`):
a stable, hardened per-user directory under the temp dir, with a private
unique-directory fallback (behind the stable virtual prefix
`/dcode-artifacts-fallback`) when that predictable directory is unusable.

Archives are subject to a retention sweep. `sweep_offloaded_history` deletes
archives older than `history.retention_days` (default
`HISTORY_RETENTION_DAYS_DEFAULT = 30`; `0` disables the sweep), resolved through
the config manifest so env var, `config.toml`, and `dcode config` all agree.
`_delete_expired_archive` re-checks expiry on the open descriptor with `fstat`
immediately before `unlink`, serializing against a concurrent archive refresh so
a rewrite that just updated the mtime is not deleted out from under the writer.
`delete_offloaded_history(thread_id)` removes a single thread's archive
best-effort. In server/sandbox mode the archive lives on the sandbox backend,
not the local `~/.deepagents` directory.

## Related behavior and observability

For observed evidence of whether summarization and offload actually fired in a
sampled run — and how often — see the runtime-behavior page
(`/openwiki/runtime-behavior.md`); any such counts are scoped to that specific
pull's sample and are not general limits. For how compaction interacts with
cost accounting and sessions see
[Cost and Sessions](/openwiki/operations/cost-and-sessions.md), and for how
checkpoint state (including `_summarization_event` and
`_summarization_session_id`) persists see
[State Persistence](/openwiki/concepts/state-persistence.md).
