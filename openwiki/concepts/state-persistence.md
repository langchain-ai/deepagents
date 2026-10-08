---
type: persistence architecture
title: State, Checkpoints, and Sessions
description: Explains how LangGraph checkpoints, dcode's local SQLite sessions, thread ownership fencing, durable titles, and separate graph stores relate. Covers restoration, isolation, deletion, inspection, and the distinct Talon persistence model.
tags: [deepagents, dcode, persistence, checkpoints, sqlite, thread-ownership, recovery, thread-inspection]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-08T08:07:53.482Z
sources:
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-1f9226665e99f6f846936c59
    resource: repo://libs/code/deepagents_code/built_in_skills/deepagents-thread-inspector/scripts/inspect_sessions.py
  - id: openwiki-source-73a12d41c3ec5c3f079ed79e
    resource: repo://libs/code/deepagents_code/built_in_skills/deepagents-thread-inspector/SKILL.md
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-0f8622164498a685abc913d5
    resource: repo://libs/code/deepagents_code/sessions.py
  - id: openwiki-source-91c9283d1547adfffd627c43
    resource: repo://libs/code/deepagents_code/thread_ownership.py
  - id: openwiki-source-52062c280ae38e9e9acab191
    resource: repo://libs/code/deepagents_code/thread_titles.py
  - id: openwiki-source-c8dacdfd6192dd22d24a9362
    resource: repo://libs/code/tests/integration_tests/test_pending_work_recovery.py
  - id: openwiki-source-140e3a9397d67359bab19562
    resource: repo://libs/code/tests/unit_tests/skills/test_thread_inspector.py
  - id: openwiki-source-cd2a5280cf3ca3ab491d7a8e
    resource: repo://libs/code/tests/unit_tests/test_sessions.py
  - id: openwiki-source-a5951057e151512583e7fd3f
    resource: repo://libs/code/tests/unit_tests/test_thread_ownership_transitions.py
  - id: openwiki-source-a5e918d96b1dae3f7adec3f5
    resource: repo://libs/code/tests/unit_tests/test_thread_ownership.py
  - id: openwiki-source-53083c05d51a08d395327737
    resource: repo://libs/code/tests/unit_tests/test_thread_titles.py
  - id: openwiki-source-822ae989625ba99d4c7cc08b
    resource: repo://libs/deepagents/deepagents/_messages_reducer.py
  - id: openwiki-source-07f9eac13e71bcbdb4e6994b
    resource: repo://libs/deepagents/deepagents/backends/state.py
  - id: openwiki-source-21e2b0401425a427d8cea9c1
    resource: repo://libs/deepagents/deepagents/backends/store.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-995d5d95882808a64071f617
    resource: repo://libs/talon/deepagents_talon/archive_saver.py
  - id: openwiki-source-c2be68f237284dc06b9c12f7
    resource: repo://libs/talon/deepagents_talon/checkpoint_backends.py
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-2318fb8a25701a5cdae717fe
    resource: repo://libs/talon/deepagents_talon/history_vector_backends.py
  - id: openwiki-source-811fef57cecdbee2ba06a7b5
    resource: repo://libs/talon/deepagents_talon/store_archive.py
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-c996df77875d3c6b30ca07cf
    resource: repo://libs/talon/tests/unit_tests/test_archive_saver.py
  - id: openwiki-source-628fd919fd2bdb09579bfb16
    resource: repo://libs/talon/tests/unit_tests/test_checkpoint_backends.py
generated: { by: "openwiki/0.4.2", at: "2026-10-08T08:07:53.482Z" }
---

# State, Checkpoints, and Sessions

Persistence is deliberately divided by scope and owner. A dcode session is a LangGraph thread checkpointed in local `sessions.db`; it is neither a workspace binding nor a durable record of every auxiliary subsystem. DeepAgents graph state and its file backends have their own semantics, while Talon has a separate assistant-scoped checkpoint, archive, vector, and scheduler model. Do not treat any one of these stores as a transactional backup of the others.

## What belongs where

| Layer | Scope and authority | Persistence meaning |
| --- | --- | --- |
| dcode sessions | A LangGraph `thread_id` in the local SQLite sessions database | Resumable graph checkpoints and writes, checkpoint metadata for discovery, and a separate durable thread-name table. |
| Local thread ownership | One local client process per dcode thread | Filesystem leases and fencing tokens prevent an old server/client from mutating a thread after release, takeover, or token rotation. |
| Workspace binding | A remote-server thread registration and validated workspace descriptor | Binds operations to a workspace; it is distinct from checkpoint storage and may need re-registration after server restart. |
| Pending-work recovery | The current graph snapshot and remote runs | Abandons unfinished nodes/tasks/interrupts rather than letting a resumed session execute stale tool work. |
| DeepAgents `StateBackend` | Current graph thread's `files` channel | Checkpointed only with graph state. |
| DeepAgents `StoreBackend` | Caller-selected `BaseStore` namespace | Cross-thread file records, durable only if the supplied store is durable. |
| Cost and offload records | dcode-owned side stores and files | Separate from checkpoints; deletion explicitly cleans them up rather than SQLite automatically owning them. |
| Talon records | Assistant-local home plus independently configured checkpoint/archive stores | A different application persistence model; Talon archive and vector data are not dcode session history. |

## dcode session database and thread identity

`deepagents_code.sessions` resolves `DEFAULT_STATE_DIR / "sessions.db"` after hardening the state directory. It opens `aiosqlite` through a cancellation-conscious wrapper: application connections use an extended lock timeout, worker threads are joined after close, and an opening-time guard preserves a SQLite handle that cancellation might otherwise leak.

New threads receive a full UUID7 string. UUID7 is time-ordered, so newly created IDs naturally sort by creation time; old short hexadecimal IDs remain valid and are listed alongside UUID7 threads. IDs identify sessions, not titles: a meaningful title is separately stored and can be changed without rewriting conversation history.

The database is LangGraph's SQLite checkpoint schema, not a hand-maintained conversation table. `list_threads()` derives one row per thread from checkpoint metadata and makes the common listing path index-only with `idx_dcode_threads_list_v2`, avoiding a scan of large serialized checkpoint blobs. The one-time index build may be slow on an existing large database, but a failure is non-fatal: listing remains correct and falls back to a slower scan. Listings can filter by agent, branch, or exact `cwd` string; a `cwd` filter excludes legacy rows with no stored path. They can sort by latest activity or by creation time, and their in-memory list, message-count, and initial-prompt caches are presentation accelerators rather than sources of truth.

Names are intentionally loaded after the limited metadata query so the covering index stays compact. `dcode_thread_names` is the authoritative, durable name record. For databases predating that table, readers fall back to `thread_name` in the latest root checkpoint metadata. Renaming validates a printable, single-line trimmed name of 1–50 characters, starts an immediate SQLite transaction, requires that the thread exists, and atomically upserts the durable row; `only_if_unnamed=True` lets concurrent automatic naming select only one winner and preserves an existing generated or manual title. The latest checkpoint metadata is also updated for legacy interoperability, but later checkpoints cannot erase the durable title. Thread deletion removes the name record.

### Generated titles are proposals, not state

`generate_thread_name()` provides the text for an optional title; persistence still happens through `rename_thread()`. It builds a bounded input from visible human and AI messages only, excluding system, internal/control, tool, and user-shell messages, then invokes a tool-free chat model with callbacks disabled. The returned text is normalized to the same one-line, printable, 50-character limit as manual names. The app runs generation outside the message pump, charges it to the thread, and uses `only_if_unnamed=True` for automatic titles, so an automatic result cannot overwrite a title that was set while the model request was in flight. Model initialization and its cancellation cleanup are awaited before cancellation escapes, avoiding a background provider-settings mutation after a timeout.

The messages channel may be a `DeltaChannel`, so the newest checkpoint need not contain an inline message list. For display, dcode reconstructs counts by replaying root-namespace `writes` in checkpoint/task/index order and excludes subgraph writes. It intentionally includes pending writes because the normal head-state read applies them; dcode does not create time-travel branches, so that full linear fold represents the visible head. Initial prompts instead come from the first messages write when possible, because partial checkpoints may omit `messages`.

`get_checkpointer()` yields an ownership-fenced `AsyncSqliteSaver`. `save_thread_seed()` is the narrow bridge for discovering a remote handoff locally: it writes an initial seed only when no local checkpoint exists and later activity remains authoritative on the connected server. It holds or obtains the thread lease for the seed, stamps agent and workspace metadata, and releases only a lease it acquired itself.

## Listing, lookup, and UI selection

`list_threads()` is the normal structured discovery API. The CLI uses it with configurable limits, optional verbose checkpoint enrichment, and text or JSON output. The `/threads` selector initially uses recent cached rows when available, then loads and enriches rows in the background; it supports filtering/searching and presents name, identifiers, agent, timestamps, branch, location, prompt, and message-count fields according to user configuration. A `@@` composer query searches cached thread ID, durable/fallback name, initial prompt, agent, branch, and cwd, then inserts the unambiguous durable token `@@(thread:THREAD_ID)` rather than display text.

Resume is more conservative than browsing. An explicit ID must exist and be reserved before it is adopted; a missing ID can yield similar prefix suggestions before a fresh UUID7 thread is created. For a bare “most recent” request on locally managed storage, the app obtains IDs in checkpoint-recency order and atomically tries each lease, skipping occupied threads; if all candidates are open elsewhere it offers a new thread rather than reading or loading one. Resume policy can reject an unverifiable or too-old `updated_at`, and a stored cwd can prompt a workspace choice. A selection conflict leaves the picker open with its filter and selection intact.

## Local ownership, tokens, and fenced writes

A dcode SQLite database may be shared by the UI and its independently running `langgraph dev` process. `thread_ownership` therefore creates a per-database `.owners` directory (mode `0700`), hashes each `thread_id` for its lock-file stem, and requires operating-system file locks rather than accepting a soft-lock fallback. The reservation is process-scoped in an in-memory registry and cross-process through a `.client` file lock; different threads can be active concurrently.

Acquiring a lease writes a random owner token to the matching `.owner` file while retaining the client lock. A mutation takes a separate writer gate, validates both that its token still matches and that the client reservation is still live, then proceeds. `rotate()` takes the same gate before replacing the token, so it waits for an in-flight mutation and invalidates prior writers without releasing the reservation. Operations defer cancellation until their lock/file mutation completes, preventing an interrupted cleanup from exposing a thread prematurely.

```mermaid
stateDiagram-v2
    [*] --> Available
    Available --> Reserved: acquire client lease and token
    Reserved --> Writing: writer gate and token validation
    Writing --> Reserved: mutation completes
    Reserved --> Rotating: replacement server or client
    Rotating --> Reserved: token replaced
    Reserved --> Released: client releases lease
    Released --> Available
    Writing --> StaleRejected: token changed or lease absent
    Rotating --> StaleRejected: old token attempts write
    StaleRejected --> [*]
```
*The local lifecycle fences SQLite mutations: reservation precedes writes, rotation preserves the reservation while invalidating old tokens, and stale writers cannot write after release or takeover.*

`owned_saver_class()` applies this guard to checkpoint `aput`, pending `aput_writes`, and `adelete_thread`. The saved config retains `x-dcode-thread-owner`; a missing, released, changed, or stale token is rejected. Deletion first tries to reserve the thread, refuses an open thread, holds the reservation through the guarded database deletion, and releases afterward. Consequently an old server cannot append a checkpoint after the client crashes, changes workspace/server, or hands ownership to a successor.

`RemoteAgent` propagates the locally held token into mutation configurations and request headers. It does **not** claim reads: state reads can remain useful for discovery. An externally managed server can set `local_ownership=False`, which intentionally disables this local fencing because its storage is separate.

## Remote thread and workspace binding

The client launches a temporary `langgraph dev` workspace whose generated `checkpointer.py` reads `DEEPAGENTS_CODE_SERVER_DB_PATH` and instantiates the ownership-fenced SQLite saver. Startup resolves a required project workspace, then calls `RemoteAgent.set_workspace()` with the user cwd, workspace policy claim, and policy fingerprint.

A workspace is not encoded in the SQLite checkpoint. Before thread-scoped operations, the remote client requests a server-validated workspace descriptor and caches it per thread. A workspace switch replaces that cached binding; policy and fingerprint must be configured together. Checkpoint persistence and the dev server's HTTP thread registration are also separate: `aensure_thread()` idempotently creates the remote thread record with `if_exists="do_nothing"`, allowing a persisted checkpoint to be used after a server restart before a subsequent state mutation.

When replacing the server for a workspace change, the application rotates the active thread token before the old server can continue writing. This fencing is retained even if stopping the old server, starting the replacement, or the switch itself fails; the client that remains active must obtain/use the current token.

## Recovering unfinished graph work

A resumed state is considered pending when its snapshot has a queued `next` node, tasks, or interrupts. Resuming such work blindly is unsafe: an interrupted model turn may be queued to run a tool after its former client has gone away.

`RemoteAgent.aabandon_pending_work()` first best-effort cancels pending/running server runs. It reads the checkpoint, creates terminal error `ToolMessage` responses only for unanswered tool calls in the trailing turn, attributes those to the `tools` node, and then updates state at `__end__`. It verifies that no pending work remains. If a state update initially receives HTTP 409 because the cancelled run has not settled, `aupdate_state()` cancels active runs concurrently with bounded waits and retries once. This recovery intentionally prevents stale tool execution rather than attempting to resume it.

## Read-only thread inspection

For a traced conversation, prefer LangSmith tooling. For offline, untraced, or local-store inspection, use the built-in `deepagents-thread-inspector` skill and its script—not ad hoc blob decoding. **Inspection is read-only.** The script opens the database using SQLite URI `mode=ro`, requires the `checkpoints` and `writes` tables, and emits JSON without changing session rows.

Resolve `SKILL_DIR` to the installed skill directory, then use a full ID or unique prefix. Do not infer the current thread from the most-recent list: obtain the actual current ID or ask for it.

```bash
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode latest-turn
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode summary
python3 "$SKILL_DIR/scripts/inspect_sessions.py" THREAD_ID --mode transcript
python3 "$SKILL_DIR/scripts/inspect_sessions.py" --list 20
```

The default path is `$DEEPAGENTS_HOME/.state/sessions.db` (or `~/.deepagents/.state/sessions.db`); `DEEPAGENTS_SESSIONS_DB` overrides it. The script rejects unsupported profile-home forms rather than silently inspecting a database dcode does not use. Its thread-prefix lookup treats `%` and `_` literally, accepts only a unique root-namespace match, and reports an ambiguous prefix rather than choosing one.

For a selected root thread, the script prefers materialized messages from the newest checkpoint and applies that checkpoint’s pending writes; if this is absent, malformed, or undecodable, it replays root `messages` writes in checkpoint order using LangGraph’s canonical message semantics. It emits warnings while preserving as much recoverable output as possible: corrupt metadata, skipped writes, and malformed overwrites are surfaced rather than represented as a trustworthy empty transcript. Summaries prefer `dcode_thread_names` over checkpoint metadata and expose counts, metadata, and turn information on request. Content and tool arguments are bounded by `--max-content` (default 4,000), and reasoning/thinking blocks are omitted from rendered content.

Database deserialization is for **trusted local Deep Agents state only**. Do not deserialize an untrusted database, expose unrelated credentials, tokens, personal data, or hidden reasoning that local records might contain, or mutate/delete records unless the user separately and explicitly requests that action.

## Separation from graph, file, cost, and offload state

DeepAgents `messages` uses a `DeltaChannel` with snapshots every 50 updates. Its reducer depends on LangGraph-assigned stable IDs to deterministically replace, remove, or reset messages during replay. That graph-level mechanism explains why dcode's session browser may reconstruct the message count from writes; it does not make a browser cache a persistence layer.

`StateBackend` is a thread-scoped virtual filesystem over the graph `files` channel. It reads fresh graph state and queues partial updates through LangGraph config keys, giving same-superstep read-your-writes behavior; files persist only if the graph's checkpointer persists them. `StoreBackend`, by contrast, uses an explicit or execution-context `BaseStore` and a validated caller-selected namespace, so it can span threads that deliberately share that namespace. Neither backend is dcode's SQLite session metadata, workspace binding, or Talon archive.

dcode also owns state outside the checkpoint schema. `delete_thread()` deletes checkpoint and write rows under a lease, removes the durable name and separately stored side-question costs, and then best-effort deletes offloaded history. The return value reports checkpoint deletion rather than offload cleanup success. Thus retention, backup, and deletion plans must include these records separately; a checkpoint deletion is not a general garbage collector. For accounting semantics and operator-facing costs, see [Cost and sessions](/openwiki/operations/cost-and-sessions.md).

## Talon is a separate persistence domain

Talon uses an assistant-local home with validated assistant IDs, direct-child state paths, and mode `0700` protections. It retains local model/conversation selections and cron JSON there. Its graph checkpointer is selected independently from its history archive: `DEEPAGENTS_TALON_CHECKPOINT_URI` supports built-in SQLite/file, PostgreSQL, MongoDB, or one plugin; `DEEPAGENTS_TALON_HISTORY_URI` independently selects SQLite, MongoDB, PostgreSQL, or one plugin. The default checkpoint is assistant-local SQLite; remote checkpoint thread IDs are not automatically assistant-namespaced, and changing URIs does not migrate data.

For a configured host, Talon wraps its checkpointer and archive in `ConversationSaver`. Only trusted, root checkpoint writes with Talon chat scope are eligible for archival. It registers the session's scope, persists the graph checkpoint before idempotently appending archive revisions, and then acknowledges archive completion. These are independent stores, not a transaction: archive failure after checkpoint success propagates as a repair condition, and cancellation is shielded until the sequence finishes.

Talon archive metadata, optional vectors, and operational cron records remain distinct. `StoreRecords` provides an in-process lock and redo-journal recovery for bounded idempotent archive mutations, not distributed coordination. The archive binds sessions to a trusted chat scope; vector indexes are separate derived data protected by an embedding fingerprint and can be explicitly rebuilt. Cron JSON uses atomic fsync-and-replace publication but only coordinates live stores in one process. Talon is experimental and is not a production-grade multi-tenant security boundary.

## Operational guidance and focused tests

- Back up, retain, and delete dcode checkpoints/writes, durable thread names, side costs, offload files, DeepAgents `BaseStore` records, and Talon data independently.
- Do not bypass `get_checkpointer()` or the generated ownership-aware checkpointer for a shared local dcode database. Acquire/resume the thread before mutating it, preserve the token through writes, and rotate it before server replacement.
- Treat a workspace switch as a binding/server lifecycle operation, not a move of checkpoint data. Re-register persistent threads with the remote server as needed.
- Use the read-only inspector for trusted local data; treat its reconstruction warnings and output truncation as limits on conclusions.
- Exercise `test_sessions.py` for naming, mixed IDs, indexing, listing, and deletion; `test_app_thread_ownership.py` and ownership-transition tests for reservation behavior; `test_thread_inspector.py` for read-only lookup and reconstruction; and `test_pending_work_recovery.py` for safe abandonment.

See [Code agent architecture](/openwiki/architecture/code-agent.md), [Subagents and skills](/openwiki/concepts/subagents-skills.md), [Cost and sessions](/openwiki/operations/cost-and-sessions.md), [Run a dcode session](/openwiki/workflows/run-dcode-session.md), and [Testing guide](/openwiki/testing/testing-guide.md).
