---
type: testing guide
title: Testing Guide
description: Focused dcode test routing for server lifecycle, runtime compatibility patches, workspace ownership, middleware integrity, and virtualized Textual transcripts. Use package Makefile targets as the command authority and prefer deterministic seams over process or model-backed tests.
tags: [testing, pytest, dcode, textual, middleware, lifecycle]
sources:
  - id: openwiki-source-e6ad64b6a7d1df658edc7234
    resource: repo://libs/code/deepagents_code/_checkpointer_adapter.py
  - id: openwiki-source-b07cec8e891bf1f14e503026
    resource: repo://libs/code/deepagents_code/_langsmith_runtime.py
  - id: openwiki-source-526a6c8ea414f4f02ee9c4a9
    resource: repo://libs/code/deepagents_code/_run_queue_wakeup.py
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-8896ee5af2462c92e51bfb6e
    resource: repo://libs/code/deepagents_code/local_context.py
  - id: openwiki-source-528731d4c0ef5dde5cf54f4c
    resource: repo://libs/code/deepagents_code/memory_guard.py
  - id: openwiki-source-91c9283d1547adfffd627c43
    resource: repo://libs/code/deepagents_code/thread_ownership.py
  - id: openwiki-source-2d3ba1d75dbc72b01448e124
    resource: repo://libs/code/deepagents_code/tui/widgets/message_store.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-36838b2ce6fb7679d6e662cb
    resource: repo://libs/code/tests/integration_tests/test_headless_resume.py
  - id: openwiki-source-6684124c441015e6f9246319
    resource: repo://libs/code/tests/unit_tests/conftest.py
  - id: openwiki-source-7f916fdd682c58125bb1c675
    resource: repo://libs/code/tests/unit_tests/test_checkpointer_adapter.py
  - id: openwiki-source-ceb0295160537174ea07a3f2
    resource: repo://libs/code/tests/unit_tests/test_langsmith_runtime.py
  - id: openwiki-source-86a8d168faa54044d4aa6ec7
    resource: repo://libs/code/tests/unit_tests/test_local_context.py
  - id: openwiki-source-d965874b5a639c3205f2eadd
    resource: repo://libs/code/tests/unit_tests/test_memory_guard.py
  - id: openwiki-source-7e558b3c5b3acbc58f79b60c
    resource: repo://libs/code/tests/unit_tests/test_run_queue_wakeup.py
  - id: openwiki-source-f598809da8d8fbff2d7ae090
    resource: repo://libs/code/tests/unit_tests/test_server_manager.py
  - id: openwiki-source-a5e918d96b1dae3f7adec3f5
    resource: repo://libs/code/tests/unit_tests/test_thread_ownership.py
  - id: openwiki-source-7b9fdc9826d14adc6230d61c
    resource: repo://libs/code/tests/unit_tests/test_transcript_virtualization.py
  - id: openwiki-source-b7beeddb49bcfbe0565494c8
    resource: repo://libs/code/tests/unit_tests/tui/widgets/test_autocomplete.py
  - id: openwiki-source-585649cb24fc70035488ac0f
    resource: repo://libs/code/tests/unit_tests/tui/widgets/test_message_store.py
verified:
  - by: openwiki/0.4.2
    at: 2026-10-25T08:05:05.473Z
generated: { by: "openwiki/0.4.2", at: "2026-10-25T08:05:05.473Z" }
---

# Testing Guide

For dcode changes, test the observable contract at the narrowest deterministic seam. Use a fake backend, temporary SQLite database, controlled environment, patched collaborator, or Textual `run_test` pilot before adding a subprocess test. Reserve process coverage for a boundary that cannot be represented faithfully—such as CLI/server startup or import-time profile state—and reserve model-backed evaluation for an explicit evaluation run. This complements [Development](../operations/development.md), [Run a dcode session](../workflows/run-dcode-session.md), and the architectural [code-agent](../architecture/code-agent.md) and [source map](../architecture/source-map.md) pages.

## Run the package authority

Run commands through `libs/code/Makefile`; do not replace its socket, parallelism, timeout, or coverage policy with ad-hoc pytest invocations. `make test` defaults `TEST_FILE` to `tests/unit_tests/`, runs with xdist, disables network sockets except Unix sockets, and accepts extra pytest arguments through `PYTEST_EXTRA`. `integration_tests` switches the target to `tests/integration_tests/` and applies its 30-second timeout.

```bash
make -C libs/code test TEST_FILE=tests/unit_tests/test_server_manager.py
make -C libs/code test TEST_FILE=tests/unit_tests/test_thread_ownership.py
make -C libs/code test TEST_FILE=tests/unit_tests/test_transcript_virtualization.py
make -C libs/code test TEST_FILE=tests/unit_tests/tui/widgets/test_message_store.py
make -C libs/code test PYTEST_EXTRA='-k wakeup' TEST_FILE=tests/unit_tests/test_run_queue_wakeup.py
make -C libs/code integration_tests TEST_FILE=tests/integration_tests/<focused_test>.py
make -C libs/code lint
```

`coverage` is the dedicated XML-report target; `update-snapshots` is limited to smoke snapshots. The dcode lint target also validates the generated command catalog and process working-directory rules. Unit-test collection establishes a synthetic `DEEPAGENTS_HOME` before imports, snapshots and restores `os.environ`, and clears global dotenv, tracing, and related runtime state so tests do not inherit a developer profile, credentials, or daemon discovery.

## Select the seam

```mermaid
flowchart TD
    Change["Changed observable contract"] --> Boundary{"Can a controlled collaborator prove it"}
    Boundary -->|Yes| Unit["Unit test with fake state or temporary files"]
    Boundary -->|No| Integration["Focused local process or persistence test"]
    Unit --> Make["Package Makefile test target"]
    Integration --> Make
    Make --> Lint["Package Makefile lint target"]
```

*Start at the controlled unit seam; cross a process boundary only when it is the behavior under test.*

- **Server lifecycle and configuration:** patch `ServerProcess`, runtime scaffolding, and `RemoteAgent`; assert configuration transfer, readiness, workspace binding, and cleanup rather than spawning a server.
- **Private upstream compatibility:** seed real lightweight upstream state, install the patch, then assert both the optimized path and the safe fallback when the upstream shape does not qualify.
- **Concurrency and durable state:** use a temporary sessions database and real locks; coordinate with events instead of sleeps. A small helper subprocess is appropriate when the invariant is explicitly cross-process.
- **Middleware:** invoke the sync and async hook with fake tool handlers or backends. Verify the protected output and failure response, not the middleware's incidental helper calls.
- **Textual UI:** use `App.run_test` and its pilot only for actual mount, scroll, focus, or interaction behavior. Keep store arithmetic and autocomplete ranking as ordinary unit tests.

## Server startup, handoff, and cleanup

`start_server_and_get_agent` captures the project context, validates MCP configuration, derives one `ServerConfig`, transfers it to the server environment, creates and scaffolds a temporary runtime, and waits for the `agent` graph. Only after readiness does it construct a `RemoteAgent` and bind the user workspace plus a session-workspace claim and fingerprint. The returned server is then owned by the successful caller; `server_session` is the higher-level lifecycle owner.

```mermaid
sequenceDiagram
    participant Caller
    participant Config as ServerConfig
    participant Runtime as Temporary Runtime
    participant Proc as ServerProcess
    participant Remote as RemoteAgent
    Caller->>Config: derive from project and CLI inputs
    Caller->>Runtime: apply environment and scaffold files
    Caller->>Proc: start and wait for agent graph
    Proc-->>Caller: ready URL
    Caller->>Remote: create and bind workspace claim
    Caller-->>Caller: hand off server on success
    Caller->>Proc: stop if startup does not hand off
```

*Startup hands a server to its caller only after graph readiness and workspace binding; every earlier failure path reaps it.*

Use `test_server_manager.py` for these contracts:

- Round-trip a `ServerConfig` through the environment and test the filesystem allowlist as a fail-closed boundary. An explicit allowlist must include `read_file`; empty, malformed, non-list, and non-string serialized values are invalid.
- Capture configuration passed into the process seam. Relative explicit MCP paths must be normalized from the captured project context, while server-owned MCP/trust policy is absent from a managed workspace claim.
- Test generated runtime contents directly: source checkouts use a local package dependency, the generated graph map exposes only `agent`, and the HTTP app uses custom route authentication.
- Block `wait_for_graph_ready`, cancel the startup task, and schedule a loop callback. Assert that `stop()` runs outside the event loop. On a readiness failure, assert abandoned startup gets a short grace period except when sandbox cleanup requires the full graceful window.

For the real CLI/server/persistence boundary, keep a separate integration test such as headless resume. Give helper and CLI subprocesses a temporary `HOME` and `DEEPAGENTS_HOME` before importing dcode, because profile paths and the sessions database are captured at import time. This prevents a resume test from touching a developer profile.

## Runtime compatibility patches

dcode carries narrowly scoped patches over private upstream APIs. Their tests should be contract tests, not broad mocks: pin the expected upstream shape, show the improvement with realistic state, and prove that version drift degrades safely.

### Checkpointer delta history

The LangGraph API custom checkpointer adapter inherits the base delta-history method, which prevents normal attribute forwarding to an inner saver. `install_delta_history_forwarding()` patches only adapters that still expose the base methods. The replacement delegates to an inner saver only if that saver overrides the base operation; otherwise it preserves the base ancestor walk.

In `test_checkpointer_adapter.py`, seed a `StateGraph` with a `DeltaChannel` into a temporary `AsyncSqliteSaver`. Compare the adapter's original result to the patched result and spy on the SQLite fast path. A companion saver deliberately lacking an override proves the fallback still returns the normal history. Keep the server-only import environment minimal (`REDIS_URI`) rather than starting a server.

### LangSmith and run scheduling

`install_untraced_inline_override()` is process-global and belongs only in the server process. It runs the two known traceable bookkeeping callbacks inline when tracing is off; any other call still uses LangSmith's default thread hop because it may perform I/O. Test a traced and untraced decorated coroutine: untraced success and failure must avoid `run_in_executor`, while tracing still preserves nested run relationships and the executor path.

`install_run_queue_wakeup()` replaces the in-memory runtime's fixed scheduler poll with an event signaled after `Runs.put`, while leaving the upstream claim operation and fallback polling interval intact. It installs idempotently only when `Runs.next` and `Runs.put` have the pinned static-method signatures; an unexpected shape must leave the runtime unmodified. Reset both methods and the wakeup object per test, then use an event barrier to show a pending insert wakes a waiting `Runs.next` without a timing sleep.

## Workspace ownership and checkpoint writers

Thread ownership protects a locally persisted thread from independent clients and stale server processes. Acquiring a thread produces a lease token; mutations carry that token in `x-dcode-thread-owner`. Before a checkpoint write, `writer_guard` serializes writers through local and filesystem gates and verifies both the recorded token and the client reservation. A released, crashed, or superseded client therefore fences its old server from writing.

```mermaid
sequenceDiagram
    participant Client as Client Lease
    participant Guard as Writer Guard
    participant DB as SQLite Saver
    Client->>Client: acquire thread and token
    Client->>Guard: mutation with token
    Guard->>Guard: take writer gate and validate lease
    Guard->>DB: checkpoint write or delete
    DB-->>Guard: result
    Guard-->>Client: release gate
```

*The writer gate makes ownership validation and the database mutation one fenced operation.*

Use `test_thread_ownership.py` with a temporary database to cover failure ordering that a mock cannot establish:

- A helper subprocess cannot acquire an already leased thread, but another thread remains available; releasing the lease permits a replacement owner.
- A client crash, ownership rotation, or failed restart invalidates an old token. The old server must reject writes, while the successor token can write.
- The ownership-fenced saver rejects tokenless or stale `aput` and `aput_writes`, and forbids deletion while another client owns the thread. Reads remain possible and deletion succeeds once the lease is released.
- Cancellation is safety-critical: cancellation during acquisition or deletion must wait for the blocking operation, release reservation/gates afterward, and never expose a half-finished database mutation. Concurrent local writers queue through the local async gate rather than polling each other.

## Local context and managed memory

`LocalContextMiddleware` gathers project context through the active backend, so the same detection runs in a local shell or remote sandbox. Its local-context snapshot, fingerprint, and refresh cutoff are private graph state: they do not become public output or trace state. When summarization reaches a valid cutoff, a changed snapshot is appended as a tagged message rather than replacing the saved initial snapshot. A forked subagent on the same thread reuses the parent snapshot; a different thread detects its own context.

Test both `before_agent` and `abefore_agent` with concrete sync or async fake backends. Cover malformed/out-of-range summarization cutoffs, deduplication by snapshot fingerprint, and escaping of context embedded in message content. A dotenv test must demonstrate that project credentials can load while `BASH_ENV` cannot influence backend startup detection.

`ManagedMemoryGuardMiddleware` protects marker-delimited onboarding-name blocks in configured memory files. It allows unrelated edits, but if `edit_file` or `write_file` changes or removes a managed block, it restores that block while retaining unrelated edits and returns a tool error. Deletes of a guarded file or a parent directory are rejected before the tool runs when a managed block exists; an existing unreadable file also fails closed. Test the synchronous and asynchronous wrappers, partial-marker removal, preserving unrelated edits, and the no-follow symlink restore hardening where `os.O_NOFOLLOW` is available.

## Virtualized transcripts and completion

The message store retains lightweight `MessageData` for the full transcript while mounting only a visible window of Textual widgets. Archived rows are represented by spacers whose height comes from measured or default height hints. Streaming and live-tool rows are protected against pruning, with independent protection reasons so releasing one reason cannot accidentally unprotect another. On resume, `bulk_load` mounts a bounded tail rather than all historic widgets.

Transcript virtualization must be tested at two seams:

1. In `tui/widgets/test_message_store.py`, test store ranges, hydration thresholds, edge hydration, height-hint clamping and scaling, bounded tail loading, and protection invariants with `MessageData` only.
2. In `test_transcript_virtualization.py`, run a real `DeepAgentsApp` pilot. Assert wheel/trackpad, keyboard, and scrollbar movement produce `_ChatScroll.Scrolled` when the vertical offset changes; then prune a small window and prove scrolling toward either spacer remounts the archived boundary widgets. Also assert spacers occupy the archived row height, so the view can reach a history edge instead of stalling.

For autocomplete, keep fuzzy ranking and controller behavior outside a Textual pilot. Compare optimized fuzzy search against a simple reference scorer over a fixed corpus, and test controller handling with a mocked completion view. Treat project filenames as untrusted display/input data: unsafe control or deceptive names must not enter suggestions or the chat input. Thread completion is a separate `@@` grammar that searches durable thread metadata and inserts the canonical `@@(thread:<uuid>)` token.

## Completion checklist

1. Identify the observable contract and choose a controlled unit seam first.
2. Run the focused `libs/code` Makefile target; use `PYTEST_EXTRA` only to add pytest selection or diagnostics.
3. Own environment, temporary files, database rows, global patch state, and cleanup in the test fixture.
4. For private-API patches, assert both the qualifying fast path and an upstream-drift fallback.
5. For cancellation or ownership behavior, coordinate with events and assert cleanup/fencing after cancellation rather than relying on sleeps.
6. Escalate to an integration test only for a real process, import-time configuration, durable persistence, or CLI/server contract.
7. Run `make -C libs/code lint` before completion.
