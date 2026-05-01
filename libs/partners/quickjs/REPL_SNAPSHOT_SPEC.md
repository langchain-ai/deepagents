# REPL Snapshot Spec (Between-Turn State Restore)

## Summary

Add explicit REPL snapshotting to `langchain-quickjs` so JS state can persist **between agent turns** without keeping QuickJS contexts alive indefinitely.

Lifecycle:

1. `after_agent`: create a QuickJS snapshot from the current thread REPL and store it in middleware private state.
2. `before_agent`: if a snapshot is present, create a fresh REPL slot and restore snapshot state into it.
3. Keep current slot-eviction behavior after each turn to avoid long-lived thread/Runtime accumulation.

This uses the snapshot API introduced in `quickjs-rs` PR #21.

## Current Behavior (As Of This Branch)

- REPL state persists during a single agent execution.
- `REPLMiddleware.after_agent` / `aafter_agent` evict the thread slot (`_registry.evict(...)`), so state is lost after each turn.
- System prompt explicitly says state does not persist across turns.

Relevant files:

- `libs/partners/quickjs/langchain_quickjs/middleware.py`
- `libs/partners/quickjs/langchain_quickjs/_repl.py`
- `libs/partners/quickjs/langchain_quickjs/_prompt.py`
- `libs/partners/quickjs/tests/unit_tests/test_repl_middleware.py`
- `libs/partners/quickjs/tests/unit_tests/smoke_tests/snapshots/*.md`

## quickjs-rs API Shape (From PR #21)

Public Python surface:

- `Context.create_snapshot(...) -> Snapshot`
- `Context.create_snapshot_async(...) -> Snapshot`
- `Runtime.restore_snapshot(snapshot, ctx, inject_globals: bool = True) -> None`
- `Snapshot.to_bytes() -> bytes`
- `Snapshot.from_bytes(data: bytes | bytearray | memoryview) -> Snapshot`

Snapshot options:

- `on_unserializable`: `"tombstone" | "error"` (default `"tombstone"`)
- `on_missing_name`: `"skip" | "tombstone" | "error"` (default `"skip"`)

Important constraints enforced by `quickjs-rs`:

- snapshot creation fails if async eval is in flight,
- snapshot creation fails if async host tasks are pending,
- snapshot creation fails for module-touched contexts in V1,
- `restore_snapshot(..., inject_globals=True)` is required to materialize values into globals.

## Proposed Design

### 1) Add middleware private snapshot state

Define a middleware-specific state schema:

- private key name: `_quickjs_snapshot_payload`
- value: serialized snapshot payload

Proposed type:

- `NotRequired[Annotated[bytes, PrivateStateAttr]]`

### 2) `before_agent` / `abefore_agent`: restore snapshot

Hook behavior:

- resolve thread id via `_resolve_thread_id(self._fallback_thread_id)`;
- read `_quickjs_snapshot_payload` from `state`;
- if absent: no-op;
- if present:
  - get/create REPL slot via registry,
  - call `restore_snapshot` into that slot’s fresh context,
  - ensure `inject_globals=True`.

Failure policy (proposed):

- log warning,
- clear `_quickjs_snapshot_payload` in returned state update to avoid repeated restore failures,
- continue with fresh REPL (fail-open).

### 3) `after_agent` / `aafter_agent`: snapshot then evict

Hook behavior:

- resolve thread id;
- if no slot exists for this thread: no-op (preserve existing snapshot state);
- if slot exists:
  - create snapshot from context (`create_snapshot` / `create_snapshot_async`),
  - serialize to bytes (`Snapshot.to_bytes()`),
  - return state update with `_quickjs_snapshot_payload`,
  - evict slot (close runtime/worker) as today.

Failure policy (proposed):

- log warning,
- clear `_quickjs_snapshot_payload` (avoid restoring stale snapshot from older turns),
- still evict slot.

### 4) `_Registry` / `_ThreadREPL` additions

Add minimal APIs to avoid creating accidental empty slots in `after_agent`:

- `_Registry.get_if_exists(thread_id) -> _ThreadREPL | None`
- or `_Registry.take_slot(thread_id) -> _Slot | None`

Add REPL helpers:

- `_ThreadREPL.create_snapshot(...) -> bytes`
- `_ThreadREPL.acreate_snapshot(...) -> bytes` (if async path needed)
- `_ThreadREPL.restore_snapshot(payload: bytes, *, inject_globals: bool = True) -> None`

Implementation detail:

- snapshot/restore calls must run on the slot’s worker thread, same as existing QuickJS context operations.

### 5) System prompt update

Update the REPL prompt text in `_prompt.py`:

- remove/replace “DO NOT persist across multiple turns”
- replace with explicit between-turn persistence statement by default.
- include a conditional note in prompt rendering when persistence is disabled via flag.

## API / Compatibility

### Behavior change

This remains behind a constructor flag, but defaults to on:

- `snapshot_between_turns: bool = True`

### Public API signatures

No required breaking signature changes to existing middleware methods.

Constructor addition:

- `snapshot_between_turns: bool = True`

### quickjs-rs dependency strategy

Use a git-backed `quickjs-rs` source pointing to the in-progress snapshot branch:

- repo: `https://github.com/langchain-ai/quickjs-rs.git`
- branch: `hunter/snapshot`

Planned local-dev source entry in `libs/partners/quickjs/pyproject.toml`:

```toml
[tool.uv.sources]
deepagents = { path = "../../deepagents", editable = true }
quickjs-rs = { git = "https://github.com/langchain-ai/quickjs-rs.git", branch = "hunter/snapshot" }
```

Follow-up after branch stabilization:

- replace branch tracking with a fixed `rev` (commit pin) for reproducibility.

## Test Plan

### Unit tests (`test_repl_middleware.py`)

Add/update tests for:

- `before_agent` restores snapshot payload into fresh slot.
- `after_agent` saves snapshot payload and evicts slot.
- `aafter_agent` async variant.
- restore failure clears payload and continues.
- snapshot failure clears payload and evicts.

Update existing eviction-only tests to new semantics.

### End-to-end tests

Add a test with same `thread_id` across two agent runs:

- turn 1 sets `globalThis.counter = 1`;
- turn 2 reads/increments `counter`;
- assert persisted value across turns.

Add isolation test:

- thread A and thread B maintain independent restored snapshots.

### Smoke snapshot tests

Update prompt snapshots under:

- `libs/partners/quickjs/tests/unit_tests/smoke_tests/snapshots/`

### Optional regression tests

- PTC tool bridges still work after restore.
- skills import still works after restore/evict cycle.

## Decisions (Locked)

1. Keep between-turn persistence behind a flag with default-on: `snapshot_between_turns=True`.
2. Store `_quickjs_snapshot_payload` as raw `bytes` initially.
3. Keep quickjs-rs snapshot defaults (`on_missing_name="skip"`, `on_unserializable="tombstone"`).
4. On snapshot or restore failure, clear stored payload.

## Implementation Checklist

1. Add middleware state schema with private snapshot field.
2. Add `snapshot_between_turns: bool = True` constructor flag.
3. Implement `before_agent`/`abefore_agent` restore flow.
4. Extend `_Registry` + `_ThreadREPL` with snapshot helpers.
5. Implement `after_agent`/`aafter_agent` snapshot-save + eviction.
6. Update system prompt text and smoke snapshots.
7. Add/adjust unit + e2e tests for lifecycle and isolation.
8. Update `libs/partners/quickjs/pyproject.toml` to use git-backed `quickjs-rs` (`hunter/snapshot`).
9. Document behavior in `libs/partners/quickjs/README.md`.
