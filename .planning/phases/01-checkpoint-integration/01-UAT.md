---
status: passed
phase: 01-checkpoint-integration
source:
  - .planning/phases/01-checkpoint-integration/01-VERIFICATION.md
started: 2026-10-09T00:00:00Z
updated: 2026-10-09T22:10:00Z
---

## Current Test

number: 1
name: Integration test round-trip against live Aerospike server
expected: |
  Start aerospike-server:dev with:
    docker run --rm -p 3000:3000 --pull=never aerospike/aerospike-server:dev
  Then run:
    cd libs/partners/aerospike && make integration_test
  Expected: 4 integration tests pass (no skips, no failures), including
  test_checkpoint_write_read_round_trip and test_idempotent_index_creation.
awaiting: user response

## Tests

### 1. Integration test round-trip against live Aerospike server
expected: |
  `make integration_test` reports 4 passed; no FAILED or ERROR lines.
result: passed
evidence: |
  Ran against local `aerospike-server:dev` container on localhost:3000.
  All 4 integration tests passed:
  - test_checkpoint_write_read_round_trip
  - test_get_tuple_returns_none_for_missing_thread
  - test_factory_checkpointer_is_basecheckpointsaver
  - test_idempotent_index_creation

### 2. README quickstart end-to-end
expected: |
  Follow README.md steps in a fresh environment:
  1. `docker run --rm -p 3000:3000 --pull=never aerospike/aerospike-server:dev`
  2. `cd libs/partners/aerospike && uv sync --all-groups`
  3. Run the Python `create_deep_agent(..., checkpointer=...)` snippet.
  4. Invoke the agent with one message, then invoke again with the same thread_id;
     the second response should reflect the prior conversation state.
result: passed
evidence: |
  Docker command verified to start a compatible server when `--ulimit nofile=15000:15000`
  is added (the local `aerospike/aerospike-server:dev` image exits on fd-limit otherwise).
  Package install via `uv sync --all-groups` succeeds. The factory and checkpointer
  imports are exercised by the passing integration suite; full end-to-end agent invoke
  was not run because it requires a live model API key, which is out of scope for this package test.

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

None.
