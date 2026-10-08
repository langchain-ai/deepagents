# Roadmap: deepagents-aerospike Partner Package

## Overview

This roadmap delivers `libs/partners/aerospike/` — a thin partner package that wires Aerospike as a drop-in LangGraph checkpointer and cross-session memory store for deepagents users. Phase 1 builds the package skeleton and full checkpoint integration so agents can persist state to Aerospike immediately. Phase 2 adds the store integration so cross-session memory works end-to-end. Phase 3 produces design documentation, a manually executed validation walkthrough, and benchmarks comparing Aerospike against existing backends.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Checkpoint Integration** - Package skeleton, checkpoint factory, unit and integration tests, README quickstart (2 plans) (completed 2026-10-08)
- [ ] **Phase 2: Store Integration** - Store factory, StoreBackend wiring, store round-trip integration tests
- [ ] **Phase 3: Docs, Validation, and Benchmark** - DESIGN.md, AERO_VALIDATION.md executed, benchmark script and reports

## Phase Details

### Phase 1: Checkpoint Integration

**Goal**: Users can wire Aerospike as a LangGraph checkpointer in a deepagents agent and persist thread state across runs.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PKG-01, PKG-02, PKG-03, CHK-01, CHK-02, CHK-03, CHK-04, TST-01, TST-02, TST-04, DOC-02
**Success Criteria** (what must be TRUE):

  1. User installs the partner package with `uv` and dependencies resolve from the local `langgraph-aerospike` checkout without error.
  2. User calls `create_aerospike_checkpointer(host, port, namespace, ttl)` and passes the result to `create_deep_agent(checkpointer=...)` — the agent starts without error and the client is constructed exactly once.
  3. An agent run persists checkpoint state to Aerospike; a subsequent run on the same thread ID resumes from that saved state.
  4. Unit tests (`tests/unit_tests/`) pass with no network; integration tests (`tests/integration_tests/`) pass against `aerospike-server:dev` with `--pull=never`.
  5. README quickstart contains a copy-pasteable snippet (server one-liner + SDK install + `create_deep_agent` usage) that a new user can follow without reading source code.

**Plans**: 2/2 plans executed

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Package skeleton + create_aerospike_checkpointer factory + unit tests (Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Integration tests (round-trip, CHK-03, CHK-04) + README quickstart (Wave 2)

### Phase 2: Store Integration

**Goal**: Users can wire Aerospike as the cross-session memory store in a deepagents agent and reliably read back files and memory across sessions.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: STR-01, STR-02, STR-03, TST-03
**Success Criteria** (what must be TRUE):

  1. User calls `create_aerospike_store(host, port, namespace, ttl)` and passes the result to `create_deep_agent(store=...)` — the agent starts without error.
  2. Files and memory put by one agent session are retrievable by a subsequent session using the same store instance and namespace.
  3. Store integration tests (put, get, delete, list namespace) all pass against `aerospike-server:dev`.

**Plans**: TBD

### Phase 3: Docs, Validation, and Benchmark

**Goal**: The integration is documented thoroughly, manually validated with pasted command output, and benchmarked against existing backends with generated reports.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DOC-01, VAL-01, VAL-02, BNK-01, BNK-02, BNK-03
**Success Criteria** (what must be TRUE):

  1. `DESIGN.md` explains checkpoint-to-Aerospike key/bin mapping, store-to-Aerospike namespace mapping, TTL semantics, single-client rationale, and dev-stack requirement clearly enough that a reader can predict how a checkpoint maps to records without reading source code.
  2. `AERO_VALIDATION.md` contains pasted `aql`/`asadm` output confirming records appear after agent runs, and the automated section shows lint, type-check, and test results all passing.
  3. The benchmark script runs to completion and produces `reports/REPORT.md`, `reports/REPORT.html`, `reports/REPORT.pdf`, and `reports/histograms/*.png`.
  4. The report contains latency numbers for checkpoint and store ops compared to SQLite, Postgres, MongoDB, and InMemoryStore across small, medium, and large data volumes.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Checkpoint Integration | 2/2 | Complete    | 2026-10-08 |
| 2. Store Integration | 0/TBD | Not started | - |
| 3. Docs, Validation, and Benchmark | 0/TBD | Not started | - |
