# Requirements: deepagents-aerospike Partner Package

**Defined:** 2026-09-30
**Core Value:** Deepagents agents can opt into Aerospike for sub-millisecond state reads, per-record TTL expiry, and durable cross-session memory — through existing LangGraph extension points with no core changes.

## v1 Requirements

### Package Infrastructure

- [x] **PKG-01**: Partner package skeleton `libs/partners/aerospike/` matches `libs/partners/quickjs` conventions (hatchling build backend, uv.lock, Makefile with test/lint/format/bench targets, ruff config, pytest config in pyproject.toml, CHANGELOG.md)
- [x] **PKG-02**: Package declared dependencies on editable-installed `langgraph-checkpoint-aerospike` and `langgraph-store-aerospike` from `/home/ubuntu/git/langgraph-aerospike` local checkout
- [x] **PKG-03**: Dev dependency on `aerospike-sdk` from `aie/vector` branch (editable install) documented in quickstart and declared in dev extras

### Checkpoint Integration

- [x] **CHK-01**: `create_aerospike_checkpointer(host, port, namespace, ttl)` factory constructs and returns `AerospikeSaver` with a single client instance
- [x] **CHK-02**: Client is constructed once at factory call; same instance reused across all checkpoint operations (no per-request construction)
- [x] **CHK-03**: Factory integrates cleanly as `create_deep_agent(checkpointer=create_aerospike_checkpointer(...))`
- [x] **CHK-04**: Secondary indexes created idempotently on connect (verify `AerospikeSaver` behavior and document in DESIGN.md)

### Store Integration

- [ ] **STR-01**: `create_aerospike_store(host, port, namespace, ttl)` factory constructs and returns `AerospikeStore` instance
- [ ] **STR-02**: `AerospikeStore` instance passes directly to `StoreBackend(store=...)` for deepagents file/memory operations
- [ ] **STR-03**: Factory integrates cleanly as `create_deep_agent(store=create_aerospike_store(...))`

### Testing

- [x] **TST-01**: Unit tests (no network) for factory config validation and namespace validation in `tests/unit_tests/`
- [x] **TST-02**: Integration tests using real `aerospike-server:dev` (`--pull=never`) for checkpointer round-trip (write checkpoint, read it back, resume thread)
- [ ] **TST-03**: Integration tests for store round-trip (put file, get file, delete file, list namespace)
- [x] **TST-04**: Tests separated into `unit_tests/` and `integration_tests/` directories with skip-if-unreachable fixture for integration suite

### Documentation

- [ ] **DOC-01**: `DESIGN.md` — checkpoint-to-Aerospike key/bin mapping, store-to-Aerospike namespace mapping, TTL semantics, dev-stack requirement, single-client rationale
- [x] **DOC-02**: `README.md` quickstart with `aerospike-server:dev` one-liner (with `--pull=never`), `aie/vector` SDK install steps, and minimal `create_deep_agent` usage snippet for both checkpointer and store

### Validation and Benchmark

- [ ] **VAL-01**: `AERO_VALIDATION.md` written with manual walkthrough (write via deepagents, inspect in `aql`/`asadm` with pasted output) and automated pass (lint + type-check + tests)
- [ ] **VAL-02**: Every command in `AERO_VALIDATION.md` run and output pasted before reporting done
- [ ] **BNK-01**: `benchmark/` script timing checkpoint ops against SQLite and Postgres, and store ops against MongoDB and InMemoryStore, across three data-volume scales (small/medium/large)
- [ ] **BNK-02**: Benchmark generates `reports/` (REPORT.md, REPORT.html styled, REPORT.pdf landscape, `reports/histograms/*.png`) from a single Markdown render
- [ ] **BNK-03**: Benchmark run and all three report formats opened and verified before reporting done

## v2 Requirements

### Vector Search

- **VEC-01**: `AerospikeStore` vector search (`store.search(query=...)`) wired into deepagents memory retrieval for semantic similarity lookup
- **VEC-02**: Design doc section covering brute-force Top-K cost model and dev-only stack dependency

### Additional Backends

- **ADD-01**: Explore wiring Aerospike as a `ContextHub` backend for prompt injection context
- **ADD-02**: Explore `SandboxBackend`-equivalent using Aerospike for sandboxed file operations

## Out of Scope

| Feature | Reason |
|---------|--------|
| Aerospike Graph | All access patterns are KV; no multi-hop traversal needed |
| Changes to `libs/deepagents/` or `libs/code/` core | Companion package only; core must remain unmodified |
| Publishing to PyPI | Dev SDK (`aie/vector` branch) cannot be a published transitive dep |
| Mocking Aerospike wire protocol in integration tests | Real `aerospike-server:dev` only; mocks go in unit tests |
| Memory middleware reimplementation | `StoreBackend` already adapts `BaseStore`; middleware unchanged |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | Phase 1 | Complete |
| PKG-02 | Phase 1 | Complete |
| PKG-03 | Phase 1 | Complete |
| CHK-01 | Phase 1 | Complete |
| CHK-02 | Phase 1 | Complete |
| CHK-03 | Phase 1 | Complete |
| CHK-04 | Phase 1 | Complete |
| STR-01 | Phase 2 | Pending |
| STR-02 | Phase 2 | Pending |
| STR-03 | Phase 2 | Pending |
| TST-01 | Phase 1 | Complete |
| TST-02 | Phase 1 | Complete |
| TST-03 | Phase 2 | Pending |
| TST-04 | Phase 1 | Complete |
| DOC-01 | Phase 3 | Pending |
| DOC-02 | Phase 1 | Complete |
| VAL-01 | Phase 3 | Pending |
| VAL-02 | Phase 3 | Pending |
| BNK-01 | Phase 3 | Pending |
| BNK-02 | Phase 3 | Pending |
| BNK-03 | Phase 3 | Pending |

**Coverage:**

- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-30*
*Last updated: 2026-09-30 after initial definition*
