# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-30

### Added

- `create_aerospike_checkpointer` factory function for wiring Aerospike as a LangGraph checkpoint store.
- Unit tests for factory config validation (host, port, namespace, TTL).
- Integration test skeleton with session-scoped fixture that skips when Aerospike is unreachable.
- README quickstart showing `create_deep_agent(checkpointer=...)` usage.
