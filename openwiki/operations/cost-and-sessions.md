---
type: operations reference
title: dcode Cost, Sessions, and Context Operations
description: Operate dcode's checkpointed cost estimates, session ownership, prompt-cache expiry guidance, and server-side context offload. This page distinguishes durable graph accounting from client-only presentation and explains failure-safe context operations.
tags: [dcode, sessions, cost-tracking, prompt-cache, offload, thread-ownership]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-06T08:06:27.683Z
sources:
  - id: openwiki-source-c0415071c1e2979d2795bd05
    resource: repo://libs/code/deepagents_code/cold_cache.py
  - id: openwiki-source-f2ac9d5fb6c7c6a21f241281
    resource: repo://libs/code/deepagents_code/cost_tracking.py
  - id: openwiki-source-ea1089f0d7536fbc96c64866
    resource: repo://libs/code/deepagents_code/offload_api.py
  - id: openwiki-source-c100a7d2ff8c43af8ad1b816
    resource: repo://libs/code/deepagents_code/offload_middleware.py
  - id: openwiki-source-0f8622164498a685abc913d5
    resource: repo://libs/code/deepagents_code/sessions.py
  - id: openwiki-source-91c9283d1547adfffd627c43
    resource: repo://libs/code/deepagents_code/thread_ownership.py
  - id: openwiki-source-5f08fb59ac37d796df875608
    resource: repo://libs/code/deepagents_code/tui/modals/_cost_breakdown.py
  - id: openwiki-source-f8c8eb69e25f569e0f8a5adb
    resource: repo://libs/code/deepagents_code/tui/modals/cost_breakdown.py
  - id: openwiki-source-2c41bc0b19795204a48854ee
    resource: repo://libs/code/deepagents_code/tui/widgets/status.py
  - id: openwiki-source-11d6c59d85493653aee76558
    resource: repo://libs/code/tests/unit_tests/test_app.py
  - id: openwiki-source-0f0d55280cd10c91f60f7af1
    resource: repo://libs/code/tests/unit_tests/test_cold_cache.py
  - id: openwiki-source-8574be7f7f29e3e1dd328837
    resource: repo://libs/code/tests/unit_tests/test_js_cost_tracking.py
  - id: openwiki-source-bfb9f0ea03fdda310b93ef72
    resource: repo://libs/code/tests/unit_tests/tui/widgets/test_status.py
generated: { by: "openwiki/0.4.2", at: "2026-10-06T08:06:27.683Z" }
---

# dcode Cost, Sessions, and Context Operations

## Operational model

dcode reports **estimated** model cost, not provider invoices, reservations, budgets, or a spending gate. Reconcile charges with the provider: an unpriceable request is omitted from the dollar estimate but is retained as an unpriceable request in structured accounting when usage is available.

The graph checkpoint is the durable accounting authority. The terminal client renders streamed and checkpoint-derived values; it does not write the lifetime total. Side questions are intentionally separate: their recorder can outlive the graph run, so their subtotal is persisted in the sessions database and is merged with graph cost only for client presentation.

```mermaid
flowchart TD
    Call["Completed model call"] --> Recorder["Process-wide recorder"]
    Recorder --> Middleware["Cost tracking middleware"]
    Middleware --> Checkpoint["Private cost channels in checkpoint"]
    Operation["Server offload summarizer"] --> Prepared["Prepared operation cost"]
    Prepared --> Checkpoint
    Checkpoint --> Stream["Absolute session cost event"]
    Stream --> Client["Read-only client display"]
    Side["Side question"] --> SideDB["Sessions SQLite subtotal"]
    SideDB --> Client
```

*Durable graph cost and side-question cost have separate persistence paths; the client combines them only to display a session view.*

`CostTrackingMiddleware` owns `_session_cost_usd` and `_session_cost_breakdown`, both additive private state channels. It drains completed call records after model steps and again when the agent finishes. It returns records to the recorder if pricing processing fails, because accounting must not fail a user turn. Nested agents first persist local cost and then stage an owner-scoped transfer for the parent graph, avoiding a loss when an interrupt stops a child after it has spent.

Pricing is best effort. `estimate_cost` is the sole `genai-prices` integration; unsupported models or malformed usage produce no price rather than failing the invocation. The structured version-one breakdown records request counts, token categories, dollar categories, and completeness flags. The `session_cost` stream event carries an **absolute** thread total and optional absolute breakdown, allowing a client to converge after missed events and discard events for a thread it no longer displays.

### Cost operation invariants

- Do not treat `_session_cost_usd` as an invoice. It covers priceable completed requests only.
- Preserve both the dollar total and structured breakdown on changes. Old checkpoints may contain a dollar total without historical detail; merging malformed or legacy detail marks `historical_complete` false.
- A direct model operation outside normal model middleware, such as server offload, must use `prepare_operation_cost()` and settle it exactly once. Its destructive recorder drain means even a zero-dollar prepared update must be committed with the state update or rolled back.
- For a failed offload checkpoint write, rollback only when a readback proves the checkpoint is unchanged. If it advanced or cannot be read, retain the claimed records to avoid a possible double charge; the result is deliberately indeterminate rather than silently retried.

## Entire-thread breakdown and read-only UI

The status bar opens **Token & Cost Breakdown** only when a left, first click lands on the rendered cost span. The application formats the current `_session_cost_usd` plus `_session_cost_breakdown`; it is a live, read-only view of graph accounting, not a status estimate or side-question subtotal. A second modal is not stacked. While open, it refreshes every 0.5 seconds and retains its prior text if the provider fails.

The entire-thread formatter refuses a table unless it sees a mapping with `version == 1` and `historical_complete is True`. This prevents a dollar-only legacy history from appearing as complete token detail. For valid detail, parent input/output rows are inclusive of cache-creation, cache-read, and reasoning subsets. Invalid fields render as `unavailable`; incomplete categories carry `(partial)`; a positive difference between directional and total cost is reported as directionless/unattributed; and a mismatch between priced and total requests warns that costs are partial.

The modal sanitizes control characters, renders non-markup content, and copies the same current sanitized plain text. `Esc` closes it; a failed clipboard operation produces a warning. These choices make accounting output and any unexpected text safe for terminal presentation rather than an interactive control surface.

## Prompt-cache expiry and cold-cache guidance

Cache expiry is advisory context economics, not durable cost accounting. A checkpointed cache-activity record holds the request time, effective model spec, endpoint identity, and cache-identity parameters. Invalid checkpoint values are discarded rather than interpreted as a warm cache. The cache identity deliberately includes only cache-affecting parameters and, for documented providers, reasoning effort; unrelated settings should not create a false “identity changed” warning.

`ConfigurableModelMiddleware` records effective cache identity around main-model requests. It can add a thread-based OpenAI `prompt_cache_key` and Fireworks session affinity without overwriting caller settings; cross-provider swaps strip Anthropic-only `cache_control` settings. Auxiliary/subagent requests do not overwrite the main conversation's cache identity.

A prompt-cache policy is resolved only for an official provider endpoint or an exact host explicitly listed in `[warnings].trusted_cache_endpoints`. Cross-wire-format routes and unknown or untrusted endpoints resolve to no policy instead of claiming a retention guarantee. The UI can distinguish an expired cache, an identity change, and unknown age; cache retention that is only a provider minimum is presented as “may be cold,” not as proof of expiry.

Operationally, set `[warnings].cache_prompt` to `expiry`, `send`, or `off`. In expiry mode, the client may offer a handoff once for a particular thread and expiry window, subject to the cold-cache threshold and suppression choices. A cache-expiring notification waits for request reconciliation so it does not race a turn that renewed the cache. Treat estimates and expiry indicators as guidance; provider-side cache behavior remains authoritative.

## Session persistence and exclusive ownership

Sessions use SQLite checkpoints. `get_checkpointer()` supplies an ownership-fenced `AsyncSqliteSaver`, not a general unguarded writer. A client reserves a thread using operating-system file locks and a random fencing token stored in an owner file. Each checkpoint mutation enters a per-thread writer gate and validates both the live client reservation and supplied token. Releasing or rotating a lease therefore prevents a stale process from writing after another client takes over.

```mermaid
sequenceDiagram
    participant Client as Client
    participant Lease as Thread lease
    participant Saver as Owned SQLite saver
    participant DB as Sessions SQLite
    Client->>Lease: acquire thread reservation
    Lease-->>Client: fencing token
    Client->>Saver: checkpoint mutation with token
    Saver->>Lease: validate reservation and token
    Lease-->>Saver: writer allowed
    Saver->>DB: persist checkpoint or writes
    Client->>Lease: release or rotate on transition
```

*The saver checks ownership at the mutation boundary, so holding an old configuration is insufficient to write a released or re-owned thread.*

Deleting a thread first acquires the same reservation and refuses with `BlockingIOError` if another client owns it. It deletes the SQLite checkpoint/write rows and side-question cost, then best-effort removes offloaded archive material. Its Boolean return reports checkpoint deletion only, not cleanup success.

## Server-side `/offload` and handoff

The built-in server-side offload path serializes attempts per thread, requires an idle registered thread with no pending graph work, reads a checkpoint, then verifies that checkpoint has not changed before committing. The server reconstructs model context from checkpointed settings rather than trusting client-supplied model fields. Its checkpoint update is allowlisted: offload may write summary and private cost channels but must not write `messages`.

```mermaid
sequenceDiagram
    participant CLI as Client
    participant API as Offload API
    participant Graph as Server offload operation
    participant Ledger as Prepared operation cost
    participant Store as Thread checkpoint
    participant Archive as Conversation archive
    CLI->>API: offload with operation id
    API->>Store: require idle and read checkpoint
    API->>Graph: summarize server-read state
    Graph-->>API: summary update and deferred archive
    API->>Ledger: drain and price summarizer calls
    API->>Store: commit summary and cost update
    API->>Archive: append archive
    API->>Store: link archive path
    API-->>CLI: compacted result
```

*This depicts the normal implementation path: state and operation cost are committed before a deferred archive is appended and linked.*

For ordinary offload, summary state and operation cost are committed before archive append; a failed append leaves the summary reservation in place. The archive link is read back after a failed linking write: a confirmed absent link rolls back the appended archive, while an unreadable result is reported as indeterminate. Cancellation does not abandon a commit halfway through: the API waits for the commit task to settle, then re-raises cancellation.

A cache-expiry handoff differs from compaction: it summarizes for a new thread while the source retains full context. The source receives only the prepared cost channels, and the recovery transcript is saved under a source-owned archive prefix. The resulting archive may be marked ephemeral when persistent offload storage is unavailable.

## Focused verification and safe changes

- Run `tests/unit_tests/test_cost_tracking.py` for recorder draining, price fallbacks, middleware transfers, and prepared operation-cost settlement.
- Run `tests/unit_tests/test_js_cost_tracking.py`, `tests/unit_tests/test_app.py`, and `tests/unit_tests/tui/widgets/test_status.py` when changing breakdown formatting, refresh/copy behavior, or cost-span interaction.
- Run `tests/unit_tests/test_cache_expiry.py`, `tests/unit_tests/test_cold_cache.py`, and `tests/unit_tests/test_configurable_model.py` for cache identity, endpoint trust, expiry prompting, and request instrumentation.
- Run `tests/unit_tests/test_sessions.py`, `tests/unit_tests/test_thread_ownership.py`, and `tests/unit_tests/test_thread_ownership_transitions.py` when changing checkpoint lifecycle, leases, or session transitions.
- Run `tests/unit_tests/test_offload_api.py` and `tests/unit_tests/test_offload.py` for server-side compaction, archive behavior, hooks, cancellation, and settlement.

Related material: [Code agent architecture](../architecture/code-agent.md), [Profiles and models](../concepts/profiles-models.md), [State persistence](../concepts/state-persistence.md), [Testing guide](../testing/testing-guide.md), and [Run a dcode session](../workflows/run-dcode-session.md).
