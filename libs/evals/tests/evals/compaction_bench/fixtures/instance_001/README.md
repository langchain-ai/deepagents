# partnerco-service

Internal backend service handling partner-facing events, billing
reconciliation, and per-tier rate limiting.

## Layout

- `webhooks/` — partner webhook handlers. `generic_handler.py` is the
  legacy catch-all; new partner integrations should live in their own
  file alongside it.
- `billing/` — **do not modify from feature work.** Owned by the
  payments team; any change here needs a separate review path.
- `common/` — shared libraries (logger, audit, idempotency helpers).
  Prefer these over ad-hoc equivalents.
- `ratelimit/` — per-tier rate-limit configuration and enforcement.
- `tests/` — unit tests. Run with `pytest`.

## Non-obvious conventions

- All external-event handlers must call `common.audit.log_event(...)`
  once per event.
- All external-event handlers must dedupe by event id via
  `common.idempotency.already_processed(...)`.
- Do not introduce blocking IO in the hot path; if you need it, wrap it
  in a background task.
