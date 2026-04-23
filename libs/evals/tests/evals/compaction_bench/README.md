# Compaction benchmark (large-scale)

End-to-end evaluation of conversation-summarization / compaction techniques
under realistic long-session conditions. The agent works through an
incremental multi-feature development task in a simulated backend service.
Compaction fires 2–3 times per run, and the task produces per-failure-mode
scores rather than a single scalar.

This is the **simplified v1**. See the design-doc chat history for the
broader vision; the cuts are documented below.

## What this benchmark measures

Six failure-mode categories, scored independently:

| Category | What it catches |
|----------|-----------------|
| `goal_drift` | Long-introduced user constraints being violated in later features |
| `decision_history` | Already-rejected alternatives being re-explored after compaction |
| `artifact_continuity` | Same canonical files being edited across features, not re-created |
| `tool_state` | Repo-layout knowledge lost across compactions (re-reads, re-greps) |
| `direct_recall` | LLM-judge-graded recall in the final review/NOTES.md turn |
| `overall_correctness` | Does the test suite still pass at the end |

A single `weighted_total` is also reported for convenience. It is **not**
the primary comparison surface.

## Scope — v1 simplifications

What the broader design doc specified vs. what ships in v1:

| Dimension | Full design | v1 |
|-----------|-------------|----|
| Configs | 4 (`none`, `default`, `aggressive`, `golden_injected`) | **2**: `deepagents`, `openai_compact`, both under a v1-tuned 15k aggressive trigger (see note below) |
| Baseline `none` | Every run | One-time validation gate on task-spec changes |
| Techniques | 8 | **2**, with a protocol that makes more trivial to add |
| Instances | 20 templated | **3 hand-authored**, no templating yet |
| Seeds | 3 per cell | **1**, bump on high-variance checkpoints |
| Features per instance | 4 (A, B, C, D refactor) | **3** (A, B, C) + review (Part E) |
| Constraints | 11 | **7** (C1–C6, plus C7/C8 merged, plus C9/C10 merged) |
| Rejections | 3 | **2** (A1 extend-generic, A2 new-dep) |
| Checkpoints | 28 | **~15** |
| Token target | 175k | **~20-40k total transcript** (enough to fire the tuned 15k trigger 1-2 times) |
| Harbor sandbox | Required | **Not used**; `FilesystemBackend` on `tmp_path` |
| Goldens | Authored for every instance | **Deferred to v2** |
| CI cadence | Weekly/monthly/quarterly | **Manual dispatch only** in v1 |

Everything that was cut is additive later. Nothing is load-bearing on the
five "non-negotiables" called out in the design doc.

## Directory layout

```
compaction_bench/
├── README.md                         # This document
├── task_spec.py                      # Enums, Checkpoint/Instance dataclasses
├── scorecard.py                      # Result schema + per-category aggregation
├── graders.py                        # All graders (deterministic + trajectory + judge)
├── techniques.py                     # SummarizationTechnique Protocol + adapters
├── instance_001_partnerco.py         # First hand-authored instance
└── fixtures/
    └── instance_001/                 # Mini-repo that instance_001 operates on
```

Unit tests for graders live in
`libs/evals/tests/unit_tests/test_compaction_bench_graders.py` so they
run under `make test` with no network or LangSmith requirement.

## Task shape

Each instance is one continuous agent session. The user drives the session
through ~20 user-facing turns (the agent's internal tool-calling steps
bring the total to ~100+). The phases:

1. **Feature A — new webhook handler.** Introduces constraints C1–C6
   (module off-limits, no new deps, idempotency, logger, audit, latency).
   Rejects branch A1 (extending a generic handler) because it would pull
   in a new dependency (violating C2).
2. **Feature B — retry logic.** Introduces constraints C7+C8 (exponential
   backoff, max N attempts). Rejects branch A2 (using an external retry
   library) because it violates C2.
3. **Feature C — per-tier rate limiting.** Introduces constraints C9+C10
   (3 tiers with per-partner overrides). Tests that C1 (module off-limits)
   is still remembered — a rate-limiter patch in that module would
   satisfy the feature request but violate C1.
4. **Part E — review.** The user asks the agent to write `NOTES.md`
   summarizing all the work, the alternatives it rejected and why, and
   which feature introduced which constraint. This is the direct-recall
   test.

### v1 token-budget tuning

The design doc called for an 80k aggressive trigger, which assumed a
production-sized mini-repo (~150 files). The v1 fixture is small (~16
files, ~3.5k tokens) to keep the spin-up quick, so 80k is unreachable
in practice. The trigger is tuned down to **15_000 tokens** for v1;
compaction reliably fires once or twice across the scripted 18 turns
(typically mid-Feature-C and around Part E), which is enough signal
to measure goal-drift, decision-history, and artifact-continuity
across a real summarization event.

The preflight test
`libs/evals/tests/unit_tests/test_compaction_bench_token_budget.py`
catches accidental regressions (fixture trimmed too aggressively,
trigger typo, etc.) before they degrade the eval silently.

v2 expands the fixture toward production scale and restores the 80k
trigger for 2-3 compaction events per run.

## Adding a new instance (v1 pattern)

1. Create `instance_002_<domain>.py` alongside `instance_001_partnerco.py`.
2. Create `fixtures/instance_002/` with the mini-repo this instance
   operates on. Keep the file count (~10–15) and shape similar to
   instance_001 so graders stay simple.
3. Populate the same `Instance(...)` structure: phase-tagged user
   messages, introduced-constraints metadata, rejection-turn metadata.
4. Re-run the grader unit tests to confirm nothing instance-specific
   leaked into shared code.

Templating comes in v2, once 3 instances make it clear what's
template-worthy and what's instance-specific.

## Adding a new technique

See `techniques.py`. Implement the `SummarizationTechnique` Protocol,
register the technique in the top-level module dict, and the runner
will pick it up automatically.

## Open items carried over from the full design

Resolved in v2, not v1:

- Golden-summary injection and the `golden_injected` config.
- Templating / YAML instance generation.
- Full LangSmith dataset publishing and CI cron.
- Judge validation at the 75-sample / kappa ≥ 0.8 level (v1 uses a
  smaller validation set; see `graders.py` for the pinned judge model
  and prompts).
- Cross-instance normalization against a per-instance `none` baseline.
