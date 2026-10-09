# Why token and cost metrics were missing from the experiment view

Three separate bugs, stacked. Each one hid the next, so fixing the first two
changed nothing visible — worth recording, because the last one fails silently
and would bite anyone adding a large-valued metric.

## 1. The cost data was on a different run

`aevaluate` traces the **target function**. Ours only makes an HTTP call to a
deployed agent and waits, so its run contains no LLM spans at all — and LangSmith
computes cost from token usage on LLM spans. It correctly reported nothing:

| run | `total_tokens` | `total_cost` |
| --- | --- | --- |
| experiment run (`AsyncTarget`) | 0 | `None` |
| deployment run (`browser-agent`) | 56,442 | $0.0246 |

The numbers existed the whole time, on the agent's own trace in the deployment's
project. Nothing was broken; we were reading the wrong run.

**Fix:** the target returns the deployment's `run_id`, and evaluators look that run
up and report its totals as experiment metrics.

## 2. The lookup rate-limited itself

First implementation called `POST /runs/query` from two evaluators (`tokens` and
`cost_usd`), each with six retries, for every example. That is up to 180 requests
for a 15-task run, and the API started returning:

```
{"detail":"Rate limit exceeded."}
```

The evaluators swallowed the failure and returned 0, so every task reported zero
cost rather than an error. Worse, the limit persisted long enough that a *single*
later request still 429'd.

**Fix:** one cached fetch per run id shared by both evaluators, 429-aware backoff,
and `GET /api/v1/runs/{id}` instead of `POST /runs/query` — a single-run read is
much cheaper against the limiter.

## 3. The real blocker: feedback scores are capped

With the lookup working, metrics still did not appear. The experiment log had the
answer:

```
422 Unprocessable entity: invalid feedback part for feedback.01a121ba-...:
score must be between -99999.9999 and 99999.9999 inclusive, was 173711
```

**LangSmith caps a feedback score at ±99,999.9999.** A browsing task routinely
burns 100k–500k tokens, so `tokens` was out of range.

The failure mode is what makes this worth writing down: the rejection is for the
whole **multipart batch**, not the offending score. One over-range `tokens` took
`cost_usd` down with it — a metric that was never out of range and looked, from
the UI, simply absent. There is no error in the experiment view; the metric just
is not there.

**Fix:** report `tokens_k` (thousands) instead of raw tokens. 500k tokens becomes
`500.0`, comfortably in range, and the unit is in the metric name.

## What to take from this

- Check the **ingest log**, not just the UI, when a metric silently fails to appear.
- Keep feedback scores small. Anything that can exceed ~100k — tokens, bytes,
  milliseconds over a long run — needs a scaled unit.
- One bad score in a batch can hide every other metric in that batch, so an
  unrelated missing metric is a symptom worth chasing rather than ignoring.
- Prefer `GET /runs/{id}` over `POST /runs/query` for per-example lookups, and
  cache: evaluators run per example and multiply quickly.
