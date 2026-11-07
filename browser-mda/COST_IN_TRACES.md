# Why cost does not appear in the experiment view

Short version: the cost is recorded, in the deployment's trace. The experiment's
trace is a different trace, and LangSmith prices the experiment's own trace. The
documented mechanism for joining the two does not work.

## How the native Cost/Tokens columns are computed

LangSmith derives them from **LLM spans inside the experiment run's trace tree**.
Evaluate a local chain and this is automatic: `evaluate()` calls your code in
process, the model calls nest under the target run, and the columns populate with
no configuration.

## Why that does not happen for a Managed Deep Agent

An MDA agent exists as a deployment. There is no in-process graph to call, so the
evaluation target makes an HTTP request and polls:

```
experiment run (AsyncTarget)   ──HTTP──▶   deployment run (browser-agent)
  no LLM spans                               every LLM span
  tokens 0 / cost None                       tokens 56,382 / cost $0.0235
  trace A                                    trace B
```

Both are traced. Nothing is lost. They are simply two traces, and the experiment
view reads trace A.

This is not specific to our harness: it follows from evaluating a *deployment*
rather than in-process code.

## What was tried

### 1. Distributed tracing headers — ignored

`RunTree.to_headers()` emits `langsmith-trace` and `baggage`, and the LangGraph
SDK's `get_client(headers=...)` forwards them. The deployment started its own
trace anyway; experiment runs stayed at `tokens=0, cost=None` with unrelated
trace ids.

### 2. `runs.create(langsmith_tracing=...)` — accepted, no effect

The SDK exposes exactly the intended hook:

```python
LangSmithTracing = {"project_name": str, "example_id": str}
```

documented as routing traces to a project or associating them with a dataset
example. Passed on every run.

The inputs were verified valid rather than assumed. Inside an `aevaluate`
target:

```
run_tree present:     True
reference_example_id: d43b384c-b591-411e-a4ab-0dfc54411127
session_name:         rt-probe-11b7abf0
```

Both values were sent. Afterwards, every deployment run still reports:

```
reference_example_id: None
```

and no project named after the experiment exists. The parameter has no
observable effect.

**This is the finding worth reporting.** Not "cost is hard to obtain here", but
"the documented mechanism for linking a deployment run to a dataset example does
not work".

### 3. Custom feedback metrics — works, but wrong shape

Reading the deployment run by id and reporting `tokens_k`/`cost_usd` as feedback
does surface the numbers, as custom columns rather than the native ones.

Two problems. It is a workaround for a platform gap, and it has its own failure
mode: **LangSmith caps a feedback score at ±99,999.9999**, a browsing task burns
100k-500k tokens, and an over-range score 422s the *entire* multipart batch --
silently dropping every other metric in it, with no error in the UI. Reporting
thousands (`tokens_k`) avoids it. See `benchmarks/webvoyager/COST_METRICS.md`.

### 4. Harbor — works, for agents that do not need a sandbox

Harbor runs the agent *inside* the traced trial, which is the condition LangSmith
needs. A completed trial produced native values on the experiment row:

```
webvoyager-apple-processor__UBieYSN
  total_tokens: 22,677
  total_cost:   $0.0259
  feedback:     reward 0.0 · found_answer 1.0 · correct_chip 0.0
```

No custom fields. This is the right answer and it demonstrably works.

It is unusable for a browsing agent today for an unrelated reason: Harbor eval
trials get a disk-only backend, so `execute` is absent and the agent cannot reach
a browser. See `MDA_EVALS_FEEDBACK.md` issue 9.

## Where that leaves cost reporting

Until either gap closes, cost for a sandbox-dependent MDA agent is read from the
deployment's own runs by id -- real numbers from the real trace, reported
alongside the experiment rather than inside it (`benchmarks/webvoyager/report.py`).

Current suite, 15 WebVoyager tasks against the deployed agent:

| metric | value |
| --- | --- |
| success | 12/15 (80.0%) |
| steps | median 12, mean 19.2, max 57 |
| latency | median 73s, mean 127s, max 322s |
| tokens | median 102k, total 3.17M |
| cost | median $0.083/task, total $1.53 |
| cost per success | $0.127 |

## What would fix it

1. Honour `langsmith_tracing` on `runs.create`, so a deployment run can be
   attached to an experiment and dataset example. This is the smallest change and
   it unblocks every "evaluate a deployment" workflow, not just ours.
2. Give Harbor eval trials a sandbox backend with execution, so agents that need
   a shell can be evaluated in the environment they actually run in.

Either one is sufficient. The first is more general; the second is needed anyway
for issue 9.
