# `libs/evals` agent guide

Quick reference for agents (and humans) running the Deep Agents eval suite.
The canonical interface is the `deepagents-evals` console script, installed with this package. The `Makefile` targets remain available for parity with CI.

## Canonical entry point

```sh
deepagents-evals --help
deepagents-evals <subcommand> --help
```

Subcommands:

| Subcommand     | Purpose                                                           |
| -------------- | ----------------------------------------------------------------- |
| `run`          | Run the eval suite once (single trial).                           |
| `trials`       | Run the eval suite N times and aggregate metrics.                 |
| `aggregate`    | Aggregate previously-written trial reports.                       |
| `radar`        | Generate a radar chart from results.                              |
| `catalog`      | Regenerate or check `EVAL_CATALOG.md`.                            |
| `model-groups` | Regenerate or check `MODEL_GROUPS.md`.                            |
| `list`         | Discover categories / tiers / models / evals.                     |

Most subcommands accept:

- `--json` — emit machine-readable JSON on stdout.
- `--dry-run` — print the underlying invocation without executing.

## Discovery

Before kicking off a run, ask the CLI what's available — no source-grepping required:

```sh
deepagents-evals list categories                  # eval categories
deepagents-evals list tiers                       # e.g. baseline | hillclimb
deepagents-evals list models --json               # full eval-tagged registry
deepagents-evals list models --group set0         # one preset
deepagents-evals list models --provider anthropic # one provider
deepagents-evals list evals --category memory     # eval functions in a category
```

## Common workflows

```sh
# Single trial against one model.
deepagents-evals run --model claude-opus-4-7

# Restrict to a category and tier, and write a JSON report.
deepagents-evals run \
    --model openai:gpt-5.5 \
    --eval-category memory \
    --eval-tier baseline \
    --report evals_report.json

# Three trials with stats aggregation.
deepagents-evals trials --model openai:gpt-5.5 --trials 3

# Re-run only the failures from a prior trial sweep.
deepagents-evals trials \
    --model openai:gpt-5.5 \
    --trials 1 \
    --retry-failed trial_runs/trials_summary.json

# Aggregate CI artifacts after a fan-out workflow.
deepagents-evals aggregate ./downloaded-artifacts --summary-out summary.json
```

## Default model env var

Set `DEEPAGENTS_EVALS_MODEL` once and omit `--model`:

```sh
export DEEPAGENTS_EVALS_MODEL=claude-sonnet-4-6
deepagents-evals run
deepagents-evals trials --trials 3
```

`scripts/run_trials.py` honors the same env var when invoked directly,
and supports its own `--json` flag for compact stdout output.

## Exit codes

| Code | Meaning                                                                                                                                                                                                                          |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`  | Success.                                                                                                                                                                                                                         |
| `1`  | Eval failures. `run` saw a non-zero `pytest` exit; `trials` / `aggregate` produced a summary whose aggregated `counts.failed.mean` is greater than zero; `radar` failed.                                                         |
| `2`  | Configuration error: missing `--model`, model-registry import failed, or a `--check` drift detector (`catalog --check`, `model-groups --check`) found that a generated file is stale. `argparse` usage errors also exit `2`.     |
| `3`  | No usable reports: `trials` / `aggregate` produced no summary, or `--retry-failed` could not parse any prior reports.                                                                                                            |

Use these codes to drive automation; do not parse human-readable output.

The `pytest_reporter` plugin rewrites the per-trial pytest exit status to `0` even when individual evals fail (so a CI shell step doesn't fail the workflow). The CLI therefore reads `trials_summary.json`'s aggregated `counts.failed.mean` to decide whether to return `1`, not the per-trial `pytest_returncode` field.

## Required environment

The eval suite refuses to start without LangSmith tracing enabled:

```sh
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```

Provider keys (any of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) are required to match the chosen `--model`.

## `trials_summary.json` schema

`deepagents-evals trials` and `deepagents-evals aggregate` write a summary
file with this shape:

```jsonc
{
  "n_trials": 3,
  "model": "openai:gpt-5.5",
  "sdk_version": "0.5.7",
  "metrics": {
    "correctness":       {"n": 3, "mean": 0.84, "median": 0.85, "stdev": 0.02, "min": 0.82, "max": 0.86},
    "solve_rate":        {"n": 3, "mean": 0.71, "median": 0.70, "stdev": 0.03, "min": 0.68, "max": 0.74},
    "step_ratio":        {"n": 3, "mean": 1.10, "median": 1.10, "stdev": 0.01, "min": 1.09, "max": 1.11},
    "tool_call_ratio":   {"n": 3, "mean": 1.05, "median": 1.05, "stdev": 0.01, "min": 1.04, "max": 1.06},
    "median_duration_s": {"n": 3, "mean": 4.30, "median": 4.31, "stdev": 0.05, "min": 4.25, "max": 4.34}
  },
  "counts": {
    "passed":  {"n": 3, "mean": 17.0, "median": 17, "stdev": 0.0, "min": 17, "max": 17},
    "failed":  {"n": 3, "mean":  3.0, "median":  3, "stdev": 0.0, "min":  3, "max":  3},
    "skipped": {"n": 3, "mean":  0.0, "median":  0, "stdev": 0.0, "min":  0, "max":  0},
    "total":   {"n": 3, "mean": 20.0, "median": 20, "stdev": 0.0, "min": 20, "max": 20}
  },
  "category_scores": {
    "memory":          {"n": 3, "mean": 0.83, "median": 0.83, "stdev": 0.0, "min": 0.83, "max": 0.83},
    "tool_use":        {"n": 3, "mean": 0.90, "median": 0.90, "stdev": 0.0, "min": 0.90, "max": 0.90},
    "file_operations": {"n": 3, "mean": 0.78, "median": 0.78, "stdev": 0.0, "min": 0.78, "max": 0.78}
  },
  "trials": [
    {
      "trial_index": 1,
      "created_at": "2026-05-06T14:23:11+00:00",
      "passed": 17, "failed": 3, "skipped": 0, "total": 20,
      "correctness": 0.85,
      "solve_rate": 0.70,
      "step_ratio": 1.10,
      "tool_call_ratio": 1.05,
      "median_duration_s": 4.31,
      "category_scores": {"memory": 0.83, "tool_use": 0.90, "file_operations": 0.78},
      "experiment_urls": ["https://smith.langchain.com/..."],
      "pytest_returncode": 0
    }
  ]
}
```

Notes on the per-trial entries:

- `pytest_returncode` is populated by the trial runner only on the
  live-execution path. It is **not** written by `pytest_reporter`, so it
  may be missing from individual `evals_report_trial_NNN.json` files and
  from summaries produced via `--aggregate-only`.
- `pytest_reporter` rewrites pytest's session exit status to `0` even when
  tests fail, so `pytest_returncode` is not a reliable failure signal —
  use `counts.failed.mean` instead.

Per-trial `evals_report_trial_NNN.json` files written by `pytest_reporter` contain the metrics shown above and additionally carry a `failures` array used by `--retry-failed`:

```jsonc
{
  "failures": [
    {
      "test_name": "tests/evals/test_memory.py::test_memory_recall[claude-sonnet-4-6]",
      "category": "memory",
      "failure_message": "AssertionError: ..."
    }
  ]
}
```

## Relationship to the `Makefile`

`make evals MODEL=...` and `make evals-trials MODEL=... TRIALS=...` still work and remain the form CI invokes. The console script is a strict superset — every flag the Makefile passes through to pytest is exposed as a first-class option on `deepagents-evals run` / `trials`, plus the discovery and JSON-output features the Makefile cannot offer.
