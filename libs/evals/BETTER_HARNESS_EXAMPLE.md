# Better Harness Example

This is the shareable end-to-end example for the better-harness workflow:

1. define a representative sample split
2. split it into optimization and holdout
3. hill-climb on the optimization cases
4. check the holdout for non-regression
5. validate the resulting prompt on a separate acceptance eval slice
6. write all prompts, scores, and artifacts to one run directory

The entrypoint is [`scripts/run_better_harness_example.py`](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/scripts/run_better_harness_example.py).

## Requirements

From `libs/evals/`:

```bash
uv sync --group test

export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export ANTHROPIC_API_KEY=...
```

For GLM-5:

```bash
export BASETEN_API_KEY=...
```

## Fast Smoke Run

This runs a small built-in slice that still exercises the full workflow:

- optimization: `tool_indirect_email_report`, `followup_vague_send_report`
- holdout: `tool_direct_slack_dm`
- acceptance: one `tool_selection` test and one `followup_quality` test

```bash
uv run python scripts/run_better_harness_example.py \
  --model claude-sonnet-4-6 \
  --smoke \
  --max-iterations 1 \
  --output-dir artifacts/better-harness-example-sonnet-smoke
```

Validated live on April 7, 2026:

- workflow report: [artifacts/better-harness-example-sonnet-smoke/workflow.md](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/better-harness-example-sonnet-smoke/workflow.md)
- result: baseline `1/2` acceptance, optimized `2/2`
- selected module: `clarify_semantics_without_reasking`

## Full Run

This uses the full built-in representative sample and the focused acceptance slice:

- optimization benchmark: `5` cases
- holdout benchmark: `4` cases
- acceptance: `tests/evals/test_tool_selection.py` + `tests/evals/test_followup_quality.py`

```bash
uv run python scripts/run_better_harness_example.py \
  --model claude-sonnet-4-6 \
  --max-iterations 2 \
  --output-dir artifacts/better-harness-example-sonnet
```

You can also run GLM-5:

```bash
uv run python scripts/run_better_harness_example.py \
  --model baseten:zai-org/GLM-5 \
  --max-iterations 2 \
  --output-dir artifacts/better-harness-example-glm5
```

## What Gets Written

Each run directory contains:

- `split.json` / `split.md`: optimization, holdout, and acceptance definitions
- `optimization/report.json` / `optimization/report.md`: hill-climb results
- `variants/baseline.json`: frozen baseline prompt used for the run
- `variants/optimized.json`: baseline prompt plus the selected modules
- `acceptance/<model>/baseline-*.json`: acceptance results for the baseline prompt
- `acceptance/<model>/optimized-*.json`: acceptance results for the optimized prompt
- `acceptance/comparison.json` / `acceptance/comparison.md`: acceptance summary
- `workflow.json` / `workflow.md`: top-level summary of the whole run

## How It Works

The example does not depend on product-level prompt switching. It freezes a baseline prompt inside [`variants.py`](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/deepagents_evals/better_harness/variants.py), patches `deepagents.graph.BASE_AGENT_PROMPT` in-process for the optimizer benchmark, and uses a small pytest plugin at [`pytest_plugin.py`](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/deepagents_evals/better_harness/pytest_plugin.py) to load baseline and optimized prompt variants inside acceptance eval subprocesses.

That keeps the example reproducible even if the shared Deep Agents prompt changes later.

## Tests

The example is covered by:

```bash
uv run --group test pytest tests/unit_tests/test_better_harness_optimizer.py
uv run --group test pytest tests/unit_tests/test_focused_harness_comparison.py
uv run --group test pytest tests/unit_tests/test_better_harness_example.py
uv run --group test ruff check \
  deepagents_evals/better_harness \
  scripts/run_better_harness_example.py \
  tests/unit_tests/test_better_harness_example.py
```
