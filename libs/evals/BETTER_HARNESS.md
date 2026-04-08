# Better Harness

This repo now includes a small eval-driven harness-tuning workflow for Deep Agents, plus concrete harness improvements validated on both Anthropic Sonnet 4.6 and Baseten GLM-5.

If you want the simplest shareable end-to-end example, start with [`BETTER_HARNESS_EXAMPLE.md`](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/BETTER_HARNESS_EXAMPLE.md).

## What Changed

- Added an experimental better-harness runner in `deepagents_evals/better_harness/`.
- Added a CLI entry point at `scripts/run_better_harness.py`.
- Added a small benchmark with optimization and holdout splits rooted in the existing eval behaviors.
- Added prompt-module search primitives and unit tests for the hill-climber.
- Updated the default `create_deep_agent()` base prompt to:
  - ask fewer redundant followups
  - avoid re-asking schedule details already given
  - use stronger defaults for summaries/briefs
  - take direct send actions when the target fields are already present
  - bound search-then-deliver workflows so they do not loop on near-duplicate searches
  - restate key identifiers like issue titles and email subjects in completion confirmations
- Added a compact GLM-5-specific prompt supplement for the highest-value failure patterns that still needed extra salience.
- Added recursion limits to the experimental benchmark runner so short tool-use cases fail fast instead of hanging.
- Added an additional benchmark case for issue-create-then-notify confirmation quality.

## Key Files

- `libs/deepagents/deepagents/graph.py`
- `libs/deepagents/deepagents/_models.py`
- `libs/evals/deepagents_evals/better_harness/assertions.py`
- `libs/evals/deepagents_evals/better_harness/benchmarks.py`
- `libs/evals/deepagents_evals/better_harness/focused_comparison.py`
- `libs/evals/deepagents_evals/better_harness/optimizer.py`
- `libs/evals/deepagents_evals/better_harness/prompt_modules.py`
- `libs/evals/scripts/run_better_harness.py`
- `libs/evals/scripts/run_focused_harness_comparison.py`

## How To Run

From `libs/evals/`:

```bash
uv sync --group test

# Experimental hill-climber on the better-harness benchmark
uv run python scripts/run_better_harness.py \
  --model claude-sonnet-4-6 \
  --max-iterations 2 \
  --report-json artifacts/better-harness-sonnet/report.json \
  --report-md artifacts/better-harness-sonnet/report.md

uv run python scripts/run_better_harness.py \
  --model baseten:zai-org/GLM-5 \
  --max-iterations 2 \
  --report-json artifacts/better-harness-glm5/report.json \
  --report-md artifacts/better-harness-glm5/report.md

# Focused baseline-vs-improved comparison on the same branch
uv run python scripts/run_focused_harness_comparison.py \
  --model claude-sonnet-4-6 \
  --model baseten:zai-org/GLM-5 \
  --output-dir artifacts/focused-comparison-v2

# Resume or rebuild the aggregate report from existing per-slice JSON files
uv run python scripts/run_focused_harness_comparison.py \
  --model claude-sonnet-4-6 \
  --model baseten:zai-org/GLM-5 \
  --output-dir artifacts/focused-comparison-v2 \
  --reuse-existing
```

From `libs/deepagents/`:

```bash
uv run --group test pytest tests/unit_tests/smoke_tests/test_system_prompt.py
uv run --group test pytest tests/unit_tests/test_models.py
```

## Results

The focused comparison below uses the same 14-case slice on the current branch:

- `tests/evals/test_tool_selection.py` (`8` tool-use cases)
- `tests/evals/test_followup_quality.py` (`6` conversation/followup cases)

### Cross-Model Summary

| Model | Legacy Tool Use | Legacy Conversation | Legacy Combined | Improved Tool Use | Improved Conversation | Improved Combined |
| --- | --- | --- | --- | --- | --- | --- |
| `claude-sonnet-4-6` | `7/8` | `2/6` | `9/14` | `8/8` | `6/6` | `14/14` |
| `baseten:zai-org/GLM-5` | `6/8` | `1/6` | `7/14` | `8/8` | `6/6` | `14/14` |

Fresh same-branch rerun artifacts from April 7, 2026:

- Combined summary: [artifacts/focused-comparison-v2/comparison.md](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/comparison.md)
- Sonnet legacy tool-use: [artifacts/focused-comparison-v2/claude-sonnet-4-6/legacy-tool_use.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/claude-sonnet-4-6/legacy-tool_use.json)
- Sonnet legacy conversation: [artifacts/focused-comparison-v2/claude-sonnet-4-6/legacy-conversation.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/claude-sonnet-4-6/legacy-conversation.json)
- Sonnet improved tool-use: [artifacts/focused-comparison-v2/claude-sonnet-4-6/improved-tool_use.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/claude-sonnet-4-6/improved-tool_use.json)
- Sonnet improved conversation: [artifacts/focused-comparison-v2/claude-sonnet-4-6/improved-conversation.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/claude-sonnet-4-6/improved-conversation.json)
- GLM legacy tool-use: [artifacts/focused-comparison-v2/baseten-zai-org-glm-5/legacy-tool_use.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/baseten-zai-org-glm-5/legacy-tool_use.json)
- GLM legacy conversation: [artifacts/focused-comparison-v2/baseten-zai-org-glm-5/legacy-conversation.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/baseten-zai-org-glm-5/legacy-conversation.json)
- GLM improved tool-use: [artifacts/focused-comparison-v2/baseten-zai-org-glm-5/improved-tool_use.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/baseten-zai-org-glm-5/improved-tool_use.json)
- GLM improved conversation: [artifacts/focused-comparison-v2/baseten-zai-org-glm-5/improved-conversation.json](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/focused-comparison-v2/baseten-zai-org-glm-5/improved-conversation.json)

### Optimizer Benchmark

The small internal hill-climber selected the same two prompt modules for both models:

- `clarify_semantics_without_reasking`
- `summary_defaults_and_delivery`

Optimizer reports:

- Sonnet: [artifacts/better-harness-sonnet/report.md](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/better-harness-sonnet/report.md)
- GLM-5: [artifacts/better-harness-glm5/report.md](/Users/vivektrivedy/Documents/better-harness/deepagents/libs/evals/artifacts/better-harness-glm5/report.md)

| Model | Optimization Baseline | Optimization Final | Holdout Baseline | Holdout Final |
| --- | --- | --- | --- | --- |
| `claude-sonnet-4-6` | `1/5` | `4/5` | `2/4` | `2/4` |
| `baseten:zai-org/GLM-5` | `0/5` | `5/5` | `2/4` | `2/4` |

The optimizer benchmark is useful for fast prompt hill-climbing, but the real acceptance gate for shipping here was the focused pytest slice above.

### What Closed the Gaps

- Stronger rules for treating `every week` and explicit times as already-specified schedules.
- A stronger monitoring rule that asks for alert signals/thresholds and alert delivery, not just "what system" to monitor.
- A tighter summary rule that treats email as the source inbox and prioritizes detail-level clarification.
- Bounded search-then-deliver guidance so the agent searches once, drafts, sends, and stops.
- Confirmation guidance that repeats the key title, subject, recipient, or channel after acting.
- A short GLM-5-specific supplement for the same high-value rules when the generic prompt was not salient enough.

## Notes

- The better-harness runner is intentionally small and easy to modify. It is best for tight, behavior-specific hill climbs on short eval subsets.
- The focused comparison runner replays the same eval slice against the shipped prompt and the frozen legacy prompt by setting `DEEPAGENTS_BASE_PROMPT_VARIANT=legacy` internally.
- The `focused-comparison-v2` artifacts are the apples-to-apples comparison for the current branch. Earlier exploratory artifacts in `artifacts/` came from separate runs during development and may not match the same-branch legacy replay exactly, especially on LLM-judge-heavy conversation cases.
- `scripts/run_focused_harness_comparison.py` now supports `--reuse-existing`, so you can resume interrupted multi-model runs or regenerate the combined report without paying for already-completed slices again.
- The runner includes a frozen legacy base prompt for optimization so you can compare candidate prompt modules against the pre-tuned harness while still shipping the improved default prompt in `graph.py`.
- The benchmark runner now applies a small recursion limit per case so search loops fail fast during optimization instead of hanging the entire run.
- Baseten eval runs now set a `90s` timeout in `tests/evals/conftest.py` so legacy GLM failures fail fast instead of hanging on provider stalls. In the legacy GLM baseline, this converts the search-email miss into a bounded `429`/timeout-style failure instead of an indefinite wait.
- LLM-judge-heavy benchmarks are slower and noisier than deterministic assertions. For future runs, prefer expanding the benchmark with more deterministic cases as you identify stable failure patterns.
