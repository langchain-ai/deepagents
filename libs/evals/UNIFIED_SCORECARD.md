# Model Scorecard — Unified Evals

GH aggregate pass@k / avg@k from `.github/workflows/unified_evals.yml`. pass@k = fraction of tasks solved in ≥1 of k rollouts; avg@k = mean reward across rollouts; rewards are binary (0/1).

## Lite — pass@k and avg@k by model and category

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lite-scorecard-dark.svg">
  <img alt="Grouped bar chart of lite pass@k (top panel) and avg@k (bottom panel) by category (autonomous, conversation, context) across GPT-5.6 terra, sol, luna, Claude Opus 4.8, Claude Sonnet 5, and GLM-5.2, sorted by macro pass@k" src="assets/lite-scorecard-light.svg">
</picture>

Context is now graded by the faithful Letta **`model_judge`** (rubric-based LLM judge) matching upstream Context-Bench, **not exact string match** — verbose-but-correct answers are no longer zeroed, confirming the earlier low context scores were an output-format artifact, not a retrieval gap. All six models below are re-graded (2026-07-20). By **macro avg@k** the lite order is **sol 0.501 > terra 0.449 > luna 0.356 > opus 0.330 > Sonnet 5 0.297 > GLM-5.2 0.133**. luna's context is judged by `gpt-5.6-terra` (independent, to avoid self-grading); all others by `gpt-5.6-luna`.

> **Pending recalibration.** Under the faithful judge the current lite context set saturates (four of six models sit at 0.750 context pass@k), so it barely discriminates. Next: re-run the full 30-task context set for terra + sol under this judge and re-pick the lite context subset to the difficulty frontier (tasks strong models don't saturate). The chart below is stale — it still shows the old pass@k/avg@k panels and will be regenerated as **avg@k-only (macro + micro)** once the lite context set is finalized.

## GPT-5.6 terra

### Full (default)

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.268 | 0.183 | 82 |
| conversation (tau3-subset) | 0.467 | 0.389 | 30 |
| context (context-retrieval) | 0.967 | 0.811 | 30 |
| **macro** | **0.567** | **0.461** |  |
| **micro** | **0.458** | **0.359** |  |

Autonomous and conversation from run [29430259116](https://github.com/langchain-ai/deepagents/actions/runs/29430259116) · 2026-07-15 · `agent_impl=bare` · `profile=full` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~4h. Context re-graded on the recalibrated 30-task set via run [29785826811](https://github.com/langchain-ai/deepagents/actions/runs/29785826811) (faithful `model_judge`, judge `gpt-5.6-luna`).

autonomous includes 14 of 246 trials that errored (agent/verifier timeouts and one OOM) and are scored as failures. Aggregated from the run's artifacts (one shard recovered from the retry attempt); no tasks were re-run.

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.455 | 11 |
| context (context-retrieval) | 0.750 | 0.625 | 8 |
| **macro** | **0.626** | **0.449** |  |
| **micro** | **0.588** | **0.412** |  |

Autonomous and conversation from run [29509108062](https://github.com/langchain-ai/deepagents/actions/runs/29509108062) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~25m. Context re-graded 2026-07-20 via run [29778667492](https://github.com/langchain-ai/deepagents/actions/runs/29778667492) with the faithful Letta `model_judge` (judge `gpt-5.6-luna`), up from an exact-match `0.625`.

One conversation shard hit a transient Docker-daemon failure; recovered from the run's artifacts (merged the retried shard), no tasks re-run.

## GPT-5.6 luna

### Full (default)

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.159 | 0.114 | 82 |
| conversation (tau3-subset) | 0.367 | 0.322 | 30 |
| context (context-retrieval) | 0.933 | 0.900 | 30 |
| **macro** | **0.486** | **0.445** |  |
| **micro** | **0.366** | **0.324** |  |

Autonomous and conversation from run [29272737912](https://github.com/langchain-ai/deepagents/actions/runs/29272737912) · 2026-07-13 · `agent_impl=bare` · `profile=full` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`af2e862`. Context re-graded on the recalibrated 30-task set via run [29785840733](https://github.com/langchain-ai/deepagents/actions/runs/29785840733) (faithful `model_judge`, judged by `gpt-5.6-terra`, independent of luna).

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.333 | 0.178 | 15 |
| conversation (tau3-subset) | 0.273 | 0.182 | 11 |
| context (context-retrieval) | 0.750 | 0.708 | 8 |
| **macro** | **0.452** | **0.356** |  |
| **micro** | **0.412** | **0.304** |  |

Autonomous and conversation from run [29509109809](https://github.com/langchain-ai/deepagents/actions/runs/29509109809) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~50m. Context re-graded 2026-07-20 via run [29779192946](https://github.com/langchain-ai/deepagents/actions/runs/29779192946) with the faithful Letta `model_judge`, judged by **`gpt-5.6-terra`** (independent, to avoid luna self-grading), up from an exact-match `0.500`.

---

2026-07-17 lite runs (sol, opus, Sonnet 5, GLM-5.2). Their context columns have since been re-graded (2026-07-20) with the faithful Letta `model_judge`; the old exact-match context is superseded (see each model's note). Judge `gpt-5.6-luna` is independent for all four.

## GPT-5.6 sol

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.485 | 11 |
| context (context-retrieval) | 0.750 | 0.750 | 8 |
| **macro** | **0.626** | **0.501** |  |
| **micro** | **0.588** | **0.451** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context re-graded 2026-07-20 via run [29778028063](https://github.com/langchain-ai/deepagents/actions/runs/29778028063) with the faithful Letta `model_judge` (`rubric.txt`, judge `gpt-5.6-luna`), which lifts context from an exact-match `0.000` — sol solves 6/8 context tasks.

## Claude Opus 4.8

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.467 | 0.289 | 15 |
| conversation (tau3-subset) | 0.364 | 0.242 | 11 |
| context (context-retrieval) | 0.500 | 0.458 | 8 |
| **macro** | **0.444** | **0.330** |  |
| **micro** | **0.441** | **0.314** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context re-graded 2026-07-20 via run [29778028063](https://github.com/langchain-ai/deepagents/actions/runs/29778028063) with the faithful Letta `model_judge` (`rubric.txt`, judge `gpt-5.6-luna`), which lifts context from an exact-match `0.125` — opus solves 4/8 context tasks.

## Claude Sonnet 5

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.267 | 0.133 | 15 |
| conversation (tau3-subset) | 0.182 | 0.091 | 11 |
| context (context-retrieval) | 0.750 | 0.667 | 8 |
| **macro** | **0.400** | **0.297** |  |
| **micro** | **0.353** | **0.245** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context re-graded 2026-07-20 via run [29778667492](https://github.com/langchain-ai/deepagents/actions/runs/29778667492) with the faithful Letta `model_judge` (judge `gpt-5.6-luna`), up from an exact-match `0.000`.

## GLM-5.2

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.133 | 0.067 | 15 |
| conversation (tau3-subset) | 0.000 | 0.000 | 11 |
| context (context-retrieval) | 0.625 | 0.333 | 8 |
| **macro** | **0.253** | **0.133** |  |
| **micro** | **0.206** | **0.108** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context re-graded 2026-07-20 via run [29778667492](https://github.com/langchain-ai/deepagents/actions/runs/29778667492) with the faithful Letta `model_judge` (judge `gpt-5.6-luna`), up from an exact-match `0.125`.