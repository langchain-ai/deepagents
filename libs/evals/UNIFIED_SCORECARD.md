# Model Scorecard — Unified Evals

GH aggregate pass@k / avg@k from `.github/workflows/unified_evals.yml`. pass@k = fraction of tasks solved in ≥1 of k rollouts; avg@k = mean reward across rollouts; rewards are binary (0/1).

## Lite — micro & macro avg@k by model

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lite-scorecard-dark.svg">
  <img alt="Grouped bar chart of lite micro and macro avg@k across GPT-5.6 sol, GPT-5.6 terra, Claude Opus 4.8, GPT-5.6 luna, Claude Sonnet 5, and GLM-5.2, sorted by micro avg@k" src="assets/lite-scorecard-light.svg">
</picture>

Context is graded by the faithful Letta **`model_judge`** (rubric-based, phrasing/name/number tolerant) matching upstream Context-Bench. The lite context set is the difficulty frontier by combined terra+luna performance. luna is judged by `gpt-5.6-terra` (independent), all others by `gpt-5.6-luna`. Lite order by **micro avg@k**: **sol 0.481 > terra 0.363 > opus 0.323 > luna 0.314 > Sonnet 5 0.235 > GLM-5.2 0.167**. Per-model tables below also carry pass@k.

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
| context (context-retrieval) | 0.750 | 0.417 | 8 |
| **macro** | **0.626** | **0.380** |  |
| **micro** | **0.588** | **0.363** |  |

Autonomous and conversation from run [29509108062](https://github.com/langchain-ai/deepagents/actions/runs/29509108062) · 2026-07-16. Context from run [29787062499](https://github.com/langchain-ai/deepagents/actions/runs/29787062499) · faithful `model_judge`, judge `gpt-5.6-luna`.

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
| context (context-retrieval) | 0.875 | 0.750 | 8 |
| **macro** | **0.494** | **0.370** |  |
| **micro** | **0.441** | **0.314** |  |

Autonomous and conversation from run [29509109809](https://github.com/langchain-ai/deepagents/actions/runs/29509109809) · 2026-07-16. Context from run [29787074607](https://github.com/langchain-ai/deepagents/actions/runs/29787074607) · faithful `model_judge`, judged by `gpt-5.6-terra` (independent of luna).

## GPT-5.6 sol

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.485 | 11 |
| context (context-retrieval) | 1.000 | 0.875 | 8 |
| **macro** | **0.709** | **0.542** |  |
| **micro** | **0.647** | **0.481** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17. Context from run [29787062499](https://github.com/langchain-ai/deepagents/actions/runs/29787062499) · faithful `model_judge`, judge `gpt-5.6-luna`.

## Claude Opus 4.8

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.467 | 0.289 | 15 |
| conversation (tau3-subset) | 0.364 | 0.242 | 11 |
| context (context-retrieval) | 0.875 | 0.500 | 8 |
| **macro** | **0.569** | **0.344** |  |
| **micro** | **0.530** | **0.323** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17. Context from run [29787062499](https://github.com/langchain-ai/deepagents/actions/runs/29787062499) · faithful `model_judge`, judge `gpt-5.6-luna`.

## Claude Sonnet 5

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.267 | 0.133 | 15 |
| conversation (tau3-subset) | 0.182 | 0.091 | 11 |
| context (context-retrieval) | 0.875 | 0.625 | 8 |
| **macro** | **0.441** | **0.283** |  |
| **micro** | **0.383** | **0.235** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17. Context from run [29787062499](https://github.com/langchain-ai/deepagents/actions/runs/29787062499) · faithful `model_judge`, judge `gpt-5.6-luna`.

## GLM-5.2

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.133 | 0.067 | 15 |
| conversation (tau3-subset) | 0.000 | 0.000 | 11 |
| context (context-retrieval) | 0.875 | 0.583 | 8 |
| **macro** | **0.336** | **0.217** |  |
| **micro** | **0.265** | **0.167** |  |

Autonomous and conversation from run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17. Context from run [29787062499](https://github.com/langchain-ai/deepagents/actions/runs/29787062499) · faithful `model_judge`, judge `gpt-5.6-luna`.