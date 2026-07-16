# Model Scorecard — Unified Evals

GH aggregate pass@k / avg@k from `.github/workflows/unified_evals.yml`. pass@k = fraction of tasks solved in ≥1 of k rollouts; avg@k = mean reward across rollouts; rewards are binary (0/1).

## openai:gpt-5.6-terra

### Full (default)

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.268 | 0.183 | 82 |
| conversation (tau3-subset) | 0.467 | 0.389 | 30 |
| context (context-retrieval) | 0.667 | 0.567 | 30 |
| **macro** | **0.467** | **0.379** |  |
| **micro** | **0.394** | **0.308** |  |

Run [29430259116](https://github.com/langchain-ai/deepagents/actions/runs/29430259116) · 2026-07-15 · `agent_impl=bare` · `profile=full` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~4h

autonomous includes 14 of 246 trials that errored (agent/verifier timeouts and one OOM) and are scored as failures. Aggregated from the run's artifacts (one shard recovered from the retry attempt); no tasks were re-run.

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.222 | 15 |
| conversation (tau3-subset) | 0.545 | 0.333 | 11 |
| context (context-retrieval) | 0.375 | 0.250 | 8 |
| **macro** | **0.440** | **0.269** |  |
| **micro** | **0.441** | **0.265** |  |

Run [29451389710](https://github.com/langchain-ai/deepagents/actions/runs/29451389710) · 2026-07-15 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~1h

## openai:gpt-5.6-luna

### Full (default)

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.159 | 0.114 | 82 |
| conversation (tau3-subset) | 0.367 | 0.322 | 30 |
| context (context-retrieval) | 0.733 | 0.411 | 30 |
| **macro** | **0.420** | **0.282** |  |
| **micro** | **0.324** | **0.221** |  |

Run [29272737912](https://github.com/langchain-ai/deepagents/actions/runs/29272737912) · 2026-07-13 · `agent_impl=bare` · `profile=full` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`af2e862`

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.156 | 15 |
| conversation (tau3-subset) | 0.273 | 0.212 | 11 |
| context (context-retrieval) | 0.375 | 0.125 | 8 |
| **macro** | **0.349** | **0.164** |  |
| **micro** | **0.353** | **0.167** |  |

Run [29304843581](https://github.com/langchain-ai/deepagents/actions/runs/29304843581) · 2026-07-14 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`af2e862`