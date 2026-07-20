# Model Scorecard — Unified Evals

GH aggregate pass@k / avg@k from `.github/workflows/unified_evals.yml`. pass@k = fraction of tasks solved in ≥1 of k rollouts; avg@k = mean reward across rollouts; rewards are binary (0/1).

## Lite — pass@k and avg@k by model and category

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lite-scorecard-dark.svg">
  <img alt="Grouped bar chart of lite pass@k (top panel) and avg@k (bottom panel) by category (autonomous, conversation, context) across GPT-5.6 terra, sol, luna, Claude Opus 4.8, Claude Sonnet 5, and GLM-5.2, sorted by macro pass@k" src="assets/lite-scorecard-light.svg">
</picture>

pass@k (top) and avg@k (bottom), sorted by macro pass@k. Context is now graded by the faithful Letta **`model_judge`** (rubric-based LLM judge, `gpt-5.6-luna`) matching upstream Context-Bench, **not exact string match** — verbose-but-correct answers are no longer zeroed. This confirms the earlier low context scores were an output-format artifact, not a retrieval gap. **Re-graded so far:** GPT-5.6 sol context 0.000 → 0.750 and Claude Opus 4.8 context 0.125 → 0.500 (run [29778028063](https://github.com/langchain-ai/deepagents/actions/runs/29778028063)). GPT-5.6 terra, luna, Sonnet 5, and GLM-5.2 context below still show the **old exact-match** numbers pending re-grade (run [29778667492](https://github.com/langchain-ai/deepagents/actions/runs/29778667492)); the chart above regenerates once all are re-graded. Judging luna with a luna grader is self-grading (noted where it applies).

## GPT-5.6 terra

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
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.455 | 11 |
| context (context-retrieval) | 0.625 | 0.375 | 8 |
| **macro** | **0.584** | **0.365** |  |
| **micro** | **0.559** | **0.353** |  |

Run [29509108062](https://github.com/langchain-ai/deepagents/actions/runs/29509108062) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~25m

One conversation shard hit a transient Docker-daemon failure; recovered from the run's artifacts (merged the retried shard), no tasks re-run.

## GPT-5.6 luna

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
| autonomous (harbor-index) | 0.333 | 0.178 | 15 |
| conversation (tau3-subset) | 0.273 | 0.182 | 11 |
| context (context-retrieval) | 0.500 | 0.333 | 8 |
| **macro** | **0.369** | **0.231** |  |
| **micro** | **0.353** | **0.216** |  |

Run [29509109809](https://github.com/langchain-ai/deepagents/actions/runs/29509109809) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~50m

---

2026-07-17 lite runs. Context is graded by exact string match on the answer file, so the context column below is output-format-sensitive. Judge `gpt-5.6-luna` is independent for all four.

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
| context (context-retrieval) | 0.000 | 0.000 | 8 |
| **macro** | **0.149** | **0.075** |  |
| **micro** | **0.176** | **0.088** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`.

## GLM-5.2

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.133 | 0.067 | 15 |
| conversation (tau3-subset) | 0.000 | 0.000 | 11 |
| context (context-retrieval) | 0.125 | 0.042 | 8 |
| **macro** | **0.086** | **0.036** |  |
| **micro** | **0.088** | **0.039** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`.