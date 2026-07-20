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
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.455 | 11 |
| context (context-retrieval) | 0.625 | 0.375 | 8 |
| **macro** | **0.584** | **0.365** |  |
| **micro** | **0.559** | **0.353** |  |

Run [29509108062](https://github.com/langchain-ai/deepagents/actions/runs/29509108062) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~25m

One conversation shard hit a transient Docker-daemon failure; recovered from the run's artifacts (merged the retried shard), no tasks re-run.

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
| autonomous (harbor-index) | 0.333 | 0.178 | 15 |
| conversation (tau3-subset) | 0.273 | 0.182 | 11 |
| context (context-retrieval) | 0.500 | 0.333 | 8 |
| **macro** | **0.369** | **0.231** |  |
| **micro** | **0.353** | **0.216** |  |

Run [29509109809](https://github.com/langchain-ai/deepagents/actions/runs/29509109809) · 2026-07-16 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac` · wall ~50m

---

**6-model lite stress run — 2026-07-17, run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741).** Two Baseten models are excluded from the tables below: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` did not serve (per-trial `reward` null with pervasive 5xx / 401 / 404 / 429 / timeout signatures across trials), and `thinkingmachines/inkling` is dropped with it. The **context** column is low across all four models (0.000–0.125) versus the 2026-07-16 terra/luna lite (0.625 / 0.500). This is a genuine output-format / instruction-following gap, not a harness regression: each task instructs "write your final answer (and nothing else)" and is graded by exact string match, and these models compute the right entity but write a full explanatory sentence — reproduced on an independent context-only rerun ([29763976690](https://github.com/langchain-ai/deepagents/actions/runs/29763976690)). terra/luna scored higher on 2026-07-16 by emitting the bare answer. Read context here as a format / instruction-following signal, not pure retrieval. cb-cloud-5 is additionally under-specified (its gold picks the owner's first-listed state via an unordered `LIMIT 1`, and Adrian's owner is multi-state), so every model misses it regardless. Judge `gpt-5.6-luna` is independent for all four (not in the set).

## openai:gpt-5.6-sol

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.400 | 0.267 | 15 |
| conversation (tau3-subset) | 0.727 | 0.485 | 11 |
| context (context-retrieval) | 0.000 | 0.000 | 8 |
| **macro** | **0.376** | **0.251** |  |
| **micro** | **0.412** | **0.275** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context: format / instruction-following gap (see run note).

Note: the 0.000 context is an instruction-following / output-format failure, not a retrieval failure. Each task instructs "write your final answer (and nothing else) to `/app/answer.txt`" and context is graded by exact string match. On 2 of 3 spot-checked tasks sol computed the correct entity (cb-cloud-15 "Julie Guzman", cb-cloud-49 "Gregory Luna") but wrote a full explanatory sentence anyway, so it scored 0; cb-cloud-5 was a genuine miss.

Reproduced on an independent context-only rerun ([29763976690](https://github.com/langchain-ai/deepagents/actions/runs/29763976690), 2026-07-20, harbor `4efafd8d`): 0/8 again, same empty-final-answer / verbose-write pattern.

## anthropic:claude-opus-4-8

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.467 | 0.289 | 15 |
| conversation (tau3-subset) | 0.364 | 0.242 | 11 |
| context (context-retrieval) | 0.125 | 0.042 | 8 |
| **macro** | **0.318** | **0.191** |  |
| **micro** | **0.353** | **0.216** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context: format / instruction-following gap (see run note).

Context reproduced on rerun [29763976690](https://github.com/langchain-ai/deepagents/actions/runs/29763976690) (2026-07-20, harbor `4efafd8d`): 0.125 again (1/8, cb-cloud-26).

## anthropic:claude-sonnet-5

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.267 | 0.133 | 15 |
| conversation (tau3-subset) | 0.182 | 0.091 | 11 |
| context (context-retrieval) | 0.000 | 0.000 | 8 |
| **macro** | **0.149** | **0.075** |  |
| **micro** | **0.176** | **0.088** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context: format / instruction-following gap (see run note).

## fireworks:accounts/fireworks/models/glm-5p2

### Lite

Frozen high-signal subset (`lite_tasks.py`, difficulty-frontier tasks).

| Category | pass@k | avg@k | tasks |
| --- | --- | --- | --- |
| autonomous (harbor-index) | 0.133 | 0.067 | 15 |
| conversation (tau3-subset) | 0.000 | 0.000 | 11 |
| context (context-retrieval) | 0.125 | 0.042 | 8 |
| **macro** | **0.086** | **0.036** |  |
| **micro** | **0.088** | **0.039** |  |

Run [29593952741](https://github.com/langchain-ai/deepagents/actions/runs/29593952741) · 2026-07-17 · `agent_impl=bare` · `profile=lite` · rollouts=3 · `sandbox=docker` · `judge=gpt-5.6-luna` · harbor@`27a6eac`. Context: format / instruction-following gap (see run note).