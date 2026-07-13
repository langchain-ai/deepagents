# Unified Evals

The **unified evals** CI job ([`.github/workflows/unified_evals.yml`](../../.github/workflows/unified_evals.yml)) runs one or more models through a single, fixed battery of benchmarks and produces one cross-model comparison — a leaderboard plus a radar chart. Its purpose is to answer "how does model X stack up as a deep agent?" along a small number of **distinct capability axes**, using the same tasks, harness, and scoring for every model so the numbers are comparable.

This document explains the *decisions* behind that battery: which benchmarks we chose, what capability each one stands in for, and why we run the specific tasks we do. For how to operate the workflow (inputs, sandboxes, Harbor setup), see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Why this, not the pytest eval suite

The SDK also has a pytest eval suite ([`tests/evals/`](tests/evals/), catalogued in [`EVAL_CATALOG.md`](EVAL_CATALOG.md)). It measures a **different thing**, and the two are complementary:

- **The pytest evals measure specific agent *behaviors* in controlled scenarios.** Does the agent pick the right tool for an intent, prefer `edit` over a full rewrite, read and write files in parallel, recover from a truncated read, keep a todo list, use memory, summarize faithfully. Each eval asserts one narrow, known-correct behavior, grouped into capability areas (file operations, retrieval, tool use, memory, conversation, summarization, …). The signal is **diagnostic** — a failure tells you *which behavior* broke.
- **The unified evals measure end-to-end task *success* on external benchmarks.** Can the agent actually finish a sandboxed autonomous task, resolve a multi-turn support conversation, or answer a multi-hop question over a large corpus — scored `pass@k` / `avg@k` and rolled up into a **cross-model comparison**. The signal is **holistic and comparative** — how capable the agent is end-to-end, and how models rank against each other, on tasks authored outside this repo.

Reach for the unified evals when the question is *"how good is model X, versus Y, as a deep agent?"* — model selection, a capability scorecard, or tracking progress against the external Harbor / τ³ / Context-Bench ecosystem. Reach for the pytest suite when the question is *"does the SDK still do the right thing in this specific case?"* — catching behavioral regressions as the code changes. An agent can top the pytest behaviors and still stall on end-to-end tasks (or the reverse), which is exactly why both exist.

## The three categories

A "deep agent" is not one skill, so a single benchmark can't score one. We split the evaluation into three capability categories, and map each to one benchmark. This mapping is the source of truth in [`unified_prep.py`](../../.github/scripts/unified_prep.py) (`CATEGORY_MAP`):

| Category | Capability it stands for | Benchmark | Harness |
|---|---|---|---|
| **autonomous** | End-to-end task execution in a real, sandboxed computer/terminal environment | [`harbor-index/harbor-index-1.0`](https://github.com/laude-institute/harbor) (Harbor registry) | bare · dcode |
| **conversation** | Multi-turn, tool-using dialogue against a simulated user, following a policy | [`tau3-subset`](https://github.com/sierra-research/tau2-bench) (τ³-bench) | tau3 |
| **context** | Retrieval + reasoning over a large, multi-file corpus | [`context-retrieval-evals`](https://github.com/letta-ai/letta-evals) (Context-Bench) | bare · dcode |

The default run exercises all three (`categories: "autonomous,conversation,context"`). A radar chart is only meaningful with the full set, so it is emitted only when all three categories run.

The `autonomous` and `context` categories run a deep-agents graph: by default the **bare** `create_deep_agent` — the SDK agent with no product scaffolding, which keeps the score a measure of the *model* rather than of a harness wrapped around it — with **`dcode`** (deep-agents-code, the full product agent) selectable as an option. The `conversation` category runs the τ³ runtime, which supplies a **user simulator** the agent must converse with rather than a static prompt.

## Task-selection philosophy

Four principles cut across all three categories and explain why the task sets look the way they do:

1. **Reuse credible external benchmarks; don't invent tasks.** Every category is sourced from an established, independently-authored benchmark (Harbor / Terminal-Bench, τ³-bench, Context-Bench). This keeps the eval honest — we aren't grading ourselves on tasks we wrote to flatter our agent.
2. **Curate small subsets by *measured* difficulty, not by hand.** Where we take a subset, tasks are tiered by the **empirical pass rate of a strong reference model (Opus 4.8)** over multiple rollouts, not by an author's guess at how hard they are.
3. **Optimize for a discriminating spread with headroom, not saturation.** A benchmark every model solves (or every model fails) ranks nothing. We deliberately keep the scarce *intermittent* tasks — the ones a strong model solves only some of the time — because they carry the most signal, and we weight toward hard tasks so the set doesn't saturate as models improve.
4. **Keep set sizes small enough to fit the CI budget while preserving signal.** Subsets are sized (~30 tasks/category) to fit GitHub's 6-hour per-job limit and the workflow's concurrency caps, run at 3 rollouts each.

## Category detail

### Autonomous — `harbor-index/harbor-index-1.0`

**What it measures.** Whether the agent can take a task to completion in a real, sandboxed environment — writing and running code, using the terminal, and manipulating files — graded by each task's own verification harness rather than by an LLM judge.

**Why this benchmark.** This is the flagship "can it actually do the job" axis. We run `harbor-index/harbor-index-1.0`, the curated autonomous-agent task index from the [Harbor](https://github.com/laude-institute/harbor) registry. It is the same sandboxed-verifiable-task family as [Terminal-Bench 2](https://github.com/laude-institute/terminal-bench-2) — the suite's original Harbor benchmark, spanning 90+ tasks across software engineering, biology, security, gaming, and more. Running through Harbor means each task ships its own environment image and grader, so a pass is objective and reproducible. (`terminal-bench/terminal-bench-2-1` is the closely-related sibling dataset selectable in the standalone [`harbor.yml`](../../.github/workflows/harbor.yml) workflow.)

**Why these tasks.** We run the benchmark's own published index as authored, rather than sub-selecting, so the score covers breadth and stays comparable to the wider Harbor/Terminal-Bench ecosystem. The tasks are cross-domain terminal / computer-use problems — in the Terminal-Bench family, software engineering, security, data and scientific computing, system administration, and more — each shipping its own environment and pass/fail verifier. So this category measures general "operate a computer to finish a real job" competence across domains, not a hand-picked slice.

### Conversation — `tau3-subset` (τ³-bench)

**What it measures.** Multi-turn customer-service-style dialogue: the agent must converse with a simulated user, call domain tools, and follow a written policy to resolve the user's issue.

**Why this benchmark.** [τ³-bench](https://github.com/sierra-research/tau2-bench) (the Harbor dataset `sierra-research/tau3-bench`; τ³ ships inside the `tau2-bench` repo) is a standard for tool-using conversational agents. The `conversation` category runs it through the `tau3` harness, whose user simulator (an OpenAI model, currently `gpt-5.2`) drives a live back-and-forth — this is the only category that scores *dialogue* rather than a single-shot task.

**Why these tasks.** We run a curated **30-task subset** ([`tau3_subset.py`](deepagents_evals/tau3_subset.py)) drawn from two τ³ domains that exercise different conversational skills:

- **`banking_knowledge` (24 tasks)** — the user asks a policy or eligibility question; the agent must retrieve the correct answer from the domain's knowledge base and state it. Measures grounded question-answering under a policy.
- **`telecom` (6 tasks)** — multi-step service-issue troubleshooting (APN settings, SIM-card PIN, airplane mode, overdue-bill suspension); the agent must diagnose a broken-service scenario and drive it to resolution with tools, across a live back-and-forth. Measures procedural, multi-turn problem-solving.

Within those domains the subset is a *difficulty probe* — a behavior spread across models — not leaderboard parity with full τ³-bench. Each task's tier is why it's included, and is the measured pass rate of `anthropic:claude-opus-4-8` over 3 rollouts at full agent timeout (easy = 3/3, medium = 1–2/3, hard = 0/3): floors any capable model should pass, an intermittent middle where models separate, and hard tasks for headroom. Opus finds most of this set hard — accepted, intentional headroom. The subset is a *living selection*: re-run and re-tier (updating each task's `justification`) as the reference model or task set changes.

### Context — `context-retrieval-evals` (Context-Bench)

**What it measures.** Extracting and reasoning over information spread across a multi-file corpus. Every task ships the **whole** 10-file corpus so the agent can't infer which files matter — it must retrieve, join, and aggregate to answer.

**Why this benchmark.** Derived from [Context-Bench](https://github.com/letta-ai/letta-evals) (Letta's `filesystem` `cloud` suite of synthetic person/vehicle/pet/account records, Apache-2.0). It isolates long-context retrieval and multi-hop joins — a capability the autonomous and conversation categories don't directly stress. Each task ships the entire 10-file corpus, so the agent can't shortcut to the relevant files; it has to search, join, and aggregate across them.

**Why these tasks.** The 30 tasks span eight query types over the same corpus, so the set measures a range of retrieval-and-reasoning operations rather than one:

- **`multi_hop_chain` · `multi_entity_comparison` (13 tasks)** — deep multi-file joins: follow a chain of relationships across files, or compare two entities each reached by its own lookup. The hardest type, and the core discriminators.
- **`aggregation` · `cross_file_counting` (4)** — sum balances or count records scattered across files.
- **`set_intersection` · `comparison_tiebreak` (6)** — find the entities satisfying several constraints at once, resolving ties.
- **`negation` · `temporal_reasoning` (7)** — exclude by a condition, or reason over dates.

Within that spread, each task earns its slot by the *role* its measured difficulty gives it. Tiers are empirical, from a calibration run on **Opus 4.8** via the **bare** `create_deep_agent` harness (3 rollouts/task; `pass@bare` = fraction solved): floors (`pass@bare` ≥ 0.90), an intermittent middle (`0.10 < pass@bare < 0.90`) that carries the most signal, and discriminators (`pass@bare` ≤ 0.10). The set is **5 easy · 9 medium · 16 hard**, weighted toward discriminators for headroom: Context-Bench is **bimodal** on a strong model (a task is solved ~1.0 or failed ~0.0), so intermittent tasks are rare — all ~9 that exist are kept, and the rest are hard so the set doesn't saturate. `pass@bare` is a *floor* (the bare default harness); a stronger harness like `dcode` is expected to solve some of the hard tier. Each task's own `pass@bare` and query type is listed in the [dataset README](datasets/context-retrieval-evals/README.md).

## Running it

Dispatched from the Actions tab (`workflow_dispatch`). Every input has a default except `models`:

- **`models`** *(required)* — comma-separated `provider:model` specs (e.g. `anthropic:claude-opus-4-8,openai:gpt-5.2`). The set of models compared in one run; everything else is applied identically across them.
- **`categories`** *(default `autonomous,conversation,context`)* — which capability axes to run. The radar chart is produced only when all three run.
- **`agent_impl`** *(default `bare`, options `bare` / `dcode`)* — the deep-agents harness for the **autonomous** and **context** categories: `bare` (`create_deep_agent`, the neutral SDK agent) or `dcode` (the deep-agents-code product agent). The **conversation** category ignores this and always uses `tau3`: `tau3` is not just a harness but the τ³-bench runtime that hosts the **user simulator** the agent has to converse with, so the category is bound to it — `bare`/`dcode` are single-shot deep-agents graphs and can't drive the multi-turn simulated-user protocol.
- **`rollouts`** *(default `3`)* — trials per task, i.e. **K** in the two scores reported per `(model × category)`:
  - **pass@K** — fraction of tasks that passed at least once within K rollouts.
  - **avg@K** — passing trials (capped at K per task) ÷ expected trials (tasks × K); missing rollouts count as failures, so a partial run can't inflate the score.
- **`concurrency`** *(default `4`)* — tasks in flight per model.
- **`shard_parallel`** *(default `10`)* — parallel shards per `(model × category)`, auto-clamped to stay within the per-model and global concurrency caps.
- **`n_shards_autonomous` · `n_shards_conversation` · `n_shards_context`** *(defaults `10` · `3` · `3`)* — how each category's tasks are split across parallel jobs, sized to fit GitHub's 6-hour per-job limit.
- **`sandbox_env`** *(default `langsmith`)* — where tasks execute.
- **`force_build`** *(default `false`)* — rebuild each task's environment image/snapshot; required the first time a new dataset runs on the LangSmith sandbox.
- **`harbor_package_override`** — install an unreleased Harbor build instead of the pinned version.

The `prep` job writes a **run-configuration summary** — every input plus the values it derived (resolved model list, effective `shard_parallel` after clamping) — to the run summary, so a dispatch's exact settings are visible for debugging. The run then publishes one cross-model comparison — a leaderboard and (for full runs) a radar chart — to the same run summary.

## Changing the battery

- **Category → benchmark + harness mapping, shard defaults:** `CATEGORY_MAP` and `DEFAULT_N_SHARDS` in [`unified_prep.py`](../../.github/scripts/unified_prep.py).
- **Conversation subset + tiers:** [`deepagents_evals/tau3_subset.py`](deepagents_evals/tau3_subset.py) (re-run and update each `justification`).
- **Context subset + tiers:** [`datasets/context-retrieval-evals`](datasets/context-retrieval-evals/README.md) and its `calibration.json`.
- **Model catalog / presets:** [`.github/scripts/models.py`](../../.github/scripts/models.py) and [`MODEL_GROUPS.md`](MODEL_GROUPS.md).
