# Unified Evals

The **unified evals** CI job ([`.github/workflows/unified_evals.yml`](../../.github/workflows/unified_evals.yml)) runs one or more models through a single, fixed battery of benchmarks and produces one cross-model comparison — a leaderboard plus a radar chart. Its purpose is to answer "how does model X stack up as a deep agent?" along a small number of **distinct capability axes**, using the same tasks, harness, and scoring for every model so the numbers are comparable.

This document explains the *decisions* behind that battery: which benchmarks we chose, what capability each one stands in for, and why we run the specific tasks we do. For how to operate the workflow (inputs, sandboxes, Harbor setup), see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Running it

Dispatched from the Actions tab (`workflow_dispatch`). Every input has a default except `models`:

- **`models`** *(required)* — comma-separated `provider:model` specs (e.g. `anthropic:claude-opus-4-8,openai:gpt-5.2`). The set of models compared in one run; everything else is applied identically across them.
- **`categories`** *(default `autonomous,conversation,research`)* — which capability axes to run, replacing the default set rather than adding to it; `context` is available but not in the default set. `research` pins its own runner, sandbox, and concurrency (see below), so it needs no other input changed. The radar chart is produced whenever at least three axes run.
- **`agent_impl`** *(default `bare`, options `bare` / `dcode`)* — the deep-agents harness for the **autonomous**, **context**, and **research** categories: `bare` (`create_deep_agent`, the neutral SDK agent) or `dcode` (the deep-agents-code product agent). The **conversation** category ignores this and always uses `tau3`: `tau3` is not just a harness but the τ³-bench runtime that hosts the **user simulator** the agent has to converse with, so the category is bound to it — `bare`/`dcode` are single-shot deep-agents graphs and can't drive the multi-turn simulated-user protocol.
- **`rollouts`** *(default `3`)* — trials per task, i.e. **K** in the two scores reported per `(model × category)`:
  - **pass@K** — fraction of tasks that passed at least once within K rollouts.
  - **avg@K** — passing trials (capped at K per task) ÷ expected trials (tasks × K); missing rollouts count as failures, so a partial run can't inflate the score.
- **`concurrency`** *(default `4`)* — tasks in flight per model.
- **`shard_parallel`** *(default `10`)* — parallel shards per `(model × category)`, auto-clamped to stay within the per-model and global concurrency caps.
- **`n_shards_autonomous` · `n_shards_conversation` · `n_shards_context`** *(defaults `10` · `3` · `3`)* — how each category's tasks are split across parallel jobs, sized to fit GitHub's 6-hour per-job limit.
- **`sandbox_env`** *(default `langsmith`)* — where tasks execute.
- **`force_build`** *(default `false`)* — rebuild each task's environment image/snapshot; required the first time a new dataset runs on the LangSmith sandbox.
- **`harbor_package_override`** *(optional)* — install Harbor from an arbitrary package spec instead of the locked version, to test an unreleased Harbor build. Leave empty to use the pinned release (`harbor[langsmith] 0.20.0`, `harbor-langsmith 0.3.0`), which forwards task-environment MCP servers to the LangGraph agent and so runs every category — including `conversation` (`tau3`) — out of the box. An override is no longer required for any category.

### Dispatch it in CI (`gh workflow run`)

You don't have to use the Actions UI — dispatch the same workflow from the CLI. Only `models` is required; every other `-f` overrides a default shown above. Note the input is `agent_impls` (plural).

```bash
gh workflow run unified_evals.yml \
  -f models="anthropic:claude-opus-4-8,openai:gpt-5.2" \
  -f categories="autonomous,conversation,research" \
  -f agent_impls="bare" \
  -f rollouts="3"

# Watch it
gh run list --workflow unified_evals.yml --limit 1
gh run watch <run-id>
```

To run **just one task** (a quick smoke test), pass its exact name via `include_tasks` — it filters the tasks resolved by `categories`/`profile`, and an unknown name fails during prep:

```bash
gh workflow run unified_evals.yml \
  -f models="anthropic:claude-opus-4-8" \
  -f categories="autonomous" \
  -f include_tasks="hello-world"
```

Add `--ref <branch>` to dispatch a workflow definition other than the default branch's. No `harbor_package_override` is needed — the pinned Harbor runs every category out of the box (see below).

### Run one category from your machine (`harbor run`)

To iterate locally, run a single category's dataset directly through Harbor with the same LangGraph agent the CI job uses — `graph=bare` for the neutral SDK agent, `graph=dcode` for the product agent. Autonomous (`harbor-index/harbor-index`) is the simplest since it comes from the Harbor registry:

```bash
# From libs/evals. Export every host var the command templates below, or Harbor
# aborts at launch: ANTHROPIC_API_KEY, LANGSMITH_API_KEY, OPENAI_API_KEY,
# OPENAI_BASE_URL, and (for gateway routing) ANTHROPIC_BASE_URL. Harbor resolves
# each ${VAR} from the host and raises ValueError on an unset one — use
# ${VAR:-default} to make a reference optional, or drop the flag entirely.
make stage-harbor-local-deps          # stage checked-out packages for the sandbox install

uv run harbor run \
  --agent langgraph \
  --agent-kwarg project_path=deepagents_harbor/langgraph_project \
  --agent-kwarg config=langgraph.json \
  --agent-kwarg graph=bare \
  --agent-env 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' \
  --agent-env 'ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}' \
  --agent-env 'LANGSMITH_API_KEY=${LANGSMITH_API_KEY}' \
  --agent-env 'LANGSMITH_TRACING=true' \
  --agent-env 'OPENAI_BASE_URL=${OPENAI_BASE_URL}' \
  --agent-env 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
  --verifier-env 'OPENAI_BASE_URL=${OPENAI_BASE_URL}' \
  --verifier-env 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
  --verifier-env 'JUDGE_PROVIDER=openai' \
  --verifier-env 'JUDGE_MODELS=gpt-5.6-luna' \
  --verifier-env 'JUDGE_REPEATS=1' \
  --verifier-env 'JUDGE_CONCURRENCY=1' \
  --dataset harbor-index/harbor-index \
  --model anthropic:claude-opus-4-8 \
  --include-task-name hello-world \
  -n 4 \
  --jobs-dir harbor-jobs/unified \
  --env langsmith \
  --plugin langsmith \
  --plugin-kwarg dataset_name=harbor-index/harbor-index \
  --plugin-kwarg experiment_name=unified-local-smoke
```

`--include-task-name` (`-i`) is the local equivalent of the workflow's `include_tasks`; drop it to run the whole dataset, or repeat it to select several tasks. (`-l N` instead caps the run to the first N tasks.)

The `harbor-index` (and `tau3`) verifiers are **OpenAI LLM judges** — pass the judge's key and base URL through `--verifier-env` (`--ve`) or the verifier exits without writing a reward. `JUDGE_MODELS` is the grader (defaults to `gpt-5.6-luna`; use an independent model to avoid self-grading, e.g. `gpt-5.6-terra` when testing a Luna model), and `JUDGE_PROVIDER=openai` / `JUDGE_REPEATS` / `JUDGE_CONCURRENCY` are the config the native judge requires. `OPENAI_BASE_URL` points at whichever OpenAI-compatible endpoint holds `OPENAI_API_KEY` — OpenAI directly (`https://api.openai.com/v1`) or the LangSmith gateway (`https://gateway.smith.langchain.com/openai/v1`). `OPENAI_BASE_URL` and `OPENAI_API_KEY` are forwarded to the agent too, so switching `--model` to an `openai:` spec resolves through the same endpoint and key (the CI workflow forwards the key conditionally, per model provider).

**`judge_models` also controls the DRBench (`research`) judge.** Empty defaults every LLM-judge verifier, including tau3 and DRBench, to `gpt-5.6-luna`; the adapter registers that pinned default with upstream DRBench's OpenAI routing allowlists. Upstream's native `gpt-4o` / `gpt-4o-mini` judges and a single `openrouter/<vendor>/<model>` slug also work. An unsupported value falls back to `gpt-5.6-luna` for research with a warning in the step summary, while the other categories keep the requested suite-wide judge. Changing graders makes research scores incomparable to earlier runs and to the paper, so re-baseline rather than reading a delta.

Forward `ANTHROPIC_BASE_URL` to the agent the same way when the Anthropic model-under-test routes through the LangSmith gateway (a gateway-only `sk-ant` key 403s against the default Anthropic endpoint). Drop it to hit the Anthropic API directly.

Full local setup — env vars, `.env`, the `context` / `conversation` local datasets, and Makefile shortcuts — is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

The `prep` job writes a **run-configuration summary** — every input plus the values it derived (resolved model list, effective `shard_parallel` after clamping) — to the run summary, so a dispatch's exact settings are visible for debugging. The run then publishes one cross-model comparison — a leaderboard and (for full runs) a radar chart — to the same run summary.

## The three categories

A "deep agent" is not one skill, so a single benchmark can't score one. We split the evaluation into three capability categories, and map each to one benchmark. This mapping is the source of truth in [`unified_prep.py`](../../.github/scripts/evals/unified_prep.py) (`CATEGORY_MAP`):

| Category | Capability it stands for | Benchmark | Harness |
|---|---|---|---|
| **autonomous** | End-to-end task execution in a real, sandboxed computer/terminal environment | [`harbor-index/harbor-index`](https://github.com/laude-institute/harbor) (Harbor registry) | bare · dcode |
| **conversation** | Multi-turn, tool-using dialogue against a simulated user, following a policy | [`tau3-subset`](https://github.com/sierra-research/tau2-bench) (τ³-bench) | tau3 |
| **context** | Retrieval + reasoning over a large, multi-file corpus | [`context-retrieval-evals`](https://github.com/letta-ai/letta-evals) (Context-Bench) | bare · dcode |
| **research** | Open-ended research across a live enterprise app stack **and** the open web, synthesized into a cited report | [`drbench-evals`](https://github.com/ServiceNow/drbench) (DRBench, app mode) | bare · dcode |

The default run exercises `autonomous`, `conversation`, and `research` (`categories: "autonomous,conversation,research"`); `context` runs only when named explicitly. A radar chart needs at least three axes to be meaningful, so it is emitted only once that many categories run. Each axis plots `pass@K`, except for categories scored on a graded reward (`research`), which plot `avg@K` — a graded category's `pass@K` is 0.000 by construction, so plotting it would pin that axis at the origin for every model. For the same reason the cross-model ranking is on **macro `avg@K`**: on a `pass@K` ranking, a row whose `research` leaf succeeded was averaged against that structural 0 while a row whose leaf failed was averaged without it, so failing research paid better than completing it.

**`research`** is in the default set, and it is the odd one out in five ways worth knowing — it now configures each of these itself, so a default dispatch handles them without extra inputs:

- **It pins `sandbox_env: docker` and `runner_label: ubuntu-24.04-arm` for itself**, through `CATEGORY_MAP` in `unified_prep.py`, so one dispatch can run it beside categories that use the dispatch inputs. Upstream publishes DRBench's per-task images for arm64 only, and a runner's architecture only matters when the containers run *on* the runner — with the LangSmith sandbox they run off-runner and the label has no effect.
- **`network_mode = "public"`**, because DRBench's ground truth includes facts that exist only on the open web. It is also the only category granted `TAVILY_API_KEY`, which is what activates the agent's `web_search` tool.
- **Scores are continuous, and there are five of them.** Rather than pass/fail, each trial emits `insights_recall`, `distractor_recall`, `factuality`, and `report_quality`, plus a composite under `reward`. The composite is the **harmonic mean** of `insights_recall`, `factuality`, `report_quality`, and `1 − distractor_recall` — the paper's own aggregate (arXiv 2510.00172, Table 2), computed here because upstream's released code ships the four metrics but not the mean. Our only deviation is a 0.01 floor per component. Publish the components beside it regardless.
- **`research` is reported differently from the other categories.** Because its reward is graded, `pass@K` is 0 by construction (nothing scores a perfect 1.0) and is *suppressed* from its summary rather than shown as a result. It reports `avg@K` — the mean reward over expected trials — plus `macro_avg@K`, the mean of per-task means over the trials that actually ran. The two agree on a complete run, so a divergence means rollouts are missing. The **Overall pass@K** columns in the cross-model table still average every category, so a graded one drags them; read **Overall avg@K** instead.
- **Disk-bound, so it pins `concurrency: 1` for itself** while other categories keep the dispatch value. Each trial runs a full Nextcloud/Mattermost/Postgres/Roundcube stack from a ~1.22 GiB image on a runner with ~14 GB free, and builds a second image for the verifier (which installs upstream `drbench` and its document parsers). The agent stack is stopped before the verifier starts, so the two never run at once, but both images stay resident on disk.

See [`datasets/drbench-evals/README.md`](datasets/drbench-evals/README.md) for the runtime contract, the two credential regimes, and scoring detail.

The `autonomous`, `context`, and `research` categories run a deep-agents graph: by default the **bare** `create_deep_agent` — the SDK agent with no product scaffolding, which keeps the score a measure of the *model* rather than of a harness wrapped around it — with **`dcode`** (deep-agents-code, the full product agent) selectable as an option. The `conversation` category runs the τ³ runtime, which supplies a **user simulator** the agent must converse with rather than a static prompt.

## Task-selection philosophy

Four principles cut across all three categories and explain why the task sets look the way they do:

1. **Reuse credible external benchmarks.** Every category is sourced from an established, independently-authored benchmark (Harbor / Terminal-Bench, τ³-bench, Context-Bench). This keeps the eval honest but also doesn't preclude us from adding our own Harbor-style dataset.
2. **Curate small subsets by *measured* difficulty.** Where we take a subset, tasks are tiered by the **empirical pass rate of a strong reference model (Opus 4.8)** over multiple rollouts.
3. **Optimize for a discriminating spread with headroom.** A benchmark every model solves (or every model fails) ranks nothing. We deliberately keep the scarce *intermittent* tasks because they carry the most signal, and we weight toward hard tasks so the set doesn't saturate as models improve.
4. **Keep set sizes small enough to fit the CI budget while preserving signal.** Subsets are sized (~30 tasks/category) to fit GitHub's 6-hour per-job limit and the workflow's concurrency caps, run at 3 rollouts each.

## Category detail

### Autonomous — `harbor-index/harbor-index`

**What it measures.** Whether the agent can take a task to completion in a real, sandboxed environment — writing and running code, using the terminal, and manipulating files — graded by each task's own verification harness rather than by an LLM judge.

**Why this benchmark.** This is the flagship "can it actually do the job" axis. We run `harbor-index/harbor-index`, the curated autonomous-agent task index from the [Harbor](https://github.com/laude-institute/harbor) registry. It is the same sandboxed-verifiable-task family as [Terminal-Bench 2](https://github.com/laude-institute/terminal-bench-2) — the suite's original Harbor benchmark, spanning 90+ tasks across software engineering, biology, security, gaming, and more. Running through Harbor means each task ships its own environment image and grader, so a pass is objective and reproducible. (`terminal-bench/terminal-bench-2-1` is the closely-related sibling dataset selectable in the standalone [`harbor.yml`](../../.github/workflows/harbor.yml) workflow.)

**Why these tasks.** We run the benchmark's own published index as authored, rather than sub-selecting, so the score covers breadth and stays comparable to the wider Harbor/Terminal-Bench ecosystem. The tasks are cross-domain terminal / computer-use problems — in the Terminal-Bench family, software engineering, security, data and scientific computing, system administration, and more — each shipping its own environment and pass/fail verifier. So this category measures general "operate a computer to finish a real job" competence across domains, not a hand-picked slice.

### Conversation — `tau3-subset` (τ³-bench)

**What it measures.** Multi-turn customer-service-style dialogue: the agent must converse with a simulated user, call domain tools, and follow a written policy to resolve the user's issue.

**Why this benchmark.** [τ³-bench](https://github.com/sierra-research/tau2-bench) (the Harbor dataset `sierra-research/tau3-bench`; τ³ ships inside the `tau2-bench` repo) is a standard for tool-using conversational agents. The `conversation` category runs it through the `tau3` harness, whose user simulator (an OpenAI model, currently `gpt-5.2`) drives a live back-and-forth — this is the only category that scores *dialogue* rather than a single-shot task.

**How the harness works (tau3 = bare DA + MCP user sim).** `tau3` is not a different agent from the deep-agents categories: it is the same `create_deep_agent`. What changes is only what the graph is wired to. Where `bare` / `dcode` attach a local shell backend (filesystem and command tools), the `tau3` graph attaches the task environment's `tau3-runtime` **MCP** tools (`start_conversation`, `send_message_to_user`, `end_conversation`, plus the domain tools) that Harbor forwards from the sandbox into `configurable["mcp_servers"]`. The **simulated user lives on that MCP server, not in the agent**: the agent holds the conversation by calling `send_message_to_user`, which returns the user's next turn, and a system prompt tells it to converse via those tools rather than finish silently. So the conversation category is really bare deep-agent capability measured through an MCP-hosted user simulator, which is exactly why it depends on Harbor forwarding task-environment MCP servers — the pinned `harbor[langsmith] 0.20.0` does this, so no `harbor_package_override` is needed.

**Why these tasks.** We run a curated **30-task subset** ([`tau3_subset.py`](deepagents_evals/tau3_subset.py)) drawn from two τ³ domains that exercise different conversational skills:

- **`banking_knowledge` (24 tasks)** — the user asks a policy or eligibility question; the agent must retrieve the correct answer from the domain's knowledge base and state it. Measures grounded question-answering under a policy.
- **`telecom` (6 tasks)** — multi-step service-issue troubleshooting (APN settings, SIM-card PIN, airplane mode, overdue-bill suspension); the agent must diagnose a broken-service scenario and drive it to resolution with tools, across a live back-and-forth. Measures procedural, multi-turn problem-solving.

Within those domains the subset is a *difficulty probe* — a behavior spread across models — not leaderboard parity with full τ³-bench. Each task's tier is why it's included, and is the measured pass rate of `anthropic:claude-opus-4-8` over 3 rollouts at full agent timeout (easy = 3/3, medium = 1–2/3, hard = 0/3): floors any capable model should pass, an intermittent middle where models separate, and hard tasks for headroom. Opus finds most of this set hard — accepted, intentional headroom. The subset is a *living selection*: re-run and re-tier (updating each task's `justification`) as the reference model or task set changes.

### Context — `context-retrieval-evals` (Context-Bench)

**What it measures.** Extracting and reasoning over information spread across a multi-file corpus. Every task ships the **whole** 10-file corpus so the agent can't infer which files matter — it must retrieve, join, and aggregate to answer.

**Why this benchmark.** Derived from [Context-Bench](https://github.com/letta-ai/letta-evals) (Letta's `filesystem` `cloud` suite of synthetic person/vehicle/pet/account records, Apache-2.0). It isolates long-context retrieval and multi-hop joins — a capability the autonomous and conversation categories don't directly stress. Each task ships the entire 10-file corpus, so the agent can't shortcut to the relevant files; it has to search, join, and aggregate across them.

**Why these tasks.** The 30 tasks span eight query types over the same corpus, so the set measures a range of retrieval-and-reasoning operations rather than one:

- **`multi_hop_chain` · `multi_entity_comparison` (16 tasks)** — deep multi-file joins: follow a chain of relationships across files, or compare two entities each reached by its own lookup.
- **`aggregation` · `cross_file_counting` (5)** — sum balances or count records scattered across files.
- **`set_intersection` · `comparison_tiebreak` (5)** — find the entities satisfying several constraints at once, resolving ties.
- **`negation` · `temporal_reasoning` (4)** — exclude by a condition, or reason over dates.

The subset is a paired, six-rollout representative sample from the full 100-task Context-Bench cloud suite, calibrated on **gpt-5.6-terra** and **gpt-5.6-luna** with the **bare** `create_deep_agent` harness. In the [source run](https://github.com/langchain-ai/deepagents/actions/runs/29881672853), Terra scored 510/600 (85.0%) and Luna 552/600 (92.0%); this 30-task sample preserves that profile at 153/180 (85.0%) and 166/180 (92.2%). It also preserves source difficulty coverage: **2 easy · 10 medium · 18 hard**. These are original Context-Bench source strata, not post-hoc model tiers. The selection preserves aggregate measurement rather than targeting a cross-model leaderboard order; it is therefore appropriate for tracking the context capability without overstating a model-pair gap. The [dataset README](datasets/context-retrieval-evals/README.md) records each task's paired result and query type.

**Lite Context slice.** `profile=lite` uses a frozen 10-task subset: `cb-cloud-48`, `cb-cloud-1`, `cb-cloud-21`, `cb-cloud-49`, `cb-cloud-65`, `cb-cloud-69`, `cb-cloud-57`, `cb-cloud-9`, `cb-cloud-7`, and `cb-cloud-4`. It retains every query type and adds a second hard `multi_entity_comparison` and `multi_hop_chain`; its source mix is **1 easy · 3 medium · 6 hard**. The slice was selected from the completed [six-model, three-rollout full-30 run](https://github.com/langchain-ai/deepagents/actions/runs/29883830538): it preserves that run's observed ContextBench order (Sol > Luna > Sonnet > Terra > Opus > GLM), while excluding the sole task with an errored trial. This preserves the measured full-corpus profile for inexpensive monitoring; it does not target an external leaderboard order.

## Why this, not the pytest eval suite

The SDK also has a pytest eval suite ([`tests/evals/`](tests/evals/), catalogued in [`EVAL_CATALOG.md`](EVAL_CATALOG.md)). It measures a **different thing**, and the two are complementary:

- **The pytest evals measure specific agent *behaviors* in controlled scenarios.** Does the agent pick the right tool for an intent, prefer `edit` over a full rewrite, read and write files in parallel, recover from a truncated read, keep a todo list, use memory, summarize faithfully. Each eval asserts one narrow, known-correct behavior, grouped into capability areas (file operations, retrieval, tool use, memory, conversation, summarization, …). The signal is **diagnostic** — a failure tells you *which behavior* broke.
- **The unified evals measure end-to-end task *success* on external benchmarks.** Can the agent actually finish a sandboxed autonomous task, resolve a multi-turn support conversation, or answer a multi-hop question over a large corpus — scored `pass@k` / `avg@k` and rolled up into a **cross-model comparison**. The signal is **holistic and comparative** — how capable the agent is end-to-end, and how models rank against each other, on tasks authored outside this repo.

Reach for the unified evals when the question is *"how good is model X, versus Y, as a deep agent?"* — model selection, a capability scorecard, or tracking progress against the external Harbor / τ³ / Context-Bench ecosystem. Reach for the pytest suite when the question is *"does the SDK still do the right thing in this specific case?"* — catching behavioral regressions as the code changes. An agent can top the pytest behaviors and still stall on end-to-end tasks (or the reverse), which is exactly why both exist.

## Changing the battery

- **Category → benchmark + harness mapping, shard defaults:** `CATEGORY_MAP` and `DEFAULT_N_SHARDS` in [`unified_prep.py`](../../.github/scripts/evals/unified_prep.py).
- **Conversation subset + tiers:** [`deepagents_evals/tau3_subset.py`](deepagents_evals/tau3_subset.py) (re-run and update each `justification`).
- **Context subset + tiers:** [`datasets/context-retrieval-evals`](datasets/context-retrieval-evals/README.md) and its `calibration.json`.
- **Model catalog / presets:** [`.github/scripts/evals/models.py`](../../.github/scripts/evals/models.py) and [`MODEL_GROUPS.md`](MODEL_GROUPS.md).
