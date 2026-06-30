# Harbor eval run playbook

How to run terminal-bench (and tau3) evaluations through Harbor via the
`📊 Evals - Harbor` GitHub Actions workflow (`.github/workflows/harbor.yml`),
and how to read the results.

> Scored / leaderboard-style numbers should go through this workflow, not a local
> `harbor run`. Local `harbor run --env docker` is for debugging a single task
> (see [Debugging one task locally](#debugging-one-task-locally)).

---

## 1. Mental model

```
workflow_dispatch
  └─ prep job        → builds a matrix of {model} × {shard}
       └─ harbor job (one per model×shard, parallel)
            └─ per task × rollout = one trial = one LangSmith trace
```

- **Model matrix** comes from `models` (a named group) or `models_override`
  (explicit comma-separated slugs). Groups are defined in
  [`../MODEL_GROUPS.md`](../MODEL_GROUPS.md).
- **Sandbox**: `langsmith` (managed remote sandboxes; default) or `docker`
  (built on the runner).
- **Agent**: `dcode` (the Deep Agents Code CLI harness) or `bare`
  (`create_deep_agent`), or `tau3` for tau3-bench.
- Each `{model, shard}` runs as its own job on its own runner at the per-job
  `concurrency`.

---

## 2. Dispatch a run

**GitHub UI:** Actions → "📊 Evals - Harbor" → Run workflow → fill inputs.

**CLI:**

```bash
gh workflow run harbor.yml --ref main \
  -f models_override="fireworks:accounts/fireworks/models/glm-5p2" \
  -f sandbox_env=langsmith \
  -f agent_impl=dcode \
  -f dataset="terminal-bench/terminal-bench-2-1" \
  -f rollouts_per_task=5 \
  -f concurrency=8

# find the run + watch it
gh run list -R langchain-ai/deepagents -w harbor.yml -L 3
gh run watch <run-id> -R langchain-ai/deepagents
```

`--ref` may be a branch or tag — useful for testing an in-progress change.

---

## 3. Inputs reference

| Input | Default | What it does |
|---|---|---|
| `dataset` | `terminal-bench/terminal-bench-2` | Dataset dropdown (tb-2, tb-2-1, tau3-bench). |
| `dataset_override` | _(empty)_ | Arbitrary `owner/dataset` ref; wins over the dropdown. |
| `models` | _(empty)_ | Named model group (`all`, `frontier`, `fireworks`, a single slug, …). See `MODEL_GROUPS.md`. |
| `models_override` | _(empty)_ | Explicit comma-separated slugs; wins over `models`. |
| `sandbox_env` | `langsmith` | `langsmith` (managed) or `docker` (on-runner). |
| `agent_impl` | `dcode` | `dcode`, `bare`, or `tau3`. |
| `n_tasks` | `0` | Max tasks to run (`0` = all). Caps the selected set. |
| `include_tasks` | _(empty)_ | Space-separated task-name globs (e.g. `terminal-bench/kv-store-grpc`). Empty = all. |
| `rollouts_per_task` | `1` | Attempts per task. Use **5** for a scored run. |
| `concurrency` | `1` | Parallel sandbox slots per job. |
| `agent_timeout_multiplier` | `1.0` | Scales the agent **execution** timeout only (setup/env-build budgets unaffected). Use `<1` to cap per-rollout wall-clock cheaply; keep at **`1.0`** for scored runs. |
| `disable_verification` | `false` | Skip the task verifier (tests). For setup/concurrency stress tests where pass/fail isn't the goal. |
| `n_shards` | `1` | Split the dataset's tasks across N parallel shard jobs per model. `1` = single job over all tasks. >1 spreads orchestration load; **diagnostics-only** (see below). |
| `harbor_package_override` | _(empty)_ | Install an arbitrary Harbor build (e.g. `harbor[langsmith] @ git+https://github.com/owner/harbor.git@ref`) instead of the pinned version, to test an unreleased Harbor. Installed with `--reinstall --refresh` so re-running the same ref picks up a fresh wheel. |

---

## 4. Recipes

### Scored single-model run
```bash
gh workflow run harbor.yml --ref main \
  -f models_override="fireworks:accounts/fireworks/models/glm-5p2" \
  -f dataset="terminal-bench/terminal-bench-2-1" \
  -f rollouts_per_task=5 -f concurrency=8 \
  -f agent_timeout_multiplier=1.0 -f n_shards=1
```
Keep `agent_timeout_multiplier=1.0` and `n_shards=1` for numbers you intend to report.

### Targeted task subset (fast iteration / A-B on specific tasks)
```bash
  -f n_tasks=0 \
  -f include_tasks="terminal-bench/kv-store-grpc terminal-bench/configure-git-webserver" \
  -f rollouts_per_task=2 -f concurrency=4
```

### Sharded run (large datasets)
```bash
  -f n_shards=12 -f concurrency=6
```
Each shard runs `tasks where index % 12 == shard_index` as its own job.
**Caveat:** sharding fragments a run into N experiments/traces, so it's for fast
stress/fix runs — **scored runs use `n_shards=1`** to keep one experiment per model.

### Concurrency / setup stress test (cheap)
```bash
  -f disable_verification=true -f agent_timeout_multiplier=0.1 \
  -f rollouts_per_task=1 -f concurrency=16
```
Caps each rollout's execution wall-clock and skips the verifier — exercises sandbox
setup + concurrency without paying for full task solves.

### Test an unreleased Harbor build
```bash
  -f harbor_package_override="harbor[langsmith] @ git+https://github.com/laude-institute/harbor.git@main"
```

---

## 5. Read the results

### Pass/fail — from the artifacts (authoritative)
```bash
gh run download <run-id> -R langchain-ai/deepagents -D ./artifacts
# artifacts: harbor-N/<timestamp>/<trial>/result.json
```
Per trial, `result.json` holds:
- `verifier_result.rewards.reward` → `1.0` (pass) / `0.0` (fail)
- `exception_info.exception_type` → `AgentTimeoutError`, `NonZeroAgentExitCodeError`,
  `ResourceCreationError`, `ApiRateLimitError`, …
- `agent_result.metadata.answer_written` → whether the agent produced output
- `verifier/test-stdout.txt` (sibling file) → the grader's actual stdout + the
  failing assertion — the real "why" behind a `reward: 0.0`.

> **Gotcha:** do **not** read pass/fail from a LangSmith trace's root run `status` —
> it is `"success"` even when the task failed. The reward lives only in the artifact.

### Trajectories — LangSmith
Each run creates one experiment per model (named
`deepagents-harbor-dcode-<model>-<jobid8>`); match it to the GH run by model +
start time. A sharded run creates one experiment **per shard**. Within an
experiment, each trial is a trace whose `thread_id` is the trial name
(`<task>__<suffix>`).

---

## 6. Debugging one task locally

For root-causing a single task (not for scored numbers), run it in Docker and keep
the container to inspect it:

```bash
cd libs/evals
make stage-harbor-local-deps   # rsync local deepagents/deepagents-code into the agent
uv run harbor run \
  --agent langgraph \
  --agent-kwarg project_path=deepagents_harbor/langgraph_project \
  --agent-kwarg config=langgraph.json --agent-kwarg graph=deepagent \
  --dataset terminal-bench/terminal-bench-2-1 -i terminal-bench/<task> \
  --model <model> -n 1 --env docker --ek keep_containers=true \
  --jobs-dir /tmp/harbor-debug
# then: docker exec into the kept container to inspect /opt/harbor-langgraph-venv etc.
```

---

## 7. Cost & correctness tips

- **`rollouts_per_task=5`**, **`agent_timeout_multiplier=1.0`**, and **`n_shards=1`** for
  scored runs; the timeout multiplier and sharding are stress/diagnostics levers.
- Size **`concurrency`** to the deployment's limits — over-subscribing a model
  endpoint inflates per-call latency (queueing) and can push borderline tasks over
  their wall-clock budget.
- Use **`include_tasks`** + low `rollouts` for quick iteration; reserve full
  dataset runs for milestones.
