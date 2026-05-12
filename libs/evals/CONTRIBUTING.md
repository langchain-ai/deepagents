# Contributing to Deep Agents Evals

Behavioral evaluation suite for the Deep Agents SDK. Runs agents end-to-end against a real LLM and asserts on the resulting trajectory (tool calls, final text, file mutations).

## Quick start

From `libs/evals/`:

```bash
# Install dependencies
uv sync

# Configure API keys
export ANTHROPIC_API_KEY="sk-ant-..."  # Required: For Claude model
export LANGSMITH_API_KEY="lsv2_..."    # Required: For tracing
export LANGSMITH_TRACING=true       # Required: Enable LangSmith tracing

# All evals
make evals MODEL=claude-opus-4-7

# Specific model via raw pytest invocation
LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals --model claude-sonnet-4-6-20250514

# Single test file
LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals/test_file_operations.py
```

Results are logged to [LangSmith](https://smith.langchain.com/) under the `deepagents-evals` test suite (under Experiments tab). Set `--evals-report-file <path>` (or `DEEPAGENTS_EVALS_REPORT_FILE`) to also write a JSON summary.

## Architecture

### Two-tier assertion model

Each eval uses a `TrajectoryScorer` with two assertion tiers:

- **Success assertions** (`.success(...)`) are correctness checks that **hard-fail** the test.
  - Examples: `final_text_contains`, `file_equals`, `llm_judge`
- **Efficiency assertions** (`.expect(...)`) are trajectory-shape expectations that are **logged but never fail**.
  - Examples: expected step count, expected tool calls.

```python
scorer = (
    TrajectoryScorer()
    .expect(agent_steps=2, tool_call_requests=1)
    .success(
        final_text_contains("three", case_insensitive=True),
    )
)
```

### Key modules

| File | Purpose |
|---|---|
| `tests/evals/utils.py` | Core framework: `AgentTrajectory`, assertion classes, `TrajectoryScorer`, `run_agent` entry point |
| `tests/evals/llm_judge.py` | LLM-as-judge `SuccessAssertion` — wraps [openevals](https://github.com/langchain-ai/openevals) to grade agent answers against human-readable criteria |
| `tests/evals/conftest.py` | pytest fixtures: `--model` CLI option, `model` / `model_name` fixtures, LangSmith metadata |
| `tests/evals/external_benchmarks.py` | Runner logic for curated external benchmarks (FRAMES, Nexus, BFCL v3) with state-comparison scoring |
| `tests/evals/memory_agent_bench/` | MemoryAgentBench (ICLR 2026) runner: configs, data loading, and evaluation utils |
| `tests/evals/pytest_reporter.py` | Custom pytest plugin: collects efficiency data and prints/writes a summary report |
| `tests/evals/fixtures/` | Static test data |
| `tests/evals/data/benchmark_samples/` | Curated case data for external benchmarks |
| `tests/evals/data/bfcl_apis/` | Stateful Python API implementations for BFCL v3 tool-calling evals |
| `tests/evals/tau2_airline/` | tau2-bench airline domain: task data, database state, policy, domain models, evaluation, and multi-turn runner (derived from [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT License) |

### Test suites

| File | Category | What it evaluates |
|---|---|---|
| `test_file_operations.py` | `file_operations`, `retrieval` | File tool usage (read/write/edit/ls), parallel reads & writes, grep/glob search, seeded file state |
| `test_tool_selection.py` | `tool_use` | Picking the right tool from intent (direct, indirect, multi-step) with independent mock tools |
| `test_tool_usage_relational.py` | `tool_use` | Multi-step tool chaining with dependent data lookups (user -> location -> weather) |
| `test_todos.py` | `tool_use` | Todo list tool usage for task planning |
| `test_external_benchmarks.py` | `retrieval`, `tool_use` | FRAMES (multi-hop retrieval), Nexus (nested function composition), BFCL v3 (multi-turn stateful tool calling) |
| `test_memory.py` | `memory` | Memory recall and behavior guidance from `AGENTS.md` files, preference persistence, composite backends |
| `test_memory_multiturn.py` | `memory` | Multi-turn memory: implicit preference extraction, explicit remember instructions, transient info filtering |
| `memory_agent_bench/test_memory_agent_bench.py` | `memory` | MemoryAgentBench (ICLR 2026): long-context memory recall and QA over chunked context |
| `test_followup_quality.py` | `conversation` | Followup question relevance for underspecified requests (LLM judge) |
| `tau2_airline/test_tau2_airline.py` | `conversation` | [tau2-bench](https://github.com/sierra-research/tau-bench) airline tasks: multi-turn agent-user conversations scored on DB state accuracy and communicate info |
| `test_summarization.py` | `summarization` | Summarization middleware triggers, post-summarization task continuation, history offload to filesystem |
| `test_hitl.py` | `unit_test` | Human-in-the-loop via `interrupt_on` approvals, subagent HITL, custom interrupt configs |
| `test_subagents.py` | `unit_test` | Subagent delegation behavior |
| `test_system_prompt.py` | `unit_test` | System prompt adherence |
| `test_skills.py` | `unit_test` | Skill discovery, reading, and application from `SKILL.md` files |

## Writing a new eval

1. Create a test function marked `@pytest.mark.langsmith`. The eval framework uses `langsmith.testing` to log inputs, outputs, and feedback (correctness scores, efficiency metrics) for every run — this data powers the report summary and cross-model comparisons. `conftest.py` aborts the suite if `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are not set.
2. Accept the `model: BaseChatModel` fixture.
3. Build the agent with `create_deep_agent(model=model, ...)`.
4. Call `run_agent(agent, model=model, query=..., scorer=...)`.
5. Use `.success()` for must-pass correctness checks and `.expect()` for soft efficiency targets.

```python
@pytest.mark.langsmith
def test_example(model: BaseChatModel) -> None:
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query="What is 2 + 2?",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1)
            .success(final_text_contains("4"))
        ),
    )
```

For semantic grading where substring matching is insufficient, use the LLM judge:

```python
from tests.evals.llm_judge import llm_judge

scorer = TrajectoryScorer().success(
    llm_judge(
        "The answer mentions the capital of France is Paris.",
        "The tone is conversational, not robotic.",
    )
)
```

## Report output

After a run, the reporter plugin prints a summary:

```
========== deepagents evals summary ==========
correctness: 0.85
step_ratio: 1.10
tool_call_ratio: 1.05
solve_rate: 0.0342
median_duration_s: 3.1200
```

- **correctness** — fraction of tests that passed all success assertions
- **step_ratio** — actual steps / expected steps (micro-averaged across tests with expectations)
- **tool_call_ratio** — actual tool calls / expected tool calls
- **solve_rate** — mean of `expected_steps / duration_s` for passing tests

## Eval categories

Every eval test is tagged with a category via `@pytest.mark.eval_category("name")`. Categories group tests by capability area and power per-category reporting in CI.

Categories and their human-readable labels are defined in [`deepagents_evals/categories.json`](deepagents_evals/categories.json) — the single source of truth consumed by the radar chart generator, the CI aggregate script, and unit tests.

### Filtering by category

Run only specific categories locally or in CI:

```bash
# Single category
uv run --group test pytest tests/evals --eval-category memory

# Multiple categories
uv run --group test pytest tests/evals --eval-category memory --eval-category tool_use
```

In the GitHub Actions workflow, pass a comma-separated list via the `eval_categories` input:

```text
eval_categories: "memory,tool_use,retrieval"
```

Omit to run all categories.

### CI concurrency

Eval jobs use per-provider concurrency groups. Two jobs hitting the same provider (e.g. both `openai`) queue — the second waits for the first to finish. Jobs on different providers run in parallel, so dispatching `frontier` (anthropic + google_genai + openai) alongside a solo `openrouter` run won't block either side.

### Per-category reporting

CI runs produce a per-category correctness table in the GitHub Actions step summary, plus a JSON summary artifact (`evals-summary`) for offline analysis.

### Radar charts

Full eval runs (3+ categories) generate a radar chart comparing model scores across categories, uploaded as the `radar-chart` artifact. The chart is skipped for narrow category-filtered runs where a radar would be meaningless.

```bash
# Install chart dependencies (matplotlib)
uv sync --extra charts

# Generate from CI summary
python scripts/generate_radar.py --summary evals_summary.json -o charts/radar.png

# Generate with toy data for experimentation
python scripts/generate_radar.py --toy -o charts/radar.png
```

#### Composite radar across multiple workflow runs

Each CI run charts the models from that single dispatch. To overlay results from several runs (e.g. a bake-off across providers, where each model was dispatched as its own run) onto one radar, use the `composite_radar.py` wrapper. It downloads the `evals-summary` artifact from each run, concatenates the per-run JSON arrays, and invokes `generate_radar.py` so every entry becomes a separate trace.

Requires the `gh` CLI authenticated against the target repo. Example:

```bash
cd libs/evals
uv run python scripts/composite_radar.py \
  25403850424 25403883357 25403894412 25403919855 25403953867 \
  -o /tmp/composite-radar.png \
  --title "Composite — <describe the cohort>"
```

Output: `composite-radar.png` (light) and `composite-radar-dark.png` (dark). Add `--individual-dir <path>` to also emit one PNG per model. Use `--repo owner/name` to pull from a fork.

Notes:

- Axes are auto-detected from the union of `category_scores` keys, ordered per `EVAL_CATEGORIES` (`unit_test` is excluded by design).
- All-zero-score models are dropped by default as likely infrastructure failures. To override, run `generate_radar.py` directly with `--keep-zero-scores` against the merged summary the wrapper leaves behind under `--workdir --keep-workdir`.
- Trace label = the `model` field of each summary entry. If two runs used the same model on different configs, edit the merged `combined_summary.json` (via `--workdir --keep-workdir`) before re-rendering so the legend stays unambiguous.
- The summary artifact is uploaded by the `📋 Aggregate evals` job; runs whose aggregate job didn't complete won't have one to download.

If you need to merge by hand (no `gh` CLI, or summaries from a non-CI source), the equivalent manual procedure is:

```bash
# 1. Pick a working dir and list the run IDs you want to combine.
mkdir -p /tmp/composite-radar && cd /tmp/composite-radar
RUNS=(25403850424 25403883357 25403894412 25403919855 25403953867)

# 2. Download the evals-summary artifact from each run.
for run in "${RUNS[@]}"; do
  mkdir -p "$run"
  gh run download "$run" -R langchain-ai/deepagents -n evals-summary -D "$run"
done

# 3. Concatenate. Each artifact is a JSON array; flatten them into one array.
python3 -c "
import json, pathlib, sys
combined = []
for r in sys.argv[1:]:
    combined.extend(json.loads(pathlib.Path(f'{r}/evals_summary.json').read_text()))
pathlib.Path('combined_summary.json').write_text(json.dumps(combined, indent=2))
" "${RUNS[@]}"

# 4. Render. Run from libs/evals so uv resolves the chart extra.
cd "$REPO_ROOT/libs/evals"
uv run python scripts/generate_radar.py \
  --summary /tmp/composite-radar/combined_summary.json \
  -o /tmp/composite-radar/composite-radar.png \
  --title "Composite — <describe the cohort>"
```

### Eval catalog

[`EVAL_CATALOG.md`](EVAL_CATALOG.md) is an auto-generated quick reference listing every eval grouped by category, with links to the source definition on GitHub and the local file path.

Regenerate after adding or removing evals:

```bash
make eval-catalog
```

A drift test (`tests/unit_tests/test_eval_catalog.py`) fails CI if the file is stale.

### Adding a new category

1. Add the category name and label to `deepagents_evals/categories.json` — add it to `categories` (all), and also to `radar_categories` if it measures model capability (not SDK plumbing)
2. Tag test(s) with `pytestmark = [pytest.mark.eval_category("your_category")]` for single-category files, or per-function `@pytest.mark.eval_category("your_category")` decorators for files with mixed categories
3. Add the category to `EXPECTED_CATEGORY_MODULES` in `tests/unit_tests/test_category_tagging.py`
4. Run `make test` — drift tests will catch any mismatch

## Multi-trial runs

When you need to tell signal from noise — "did this prompt change actually move correctness, or am I looking at run-to-run variance?" — run the suite N times under the same config and look at the spread.

### Why two eval workflows

| Workflow | Question it answers | Shape |
|---|---|---|
| `evals.yml` | How do *these models* compare on this dataset? | trials = 1, models = many (presets, fan-out across providers) |
| `evals_trials.yml` | Is *this model's* number stable under the same config? | trials = N, models = exactly one |

They are kept separate because folding trials into `evals.yml` would produce a *trials × models × providers* fan-out (rarely useful) and would require a second aggregate shape (per-trial stats vs. per-model comparison) in one workflow. The trials workflow also needs a per-trial GHA job so each trial gets its own 6h budget — a 5-trial sequential run on a slow model can take 10–15h wall time and cannot fit in a single job.

### Running locally

```bash
make evals-trials MODEL=openai:gpt-5.5 TRIALS=5
```

Forward extra flags through `TRIAL_ARGS`:

```bash
make evals-trials MODEL=openai:gpt-5.5 TRIALS=3 \
    TRIAL_ARGS="--openai-reasoning-effort medium --eval-category memory"
```

Trials run sequentially. Outputs land under `libs/evals/trial_runs/`:

- `evals_report_trial_NNN.json` — one per trial, same schema as the single-run report described in [Report output](#report-output)
- `trials_summary.json` — aggregate across trials (schema below)

Each trial creates its own LangSmith experiment.

### Running in CI

Dispatch the **📊 Evals - N Trials** workflow (`evals_trials.yml`).

| Input | Notes |
|---|---|
| `model` | Single spec, e.g. `openai:gpt-5.5`. No presets — trials are single-model only |
| `trials` | 1–20 (the local CLI accepts up to 50; the workflow caps lower since a runaway runner pool is harder to recover than a stuck terminal) |
| `parallel` | Off (default): trials run sequentially. On: all trials run concurrently |
| `eval_categories`, `eval_tiers`, `openai_reasoning_effort`, `openrouter_provider`, `openrouter_allow_fallbacks` | Same semantics as `evals.yml` (the OpenRouter pin accepts a comma-separated allowlist; `openrouter_allow_fallbacks` toggles strict pin vs. soft preference) |

The `parallel` toggle exists to trade time for API burst pressure. Sequential mode keeps the per-second call rate equivalent to a single eval run, so it is safe for any provider; parallel mode finishes ~N× faster but bursts every trial's traffic at once and should only be used when the provider can absorb the load.

**Job structure:**

1. `prep` — validates inputs and emits the trial matrix (`include` array of `{trial_index, artifact_key}`) plus a `max_parallel` value (1 for sequential, N for parallel).
2. `eval-trial` — matrix job that calls the existing `_eval.yml` reusable workflow once per trial. Each trial runs on its own GHA runner with its own 6h budget; `strategy.max-parallel` controls sequential vs. parallel. Each trial uploads its report under a unique `evals-report-trial-NNN-<slug>` artifact name.
3. `aggregate-trials` — runs even when individual trials fail. Downloads every trial's report artifact, runs `scripts/run_trials.py --aggregate-only`, uploads `trials-summary` as an artifact, and posts a markdown table to the workflow summary (overall metrics, per-category breakdown, per-trial rows). If fewer reports than the requested trial count made it through, the summary banners the gap loudly and the job exits non-zero — a green check on a partial sample size is a trap.

### `trials_summary.json` schema

```jsonc
{
  "n_trials": 5,
  "model": "openai:gpt-5.5",
  "sdk_version": "0.5.6",
  "metrics": {
    // One block per scalar metric
    "correctness":       {"n": 5, "mean": 0.49, "median": 0.49, "stdev": 0.012, "min": 0.47, "max": 0.51},
    "solve_rate":        {"n": 5, "mean": ..., ...},
    "step_ratio":        {"n": 5, "mean": ..., ...},
    "tool_call_ratio":   {"n": 5, "mean": ..., ...},
    "median_duration_s": {"n": 5, "mean": ..., ...}
  },
  "counts": {
    // Pass/fail/skip/total stats across trials
    "passed":  {"n": 5, "mean": 81.6, ...},
    "failed":  {...},
    "skipped": {...},
    "total":   {...}
  },
  "category_scores": {
    // Per-category correctness across trials
    "memory":   {"n": 5, "mean": 0.62, "stdev": 0.018, ...},
    "tool_use": {"n": 5, ...}
  },
  "trials": [
    // Per-trial records preserved in dispatch order
    {"trial_index": 1, "passed": 81, "correctness": 0.47, "category_scores": {...}, "experiment_urls": [...]},
    ...
  ]
}
```

`stdev` is `null` for n < 2 (sample stdev needs n ≥ 2). Metrics some trials reported as `null` (e.g. `solve_rate` when nothing passed) are silently dropped from that metric's stats — `n` reflects how many trials actually contributed, not the trial count. Non-`null` values that aren't numeric (a sign the upstream reporter shape changed) are excluded *with* a warning so the regression surfaces.

### Interpreting the spread

Trial stdev tells you what change sizes are real:

- **stdev ≪ candidate delta** → the change is signal; ship the conclusion.
- **stdev ≈ candidate delta** → run more trials, or treat the result as noise.

For context: a 2-task pass-count swing on a 170-test suite is Δcorrectness ≈ 0.012, which is comparable to typical single-trial stdev on this benchmark. Single-run deltas in that range should not be reported as model differences without trial backing.

## Harbor / Terminal Bench 2.0

### What is Harbor?

[Harbor](https://harborframework.com/) is an evaluation framework that simplifies running agents on challenging benchmarks. It provides:

- **Sandbox environments** (Docker, Modal, Daytona, E2B, etc.)
- **Automatic test execution** and verification
- **Reward scoring** (0.0 - 1.0 based on test pass rate)
- **Trajectory logging** in ATIF format [(Agent Trajectory Interchange Format)](https://harborframework.com/docs/trajectory-format)

### What is Terminal Bench 2.0?

[Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2) is an evaluation benchmark that measures agent capabilities across several domains, testing how well an agent operates using a computer environment, primarily via the terminal. The benchmark includes 90+ tasks across domains like software engineering, biology, security, gaming, and more.

**Example tasks:**

- `path-tracing`: Reverse-engineer C program from rendered image
- `chess-best-move`: Find optimal move using chess engine
- `git-multibranch`: Complex git operations with merge conflicts
- `sqlite-with-gcov`: Build SQLite with code coverage, analyze reports

### The Deep Agent architecture

The Deep Agent harness ships with design patterns validated as good defaults across agentic tasks:

1. **Detailed System Prompt**: Expansive, instructional prompts with tool guidance and examples
2. **Planning Middleware**: The `write_todos` tool helps the agent structure thinking and track progress
3. **Filesystem**: Provides `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` for context management
4. **SubAgents**: The `task` tool spawns specialized subagents for isolated work

### Setup

```bash
# Configure API keys - Choose one approach:

# Option 1: Use .env file (recommended for local development)
cp .env.example .env
# Edit .env and add your keys - they'll be automatically loaded

# Option 2: Export directly (useful for CI/CD or quick testing)
export ANTHROPIC_API_KEY="sk-ant-..."  # Required: For Claude model
export LANGSMITH_API_KEY="lsv2_..."    # Required: For tracing
export LANGSMITH_TRACING=true          # Required: Enable LangSmith tracing
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # Optional: Default shown
# export DAYTONA_API_KEY="..."  # Optional: Only if using --env daytona
```

### Running benchmarks

```bash
# Run via Docker (sequential, all tasks)
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker

# Run via Daytona (10 concurrent trials)
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env daytona
```

### Available environments

Harbor supports multiple sandbox environments. Use the `--env` flag to select:

- `docker` - Local Docker containers (good for testing)
- `daytona` - Daytona cloud sandboxes (requires `DAYTONA_API_KEY`)
- `modal` - Modal cloud compute
- `runloop` - Runloop sandboxes

Makefile shortcuts are available for common workflows:

- `make run-terminal-bench-docker` - Run on Docker (sequential)
- `make run-terminal-bench-daytona` - Run on Daytona (40 concurrent)
- `make run-terminal-bench-modal` - Run on Modal (4 concurrent)
- `make run-terminal-bench-runloop` - Run on Runloop (10 concurrent)

### LangSmith integration

LangSmith provides tracing and observability for agent runs. The workflow:

```txt
Deep Agents -> Harbor (evaluate) -> LangSmith (analyze) -> Improve -> Repeat
```

#### Step 1: Create dataset and experiment

```bash
# Create dataset from Harbor tasks
python scripts/harbor_langsmith.py create-dataset terminal-bench --version 2.0

# Create experiment session (outputs session ID and URL)
python scripts/harbor_langsmith.py create-experiment terminal-bench --name deepagents-baseline-v1
```

#### Step 2: Run benchmark with tracing

```bash
# Option 1: For experiments (enables side-by-side comparison in LangSmith)
export LANGSMITH_EXPERIMENT="deepagents-baseline-v1"
make run-terminal-bench-daytona  # 40 concurrent trials on Daytona

# Option 2: For development (simpler project view in LangSmith)
export LANGSMITH_PROJECT="deepagents-development"
make run-terminal-bench-daytona

# Option 3: Run harbor directly (-n = concurrency; add -l N to limit tasks)
export LANGSMITH_EXPERIMENT="deepagents-baseline-v1"
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env daytona
```

#### Step 3: Add feedback scores

After the benchmark completes, push reward scores to LangSmith for filtering and analysis:

```bash
python scripts/harbor_langsmith.py add-feedback jobs/terminal-bench/2025-12-02__16-25-40 \
  --project-name deepagents-baseline-v1
```

This matches trials to traces and adds `harbor_reward` feedback (0.0-1.0) from Harbor's test results.

### Analyzing results

LangSmith captures every LLM call, tool invocation, and performance metric. Combined with Harbor reward scores (added via Step 3), you can filter runs by performance and identify patterns in successful vs. failed runs.

#### Common failure patterns

| Pattern | Symptom | Potential Fix |
|---|---|---|
| **Poor Planning** | Agent jumps into coding without reading requirements | Add upfront planning requirement to prompt |
| **Incorrect Tool Usage** | Uses `bash cat` instead of `read_file` | Improve tool descriptions with examples |
| **No Incremental Testing** | Writes 200 lines, then tests once | Prompt to test after each logical unit |
| **Hallucinated Paths** | Reads files before checking existence | Add "always `ls` before read" rule |
| **Wrong Model** | Model fails on complex reasoning | Use more capable model for hard tasks |

#### Agent-assisted analysis

Use LangSmith's Insights Agent or your own agent to analyze trajectory data across runs. Task it with identifying common failure patterns, grouping errors by category, and suggesting prompt or tool improvements.

## Resources

- [Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangSmith documentation](https://docs.langchain.com/langsmith/home)
- [Harbor GitHub](https://github.com/laude-institute/harbor)
