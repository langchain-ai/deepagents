# Deep Agents Evals

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

# All evals (default model)
make evals

# Specific model
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
