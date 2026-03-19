# Deep Agents Evals

Behavioral evaluation suite for the Deep Agents SDK. Test run agents end-to-end against a real LLM and assert on the resulting trajectory (tool calls, final text, file mutations).

## Running

From `libs/evals/`:

```bash
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

- ✅ **Success assertions** (`.success(...)`) are correctness checks that **hard-fail** the test.
  - Examples: `final_text_contains`, `file_equals`, `llm_judge`
- 📈 **Efficiency assertions** (`.expect(...)`) are trajectory-shape expectations that are **logged but never fail**.
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
| `utils.py` | Core framework: `AgentTrajectory`, assertion classes, `TrajectoryScorer`, `run_agent` entry point |
| `llm_judge.py` | LLM-as-judge `SuccessAssertion` — wraps [openevals](https://github.com/langchain-ai/openevals) to grade agent answers against human-readable criteria |
| `conftest.py` | pytest fixtures: `--model` CLI option, `model` / `model_name` fixtures, LangSmith metadata |
| `external_benchmarks.py` | Runner logic for curated external benchmarks (FRAMES, Nexus, BFCL v3) with state-comparison scoring |
| `memory_agent_bench/` | MemoryAgentBench (ICLR 2026) runner: configs, data loading, and evaluation utils |
| `pytest_reporter.py` | Custom pytest plugin: collects efficiency data and prints/writes a summary report |
| `fixtures/` | Static test data |
| `data/benchmark_samples/` | Curated case data for external benchmarks |
| `data/bfcl_apis/` | Stateful Python API implementations for BFCL v3 tool-calling evals |
| `tau2_airline/` | τ²-bench airline domain: task data, database state, policy, domain models, evaluation, and multi-turn runner (derived from [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT License) |

### Test suites

| File | What it evaluates |
|---|---|
| `test_file_operations.py` | File tool usage (read/write/edit/ls/grep/glob), parallel reads & writes, seeded file state |
| `test_skills.py` | Skill discovery, reading, and application from `SKILL.md` files |
| `test_hitl.py` | Human-in-the-loop via `interrupt_on` approvals, subagent HITL, custom interrupt configs |
| `test_memory.py` | Memory recall and behavior guidance from `AGENTS.md` files, preference persistence, composite backends |
| `test_memory_multiturn.py` | Multi-turn memory: implicit preference extraction, explicit remember instructions, transient info filtering |
| `test_summarization.py` | Summarization middleware triggers, post-summarization task continuation, history offload to filesystem |
| `test_subagents.py` | Subagent delegation behavior |
| `test_system_prompt.py` | System prompt adherence |
| `test_tool_usage_relational.py` | Multi-step tool chaining with dependent data lookups (user -> location -> weather) |
| `test_tool_selection.py` | Picking the right tool from intent (direct, indirect, multi-step) with independent mock tools |
| `test_followup_quality.py` | Followup question relevance for underspecified requests (LLM judge) |
| `test_external_benchmarks.py` | Curated hard cases from FRAMES (multi-hop retrieval), Nexus (nested function composition), and BFCL v3 (multi-turn stateful tool calling) |
| `memory_agent_bench/test_memory_agent_bench.py` | MemoryAgentBench (ICLR 2026): long-context memory recall and QA over chunked context |
| `tau2_airline/test_tau2_airline.py` | [τ²-bench](https://github.com/sierra-research/tau-bench) airline tasks: multi-turn agent-user conversations scored on DB state accuracy and communicate info |

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
