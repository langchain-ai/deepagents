# tests/evals/

Eval test suites for the Deep Agents SDK. See the [root README](../../README.md) for architecture, assertion model, how to write evals, and running instructions.

## Directory layout

- `conftest.py` — pytest fixtures (`--model`, `model` / `model_name`, LangSmith metadata)
- `utils.py` — core framework: `AgentTrajectory`, assertion classes, `TrajectoryScorer`, `run_agent`
- `llm_judge.py` — LLM-as-judge assertion via [openevals](https://github.com/langchain-ai/openevals)
- `pytest_reporter.py` — custom pytest plugin for efficiency summary reports
- `fixtures/` — static test data
- `data/` — benchmark samples and BFCL API implementations
- `memory_agent_bench/` — MemoryAgentBench (ICLR 2026) runner
- `tau2_airline/` — tau2-bench airline domain (vendored from [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT License; see root `AGENTS.md` for formatting exclusions)
