# Deep Agents Evals

End-to-end behavioral evaluation suite for the Deep Agents SDK. Each eval runs an agent against a real LLM, captures the full trajectory (tool calls, file mutations, final response), and scores it on correctness and efficiency.

See [`EVAL_CATALOG.md`](EVAL_CATALOG.md) for the full list of evals and categories.

The suite also includes [Harbor](https://github.com/laude-institute/harbor) integration for running sandboxed benchmarks like [Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2).

## Results

| Suite | CI | LangSmith |
|---|---|---|
| Evals | [evals.yml](https://github.com/langchain-ai/deepagents/actions/workflows/evals.yml) | [deepagents-evals](https://smith.langchain.com/public/d4245855-4e15-48dc-a39d-8631780a9aeb/d) |
| Harbor | [harbor.yml](https://github.com/langchain-ai/deepagents/actions/workflows/harbor.yml) | [deepagents-harbor](https://smith.langchain.com/public/e5f44462-4615-49ba-a0a1-194892dd5837/d) |

## Contributing

Architecture, writing new evals, category system, Harbor setup, and LangSmith integration are all documented in [CONTRIBUTING.md](CONTRIBUTING.md).
