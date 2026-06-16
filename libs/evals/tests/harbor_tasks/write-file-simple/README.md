# write-file-simple (Harbor task)

A **prototype** showing how a behavioral/efficiency eval from the pytest suite
can run as a containerized `harbor run` task while still producing the same
metrics — correctness *and* trajectory shape (step / tool-call counts, ratios).

It is the containerized counterpart of
[`tests/evals/test_file_operations.py::test_write_file_simple`](../../evals/test_file_operations.py).

## How it works

| Piece | Role |
|---|---|
| `instruction.md` | The prompt given to the agent. |
| `environment/Dockerfile` | `python:3.12-slim`, `WORKDIR /app`. |
| `tests/test.sh` → `tests/score.py` | Verifier. Reads the ATIF trajectory at `/logs/agent/trajectory.json` (written by `DeepAgentsWrapper`) and the filesystem end-state `/app/name.txt`, then writes a **multi-key** `/logs/verifier/reward.json`. |
| `task.toml` `[min_reward]` | Gates the trial on `correctness` (the `.success()` tier). Other keys are diagnostics (the `.expect()` tier). |

The reward dict carries every metric the pytest `TrajectoryScorer` checks:

```json
{
  "reward": 1.0,
  "correctness": 1.0,
  "file_name_contains": 1.0,
  "final_text_contains": 1.0,
  "wrote_name_file": 1.0,
  "agent_steps": 2.0,
  "tool_call_requests": 1.0,
  "step_ratio": 1.0,
  "tool_call_ratio": 1.0
}
```

`scripts/harbor_langsmith.py add-feedback` fans each key out to LangSmith as
`harbor_<key>` feedback (see `deepagents_harbor/langsmith.py`).

## Run it

```bash
cd libs/evals
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --path tests/harbor_tasks/write-file-simple \
  --model anthropic:claude-sonnet-4-6 \
  -n 1 --jobs-dir jobs/write-file-simple --env docker \
  --agent-kwarg use_cli_agent=false
```

> **Status:** the scoring logic (`compute_rewards`) is unit-tested in
> `tests/unit_tests/test_harbor_task_reward.py`. The end-to-end `harbor run`
> above requires Docker + an API key and has **not** been run in CI yet — it is
> the next validation step for this prototype.
