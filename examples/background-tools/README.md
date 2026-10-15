# Background Tools

A Deep Agent whose slow tools run in the background. Each tool is a function decorated with `@background_tool`, which compiles it into its own functional-API graph. The tool graphs are registered in `langgraph.json` next to the agent, so the Agent Server runs them as separate runs on their own threads.

When the agent calls a background tool, the call starts a run of the tool's graph and returns a task ID immediately. The agent keeps helping the user while the tool runs, and is told when it finishes.

## How it works

```python
@background_tool
def run_benchmark(steps: int = 10) -> dict[str, float]:
    """Run the performance benchmark and return timing statistics."""
    ...

agent = create_deep_agent(tools=[run_benchmark])
```

```json
{
  "graphs": {
    "agent": "./agent.py:agent",
    "run_benchmark": "./agent.py:run_benchmark"
  }
}
```

The graph name in `langgraph.json` must match the tool's `graph_id`, which defaults to the function name. Pass `@background_tool(graph_id=..., url=...)` to point at a graph registered under another name or on another deployment.

`create_deep_agent` routes background tools into `AsyncToolMiddleware`, which gives the model:

| Tool | Purpose |
| --- | --- |
| `research_report`, `run_benchmark` | One launcher per background tool, with the function's name and arguments. Returns a `task_id`. |
| `check_background_task` | Read a task's status and result. `block=true` waits up to `timeout` seconds. |
| `cancel_background_task` | Stop a running task. |
| `list_background_tasks` | List tracked tasks with live statuses. |

Before each model call, the middleware polls unfinished tasks and adds a `<system-reminder>` message for any that finished, so the model doesn't need to poll. This mirrors background tasks in Claude Code (`run_in_background`, `TaskOutput` with `block`/`timeout`, `TaskStop`, completion reminders) and Codex's unified exec polling (`write_stdin` with a yield window).

Under the hood, a launch calls `threads.create` and `runs.create` on the Agent Server; checks use `runs.get` and read the result from `threads.get`; cancellation uses `runs.cancel`. Without a `url`, these go over the in-process transport of the deployment the agent runs on.

## Run it

```bash
cd examples/background-tools
uv sync
cp .env.example .env  # add ANTHROPIC_API_KEY
uv run langgraph dev
```

Chat with the `agent` graph in LangSmith Studio, or run the scripted conversation in another terminal:

```bash
uv run python client.py
```
