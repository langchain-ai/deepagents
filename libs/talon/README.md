# DeepAgents Talon

DeepAgents Talon is the local runtime host for long-running Deep Agents. It owns
the process lifecycle for channel adapters, cron schedulers, and the agent
runtime in a single event loop.

This package is intentionally skeletal while the concrete channel, scheduler,
and backend implementations land in follow-up tickets. The public surface here
defines the integration protocols and host behavior those implementations plug
into.

## Development

```bash
uv sync --group test
uv run --group test pytest tests/
uv run deepagents-talon
```
