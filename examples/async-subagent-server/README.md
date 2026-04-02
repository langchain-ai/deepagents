# Async Subagent Server

A self-hosted [Agent Protocol](https://github.com/langchain-ai/agent-protocol) server that exposes a DeepAgents researcher as an async subagent. Use this as a starting point for hosting your own agent on any infrastructure and connecting it to a DeepAgents supervisor.

The example includes both sides of the pattern:

- **`server.py`** — the FastAPI server your subagent runs on
- **`supervisor.py`** — an interactive REPL showing how to connect to it

## Prerequisites

- `ANTHROPIC_API_KEY` — required
- `TAVILY_API_KEY` — optional; stub search is used if not set

## Quickstart

**1. Install dependencies:**

```bash
cd examples/async-subagent-server
uv sync
```

**2. Set up your environment:**

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY (and optionally TAVILY_API_KEY)
```

**3. Start the server:**

```bash
uv run uvicorn server:app --port 2024
```

**4. In another terminal, start the supervisor:**

```bash
cd examples/async-subagent-server
ANTHROPIC_API_KEY=... uv run python supervisor.py
```

Try these prompts:

```
> research the latest developments in quantum computing
> check status of <task-id>
> update <task-id> to focus on commercial applications only
> cancel <task-id>
> list all tasks
```

## Implemented endpoints

These are the Agent Protocol endpoints the DeepAgents async subagent middleware calls (via the LangGraph SDK):

| Endpoint | Purpose |
| -------------------------------------------- | -------------------------------- |
| `POST /threads` | Create a thread for a new task |
| `POST /threads/{thread_id}/runs` | Start or interrupt+restart a run |
| `GET /threads/{thread_id}/runs/{run_id}` | Poll run status |
| `GET /threads/{thread_id}` | Fetch thread state (`values.messages`) |
| `POST /threads/{thread_id}/runs/{run_id}/cancel` | Cancel a run |
| `GET /ok` | Health check |

## Swap in your own agent

Replace the `create_deep_agent` call in `server.py` with your own agent. The Agent Protocol layer stays the same regardless of what the agent does.

```python
_agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5"),
    system_prompt="You are a ...",
    tools=[your_tool],
)
```

## ⚠️ For demonstration purposes only

This example is intended to illustrate the self-hosted async subagent pattern. It does not feature authentication, rate limiting, or other features required for production use.
