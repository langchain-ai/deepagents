# deploy-coding-agent

An autonomous coding agent deployed with `deepagents deploy`. Given a task description, it plans, implements, tests, and commits changes inside a LangSmith sandbox with full shell access.

## Prerequisites

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude model access |
| `LANGSMITH_API_KEY` | Required for deploy and the LangSmith sandbox |

Copy `.env.example` to `.env` and fill in both keys.

## Deploy

```bash
deepagents deploy
```

The agent is deployed using the config in `agent.json`.

## What to try

Once deployed, open the agent in LangSmith and send it tasks like:

- `"Add a function that reverses a string and write a test for it"`
- `"Find all TODO comments in the repo and create a summary"`
- `"Refactor the main module to use dataclasses"`

The agent follows a Plan → Implement → Review → Deliver workflow defined in `AGENTS.md`.

## Structure

```
deploy-coding-agent/
├── AGENTS.md                  # Agent instructions and workflow
├── agent.json                 # Deploy config (name, model)
└── skills/
    ├── code-review/           # Code review skill with lint helper
    ├── coding-prefs/          # Coding style preferences
    └── planning/              # Task planning skill
```

> **MCP servers:** This example previously used `mcp.json` to wire in the LangChain docs MCP server. MCP servers are now workspace-level resources. Register them once with `deepagents mcp-servers add --url <url>` and reference them in a `tools.json` file.

## Query via SDK

```python
from langgraph_sdk import get_client

client = get_client(url="https://<your-deployment-url>")
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"], "agent",
    input={"messages": [{"role": "user", "content": "Add a hello_world function and test it"}]},
    stream_mode="messages",
):
    print(chunk.data, end="", flush=True)
```

Find your deployment URL in LangSmith under **Deployments**. See the [LangGraph SDK docs](https://langchain-ai.github.io/langgraph/concepts/sdk/) for more.

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [LangSmith sandbox docs](https://docs.langchain.com/deepagents/sandbox)
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
