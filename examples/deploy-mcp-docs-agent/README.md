# deploy-mcp-docs-agent

A documentation research agent deployed with `deepagents deploy`. It answers developer questions about LangChain, LangGraph, and Deep Agents by searching the live docs via MCP before relying on general knowledge.

## Prerequisites

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude model access |
| `LANGSMITH_API_KEY` | Required for deploy |

## Deploy

```bash
deepagents deploy
```

The `mcp.json` wires in the LangChain docs MCP server at `https://docs.langchain.com/mcp`. No additional setup is needed — the agent discovers and uses the docs tools automatically.

## What to try

Once deployed, open the agent in LangSmith and ask it questions like:

- `"How do I configure memory in Deep Agents?"`
- `"What's the difference between sync and async subagents?"`
- `"Show me how to add an MCP server to deepagents.toml"`
- `"What models are supported for deploy?"`

The agent always searches the docs first and cites the page it found the answer on.

## Query via SDK

```python
from langgraph_sdk import get_client

client = get_client(url="https://<your-deployment-url>")
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"], "agent",
    input={"messages": [{"role": "user", "content": "How do I add an MCP server to deepagents.toml?"}]},
    stream_mode="messages",
):
    print(chunk.data, end="", flush=True)
```

Find your deployment URL in LangSmith under **Deployments**. See the [LangGraph SDK docs](https://langchain-ai.github.io/langgraph/concepts/sdk/) for more.

## Structure

```
deploy-mcp-docs-agent/
├── AGENTS.md          # Agent instructions and answer format
├── deepagents.toml    # Deploy config (model)
└── mcp.json           # LangChain docs MCP server
```

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [MCP server docs](https://docs.langchain.com/deepagents/mcp)
