# deploy-gtm-agent

A go-to-market strategy agent deployed with `deepagents deploy`. Given a product or feature, it coordinates a **sync** market-researcher subagent and an **async** content-writer subagent to produce a full GTM plan with supporting marketing materials.

This example demonstrates the sync/async subagent pattern: market research blocks on results before strategy is written, while content creation runs in the background and is integrated when ready.

## Prerequisites

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Model access (gpt-5.4-nano) |
| `LANGSMITH_API_KEY` | Required for deploy |

Copy `.env` and fill in your keys.

## Deploy

```bash
deepagents deploy
```

The subagents defined under `subagents/` are automatically discovered and wired in at deploy time.

## What to try

Once deployed, open the agent in LangSmith and send it prompts like:

- `"We're launching a new Python SDK for AI agents next month — build me a GTM plan"`
- `"Help us position our vector database product against Pinecone and Weaviate"`
- `"We're targeting mid-market engineering teams — what channels should we prioritize?"`

The agent will kick off market research, synthesize a strategy, and produce content briefs in parallel.

## Structure

```
deploy-gtm-agent/
├── AGENTS.md              # Supervisor agent instructions
├── deepagents.toml        # Deploy config (model)
├── mcp.json               # MCP server config
├── skills/
│   └── competitor-analysis/   # Competitor analysis skill
└── subagents/
    └── market-researcher/     # Sync subagent for market research
        ├── AGENTS.md
        ├── deepagents.toml
        └── skills/
            └── analyze-market/
```

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [Subagents docs](https://docs.langchain.com/deepagents/subagents)
