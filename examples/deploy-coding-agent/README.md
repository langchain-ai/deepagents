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

The agent is deployed using the config in `deepagents.toml`. The `[sandbox]` section provisions a LangSmith coding sandbox with a Python 3.12 image so the agent can run code safely.

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
├── deepagents.toml            # Deploy config (model, sandbox)
├── deepagents.assistant-scope.toml  # Assistant-scoped config variant
├── mcp.json                   # MCP server config
└── skills/
    ├── code-review/           # Code review skill with lint helper
    ├── coding-prefs/          # Coding style preferences
    └── planning/              # Task planning skill
```

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [LangSmith sandbox docs](https://docs.langchain.com/deepagents/sandbox)
