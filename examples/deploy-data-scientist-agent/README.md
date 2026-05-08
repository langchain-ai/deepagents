# deploy-data-scientist-agent

A data scientist agent deployed with `deepagents deploy`. It analyzes one or more data files in a LangSmith sandbox, runs Python for reproducible calculations, creates charts, and writes stakeholder-ready reports.

This example reads datasets uploaded through the bundled frontend, so it works without MCP, database credentials, bundled data files, or third-party integrations.

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

The agent is deployed using the config in `deepagents.toml`. The `[sandbox]` section provisions a LangSmith Python 3.12 sandbox so the agent can run data analysis code safely.

This example also enables the bundled frontend with `[frontend]`. After deploy, open:

```text
https://<your-deployment-url>/app
```

The frontend uses `[auth] provider = "anonymous"` so it can be tried without configuring Supabase or Clerk. Use anonymous mode only for demos or private deployments because anyone with the deployment URL can access the app.

## Uploaded Data

Upload text files, CSVs, JSON, markdown files, and images from the bundled frontend. Uploaded files are written directly to the thread's sandbox under:

```text
/uploads/
```

Uploads are the only supported data input path for this example. They are available only when a sandbox is configured and are not persisted to Store or Context Hub in this first version.

## What To Try

Once deployed, open the agent in LangSmith and send prompts like:

- Upload a CSV in the frontend, then ask: `"Analyze the uploaded file and create a chart."`
- Upload multiple CSVs, then ask: `"Inspect the uploaded files and tell me which files can be joined."`
- `"Create charts from the uploaded data and explain the trends."`
- `"Use Python to validate the uploaded data and explain your method."`
- `"Write an executive report with findings, limitations, and recommendations."`

The agent follows the workflow in `AGENTS.md`: inspect data, plan, run Python, validate results, create artifacts, and report findings with caveats.

## Structure

```text
deploy-data-scientist-agent/
├── AGENTS.md                  # Agent instructions and analysis workflow
├── deepagents.toml            # Deploy config (model, sandbox)
├── mcp.json                   # Empty MCP config for this example
└── skills/
    ├── exploratory-data-analysis/
    ├── report-writing/
    ├── sample-data/
    └── visualization/
```

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [LangSmith sandbox docs](https://docs.langchain.com/deepagents/sandbox)
