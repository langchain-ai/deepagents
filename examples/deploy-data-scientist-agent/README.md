# deploy-data-scientist-agent

A data scientist agent deployed with `deepagents deploy`. It analyzes one or more data files in a LangSmith sandbox, runs Python for reproducible calculations, creates charts, and writes stakeholder-ready reports.

This example reads datasets from a bundled `skills/data/` directory so it works without MCP, database credentials, or third-party integrations. It includes a synthetic SaaS metrics CSV as demo data, but the agent instructions are generalized for arbitrary files placed in that data directory.

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

## Data Directory

Put data files for the agent under:

```text
skills/data/
```

At runtime, deploy seeds skill files under `/memories/skills/` and syncs them into the sandbox. The agent should inspect:

```text
/memories/skills/data/
```

The example includes a synthetic SaaS metrics CSV at:

```text
skills/data/sample_saas_metrics.csv
```

and at runtime:

```text
/memories/skills/data/sample_saas_metrics.csv
```

The source generator is in `scripts/generate_sample_data.py`. It writes both the local `data/sample_saas_metrics.csv` copy and the deploy-bundled `skills/data/sample_saas_metrics.csv` copy. You can add additional files under `skills/data/`; the agent will list the directory, inspect schemas, and decide whether to analyze files separately, join them, or compare them.

## What To Try

Once deployed, open the agent in LangSmith and send prompts like:

- `"Analyze the bundled SaaS metrics dataset and summarize the most important revenue trends."`
- `"Create charts showing monthly revenue by segment and churn rate by plan."`
- `"Is support response time related to NPS? Use Python and explain your method."`
- `"Inspect every file in the data folder and tell me which files can be joined."`
- `"Write an executive report with findings, limitations, and recommendations."`

The agent follows the workflow in `AGENTS.md`: inspect data, plan, run Python, validate results, create artifacts, and report findings with caveats.

## Structure

```text
deploy-data-scientist-agent/
├── AGENTS.md                  # Agent instructions and analysis workflow
├── deepagents.toml            # Deploy config (model, sandbox)
├── mcp.json                   # Empty MCP config for this example
├── data/
│   └── sample_saas_metrics.csv
├── scripts/
│   └── generate_sample_data.py
└── skills/
    ├── data/
    │   └── sample_saas_metrics.csv
    ├── exploratory-data-analysis/
    ├── report-writing/
    ├── sample-data/
    └── visualization/
```

## Regenerate The Sample Data

```bash
python3 scripts/generate_sample_data.py
```

The script uses a fixed random seed so the generated CSV is deterministic.

## Resources

- [deepagents deploy docs](https://docs.langchain.com/deepagents/deploy)
- [LangSmith sandbox docs](https://docs.langchain.com/deepagents/sandbox)
