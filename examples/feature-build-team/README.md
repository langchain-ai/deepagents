# Engineering Org

A three-level org of deep agents that builds a feature in parallel, connected by [async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents).

```text
lead
├── api_lead   → storage_engineer, routes_engineer   (/api/)
├── web_lead   → markup_engineer, client_engineer    (/web/)
└── qa_lead    → api_tester, flow_tester             (/tests/)
```

- **`lead`** writes `/PLAN.md` with the API contract, starts the three team leads at once, and relays their reports to you.
- **Team leads** split their part between two engineers, start both, and verify the result when they report back.
- **Engineers** build the files they're assigned and report `DONE:` when they're finished.

Every agent is its own graph on the same LangGraph server. Work flows down with `start_async_task`, changes flow down with `update_async_task`, and reports flow back up as task notifications that wake the idle parent. All agents share `workspace/`, so you can watch the app appear as they work.

> [!WARNING]
> The agents run shell commands on your machine through `LocalShellBackend`. Its working directory is `workspace/`, but commands aren't sandboxed. Only run it somewhere you're comfortable with that.

## Run

Create a `.env` file next to `langgraph.json` with these variables. `langgraph dev` loads it.

| Variable | Required | What it's for |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Calls the Gemini model every agent uses. |
| `DEMO_MODEL` | No | The model for every agent, as any `provider:model` string `init_chat_model` accepts. Defaults to `google_genai:gemini-4-argon`; another provider needs its own API key. |
| `LANGSMITH_API_KEY` | No | Sends traces to LangSmith. |
| `LANGSMITH_TRACING` | No | Set to `true` to turn tracing on. |

```bash
uv sync
uv run langgraph dev --no-browser --no-reload --n-jobs-per-worker 16
```

`--no-reload` keeps the server from restarting when agents write Python files, and `--n-jobs-per-worker 16` lets all ten agents run at once.

In a second terminal, start the viewer and open http://127.0.0.1:3000:

```bash
uv run uvicorn viewer:app --port 3000
```

Click **URL shortener** to start a run. While the engineers work, click **Steer: add expiry**: the lead updates the plan and redirects the running team leads, who pass the change on to their engineers instead of starting over.

## The viewer

The viewer reads everything from the local LangGraph server, so it needs no LangSmith account. Each card shows an agent's status, its latest tool call, and its latest report:

- **Working**: the agent is running.
- **Idle**: the agent is waiting, usually for its subagents to report back.
- **Done**: the agent reported `DONE:`.

Edges light up as messages move through the org: blue when a parent delegates, rose when it steers, and lime when a subagent reports back.

The viewer expects the LangGraph server at `http://127.0.0.1:2024`; set `LANGGRAPH_URL` to point it somewhere else.

## Serve what was built

```bash
cd workspace && uv run --with fastapi --with uvicorn uvicorn api.app:app --reload
```
