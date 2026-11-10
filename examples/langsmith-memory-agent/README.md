# LangSmith memory agent

A small Deep Agent that uses every kind of memory LangSmith can back. Each kind is stored where it belongs:

| Memory | What the agent sees | Storage | Scope |
|---|---|---|---|
| **Short-term** | The conversation and scratch files such as `/scratch/plan.md` | Checkpointer (thread state) | One thread |
| **Semantic** | `/memories/user/profile.md`, loaded into every prompt | LangGraph `BaseStore` (via `StoreBackend`) | One user, all threads |
| **Episodic** | `search_past_conversations` tool | LangGraph `BaseStore` (raw `put` / `search`) | One user, all threads |
| **Procedural** | `/memories/agent/AGENTS.md`, loaded into every prompt | [LangSmith Context Hub](https://docs.langchain.com/langsmith/use-the-context-hub), one commit per edit | Every user |
| **Skills** | `/memories/agent/skills/` | LangSmith Context Hub, read-only to the agent | Every user |

On LangSmith Deployment (and `langgraph dev`), the checkpointer and the store are provided by the server. You don't configure a database. Context Hub is a versioned file store in LangSmith, so shared memory can be reviewed, tagged, promoted, and edited by people outside engineering.

## How it works

Everything is in [`agent.py`](agent.py). The agent uses a `CompositeBackend` to route each path prefix to a different storage layer:

```python
CompositeBackend(
    default=StateBackend(),  # scratch files: thread-scoped
    routes={
        "/memories/user/": StoreBackend(namespace=lambda rt: ("memory-agent", rt.context.user_id, "files")),
        "/memories/agent/": ContextHubBackend("-/langsmith-memory-agent"),
    },
)
```

- **`memory=[...]`** loads `/memories/agent/AGENTS.md` and `/memories/user/profile.md` into the system prompt at the start of each thread (hot memory). The agent updates them with its built-in `write_file` and `edit_file` tools.
- **`skills=[...]`** points at `/memories/agent/skills/` in Context Hub. A `FilesystemPermission` deny rule keeps the agent from rewriting curated skills.
- **`EpisodeRecorder`** middleware writes each finished turn to the store under `("memory-agent", <user_id>, "episodes")`. The `search_past_conversations` tool calls `store.search(...)` on that namespace. The full transcripts stay in the checkpointer.
- **`make_agent()`** is a graph factory. Agent Server calls it for each run, so each run pulls the latest Context Hub commit, including edits made in the LangSmith UI.

The user comes from the run's `context` (`{"user_id": "alice"}`), so you can act as several users from one API key. In production, derive the namespace from the authenticated user (`rt.server_info.user.identity`) with [custom auth](https://docs.langchain.com/langsmith/custom-auth), so callers can't read each other's memory. See [`deploy-content-writer`](../deploy-content-writer/) for an example.

## Setup

```bash
cd examples/langsmith-memory-agent
uv sync
cp .env.example .env  # add ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

Seed the Context Hub repo with the starter `AGENTS.md` and the `incident-summary` skill from [`context_hub/`](context_hub/):

```bash
uv run python seed_context_hub.py
```

The script skips a repo that already exists, so it never overwrites learned memory. Pass `--overwrite` to reset the seeded files.

## Run it

Start a local Agent Server, which includes a checkpointer and a store:

```bash
uv run langgraph dev
```

In a second terminal, run the scripted walkthrough:

```bash
uv run python demo.py
```

It plays two users across three threads and prints each memory tool call:

| Step | User / thread | What it shows |
|---|---|---|
| 1 | Alice / thread 1 | Alice's profile is written to the store |
| 2 | Alice / thread 1 | A team convention is committed to Context Hub |
| 3 | Alice / thread 1 | A checklist is drafted in `/scratch/` (thread-only) |
| 4 | Alice / thread 2 | New thread: profile loaded, past episode found, scratch file gone |
| 5 | Bob / thread 3 | Bob sees none of Alice's memory |
| 6 | Bob / thread 3 | Bob's incident summary follows the skill and Alice's team convention |

At the end, it prints each user's store items and the latest `AGENTS.md` from Context Hub.

To chat instead, open the Studio link printed by `langgraph dev`, and set `user_id` in the run context.

## See it in LangSmith

- **Context Hub**: open **Context** in LangSmith and select `langsmith-memory-agent`. Each memory edit is a commit. Edit `AGENTS.md` in the UI, and the next new thread uses your change without a redeploy.
- **Traces**: every memory read and write appears as a tool call in the trace, so you can audit what the agent saved and why.
- **Store**: browse the store with the SDK (`client.store.search_items(("memory-agent", "alice", "files"))`) or from Studio.

## Semantic search over episodes

By default, `search_past_conversations` returns the five most recent episodes. To rank them by meaning, add an index to the store in `langgraph.json`, and install the embedding provider (for example, `uv add langchain-openai`):

```json
"store": {
  "index": {
    "embed": "openai:text-embedding-3-small",
    "dims": 1536,
    "fields": ["user_message", "assistant_message"]
  }
}
```

The tool code stays the same: `store.search(namespace, query=...)` returns results ranked by similarity once an index exists.

## Deploy

```bash
uv run langgraph deploy
```

The deployment gets a managed Postgres checkpointer and store. Set `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` as deployment secrets. Then run the walkthrough against it:

```bash
uv run python demo.py --url https://<your-deployment-url>
```

## Hardening shared memory

`/memories/agent/` is read and written by every user, so one user can influence what the agent tells another. Before you put this pattern in front of many people:

- Keep personal details in user memory. The instructions tell the agent this, but nothing enforces it.
- Require human approval for shared-memory writes with an interrupt rule:

  ```python
  FilesystemPermission(operations=["write"], paths=["/memories/agent/**"], mode="interrupt")
  ```

- Use Context Hub tags to pin production to a reviewed commit, and let the agent learn on a staging copy.

## Structure

```text
langsmith-memory-agent/
├── agent.py              # Graph factory, memory routing, episode recorder, search tool
├── context_hub/          # Seed content for the Context Hub repo
│   ├── AGENTS.md
│   └── skills/incident-summary/SKILL.md
├── seed_context_hub.py   # Pushes context_hub/ to LangSmith
├── demo.py               # Scripted multi-user walkthrough
├── langgraph.json        # Agent Server config
└── tests/                # Network-free tests with a fake model and fake Context Hub
```

Run the tests with:

```bash
uv run pytest
```

## Resources

- [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory)
- [Context Hub concepts](https://docs.langchain.com/langsmith/context-engineering-concepts)
- [Manage contexts with the SDK](https://docs.langchain.com/langsmith/manage-contexts-sdk)
- [LangGraph persistence: checkpointers and stores](https://docs.langchain.com/oss/python/langgraph/persistence)
