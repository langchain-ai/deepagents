# Persistent Memory Assistant

A personal assistant that remembers facts about users across separate conversations — even when starting a completely new thread.

## What This Demonstrates

| Feature | How It's Used |
|---|---|
| `StoreBackend` | Stores `/memories/profile.md` in LangGraph's `BaseStore`, outside the conversation thread |
| `CompositeBackend` | Routes `/memories/` to `StoreBackend`, all other paths to ephemeral `StateBackend` |
| `MemoryMiddleware` | Automatically loads `/memories/profile.md` into the system prompt at the start of every conversation |
| Multi-user isolation | Each `user_id` gets its own namespace — Alice and Bob never see each other's memories |

### The Key Problem It Solves

By default, a Deep Agent's files live in `StateBackend`, scoped to a single conversation thread. Start a new thread and all files are gone.

`StoreBackend` solves this by storing files in LangGraph's `BaseStore`, which is thread-independent. Combine it with `CompositeBackend` to surgically route only the memory path to persistent storage, while keeping all other working files ephemeral.

```
/memories/profile.md  →  StoreBackend  →  persists across ALL sessions
/scratch/notes.txt    →  StateBackend  →  cleared when session ends
```

### How Memory Flows

```
Session 1 starts
  └─ MemoryMiddleware reads /memories/profile.md
       └─ CompositeBackend routes to StoreBackend
            └─ File not found → "(No memory loaded)" in system prompt

User: "I'm Alice, I prefer Python and async patterns"
  └─ Agent writes /memories/profile.md
       └─ CompositeBackend routes to StoreBackend
            └─ Saved in store under namespace ("user:alice", "memories")

Session 2 starts (new thread_id)
  └─ MemoryMiddleware reads /memories/profile.md
       └─ CompositeBackend routes to StoreBackend
            └─ File found → Alice's profile injected into system prompt

User: "What do you know about me?"
  └─ Agent answers from memory — no extra tool calls needed
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Run the built-in demo

The demo runs two sessions for the same user with a fresh store, showing that memory written in session 1 is available in session 2.

```bash
python assistant.py --demo
```

Expected output:
```
[Session 1 | thread: a3f1bc8d...]
User: Hi! I'm Alice. I'm a senior Python developer...
Assistant: Nice to meet you, Alice! I've noted your preferences...

[Session 2 | thread: 7e92cd4f...] (new conversation)
User: Hey, what do you remember about me?
Assistant: You're Alice, a senior Python developer. You're building a
FastAPI backend and prefer type hints, async/await patterns, and
concise bullet-point answers.
```

### Interactive use

```bash
# In-memory (resets on exit — good for quick testing)
python assistant.py --user alice "I'm a data scientist. I mainly use pandas and scikit-learn."
python assistant.py --user alice "What stack do I use?"

# SQLite (persists across terminal runs)
python assistant.py --store sqlite --user alice "I'm a data scientist. I mainly use pandas."
python assistant.py --store sqlite --user alice "What stack do I use?"  # new process, still remembers

# Different users have isolated memories
python assistant.py --store sqlite --user bob "I'm a Go developer building microservices"
python assistant.py --store sqlite --user alice "What do you know about me?"  # Alice's data, not Bob's
```

See [Persistent Storage](#persistent-storage) for SQLite and Postgres setup.

## Persistent Storage

The default `--store memory` resets when the process exits. Pass `--store sqlite`
or `--store postgres` to keep memory across separate terminal runs.

### SQLite (local development)

```bash
pip install langgraph-checkpoint-sqlite
# or: uv pip install -e ".[sqlite]"

python assistant.py --store sqlite --user alice "I prefer Python and FastAPI"
# memory is written to assistant_memory.db

python assistant.py --store sqlite --user alice "What do you know about me?"
# new process, same file — agent still knows Alice
```

Under the hood:

```python
import sqlite3
from langgraph.store.sqlite import SqliteStore

# isolation_level=None (autocommit) is required by SqliteStore — same as from_conn_string
conn = sqlite3.connect("assistant_memory.db", isolation_level=None)
store = SqliteStore(conn)
store.setup()  # creates schema on first run, no-op after that

agent = create_assistant(user_id="alice", store=store)
```

### PostgreSQL (production)

```bash
pip install langgraph-checkpoint-postgres
# or: uv pip install -e ".[postgres]"

export DATABASE_URL=postgresql://user:pass@localhost/mydb
python assistant.py --store postgres --user alice "I prefer Python"
```

Under the hood:

```python
import os
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string(os.environ["DATABASE_URL"]) as store:
    store.setup()
    agent = create_assistant(user_id="alice", store=store)
```

## Architecture

```python
store_backend = StoreBackend(
    store=store,
    # Each user gets a private namespace — no cross-user data leakage
    namespace=lambda rt: (f"user:{user_id}", "memories"),
)

backend = CompositeBackend(
    default=StateBackend(),           # ephemeral: session-scoped
    routes={"/memories/": store_backend},  # persistent: lives in BaseStore
)

agent = create_deep_agent(
    backend=backend,
    # MemoryMiddleware reads this file before every conversation turn.
    # If missing → "(No memory loaded)". If present → injected into system prompt.
    memory=["/memories/profile.md"],
)
```

## Extending This Example

**Multiple memory files:** Pass multiple paths to `memory` to load separate files for different concerns (preferences, project context, ongoing tasks).

```python
memory=[
    "/memories/profile.md",       # who the user is
    "/memories/projects.md",      # ongoing projects
    "/memories/preferences.md",   # style and tool preferences
]
```

**Scoped permissions:** Add `FilesystemPermission` rules so the agent can only write to its own memory directory and can't modify other paths.

```python
from deepagents import FilesystemPermission

permissions=[
    FilesystemPermission(operations=["write"], paths=["/memories/**"]),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]
```

**Multi-tenant deployment:** In a LangGraph Platform deployment, derive `user_id` from the authenticated request context instead of a CLI argument:

```python
namespace=lambda rt: (rt.server_info.user.identity, "memories")
```