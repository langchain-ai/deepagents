# Pluggable Memory Backends

This module provides pluggable backends for long-term agent memory storage.

## Overview

Long-term memory allows agents to persist information across sessions. The memory system uses a simple protocol (`MemoryBackend`) that can be implemented with different storage solutions.

## Protocol

The `MemoryBackend` protocol defines three simple methods:

```python
class MemoryBackend(Protocol):
    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a memory item by key."""
        
    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store or update a memory item."""
        
    def ls(self) -> list[str]:
        """List all memory keys."""
```

Keys are file paths (e.g., `/agent.md`, `/notes.txt`). Values are dictionaries with:
- `content`: list of strings (file lines)
- `created_at`: ISO 8601 timestamp
- `modified_at`: ISO 8601 timestamp

## Built-in Backends

### FilesystemBackend

Stores memory as JSON files on disk.

```python
from deepagents import create_deep_agent
from deepagents.memory import FilesystemBackend

backend = FilesystemBackend(
    base_path="~/.my-app/memory",
    agent_id="my-agent"  # Optional, for multi-agent isolation
)

agent = create_deep_agent(
    memory_backend=backend,
    use_longterm_memory=True,
)
```

**Features:**
- Simple file-based storage
- Human-readable JSON format
- Directory structure mirrors key paths
- Multi-agent isolation via subdirectories

**Storage format:**
```
~/.my-app/memory/
└── my-agent/          # agent_id subdirectory
    ├── agent.md.json
    └── notes.md.json
```

### SQLiteBackend

Stores memory in an SQLite database.

```python
from deepagents.memory import SQLiteBackend

backend = SQLiteBackend(
    db_path="~/.my-app/memory.db",
    agent_id="my-agent"  # Optional, for multi-agent isolation
)

agent = create_deep_agent(
    memory_backend=backend,
    use_longterm_memory=True,
)
```

**Features:**
- Embedded database (no server needed)
- ACID transactions
- Efficient querying
- Multiple agents can share one database file
- Multi-agent isolation via `agent_id` column

### StoreBackend

Adapter for LangGraph's `BaseStore` (for backward compatibility).

```python
from langgraph.store.memory import InMemoryStore
from deepagents.memory import StoreBackend

store = InMemoryStore()
backend = StoreBackend(store, agent_id="my-agent")

agent = create_deep_agent(
    memory_backend=backend,
    use_longterm_memory=True,
)
```

Or use the `store` parameter directly (legacy):

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_deep_agent(
    store=store,  # Automatically wrapped in StoreBackend
    use_longterm_memory=True,
)
```

## Multi-Agent Isolation

All backends support multi-agent isolation via the `agent_id` parameter:

```python
# Each agent gets its own isolated memory space
backend_agent1 = FilesystemBackend(base_path="~/.app/memory", agent_id="agent1")
backend_agent2 = FilesystemBackend(base_path="~/.app/memory", agent_id="agent2")

agent1 = create_deep_agent(memory_backend=backend_agent1, use_longterm_memory=True)
agent2 = create_deep_agent(memory_backend=backend_agent2, use_longterm_memory=True)

# agent1 and agent2 can't see each other's memories
```

## Custom Backends

Implement the `MemoryBackend` protocol for custom storage solutions:

```python
from typing import Any

class RedisBackend:
    """Store memory in Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", agent_id: str | None = None):
        import redis
        self.client = redis.from_url(redis_url)
        self.agent_id = agent_id or "default"
        self.index_key = f"{self.agent_id}:__index__"
    
    def _make_key(self, key: str) -> str:
        return f"{self.agent_id}:{key}"
    
    def get(self, key: str) -> dict[str, Any] | None:
        import json
        redis_key = self._make_key(key)
        value = self.client.get(redis_key)
        return json.loads(value) if value else None
    
    def put(self, key: str, value: dict[str, Any]) -> None:
        import json
        redis_key = self._make_key(key)
        self.client.set(redis_key, json.dumps(value))
        self.client.sadd(self.index_key, key)
    
    def ls(self) -> list[str]:
        keys = self.client.smembers(self.index_key)
        return sorted([k.decode() for k in keys])

# Usage
backend = RedisBackend(agent_id="my-agent")
agent = create_deep_agent(memory_backend=backend, use_longterm_memory=True)
```

## How Agents Use Memory

Agents access long-term memory via the `/memories/` path prefix:

```python
# Agent can read its memory
agent.run("read /memories/agent.md")

# Agent can write to memory
agent.run("write to /memories/notes.txt: Important information")

# Agent can list memory files
agent.run("ls /memories/")

# Agent can edit memory
agent.run("edit /memories/agent.md to add this information")
```

The special file `/memories/agent.md` is loaded into the agent's system prompt automatically.

## Backend Comparison

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **FilesystemBackend** | Development, single machine | Simple, human-readable | No ACID, file I/O overhead |
| **SQLiteBackend** | Production, embedded apps | ACID, efficient queries | Single file, not distributed |
| **StoreBackend** | LangGraph integration | Backward compatible | Requires LangGraph store |
| **Custom (e.g., Redis)** | Distributed systems | Shared across machines | Requires external service |

## Migration Guide

### From `store` parameter to `memory_backend`

**Before:**
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
agent = create_deep_agent(store=store, use_longterm_memory=True)
```

**After:**
```python
from deepagents.memory import FilesystemBackend

backend = FilesystemBackend(base_path="~/.my-app/memory")
agent = create_deep_agent(memory_backend=backend, use_longterm_memory=True)
```

**Note:** The old `store` parameter still works for backward compatibility!

## Design Decisions

- **No delete operation**: Agents can't delete from long-term memory (safety feature). Cleanup must be done externally.
- **Simple protocol**: Just 3 methods keep implementations simple and focused.
- **Keys are paths**: File-like interface is intuitive for agents.
- **Backend handles isolation**: Multi-tenancy logic lives in backends, not the core system.
