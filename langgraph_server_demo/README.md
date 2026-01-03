# LangGraph Server Demo for Deepagents

This demonstrates running your deepagents with LangGraph Server.

## Quick Start

### 1. Install dependencies with uv

```bash
# Install with uv (recommended)
uv sync

# Or install globally
uv pip install langgraph-cli httpx
```

### 2. Setup environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the server

**Development mode** (with auto-reload and LangGraph Studio):
```bash
# With uv
uv run langgraph dev

# Or if you installed globally
langgraph dev
```

This will:
- Start the server on http://localhost:2024
- Open LangGraph Studio at http://localhost:2024/studio
- Auto-reload on code changes

**Production mode**:
```bash
uv run langgraph up
```

## Using the API

### Create a thread and send a message

```bash
# Create a new thread
curl -X POST http://localhost:2024/threads

# Send a message (replace THREAD_ID with the ID from above)
curl -X POST http://localhost:2024/threads/THREAD_ID/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "deepagent",
    "input": {
      "messages": [{"role": "user", "content": "Hello! List the files in the current directory"}]
    }
  }'
```

### Stream responses

```bash
curl -X POST http://localhost:2024/threads/THREAD_ID/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "deepagent",
    "input": {
      "messages": [{"role": "user", "content": "What files are in this directory?"}]
    }
  }'
```

### Get thread history

```bash
curl http://localhost:2024/threads/THREAD_ID/state
```

## What You Get

1. **REST API** - Full HTTP API for your agent
2. **LangGraph Studio** - Visual debugging UI at http://localhost:2024/studio
3. **Thread persistence** - Conversations saved in SQLite (upgradable to Postgres)
4. **Streaming** - Real-time response streaming
5. **Multi-user** - Handle concurrent requests

## Architecture

```
┌─────────────┐
│   Client    │ (Web, Mobile, CLI)
│  (HTTP/WS)  │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│  LangGraph Server   │ (Port 2024)
│  - REST API         │
│  - Streaming        │
│  - Thread Mgmt      │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│   create_deep_agent │
│   - Planning        │
│   - File ops        │
│   - Subagents       │
│   - Todos           │
└─────────────────────┘
```

## Benefits vs CLI

| Feature | CLI | LangGraph Server |
|---------|-----|------------------|
| Single user | ✓ | ✓ |
| Multi-user | ✗ | ✓ |
| Web/Mobile UI | ✗ | ✓ |
| Visual debugging | ✗ | ✓ (Studio) |
| API access | ✗ | ✓ (REST) |
| Streaming | ✗ | ✓ (SSE/WebSocket) |
| Thread persistence | ✓ | ✓ |
| Production ready | Partial | ✓ |

## Next Steps

1. **Try LangGraph Studio** - Visual debugging is amazing!
2. **Build a web UI** - Connect a frontend to the API
3. **Deploy to production** - Use LangGraph Cloud or Docker
4. **Add authentication** - Implement auth middleware
5. **Scale horizontally** - Add more server instances

## Production Deployment

For production, you'd want to:

1. **Use PostgreSQL** instead of SQLite for checkpointer
2. **Add authentication** - API keys, JWT, etc.
3. **Deploy to LangGraph Cloud** or self-host with Docker
4. **Configure CORS** for web clients
5. **Add rate limiting** and monitoring

See: https://langchain-ai.github.io/langgraph/cloud/
