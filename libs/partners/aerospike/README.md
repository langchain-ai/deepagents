# deepagents-aerospike

Wire Aerospike as a LangGraph checkpoint store for deepagents agents — sub-millisecond state reads, per-record TTL expiry, and durable cross-session memory through existing LangGraph integration points with zero changes to core.

## Quick Start

1. **Start Aerospike server (local dev)**

   ```bash
   docker run --rm -p 3000:3000 aerospike/aerospike-server
   ```

2. **Install the package**

   ```bash
   cd /path/to/deepagents/libs/partners/aerospike
   uv sync --all-groups
   ```

3. **Use with deepagents**

   ```python
   from deepagents import create_deep_agent
   from deepagents_aerospike import create_aerospike_checkpointer

   checkpointer = create_aerospike_checkpointer(
       host="localhost",
       port=3000,
       namespace="myapp",
       ttl_minutes=1440,
   )

   agent = create_deep_agent(
       model="claude-sonnet-4-6",
       checkpointer=checkpointer,
   )

   agent.invoke(
       {"messages": [{"role": "user", "content": "Hello"}]},
       config={"configurable": {"thread_id": "user-123"}},
   )
   ```

## How It Works

- One-time connection: `create_aerospike_checkpointer` opens a single Aerospike client connection and hands it to `AerospikeSaver`. That session is reused for every `put`, `get_tuple`, and list operation, so checkpoint reads stay fast and do not open a new socket per call.
- Automatic indexing: `AerospikeSaver` creates `thread_id` secondary indexes on its checkpoint sets the first time it is instantiated. If another instance is created against the same namespace, it finds the indexes already exist and proceeds normally — index setup is idempotent.
- TTL-based expiry: if you pass `ttl_minutes`, checkpoint records expire automatically after that many minutes. This is useful for session agents that do not need infinite memory without manual cleanup.

## Advanced Usage

Users needing advanced options — TLS, custom serialization, custom set names, a custom serde, or `refresh_on_read` sliding TTL — should import `AerospikeSaver` directly from `langgraph.checkpoint.aerospike`. The `create_aerospike_checkpointer` factory exposes only the common case.

## Testing

Run unit tests (no network required):

```bash
make test
```

Run integration tests (requires a running Aerospike server):

```bash
make integration_test
```

## License

MIT
