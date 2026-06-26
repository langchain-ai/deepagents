# Vercel Sandbox in production

The [`langchain-vercel-sandbox`](../../libs/partners/vercel/) backend is
deliberately thin: you hand it a `vercel.sandbox.Sandbox` and it exposes
`execute` plus filesystem tools to your Deep Agent. **How** that sandbox is
created, secured, and reused across a conversation is left to you.

This example fills that gap with three patterns from running agent-generated
code in production:

| Pattern | File | Why |
|---|---|---|
| Pre-baked snapshots | `bake_snapshot.py` | Cold start drops from ~30-60s to a few seconds — the dependency stack ships in the image. |
| Deny-by-default network policy + credential brokering | `network_policy.py` | Treat the sandbox as hostile: block private/metadata subnets, allow only what's needed, and inject auth at the firewall so the sandbox process never holds your token. |
| Per-conversation sandbox lifecycle | `agent.py` (`get_backend`) | Each thread reuses one warm sandbox across turns and recreates it before Vercel's lifetime ceiling, instead of cold-booting every turn. |

## Quick start

### Prerequisites

- Python 3.11+
- An Anthropic API key
- A Vercel account with Sandbox access (token, team id, project id)

### Install

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/examples/vercel-sandbox-production
uv venv --python 3.11 && source .venv/bin/activate
uv sync
cp .env.example .env   # then fill in your keys
```

### (Optional) bake a snapshot

```bash
python bake_snapshot.py
# copy the printed id into .env as VERCEL_SANDBOX_SNAPSHOT_ID
```

Without a snapshot the example installs `pandas` + `matplotlib` on first boot;
with one it boots from the pre-built image.

### Run

```bash
python agent.py "Plot [3, 1, 4, 1, 5, 9, 2, 6] as a bar chart and save it to /tmp/out.png"
```

## How it fits together

```
create_deep_agent(backend=get_backend)        # agent.py
        │  get_backend is a BackendFactory: (ToolRuntime) -> VercelSandbox
        ▼
get_backend(runtime)                           # agent.py
        │  reuse a warm sandbox per thread_id; recreate before the lifetime ceiling
        ▼
Sandbox.create(source=snapshot, network_policy=...)   # agent.py + network_policy.py
        │  fast cold start + deny-by-default egress + brokered auth
        ▼
VercelSandbox(sandbox=...)                      # langchain-vercel-sandbox
        │  execute() + ls/read/write/edit/glob/grep
        ▼
Deep Agent runs untrusted code safely
```

### A note on the two timeouts

They are different units and mean different things:

- `Sandbox.create(timeout=...)` — **milliseconds**, the sandbox's wall-clock
  lifetime. Vercel caps this (max 45 min); we recreate at 25 min in
  `get_backend` so a turn never lands on a sandbox the platform is about to
  reap.
- `VercelSandbox(sandbox=..., timeout=...)` — **seconds**, the default ceiling
  for a single `execute()` command.

### Credential brokering

Set `APP_API_HOST` and `APP_API_TOKEN` in `.env` to see brokering in action.
`network_policy.py` allow-lists the host with a `transform` rule that injects
`Authorization: Bearer <token>` on egress. The sandbox can call your API
authenticated, but `os.environ` inside the sandbox contains no token — even full
RCE in the sandbox has nothing to exfiltrate.

## Going further

Patterns intentionally left out to keep this example focused, but worth adding
for a full production deployment:

- Persist generated artifacts to object storage via signed upload URLs, so file
  bytes never round-trip through the model context.
- Back the warm-sandbox cache with a shared store (Redis) for multi-worker
  servers, and tag sandboxes with org/user/thread ids for observability.
- Human-in-the-loop approval before the agent runs sensitive commands.

## Resources

- [Deep Agents sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- [`langchain-vercel-sandbox`](../../libs/partners/vercel/)
- [Vercel Sandbox docs](https://vercel.com/docs/vercel-sandbox)
