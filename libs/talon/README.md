# Deep Agents Talon

Deep Agents Talon is the local runtime host for long-running Deep Agents. It owns the process lifecycle for channel adapters, cron schedulers, and the agent runtime in a single event loop.

> **Experimental:** Talon is an experimental runtime and is subject to change or removal at any time.

Talon currently includes:

- A host process with graceful shutdown, per-conversation serialization, and `/stop` cancellation.
- A generic channel protocol plus a WhatsApp adapter backed by a loopback Node bridge.
- A persistent cron scheduler with agent-facing cron tool helpers.
- MCP tool loading from the assistant manifest directory.
- Optional LangSmith tracing for each channel or cron-triggered run.

## Quickstart

```bash
uv sync --group test
AGENT_ASSISTANT_ID=local AGENT_MODEL=openai:gpt-5.2 uv run deepagents-talon --once
```

If `AGENT_MODEL` is unset, Talon starts with the echo runtime. This is useful for checking host lifecycle and channel wiring without provider credentials.

Assistant state lives under `~/.deepagents/<assistant_id>/` by default. The host creates restrictive state directories for the materialized agent manifest, channel sessions, and cron jobs. The default local execution workspace is `/workspace`; set `DEEPAGENTS_TALON_WORKSPACE` to use a different directory.

## Fleet Exports

Talon can host an operator-unzipped LangSmith Fleet export through the `fleet-deepagents-export` library:

```bash
unzip path/to/fleet-export.zip -d ./fleet

DEEPAGENTS_TALON_FLEET_DIR=./fleet \
AGENT_ASSISTANT_ID=fleet-local \
uv run deepagents-talon --once
```

In Fleet mode, Talon uses the model from `fleet/config.json` unless `DEEPAGENTS_TALON_MODEL` or `AGENT_MODEL` is set. The Fleet loader resolves MCP registry references through LangSmith, so provide the required `LANGSMITH_API_KEY`, `LANGSMITH_TENANT_ID`, `LANGSMITH_ORGANIZATION_ID`, and when needed `LANGSMITH_USER_ID`, `BUILTIN_MCP_URL`, `LANGSMITH_HOST_URL`, and `HOST_LANGCHAIN_API_URL`. Locally-authored agents without `DEEPAGENTS_TALON_FLEET_DIR` continue to load from the assistant manifest directory and Talon's plain MCP config discovery.

OAuth-backed Fleet MCP tools must be authorized once from an interactive shell before starting a headless host. Run the host in `--once` mode with the same Fleet directory and LangSmith environment you will use in production, complete the browser authorization if prompted, then start the long-running host:

```bash
DEEPAGENTS_TALON_FLEET_DIR=./fleet \
AGENT_ASSISTANT_ID=fleet-local \
uv run deepagents-talon --once
```

During a long-running Fleet session, Talon treats a 401/403 from an MCP tool as an expired OAuth credential signal. It reloads the Fleet components once, which re-mints tokens and rebuilds MCP connections, then retries the failed graph invocation once. If authorization still fails, Talon returns a structured `mcp_auth_failed` error instead of looping.

## WhatsApp

The WhatsApp channel uses a local Node bridge packaged with this library. The Python adapter talks to the bridge over loopback only.

```bash
cd deepagents_talon/channels/whatsapp_bridge
npm install
cd ../../..

DEEPAGENTS_TALON_WHATSAPP_ENABLED=true \
DEEPAGENTS_TALON_WHATSAPP_START_BRIDGE=true \
AGENT_ASSISTANT_ID=whatsapp-local \
AGENT_MODEL=openai:gpt-5.2 \
uv run deepagents-talon --whatsapp
```

The bridge prints a QR code during pairing. By default, inbound exposure is `self`, so only messages from the paired account trigger the agent. Configure `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` with `DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or `DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS` to allow specific chats. Outbound WhatsApp messages include a `deepagents bot` header by default so self-message conversations clearly distinguish agent replies from operator messages. Set `DEEPAGENTS_TALON_WHATSAPP_BOT_HEADER` to customize that label. Markdown image/video references in assistant replies may attach files only when they are relative paths inside `DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR`, or inside `DEEPAGENTS_TALON_WORKSPACE` when no outbound media directory is configured.

Inbound voice transcription is opt-in:

```bash
DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED=true
```

When enabled without `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL`, Talon uses the same local default as the original WhatsApp example: `nvidia/parakeet-tdt-0.6b-v3` through Transformers, with ffmpeg converting inbound audio to 16 kHz mono WAV first. Set `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE=cuda` to use a GPU. The legacy example variables `SPEECH_ENABLED` and `SPEECH_DEVICE` are also accepted. Setting `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL` to a non-Parakeet model keeps the existing OpenAI SDK transcription path.

`open` exposure allows arbitrary WhatsApp senders to trigger the agent while it runs with the operator's model credentials, channel credentials, MCP tool access, and local-host access when the local execution backend is active. Enabling it requires explicit acknowledgement:

```bash
DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open
DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK=allow-arbitrary-senders
```

See `../../examples/talon-whatsapp/` for a runnable Docker Compose topology and `.env` reference.

## Tracing

LangSmith tracing is opt-in. Set both values before starting the host:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deepagents-talon
```

When enabled, Talon wraps each agent run in a LangSmith tracing context with assistant id, conversation id, trigger metadata, and source message metadata.

## MCP Tools

Talon loads MCP servers from one config file. It checks `DEEPAGENTS_TALON_MCP_CONFIG`, then `MCP_CONFIG`, then `~/.deepagents/<assistant_id>/agent/tools.json`, then `~/.deepagents/.mcp.json`. Add Talon-local servers by editing `tools.json` directly:

```json
{
  "mcpServers": {
    "linear": {
      "type": "http",
      "url": "https://mcp.example/mcp"
    }
  }
}
```

Run `deepagents-talon mcp config` to print the resolved config paths, and `deepagents-talon mcp login <server>` for OAuth-backed servers.

## Cron Observability

Cron jobs are persisted in `cron/jobs.json` under the assistant state directory. Scheduler lifecycle events are emitted through the standard Python logger as `talon_event` JSON records:

- `cron.tick`
- `cron.dispatch`
- `cron.success`
- `cron.failure`
- `cron.delivery`
- `cron.delivery_suppressed`
- `cron.delivery_failure`

These logs complement the persisted `last_status` and `last_error` fields.

## Security and Data Lifecycle

Talon is single-operator by design. It does not provide multi-tenant isolation, and channel exposure should be treated as direct access to the operator's agent.

Attacker-influenceable inputs include channel message text, voice transcripts, channel media metadata, downloaded media files when a channel adapter persists them for processing, web or search result content, MCP tool results, and imported manifest instructions. Treat all of those inputs as untrusted content entering the agent context.

Outbound data leaves Talon through these integrations:

- Model providers receive conversation text, cron prompts, voice transcripts, selected tool outputs, and system or manifest instructions.
- LangSmith receives trace metadata and serialized run inputs/outputs when `LANGSMITH_TRACING=true`.
- MCP servers receive tool arguments chosen by the model and may receive conversation-derived values.
- Tavily or other search tools receive query strings chosen by the model and may include conversation-derived values.
- Channel providers receive assistant replies and outbound media paths supplied to the channel adapter.

Sensitive local state is stored under `~/.deepagents/<assistant_id>/` by default with `0700` directories and `0600` cron files:

- `cron/jobs.json` stores cron prompts, origin conversation ids, message ids, run status, and errors. Active jobs are retained while enabled. Completed jobs are deleted on startup after `DEEPAGENTS_TALON_CRON_RETENTION_DAYS`, default `30`.
- `channels/whatsapp/` stores WhatsApp `LocalAuth` credentials and Chromium profile state. These credentials are retained until the operator deletes the directory, because automatic deletion would silently unpair the channel.
- `media/inbound/` is reserved for downloaded inbound media. Files older than `DEEPAGENTS_TALON_INBOUND_MEDIA_RETENTION_HOURS`, default `24`, are deleted on startup. The WhatsApp bridge stores downloaded inbound media under the assistant's inbound media directory and passes local paths plus MIME metadata to the host.

Conversation persistence is intentionally not durable yet. Runtime conversation state is in-memory unless a future backend explicitly adds thread persistence.

## Development

```bash
uv sync --group test
uv run --group test pytest tests/
uv run deepagents-talon
```

Focused verification:

```bash
make lint
make test
```

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
