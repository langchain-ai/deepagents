# DeepAgents Talon

DeepAgents Talon is the local runtime host for long-running Deep Agents. It owns
the process lifecycle for channel adapters, cron schedulers, and the agent
runtime in a single event loop.

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

If `AGENT_MODEL` is unset, Talon starts with the echo runtime. This is useful for
checking host lifecycle and channel wiring without provider credentials.

Assistant state lives under `~/.deepagents/<assistant_id>/` by default. The host
creates restrictive state directories for the materialized agent manifest,
channel sessions, and cron jobs.

## WhatsApp

The WhatsApp channel uses a local Node bridge packaged with this library. The
Python adapter talks to the bridge over loopback only.

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

The bridge prints a QR code during pairing. By default, inbound exposure is
`self`, so only messages from the paired account trigger the agent. Configure
`DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` with
`DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or
`DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS` to allow specific chats.
Outbound WhatsApp messages include a `deepagents bot` header by default so
self-message conversations clearly distinguish agent replies from operator
messages. Set `DEEPAGENTS_TALON_WHATSAPP_BOT_HEADER` to customize that label.

Inbound voice transcription is opt-in:

```bash
DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED=true
```

When enabled without `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL`, Talon uses the
same local default as the original WhatsApp example:
`nvidia/parakeet-tdt-0.6b-v3` through Transformers, with ffmpeg converting
inbound audio to 16 kHz mono WAV first. Set
`DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE=cuda` to use a GPU. The legacy
example variables `SPEECH_ENABLED` and `SPEECH_DEVICE` are also accepted.
Setting `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL` to a non-Parakeet model
keeps the existing OpenAI SDK transcription path.

`open` exposure allows arbitrary WhatsApp senders to trigger the agent while it
runs with the operator's model credentials, channel credentials, MCP tool access,
and local-host access when the local execution backend is active. Enabling it
requires explicit acknowledgement:

```bash
DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open
DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK=allow-arbitrary-senders
```

See `../../examples/talon-whatsapp/` for a runnable Docker Compose topology and
`.env` reference.

## Tracing

LangSmith tracing is opt-in. Set both values before starting the host:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deepagents-talon
```

When enabled, Talon wraps each agent run in a LangSmith tracing context with
assistant id, conversation id, trigger metadata, and source message metadata.

## Cron Observability

Cron jobs are persisted in `cron/jobs.json` under the assistant state directory.
Scheduler lifecycle events are emitted through the standard Python logger as
`talon_event` JSON records:

- `cron.tick`
- `cron.dispatch`
- `cron.success`
- `cron.failure`
- `cron.delivery`
- `cron.delivery_suppressed`
- `cron.delivery_failure`

These logs complement the persisted `last_status` and `last_error` fields.

## Security and Data Lifecycle

Talon is single-operator by design. It does not provide multi-tenant isolation,
and channel exposure should be treated as direct access to the operator's agent.

Attacker-influenceable inputs include channel message text, voice transcripts,
channel media metadata, downloaded media files when a channel adapter persists
them for processing, web or search result content, MCP tool results, and imported
manifest instructions. Treat all of those inputs as untrusted content entering
the agent context.

Outbound data leaves Talon through these integrations:

- Model providers receive conversation text, cron prompts, voice transcripts,
  selected tool outputs, and system or manifest instructions.
- LangSmith receives trace metadata and serialized run inputs/outputs when
  `LANGSMITH_TRACING=true`.
- MCP servers receive tool arguments chosen by the model and may receive
  conversation-derived values.
- Tavily or other search tools receive query strings chosen by the model and may
  include conversation-derived values.
- Channel providers receive assistant replies and outbound media paths supplied
  to the channel adapter.

Sensitive local state is stored under `~/.deepagents/<assistant_id>/` by default
with `0700` directories and `0600` cron files:

- `cron/jobs.json` stores cron prompts, origin conversation ids, message ids,
  run status, and errors. Active jobs are retained while enabled. Completed jobs
  are deleted on startup after `DEEPAGENTS_TALON_CRON_RETENTION_DAYS`, default
  `30`.
- `channels/whatsapp/` stores WhatsApp `LocalAuth` credentials and Chromium
  profile state. These credentials are retained until the operator deletes the
  directory, because automatic deletion would silently unpair the channel.
- `media/inbound/` is reserved for downloaded inbound media. Files older than
  `DEEPAGENTS_TALON_INBOUND_MEDIA_RETENTION_HOURS`, default `24`, are deleted on
  startup. The WhatsApp bridge stores downloaded inbound media under the
  assistant's inbound media directory and passes local paths plus MIME metadata
  to the host.

Conversation persistence is intentionally not durable yet. Runtime conversation
state is in-memory unless a future backend explicitly adds thread persistence.

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
