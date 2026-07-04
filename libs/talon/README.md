# Deep Agents Talon

Deep Agents Talon is the local runtime host for long-running Deep Agents. It owns the process lifecycle for channel adapters, cron schedulers, and the agent runtime in a single event loop.

> **Experimental:** Talon is an experimental, alpha-status runtime and is subject to change or removal at any time. It is not intended for production or enterprise use.
>
> **Security support:** Talon does not yet implement production-grade security controls such as complete human-in-the-loop (HITL) approval policy, channel administrator controls, sandbox-backed execution isolation, or multi-tenant boundaries. Channel access should be treated as direct access to the operator's agent, model credentials, MCP tools, and local host resources. We do not accept security vulnerability reports for the absence of these known, unimplemented Talon hardening features while Talon remains experimental.

Talon currently includes:

- A host process with graceful shutdown, per-conversation serialization, and `/stop` cancellation.
- A generic channel protocol plus a WhatsApp adapter backed by a loopback Node bridge.
- A persistent cron scheduler with agent-facing cron tool helpers.
- MCP tool loading from the assistant manifest directory.
- Optional LangSmith tracing for each channel or cron-triggered run.

## Quickstart

Run the commands in this README from `libs/talon`. From the repository root,
prefix `uv` commands with `--directory libs/talon`.

```bash
cd libs/talon
uv sync --group test
AGENT_ASSISTANT_ID=local AGENT_MODEL=<provider>:<model-id> uv run deepagents-talon --once
```

If `AGENT_MODEL` is unset, Talon starts with the echo runtime. This is useful for checking host lifecycle and channel wiring without provider credentials.

Assistant state lives under `~/.deepagents/<assistant_id>/` by default. The host creates restrictive state directories for the materialized agent manifest, channel sessions, and cron jobs. The default local execution workspace is the current working directory; set `DEEPAGENTS_TALON_WORKSPACE` to use a different directory. The per-invocation graph recursion limit defaults to `500`; set `DEEPAGENTS_TALON_RECURSION_LIMIT` to tune it.

## Fleet Exports

Talon can host an operator-unzipped LangSmith Fleet export through the `fleet-deepagents-export` library:

```bash
unzip path/to/fleet-export.zip -d ./fleet
```

Import the export once per Talon assistant id. The import validates `AGENTS.md` and
`config.json`, materializes the Fleet prompt, skills, and subagents into the
assistant directory, and writes Fleet MCP replacement follow-up tasks to the
assistant-scoped manifest:

```bash
AGENT_ASSISTANT_ID=fleet-local \
uv run deepagents-talon import-fleet ./fleet \
  --assistant-id fleet-local \
  --non-interactive
```

The deprecated `--channel` flag is accepted for compatibility but ignored.
Channels are selected by the normal Talon runtime flags and environment.

The manifest is written to `<assistant-home>/agent/tools.json`, where
`<assistant-home>` is `~/.deepagents/<assistant-id>` by default or
`$DEEPAGENTS_TALON_HOME/<assistant-id>` when `DEEPAGENTS_TALON_HOME` is set. Local
MCP replacement setup is best-effort: unresolved Fleet MCP tools are recorded as
setup tasks, but missing local replacements do not block import. The default
follow-up target for local MCP replacement configuration is the same manifest file,
`<assistant-home>/agent/tools.json`.

Run the imported Fleet export through the normal Talon runtime by assistant id:

```bash
AGENT_ASSISTANT_ID=fleet-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --once
```

Use `--once` as a local startup check before leaving the host running. Existing
channel flags such as `--whatsapp` and `--telegram`, or
`DEEPAGENTS_TALON_<CHANNEL>_ENABLED`, attach adapters for the run.

For Telegram, provide the Bot API token and an exposure policy:

```bash
DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN=<telegram-bot-token> \
DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS=<telegram-user-id> \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS=<telegram-chat-id> \
AGENT_ASSISTANT_ID=fleet-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon
```

For WhatsApp:

```bash
DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=self \
AGENT_ASSISTANT_ID=fleet-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --whatsapp
```

Imported Fleet agents use the model configured in the Talon runtime environment
with `DEEPAGENTS_TALON_MODEL` or `AGENT_MODEL`. The Fleet export may include MCP
registry references that require local MCP replacements, so provide the required
`LANGSMITH_API_KEY`, `LANGSMITH_TENANT_ID`, and
`LANGSMITH_ORGANIZATION_ID`. Some exports also require `LANGSMITH_USER_ID`,
`BUILTIN_MCP_URL`, `LANGSMITH_HOST_URL`, or `HOST_LANGCHAIN_API_URL`. Keep these
values in environment variables or a local secret manager; do not copy real
tokens into the manifest, README examples, or command history. Locally-authored
agents and imported Fleet agents both load from the assistant manifest directory
and Talon's plain MCP config discovery.

OAuth-backed Fleet MCP tools must be authorized once from an interactive shell before starting a headless host. Run the imported assistant in `--once` mode with the same Fleet directory and LangSmith environment you will use in production, complete the browser authorization if prompted, then start the long-running host:

```bash
AGENT_ASSISTANT_ID=fleet-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --once
```

During a long-running session, Talon treats a 401/403 from an MCP tool as an expired OAuth credential signal. It reloads MCP connections once, then retries the failed graph invocation once. If authorization still fails, Talon returns a structured `mcp_auth_failed` error instead of looping.

## Tool Approval Overrides

Set `DEEPAGENTS_TALON_INTERRUPT_ON_TOOLS` to a comma-separated list of tool names that should always require Talon's channel approval flow. This local override is additive with Fleet-provided HITL configuration and also applies to non-Fleet MCP or local runtime tools.

```bash
DEEPAGENTS_TALON_INTERRUPT_ON_TOOLS=bash,execute,github_create_pr
```

## WhatsApp

The WhatsApp channel uses a local Node bridge packaged with this library. The Python adapter talks to the bridge over loopback only.

```bash
cd deepagents_talon/channels/whatsapp_bridge
npm install
cd ../../..

DEEPAGENTS_TALON_WHATSAPP_ENABLED=true \
DEEPAGENTS_TALON_WHATSAPP_START_BRIDGE=true \
AGENT_ASSISTANT_ID=whatsapp-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --whatsapp
```

The bridge prints a QR code during pairing. By default, inbound exposure is `self`, so only messages from the paired account trigger the agent. Configure `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` with `DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or `DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS` to allow specific chats. `DEEPAGENTS_TALON_WHATSAPP_OPERATOR_ID` accepts one or more comma-separated operator IDs for `self` exposure. Outbound WhatsApp messages include a `deepagents bot` header by default so self-message conversations clearly distinguish agent replies from operator messages. Set `DEEPAGENTS_TALON_WHATSAPP_BOT_HEADER` to customize that label. Markdown image/video references in assistant replies may attach files only when they are relative paths inside `DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR`, or inside `DEEPAGENTS_TALON_WORKSPACE` when no outbound media directory is configured. `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` caps inbound and outbound channel media across providers and defaults to `1073741824` (1 GiB), but WhatsApp is clamped to `67108864` (64 MiB) because the bridge library materializes downloads in memory before writing them.

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

## Telegram

The Telegram channel uses the Bot API with long polling. Provide a bot token from BotFather and either a model or Fleet export so Talon runs the real Deep Agents runtime instead of the echo runtime:

```bash
DEEPAGENTS_TALON_TELEGRAM_ENABLED=true \
DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN=... \
DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS=123456789 \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS=-1001234567890 \
AGENT_ASSISTANT_ID=telegram-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --telegram
```

From the repository root, run the same host with:

```bash
DEEPAGENTS_TALON_TELEGRAM_ENABLED=true \
DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN=... \
DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS=123456789 \
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS=-1001234567890 \
AGENT_ASSISTANT_ID=telegram-local \
AGENT_MODEL=<provider>:<model-id> \
uv run --directory libs/talon deepagents-talon --telegram
```

In `allowlist` mode, `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS` allows private bot DMs from specific Telegram user IDs, while `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS` allows channel posts from specific channel chat IDs. `DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID` accepts one or more comma-separated operator IDs for `self` exposure. `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` caps inbound and outbound channel media across providers and defaults to `1073741824` (1 GiB); Telegram's smaller Bot API upload limits still apply. If `AGENT_MODEL`, `DEEPAGENTS_TALON_MODEL`, and `DEEPAGENTS_TALON_FLEET_DIR` are all unset, Talon uses the echo runtime and replies with the inbound text unchanged.

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

Talon is single-operator by design. It does not provide multi-tenant isolation, sandbox-backed execution isolation, production-grade HITL policy enforcement, or channel administrator boundaries. Any tool approval prompt surfaced through a channel is an experimental convenience feature, not a complete security boundary. Channel exposure should be treated as direct access to the operator's agent, model credentials, MCP tools, and local host resources.

Do not file security vulnerability reports for the absence of these known, unimplemented hardening features in Talon while it remains experimental. Reports about missing enterprise controls, channel admin gates, sandbox integrations, or production HITL policy are considered feature requests for a future production-ready runtime.

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
- `media/inbound/` is reserved for downloaded inbound media. Files older than `DEEPAGENTS_TALON_INBOUND_MEDIA_RETENTION_HOURS`, default `24`, are deleted on startup. Inbound and outbound channel media are capped by `DEEPAGENTS_TALON_MAX_MEDIA_BYTES`, default `1073741824` (1 GiB); WhatsApp is further clamped to `67108864` (64 MiB). The WhatsApp bridge stores downloaded inbound media under the assistant's inbound media directory and passes local paths plus MIME metadata to the host.

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
