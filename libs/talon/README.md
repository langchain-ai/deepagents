# Deep Agents Talon

Deep Agents Talon is the local runtime host for long-running Deep Agents. It owns the process lifecycle for channel adapters, cron schedulers, and the agent runtime in a single event loop.

> **Experimental:** Talon is an experimental, alpha-status runtime and is subject to change or removal at any time. It is not intended for production or enterprise use.
>
> **Security support:** Talon does not yet implement production-grade security controls such as complete human-in-the-loop (HITL) approval policy, channel administrator controls, sandbox-backed execution isolation, or multi-tenant boundaries. Channel access should be treated as direct access to the operator's agent, model credentials, MCP tools, and local host resources. We do not accept security vulnerability reports for the absence of these known, unimplemented Talon hardening features while Talon remains experimental.

Talon currently includes:

- A host process with graceful shutdown, per-conversation interrupt-and-continue, and `/stop` cancellation.
- A generic channel protocol plus WhatsApp, Telegram, and Discord adapters (WhatsApp is backed by a loopback Node bridge).
- A persistent cron scheduler with agent-facing cron tool helpers.
- MCP tool loading from explicit config paths or `~/.deepagents/.mcp.json`.
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

Assistant state lives under `~/.deepagents/<assistant_id>/` by default. The host creates restrictive state directories for the materialized agent manifest, channel sessions, and cron jobs, and persists conversation checkpoints in `checkpoints.sqlite` so chat history survives restarts. The default local execution workspace is the current working directory; set `DEEPAGENTS_TALON_WORKSPACE` to use a different directory. The per-invocation graph recursion limit defaults to `500`; set `DEEPAGENTS_TALON_RECURSION_LIMIT` to tune it.

## Conversation history

Talon archives channel conversations in `checkpoints.sqlite` without automatic
expiry. The agent can list, search, and read past sessions in bounded pages,
restricted to the current channel and chat. History survives context compaction;
text, tool-call arguments, and distinct message revisions are retained.

- `/new` starts a fresh context while keeping earlier sessions searchable.
- `/reset-all-history` stops active work, deletes this chat's archived sessions and
  checkpoints, and starts a fresh context. Other chats are unaffected. Cancellation
  timeouts leave history intact; deletion failures may leave a partial reset that
  you can retry.

Reset does not remove cron jobs, memory files, downloaded media, traces, or backups.
Attachment binaries and archive-tool results are not indexed. Scheduled runs do not
add conversation history, and existing checkpoints are not backfilled.

The echo runtime and unwrapped custom checkpointers do not support history tools or
reset. Custom async LangGraph checkpointers can enable history with `ConversationSaver`.

Set `DEEPAGENTS_TALON_HISTORY_URI` to `mongodb://host/database` or
`postgresql://user:password@host/database` and install the `mongodb` or `postgres`
extra (`uv sync --extra mongodb`). All three backends use the same archive; SQLite
is the default. This alpha requires fresh history storage. Checkpoints stay local.
Default SQLite uses the same store factory and assistant namespace as configured
backends, with its own connection to the checkpoint database.

For a separate SQLite database, set the URI to `sqlite:///absolute/path/history.sqlite`
or a SQLite `file:` URI, including connection options such as `?mode=rwc`.
Paths containing spaces must be percent-encoded. All archives are
namespaced by assistant ID, so assistants can share a database.

Additional backends can be installed as Python packages without changing Talon.
Register the URI scheme in the package's `pyproject.toml`:

```toml
[project.entry-points."deepagents_talon.history_backends"]
mysql = "my_history_backend:open_store"
```

The entry point is a trusted operator-installed callable that accepts the unchanged
URI and returns an async context manager yielding an initialized LangGraph
`BaseStore`. It owns connection setup and cleanup, including cancellation, and
validates its backend-specific URI requirements. Talon wraps the store in its shared
archive and verifies write access before startup completes. Built-in schemes take
precedence; unknown or duplicate plugin schemes fail startup. The plugin API is
experimental and may change with Talon.

Archives require one writer per assistant. Retrieval scans at most 500
records and raises an error if it cannot complete the page within that budget.

Set `DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH=1` to add semantic matches to keyword
search. Select an embedding adapter independently of the history database:

| Adapter | Install extra | Credentials | Inference |
| --- | --- | --- | --- |
| `local` (default) | `history-local` | None | Local CPU, lazy Qwen loading |
| `voyage` | `history-voyage` | `VOYAGE_API_KEY` | Voyage API |
| `openai-compatible` | `history-openai` | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` | HTTPS embedding API |
| `atlas` | `mongodb` | Configure the model in Atlas | Atlas Automated Embedding |

Remote adapters do not require torch or sentence-transformers. The former `history`
extra is now `history-local`. Provider packages supply the maintained API integrations;
`langchain-voyageai` and `langchain-openai` are MIT-licensed LangChain packages.

For Voyage, install `uv sync --extra history-voyage` and configure:

```sh
DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH=1
DEEPAGENTS_TALON_HISTORY_EMBED_ADAPTER=voyage
DEEPAGENTS_TALON_HISTORY_EMBED_MODEL=voyage-4-large
DEEPAGENTS_TALON_HISTORY_EMBED_DIMS=1024
DEEPAGENTS_TALON_HISTORY_EMBED_MAX_INPUT_TOKENS=32000
```

Supply `VOYAGE_API_KEY` through the environment. For OpenRouter, install
`history-openai`, select `openai-compatible`, set `BASE_URL` below to
`https://openrouter.ai/api/v1`, and supply `OPENROUTER_API_KEY`. For example,
`qwen/qwen3-embedding-8b` supports 4096 dimensions and a 32768-token context.
Verify the selected model's limits in the [Voyage documentation](https://docs.voyageai.com/docs/embeddings)
or [OpenRouter catalog](https://openrouter.ai/models?output_modalities=embeddings).

Embedding settings use the `DEEPAGENTS_TALON_HISTORY_EMBED_` prefix:

| Suffix | Meaning |
| --- | --- |
| `ADAPTER` | `local`, `voyage`, `openai-compatible`, or `atlas` |
| `MODEL` | Required for remote adapters; local defaults to `Qwen/Qwen3-Embedding-0.6B` |
| `DIMS` | Output width; required for remote client adapters |
| `MAX_INPUT_TOKENS` | Model context budget; required remotely, local defaults to 8192 |
| `BATCH_SIZE` | Local defaults to 4 (maximum 4); remote defaults to 32 (maximum 96) |
| `CONCURRENCY` | Local uses 1; remote defaults to 4 (maximum 16) |
| `QUERY_PROMPT` | Optional query instruction; only local Qwen has a default prefix |
| `BASE_URL` | Optional HTTPS endpoint without credentials, query parameters, or fragments |
| `API_KEY` | Optional environment override for the adapter's standard API key |
| `QUERY_MODEL` | Optional compatible query-time model, supported only by Atlas |

Queries retain each provider's query/document semantics on all three databases.
Inputs use UTF-8 byte counts as a conservative token bound, reserving 128 tokens
for provider instructions. Oversized documents are split without losing text and
their vectors are combined with a length-weighted mean; transcript pagination stays
unchanged. Oversized queries fall back to keyword search. Atlas requires a budget
large enough for a complete archive chunk because embedding happens server-side.
Remote indexing uses bounded batches and concurrency; errors retain pending work
for retry. Selecting a remote adapter sends archived text and queries to that
provider and may incur charges.

Vector data uses fingerprint-specific SQLite files, PostgreSQL schemas, or MongoDB
collections, keeping incompatible dimensions separate. PostgreSQL uses exact vector
search above 2000 dimensions. Metadata and vectors always use separate Store instances.
Changing a model, endpoint, dimensions, prompt, or input budget fails startup when
an existing index is incompatible. Set `DEEPAGENTS_TALON_HISTORY_REINDEX=1` explicitly
to remove the old vectors and rebuild from retained transcripts; this can incur
embedding charges. Deletion progress survives interruption. Remove the flag afterward;
it does not rebuild an already matching index. Empty old vector files/schemas/collections
remain for operator cleanup. Missing fingerprints on older indexes also require reindexing.
Reset deletes vectors even after semantic search has been disabled.

Backend plugins can optionally register `deepagents_talon.history_vector_backends`
under the same URI scheme. The vector factory receives `(uri, *, index, generation)`
and yields a separate initialized `BaseStore`; `index=None` means deletion-only mode.
It must isolate generations, own cleanup, and apply backend-specific index options.
The existing metadata factory remains unchanged. Atlas mode requires MongoDB.

`search_conversations` returns results, indexing coverage, and an opaque
`next_after` token. Continue with the same query and chat; expired tokens require
a new search. Semantic errors and timeouts fall back to keyword matches. Unknown
or pending indexing coverage means an empty page does not prove history is absent.

## Interrupt and Continue

A new message in a conversation cancels the active turn, records an interruption marker after the latest committed graph checkpoint, and starts the new message on the same thread. Partial output from the cancelled turn is not fabricated or delivered. `/stop` and `/new` also recover interrupted state; process shutdown does not. If cancellation does not finish within 30 seconds, Talon leaves the existing run isolated and does not start the new message; restart Talon to recover.

## Local Agent Activity Logs

Set `DEEPAGENTS_TALON_AGENT_ACTIVITY_LOGGING=true` to emit agent run, model activity, and tool call events to the local process logs at `INFO`. Tool inputs and outputs are redacted and truncated to 1,000 characters, but may still contain sensitive application data; enable these logs only where local log access is appropriately restricted. “Thinking” events report model-call lifecycle activity and do not expose hidden chain-of-thought.

## Tool Approval Overrides

Set `DEEPAGENTS_TALON_INTERRUPT_ON_TOOLS` to a comma-separated list of tool names that should always require Talon's channel approval flow. This local override is additive with agent-provided HITL configuration and applies to MCP or local runtime tools.

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

The Telegram channel uses the Bot API with long polling. Provide a bot token from BotFather and a model so Talon runs the real Deep Agents runtime instead of the echo runtime:

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

In `allowlist` mode, `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS` allows private bot DMs from specific Telegram user IDs, while `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS` allows channel posts from specific channel chat IDs. `DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID` accepts one or more comma-separated operator IDs for `self` exposure. `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` caps inbound and outbound channel media across providers and defaults to `1073741824` (1 GiB); Telegram's smaller Bot API upload limits still apply. If `AGENT_MODEL` and `DEEPAGENTS_TALON_MODEL` are both unset, Talon uses the echo runtime and replies with the inbound text unchanged.

## Discord

The Discord channel uses the [`discord.py`](https://discordpy.readthedocs.io/) Gateway client for real-time message delivery. Create a bot application in the [Discord Developer Portal](https://discord.com/developers/applications), copy its token, and enable the **Message Content** privileged intent under the Bot settings — without it, the bot receives events but not message text:

```bash
DEEPAGENTS_TALON_DISCORD_ENABLED=true \
DEEPAGENTS_TALON_DISCORD_BOT_TOKEN=... \
DEEPAGENTS_TALON_DISCORD_EXPOSURE=allowlist \
DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS=123456789012345678 \
DEEPAGENTS_TALON_DISCORD_ALLOWLIST_CHATS=234567890123456789 \
AGENT_ASSISTANT_ID=discord-local \
AGENT_MODEL=<provider>:<model-id> \
uv run deepagents-talon --discord
```

From the repository root, run the same host with:

```bash
DEEPAGENTS_TALON_DISCORD_ENABLED=true \
DEEPAGENTS_TALON_DISCORD_BOT_TOKEN=... \
DEEPAGENTS_TALON_DISCORD_EXPOSURE=allowlist \
DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS=123456789012345678 \
DEEPAGENTS_TALON_DISCORD_ALLOWLIST_CHATS=234567890123456789 \
AGENT_ASSISTANT_ID=discord-local \
AGENT_MODEL=<provider>:<model-id> \
uv run --directory libs/talon deepagents-talon --discord
```

`conversation_id` is the Discord channel ID, which works uniformly for DM channels and guild text channels. In `allowlist` mode, `DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS` allows DMs from specific Discord user IDs regardless of channel, while `DEEPAGENTS_TALON_DISCORD_ALLOWLIST_CHATS` allows messages from specific channel IDs (DM or guild). `DEEPAGENTS_TALON_DISCORD_OPERATOR_ID` accepts one or more comma-separated operator IDs for `self` exposure, the default mode, which only accepts DMs from those operators. Outbound text over Discord's 2000-character message limit is split into multiple separate messages sent in order; outbound media is sent as a file attachment with the caption as the message content when it fits, or as a preceding separate message otherwise. `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` caps inbound and outbound channel media across providers and defaults to `1073741824` (1 GiB). If `AGENT_MODEL` and `DEEPAGENTS_TALON_MODEL` are both unset, Talon uses the echo runtime and replies with the inbound text unchanged.

## Tracing

LangSmith tracing is opt-in. Set both values before starting the host:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deepagents-talon
```

When enabled, Talon wraps each agent run in a LangSmith tracing context with assistant id, conversation id, trigger metadata, and source message metadata.

## Chat commands

Send `/help` for a brief guide to Talon, its built-in commands (`/new`, `/stop`,
and `/mcp-reload`), and using MCP configuration and OAuth through chat. Help does
not interrupt current work or consume a pending approval or sign-in response.

## MCP Tools

Talon loads MCP servers from `~/.deepagents/.mcp.json`. Set `DEEPAGENTS_TALON_MCP_CONFIG` to use a different path. For user-level MCP servers, edit the standard file:

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

Set `"auth": "oauth"` on a remote server to enable OAuth. From WhatsApp,
Telegram, or another interactive channel, ask Talon to authenticate that configured
server. Talon calls the narrow `authenticate_mcp_server` capability, sends the
authorization link directly to the originating conversation, and waits for the same
operator to paste the full callback URL. The authorization link and callback bypass
the model context and traces. Newly discovered tools are available on the next channel
turn after login completes.

Run `deepagents-talon mcp config` to print the resolved config path. The terminal-only
`deepagents-talon mcp login <server>` flow remains available as an alternative.

On Linux/macOS, Talon can manage its MCP configuration through chat using
`get_mcp_configuration` (redacted view) and `update_mcp_server` (add, replace, or
remove one server). Updates require human approval by default and reload before
the next turn. Set `DEEPAGENTS_TALON_MCP_CONFIG_AUTO_APPROVE=true` in the host
environment to opt out; explicit tool approval policies still apply.

Use `${ENV_VAR}` references for credentials. Set `DEEPAGENTS_TALON_MCP_CONFIG`
to keep the file outside the workspace. These tools do not sandbox Talon's local
shell backend; deployments must enforce filesystem isolation separately.

After editing the configuration manually, send `/mcp-reload` through an authorized channel to
reload it without restarting Talon. The agent can also call
`reload_mcp_configuration` autonomously; that schedules the same reload before the
next agent turn.

Fleet zip exports can be materialized into a Talon-local agent directory before
starting the host:

```bash
deepagents-talon import-fleet <fleet-export.zip> [--assistant-id <id>] [--target-dir <dir>]
```

From the repository root:

```bash
uv run --directory libs/talon deepagents-talon import-fleet ./fleet-export.zip \
  --assistant-id local
```

By default, `import-fleet` writes into the selected assistant manifest directory:
`~/.deepagents/<assistant_id>/`, with subagent prompts in
`~/.deepagents/<assistant_id>/agents/`. The selected assistant id comes from
`DEEPAGENTS_TALON_ASSISTANT_ID` or `AGENT_ASSISTANT_ID`; when neither is set,
the importer uses the Fleet export filename stem. For example, `crowbar.zip`
imports into `~/.deepagents/crowbar/`. Pass `--assistant-id <id>` to select a
different assistant for the import, or `--target-dir <dir>` to write all
imported files under an explicit directory.

The importer writes Fleet prompts, skills, and subagent prompts. Talon loads local
subagents from `agents/<name>/AGENTS.md` using dcode's YAML frontmatter format:
`description` is required, `name` defaults to the directory name, and `model` is
optional. Local subagents use fork mode so they inherit the current conversation and
runtime policy; Talon also provides the standard `general-purpose` subagent unless the
assistant defines one. Fleet `tools.json` and `config.json` are ignored and are not
copied into the Talon agent directory. Talon does not support the old Fleet direct-run
startup path or its environment variables; import the zip first, then run Talon against
the materialized local assistant.

## Background Subagents

Talon loads local `agents/<name>/AGENTS.md` definitions and remote
`[async_subagents]` configuration at startup. After adding, editing, or deleting
definitions, the main agent can call `reload_subagent_configuration` to apply the
changes on subsequent turns. Ordinary turns reuse the loaded definitions. Invalid
edits retain the last valid configuration; running subagents keep their original
configuration.

`task` launches local subagents and `start_async_task` launches remote subagents.
Both return immediately. The user can continue chatting while the main agent uses
`list_subagents` to inspect work and `cancel_subagent` to cancel it. When work
finishes, its result is passed to the main agent for processing on the next idle
turn, then the main agent replies to the channel.

Workers and pending results live only in memory and are discarded on restart.
`/stop` and `/new` cancel all subagents belonging to that conversation; ordinary
messages interrupt only the main turn. Shutdown cancels all workers. Local tool
approval policy still applies; a child needing approval reports that it could not
complete the action. Remote runs cancel when their stream disconnects.

Talon allows four simultaneous subagents, retains at most 128 unprocessed jobs,
and limits each run to one hour. Completed results are capped at 64,000 characters.

## Cron Schedules

`create_job` and `edit_job` accept four schedule forms:

| Form | Kind | Example |
| --- | --- | --- |
| `in <N>{m,h}` | one-shot | `in 30m` |
| `every <N>{m,h}` | recurring | `every 6h` |
| `at <YYYY-MM-DD> <HH:MM> <tz>` | one-shot | `at 2026-09-04 13:30 America/New_York` |
| `daily at <HH:MM> <tz>` | recurring | `daily at 08:00 America/New_York` |

The wall-clock forms require an explicit IANA timezone name; there is no default
zone, and legacy POSIX aliases (`EST5EDT`) and bare UTC offsets (`+02:00`) are
rejected because they cannot express a region's future daylight-saving rules.

The agent gets that zone name from the `current_time` tool, which is always
available and reports the current date, time, and IANA timezone. Called with no
argument it uses the host's local zone; pass a zone name to read the clock
elsewhere. Its `timezone` value goes straight into a schedule string. When the
host zone name cannot be determined the tool still reports the correct local
time and UTC offset, but returns `timezone: null` and a note to ask the user
rather than guessing.

The timezone is stored on the job and pinned. `daily at 08:00 America/New_York`
fires at 08:00 New York wall-clock time no matter where the host is or which
side of a daylight-saving transition the run falls on — the next run is rebuilt
from the local date each time rather than advanced by 24 hours. Two edge cases
resolve deterministically:

- A local time skipped by a spring-forward transition snaps forward to the first
  minute that exists, so `daily at 02:30` fires at 03:00 local on that day
  rather than being skipped.
- An ambiguous local time repeated by a fall-back transition resolves to its
  earlier occurrence, so the job fires once.

Interval schedules stay phase-locked to their previous run, so a late scheduler
tick does not shift an `every 15m` job off its cadence. A one-shot `at` schedule
that has already passed is rejected at create and edit time with the resolved
instant in the error message. Because the scheduler ticks every 60 seconds, a
run lands within the minute it is due, not on the exact second.

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

- `AGENTS.md`, `skills/`, and `agents/` store the materialized assistant instructions, skills, and subagent definitions.
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
