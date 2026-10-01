# Deep Agents Talon

Deep Agents Talon is the local runtime host for long-running Deep Agents. It owns the process lifecycle for channel adapters, cron schedulers, and the agent runtime in a single event loop.

> **Experimental:** Talon is an experimental, alpha-status runtime and is subject to change or removal at any time. It is not intended for production or enterprise use.
>
> **Security support:** Talon does not yet implement production-grade security controls such as complete human-in-the-loop (HITL) approval policy, channel administrator controls, or multi-tenant boundaries. [Sandboxed execution](#sandboxed-execution) is opt-in and does not cover MCP tools. Channel access should be treated as direct access to the operator's agent, model credentials, MCP tools, and local host resources. We do not accept security vulnerability reports for the absence of these known, unimplemented Talon hardening features while Talon remains experimental.

Talon currently includes:

- A host process with graceful shutdown, per-conversation interrupt-and-continue, and `/stop` cancellation.
- A generic channel protocol plus WhatsApp, Telegram, Discord, and Slack adapters (WhatsApp is backed by a loopback Node bridge).
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

Assistant state lives under `~/.deepagents/<assistant_id>/` by default. The host creates restrictive state directories for the materialized agent manifest, channel sessions, and cron jobs, and persists conversation checkpoints in `checkpoints.sqlite` so chat history survives restarts. Offloaded conversation history and large tool results live in the assistant home’s `artifacts/` directory. The default local execution workspace is the current working directory; set `DEEPAGENTS_TALON_WORKSPACE` to use a different directory. The per-invocation graph recursion limit defaults to `500`; set `DEEPAGENTS_TALON_RECURSION_LIMIT` to tune it.

## Sandboxed execution

By default the agent's shell and file tools run on the host. Set `DEEPAGENTS_TALON_SANDBOX` to a sandbox provider to run them in a remote sandbox instead:

```bash
DEEPAGENTS_TALON_SANDBOX=langsmith AGENT_MODEL=<provider>:<model-id> uv run deepagents-talon
```

Under systemd, put the variable in the service's environment file instead.

Talon uses the `deepagents-code` sandbox providers: `langsmith` works out of the box, and `agentcore`, `daytona`, `modal`, `runloop`, and `vercel` need the matching extra, for example `uv run --with 'deepagents-code[daytona]' deepagents-talon`. Each provider reads its own credentials (such as `LANGSMITH_API_KEY` or `DAYTONA_API_KEY`) in the Talon process; they are not forwarded into the sandbox.

| Variable | Purpose |
| --- | --- |
| `DEEPAGENTS_TALON_SANDBOX` | Provider name. Unset keeps host execution. |
| `DEEPAGENTS_TALON_SANDBOX_ID` | Attach to an existing sandbox instead of creating one. Talon never deletes it. |
| `DEEPAGENTS_TALON_SANDBOX_SNAPSHOT` | Snapshot or blueprint for providers that support one (`langsmith`, `runloop`). LangSmith defaults to `talon-<assistant_id>`, because snapshot names are shared across a workspace. |
| `DEEPAGENTS_TALON_SANDBOX_SETUP` | Host path to a script run once after the sandbox starts. |

Talon creates the sandbox at startup and deletes it on shutdown, so sandbox files do not survive a restart unless you set `DEEPAGENTS_TALON_SANDBOX_ID`. If the sandbox cannot start, Talon exits with an error rather than falling back to host execution. The first LangSmith start builds the snapshot and can outlast the provider's wait; Talon then exits, and the next start uses the snapshot once it is ready. Stopping Talon with SIGTERM while the sandbox is still starting can leave it running; attach with `DEEPAGENTS_TALON_SANDBOX_ID` if that matters.

The assistant's `skills/` and `memory/` directories stay on the host so skills and memory keep working; every other path, including large tool results, lives in the sandbox. `tools.json` and other assistant state are not reachable from sandbox tools. Memory paths from `DEEPAGENTS_TALON_MEMORY_PATHS` or the manifest must sit inside the assistant's `memory/` directory; Talon ignores others with a warning. Extra `DEEPAGENTS_TALON_SKILLS_DIRS` resolve inside the sandbox. MCP tools, web tools, and channel media handling still run on the host, so the sandbox is not a multi-tenant boundary.

## Conversation history

Talon archives channel conversations in `checkpoints.sqlite` without automatic
expiry. The agent can list, search, and read past sessions in bounded pages,
restricted to the current channel and chat. History survives context compaction;
text, tool-call arguments, and distinct message revisions are retained.

- `/new` starts a fresh context while keeping earlier sessions searchable.
- `/reset-all-history` stops active work, deletes this chat's archived sessions and
  checkpoints, and starts a fresh context. Other chats are unaffected. Cancellation
  timeouts leave history intact; deletion failures may leave a partial reset that
  you can retry. Because the deletion cannot be undone and Talon does not ask for
  confirmation, this command is deliberately left out of `/help` and is not
  registered as a Discord slash command: type it in full to use it (on Slack,
  `@Talon /reset-all-history` in a thread, or `/talon reset-all-history` in a DM).

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
| `CONCURRENCY` | Indexing requests in flight; local uses 1, remote defaults to 4 (maximum 16) |
| `BYTES_PER_TOKEN` | UTF-8 bytes budgeted per token, 1-4; defaults to the worst case of 1 |
| `QUERY_PROMPT` | Optional query instruction; Qwen3-Embedding models default to Qwen's prefix |
| `SEND_DIMENSIONS` | Send the OpenAI `dimensions` parameter; set `0` for models that reject it |
| `BASE_URL` | Optional HTTPS endpoint, routable host, without credentials, query, or fragments |
| `API_KEY` | Optional environment override for the adapter's standard API key |
| `QUERY_MODEL` | Optional compatible query-time model, supported only by Atlas |

Queries retain each provider's query/document semantics on all three databases,
and the instruction prefix follows the model rather than the adapter, so a
Qwen3-Embedding model reached through OpenRouter is prompted like a local one.

Inputs use UTF-8 byte counts as a conservative token bound, reserving 128 tokens
for provider instructions. `BYTES_PER_TOKEN` converts the token limit into that
byte measure and defaults to 1, which assumes every byte can become its own token.
Natural non-ASCII text is far cheaper than that -- a CJK character is roughly three
bytes but about one token -- so the default splits transcripts a model could embed
whole. Raising it trades safety margin for fewer splits; the value is part of the
embedding fingerprint, so a change rebuilds the index.

Oversized documents are split without losing text and their vectors are combined
with a length-weighted mean, which is logged once per run because pooled documents
are compared against unpooled queries. Transcript pagination stays unchanged.
Oversized queries fall back to keyword search. Atlas requires a budget
large enough for a complete archive chunk because embedding happens server-side.
A search holds a slot of its own at both the store and the provider, so it never
queues behind indexing and may add one request above `CONCURRENCY`.

`BASE_URL` must name a routable host: address literals in loopback, private,
link-local, or reserved ranges are refused, as is `localhost`, because the
configured endpoint receives the provider API key. Abbreviated IPv4 spellings
that the C resolver still accepts, such as `127.1` and `2130706433`, are
refused as the addresses they reach. A public name that resolves
to a private address still connects, which needs resolution-time control the
embedding clients do not expose.

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

When asked, the agent can use `delete_conversations` with one session ID or a list
from `list_conversations` or `search_conversations`. This deletes those sessions'
transcripts, search indexes, and checkpoints in the current chat. The active
conversation is protected; use `/new` before asking to delete it. Failed batches
may be partially deleted and can be retried with the same IDs.

## Interrupt and Continue

A new message in a conversation cancels the active turn, records an interruption marker after the latest committed graph checkpoint, and starts the new message on the same thread. Partial output from the cancelled turn is not fabricated or delivered. `/stop` and `/new` also recover interrupted state; process shutdown does not. If cancellation does not finish within 30 seconds, Talon leaves the existing run isolated and does not start the new message; restart Talon to recover.

## Local Agent Activity Logs

Set `DEEPAGENTS_TALON_AGENT_ACTIVITY_LOGGING=true` to emit agent run, model activity, and tool call events to the local process logs at `INFO`. Tool inputs and outputs are redacted and truncated to 1,000 characters, but may still contain sensitive application data; enable these logs only where local log access is appropriately restricted. “Thinking” events report model-call lifecycle activity and do not expose hidden chain-of-thought.

## Tool Approvals

Each assistant has one fixed policy at `TalonConfig.home / "tools.json"`, normally
`~/.deepagents/<assistant_id>/tools.json`. It is a flat JSON object mapping exact
tool names to booleans: `true` requires a channel approval prompt; `false` does
not. There are no patterns or per-agent policy files. The defaults are:

```json
{
  "update_tool_approvals": true,
  "delete_conversations": true,
  "update_mcp_server": true,
  "start_async_task": true,
  "send_message": true
}
```

`ask_for_help` always requires an approval prompt when enabled, independently of
this policy, because it sends the question to another model provider.

Native and container startup create these defaults when the file is missing and
preserve existing configuration. `send_message` prompts before an agent progress
update, including one with Slack mentions. It does not gate the final reply or
host-generated messages. Existing policy files retain their settings; add
`"send_message": true` to an existing `tools.json` to opt in. Unspecified tools
default to `false`; listing, searching, and reading conversation history do not prompt by default. A `false` value controls prompting, not tool
availability or authorization. There is no migration from the old approval settings.

Read `get_tool_approvals` before editing:

- `tools` is the persisted policy; `active_tools` is the current invocation's policy.
- `persisted_revision` is the revision to use for the next write; `active_revision`
  identifies the current invocation's snapshot.
- `saved_changes_inactive` indicates that saved changes are not active in this invocation.

Call `update_tool_approvals(updates={"execute": true, "delete_conversations": true},
expected_revision=<persisted_revision>)` with an updates mapping and the revision
returned by the read. The batch is atomic compare-and-swap: a stale revision
rejects the entire write, and unrelated entries are preserved. Read again and
review before retrying; do not replace the whole file to resolve a conflict.

Saved changes activate on the next invocation without a restart. Existing turns
and tasks keep their policy snapshot. An invalid file fails closed on the next
invocation rather than silently using an older policy; repair it as the operator.

Policy self-edits are checked against the **pre-edit** policy, so disabling
`update_tool_approvals` prompting cannot bypass the approval required for that
edit. An operator is required even when its prompt is `false`. In `self` exposure,
messages identified as `from_self` qualify without an extra operator list;
otherwise only the configured channel operator IDs qualify, not chat/user
allowlists or mention matches. Configure `DEEPAGENTS_TALON_WHATSAPP_OPERATOR_ID`,
`DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID`, `DEEPAGENTS_TALON_DISCORD_OPERATOR_ID`, or
`DEEPAGENTS_TALON_SLACK_OPERATOR_ID` for the applicable channel. Unidentified senders, scheduled runs, detached workers,
and background-result follow-ups cannot edit policy. Unattended follow-ups cannot
start interactive approvals or authorization flows.

`DeepAgentRuntime` no longer accepts `interrupt_on`; embedding hosts can pass an
`approval_store=ToolApprovalStore(path)` instead. The underlying Deep Agents
`interrupt_on` graph API is unchanged. Embedding hosts are responsible for supplying
trusted `AgentRequest.metadata["tool_approval_operator"]` authorization; never copy
that value from model arguments or untrusted inbound metadata.

Keep the assistant home outside the workspace, just like MCP configuration, and
persist its parent directory rather than bind-mounting a single `tools.json`:
updates use atomic file replacement. These controls are not a sandbox boundary.
A shell running as the same UID can bypass the tool API and edit the file directly;
filesystem isolation must be enforced separately. Talon-built local subagents inherit
this policy for their attached tools. Local tool gates do not enforce policy inside
opaque remote or precompiled graphs: `start_async_task` gates delegation, not the
remote graph's internal calls.

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

Local Parakeet and Qwen embedding model downloads share a Hugging Face cache in
`$DEEPAGENTS_TALON_HOME/cache/models/huggingface` (default:
`~/.deepagents/cache/models/huggingface`), shared across assistants.

`open` exposure allows arbitrary WhatsApp senders to trigger the agent while it runs with the operator's model credentials, channel credentials, MCP tool access, and local-host access when the local execution backend is active. Enabling it requires explicit acknowledgement:

```bash
DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open
DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK=allow-arbitrary-senders
```

See `../../examples/talon/` for a runnable Docker Compose topology and `.env` reference.

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

Talon's commands are also registered as native Discord slash commands, so typing `/` in a chat with the bot offers `/help`, `/new`, `/stop`, `/mcp-reload`, `/context-doctor`, and `/model` with autocomplete. The reply arrives as that command's own response rather than as a separate message. `/reset-all-history` is deliberately not registered, because it deletes stored history irreversibly and Talon has no confirmation step; it still works when typed in full.

Registration needs the **`applications.commands`** scope alongside `bot` in the bot's invite URL. A bot invited with only `bot` still receives messages, but a guild-scoped registration is rejected. Registration runs once per process, the first time the Gateway reports ready; a failure is logged and leaves the channel connected and usable. Because Discord requires a response to every slash command, an invocation that the exposure policy refuses now receives a brief private refusal, where a typed command is silently ignored — slash commands are visible to anyone who can see the bot, so the exposure policy, not their visibility, is what restricts use.

`DEEPAGENTS_TALON_DISCORD_COMMAND_GUILD_ID` scopes registration to one guild, which applies immediately and is useful while developing; global registration can take several minutes to propagate but is the only kind that reaches DMs, so leave this unset for an operator-DM deployment. `DEEPAGENTS_TALON_DISCORD_SLASH_COMMANDS=false` disables registration entirely, leaving commands available as typed text.

`conversation_id` is the Discord channel ID, which works uniformly for DM channels and guild text channels. A public guild thread keeps its own conversation and reply destination, but its archived history is shared with its parent channel and other public threads there. Private threads and DMs retain separate history. Existing thread-scoped history is not moved into the parent channel. `/reset-all-history` in a public thread clears the parent channel's shared archive; `/new` starts a new conversation only in the current thread. In `allowlist` mode, `DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS` allows DMs from specific Discord user IDs regardless of channel, while `DEEPAGENTS_TALON_DISCORD_ALLOWLIST_CHATS` allows messages from specific channel IDs (DM or guild). `DEEPAGENTS_TALON_DISCORD_OPERATOR_ID` accepts one or more comma-separated operator IDs for `self` exposure, the default mode, which only accepts DMs from those operators. Outbound text over Discord's 2000-character message limit is split into multiple separate messages sent in order; outbound media is sent as a file attachment with the caption as the message content when it fits, or as a preceding separate message otherwise. `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` caps inbound and outbound channel media across providers and defaults to `1073741824` (1 GiB). If `AGENT_MODEL` and `DEEPAGENTS_TALON_MODEL` are both unset, Talon uses the echo runtime and replies with the inbound text unchanged.

## Slack

The Slack channel uses [`slack_sdk`](https://docs.slack.dev/tools/python-slack-sdk/) Socket Mode, so Talon opens an outbound WebSocket and needs no public HTTP endpoint. Create an app at [api.slack.com/apps](https://api.slack.com/apps) with **From a manifest**, using this manifest as a starting point:

```yaml
display_information:
  name: Talon
features:
  bot_user:
    display_name: Talon
    always_online: true
  app_home:
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  slash_commands:
    - command: /talon
      description: Run a Talon command
      usage_hint: "[help | new | stop | mcp-reload | context-doctor]"
oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - channels:history
      - groups:history
      - chat:write
      - commands
      - files:read
      - files:write
      - im:history
      - im:write
      - reactions:read
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.im
      - reaction_added
  socket_mode_enabled: true
```

Install the app to the workspace, copy the **Bot User OAuth Token** (`xoxb-`), and under **Basic Information → App-Level Tokens** create a token with the `connections:write` scope (`xapp-`):

```bash
DEEPAGENTS_TALON_SLACK_ENABLED=true \
DEEPAGENTS_TALON_SLACK_BOT_TOKEN=xoxb-... \
DEEPAGENTS_TALON_SLACK_APP_TOKEN=xapp-... \
DEEPAGENTS_TALON_SLACK_OPERATOR_ID=U0123456789 \
AGENT_ASSISTANT_ID=slack-local \
uv run --directory libs/talon deepagents-talon --slack
```

A direct message with the bot is one conversation. In channels the bot answers only when mentioned, and it replies in a thread under the mentioning message. Each thread is its own active conversation, identified as `<channel id>:<thread ts>`, so mention the bot again in the thread to continue. Archived history for channel threads is shared by channel: `list_conversations`, `search_conversations`, and the other conversation tools can see sibling threads, but their active contexts and reply destinations remain separate. DMs keep separate history. Existing thread-scoped sessions continue writing to their original archive and are not included in channel-wide history tools or resets; use `/new` to start a channel-scoped archive without moving the old history. `/reset-all-history` in a channel thread clears its channel's shared archive; `/new` starts a new conversation only in that thread. Invite the bot to a channel with `/invite @Talon` before mentioning it there.

When mentioned in an existing channel thread, Talon includes bounded preceding replies from configured operators and allowlisted users as model context. Replies from other senders are excluded, even in open or channel-allowlist exposure. Existing installations must add `channels:history` (public channels) and `groups:history` (private channels) to the bot scopes and reinstall the app. If fetching history fails or exceeds the retrieval limit, Talon tells the model history is unavailable rather than supplying stale replies.

Slack treats any message that starts with `/` as a slash command, so Talon's commands are reached through the single `/talon` command: `/talon new`, `/talon stop`, `/talon mcp-reload`, `/talon context-doctor`, and `/talon help` (the default when no argument is given). The command can have any name: an app that names it `/talon-dev` in its manifest uses `/talon-dev new`, and so on. Anything after the command name is passed along as typed, so `/talon model <provider>:<model>` behaves like `/model <provider>:<model>`. A slash command carries no thread, so `/talon` works only in a direct message with the bot. In a channel thread, mention the bot followed by the command instead, for example `@Talon /new`. Refusals from `/talon`, including one from the exposure policy, are shown only to the invoking user.

`DEEPAGENTS_TALON_SLACK_OPERATOR_ID` accepts one or more comma-separated Slack user IDs (member IDs starting with `U`) for `self` exposure, the default mode. In `allowlist` mode, `DEEPAGENTS_TALON_SLACK_ALLOWLIST_USERS` allows DMs from specific user IDs, and `DEEPAGENTS_TALON_SLACK_ALLOWLIST_CHATS` allows mentions in specific channel IDs, covering every thread in them. `open` mode also requires `DEEPAGENTS_TALON_SLACK_OPEN_ACK=allow-arbitrary-senders`. Reactions are accepted only from operators and allowlisted users; a 👍 (`:+1:`, `:thumbsup:`, or `:thumbsup_all:`) or 👎 reaction on an approval prompt approves or rejects it, as on other channels. To finish an MCP OAuth sign-in, paste the callback URL into the DM, or into the thread after mentioning the bot.

Outbound Markdown is converted to Slack `mrkdwn`. Literal `<@USER_ID>` references to Slack user IDs (starting with `U` or `W`, including bot users) in prose can notify those users; code spans, other Slack control sequences, and `@channel` remain escaped. To restrict outbound mentions, set `DEEPAGENTS_TALON_SLACK_MENTION_ALLOWLIST_USERS` to comma-separated user IDs; if unset, any valid user mention is allowed, and if set to an empty value, none are allowed. This is separate from the inbound sender allowlist. Only allow outbound mentions where agent-generated text is trusted to notify recipients. Text over 4000 characters is split across posts. Media is uploaded as a file with the caption as its comment. Inbound files are downloaded with the bot token, which is sent only to `https://files.slack.com`, and redirects are refused. `DEEPAGENTS_TALON_SLACK_MEDIA_DIR` overrides the download directory, and `DEEPAGENTS_TALON_MAX_MEDIA_BYTES` applies here as on other channels. Slack has no bot typing indicator, so none is shown while the agent works.

## Sender pairing

Sender pairing lets the operator admit a new person on Discord, Telegram, or Slack without editing env and restarting. It is off by default. It is unrelated to WhatsApp's QR pairing. WhatsApp does not support it: its bridge runs on the operator's own account, so it would answer everyone who texts them.

Set `DEEPAGENTS_TALON_DISCORD_PAIRING=enabled`, `DEEPAGENTS_TALON_TELEGRAM_PAIRING=enabled`, or `DEEPAGENTS_TALON_SLACK_PAIRING=enabled`. Pairing works with `self` and `allowlist` exposure and is refused with `open`.

On Slack, the client treats a message starting with `/` as a slash command, so the operator reaches `/pair` through `/talon` in their DM with the bot: `/talon pair approve K7QM-3XRD`, `/talon pair list`, and `/talon pair revoke <member-id>`. Slack sender ids are member ids starting with `U`, and `/talon pair list` shows them. The requester can DM the bot or mention it in a channel the app has joined. The Slack app's Messages tab must be enabled, as it is in the manifest above. The `im:write` scope lets Talon open a DM for a channel mention; existing installations must add that scope and reinstall the app.

1. An unknown sender DMs or mentions the bot in a channel. Their message is dropped before it reaches the host or the model, and the bot sends a code such as `K7QM-3XRD` only in the sender's DM, never in the channel. If the bot cannot open the DM, no request is recorded. The code is bound to that sender on that channel, expires after 1 hour, and works once.
2. The sender passes the code to the operator by any other means.
3. The operator approves it with `/pair approve K7QM-3XRD` in their own DM with the bot, or with `deepagents-talon pairing approve <channel> K7QM-3XRD`, where `<channel>` is the requester's channel (`discord`, `slack`, or `telegram`); a code only approves on the channel it was issued on. The sender is told they were approved. From then on they reach the agent wherever the bot is, the way an env operator does: in their DM, by mentioning the bot in any Slack channel it has joined, and in any Discord server channel it can read.

`/pair list` shows pending codes and paired senders. `/pair revoke <sender-id>` removes a sender, stops all in-flight work, including background workers, in every chat they have used since that chat was last idle, pauses the cron jobs they created in any chat, and stops any of those jobs' runs in progress so their results are not delivered. Paused jobs stay paused if the sender is approved again. The CLI has matching `list`, `approve`, and `revoke` subcommands. A CLI revoke takes effect on the sender's next message but cannot cancel a run in progress. It lists their enabled cron jobs and prints a `deepagents-talon pairing pause-jobs <channel> <sender-id>` command to pause them. Run that only while Talon is stopped, because the running host is the cron store's only writer.

Only an operator id from `DEEPAGENTS_TALON_<CHANNEL>_OPERATOR_ID` can run `/pair`, and only in a DM. A paired sender cannot approve anyone, and the model has no pairing tool. Codes are accepted only on those operator surfaces, so strangers have nowhere to guess them. Each sender holds at most one live code, and a channel holds at most 16; further requests are dropped silently. Set `DEEPAGENTS_TALON_<CHANNEL>_PAIRING_REPLY=false` to keep the bot silent and read pending codes from `/pair list` instead.

A paired sender is admitted in every chat the bot can see, like an env operator, but never gains operator controls: they cannot run `/pair` or change tool approval policy, and tool approval prompts for their runs go to them. On Discord, where the bot reads every message in the channels it can see, any message a paired sender posts in those channels starts a run. Telegram group chats stay ignored for everyone. With pairing enabled, `DEEPAGENTS_TALON_<CHANNEL>_ALLOWLIST_USERS` also admits DMs in `self` mode; allowlisted users stay DM-only. Env stays authoritative: env operators and allowlisted users never receive codes and cannot be revoked with `/pair`. Paired senders are stored in `pairing.json` in the assistant home. If that file is unreadable or invalid, only env senders are admitted.

A paired sender has the same access to the agent as the operator: model credentials, MCP tools, and the local host. The agent's replies to a paired sender in a shared channel are visible to everyone in that channel, including members of other organizations in a Slack Connect channel. Because background work belongs to a chat rather than a person, revoking a sender also stops other people's in-flight work in chats the sender shared with them. Anyone admitted in a thread can edit that thread's scheduled jobs, and revocation pauses only the jobs the revoked sender created. Pair only people you would hand your terminal to.

## Tracing

LangSmith tracing is opt-in. Set both values before starting the host:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deepagents-talon
```

When enabled, Talon wraps each agent run in a LangSmith tracing context with assistant id, conversation id, trigger metadata, and source message metadata.

## Chat commands

The agent can call `send_message(text)` to post a progress update to the same chat
while continuing to work. Updates do not end the turn; the final reply is sent
normally. The destination is fixed by the host, and sending is disabled once the
originating turn finishes or is superseded. Runs without a channel cannot send updates.

Send `/help` for a brief guide to Talon, its built-in commands (`/new`, `/stop`,
`/mcp-reload`, `/context-doctor`, `/model`, and `/smart-model`), and using MCP configuration and OAuth through chat. Help does
not interrupt current work or consume a pending approval or sign-in response.

Send `/context-doctor` to estimate the token cost of the configured system prompt,
memory, skill index, and all active tool schemas (including MCP). It also shows
the current conversation estimate and the last provider-reported input count,
when available. Estimates exclude middleware additions and provider overhead.
The command reads the current chat's checkpoint without calling the model,
changing history, or interrupting active work. It reports counts, not contents.

Send `/model` to see which model the current chat uses and which providers are
available, and `/model <provider>` to list one provider's models. Talon discovers
the models that support tool calling from the installed LangChain provider
packages, using the same catalog as `dcode`. It lists only providers whose API key
is set in Talon's environment. `AGENT_MODEL` (the default) is always available.
An operator can send `/model <provider:model>` to switch the chat to one of those
models, or `/model default` to switch back. The switch applies from the chat's
next turn, lasts across `/new` and restarts, and does not affect other chats.
The chat's context is sized for the selected model, so switching to a model with a
larger or smaller context window changes when history is compacted. Scheduled jobs
and subagents keep their own models. The model is built the first time a chat
selects it, so a model that cannot be loaded is reported when you switch to it.

To let Talon ask a stronger model for one-off advice, set
`DEEPAGENTS_TALON_HELP_MODEL=<provider>:<model-id>` alongside its provider credentials,
or send `/smart-model <provider:model>` as an operator. `/smart-model` shows the
assistant-wide selection, `/smart-model off` disables consultations, and
`/smart-model default` restores the environment default. The choice persists across
chats and restarts; unlike `/model`, it does not change the conversation's model.
The command selects from models Talon can discover using its configured credentials;
it cannot change provider URLs or API keys. An operator who uses two OpenAI endpoints
must configure routing in the inference proxy or provider environment directly.
When enabled, this adds `ask_for_help(question)` to the main agent only. The tool sends just the question and a fixed instruction to the
configured model, not the chat history, tools, or filesystem. Only an operator's
main conversation can use it, and a channel approval prompt is always required
before the question leaves Talon; channels without approval support and scheduled
runs cannot use it. Inspect the exact question before approving: the agent can
include private content in it, and the destination provider receives that content.
The response is advice, not an instruction to run tools. With no environment
default or saved selection, the tool is omitted. The configured provider must be
installed and credentialed; configuration and provider errors surface when the tool
is called.

Commands work as ordinary message text on every channel, and are case-insensitive
with an optional `@bot` suffix. On Discord they are additionally registered as
native slash commands, so typing `/` offers them with autocomplete and the reply
arrives as that command's own response; see [Discord](#discord) below.

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
remove one server). Updates require human approval by default through the
`update_mcp_server` entry in `tools.json` and reload before the next turn.
Setting that entry to `false` disables its prompt, not validation or secret-safety
restrictions. Unprompted updates that reuse `<redacted>` values may change only
`allowedTools` and `disabledTools`; other managed settings must remain unchanged.
To change those settings, supply `${ENV_VAR}` references instead of redacted
values, or have the operator re-enable approval. Redaction is not permission to
redirect stored credentials.

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

Talon loads local subagents from `agents/<name>/AGENTS.md` using YAML frontmatter:
`description` is required, `name` defaults to the directory name, and `model` is
optional.

## Research defaults

Talon also installs the `configuration-hardening` skill and its reference under the
assistant home's `skills/` directory, preserving existing files. Ask it to review
tool separation or minimize tools; the default main instructions also trigger a
placement review when tools or subagents change. The skill proposes scoped changes,
uses existing confirmation controls, and verifies active attachments after reload.
Sensitive-action and access reviews are advisory: it never edits HITL/Ask controls.
Existing customized main instructions need a reviewed update to add this trigger.

Talon also installs a `safety` skill under `skills/`, preserving existing customizations.
Ask for a safety preflight, or use it when committing, pushing, building/publishing images,
or adding dependencies. It provides contextual guidance, not enforced tool restrictions or
automatic approvals; optional scanning tools are not installed automatically.

On startup, homes receive any missing `AGENTS.md` files for main, `internal-research`, and
`external-research`, with defensive prompts. External research declares `web: true` in its
frontmatter, which is what attaches `fetch_url` and Tavily-backed `web_search` at
construction; the capability follows the declaration, not the directory name. Search is added
only when `TAVILY_API_KEY` is nonempty in the runtime environment; without it,
startup and reload still work and `fetch_url` remains available.
Main and internal research are constructed without them; disabling web tools leaves
external research usable without built-in web access. Internal research starts
with `tools: []`. Main passes additional reads through `task(..., tools=[...])`, such as
applicable GitHub, Notion, email, and calendar reads internally. No integrations are
connected automatically. Set persistent tools with standard `tools` frontmatter;
launch-time additions apply only to that task. Main retains filesystem, action tools, and existing
approval controls, chooses placement from the workflow, and mediates minimal
internal-to-external context.

Existing files are unchanged; missing research definitions are installed automatically.
Review the packaged `deepagents_talon/defaults/` files,
back up affected instructions, and merge the selected changes without replacing custom
content. Call `reload_subagent_configuration` and inspect `get_agent_tools`; roll back
by restoring those files and reloading. Include restored capabilities in the rollback
review. Running tasks retain their original graphs until finished or canceled.

Prompts are not a sandbox: main filesystem/shell access, injected results, classification
mistakes, shared runtime/credentials, and retrieval of private destinations remain
operator-managed risks. The benign fixtures in `tests/unit_tests/fixtures/research_injections.json`
exercise missing capabilities and approval gates with scripted calls, not model refusal
or guaranteed public-only retrieval. Evaluate prompt behavior separately with your model.

## Background Subagents

Talon loads local `agents/<name>/AGENTS.md` definitions and remote
`[async_subagents]` configuration at startup. After adding, editing, or deleting
definitions, the main agent can call `reload_subagent_configuration` to apply the
changes on subsequent turns. Ordinary turns reuse the loaded definitions. Invalid
edits retain the last valid configuration; running subagents keep their original
configuration.

Subagents use fresh task context; fork is unsupported. Attach local tools with
`tools: [exact_tool_name]` (omitted means none); named agents start with those configured tools.
Add `web: true` to grant whichever web tools the runtime has, without naming them; an agent
without it never receives them, whatever its directory is called.
There is no automatic general-purpose agent; delegate to a research role or another
configured agent. Pass a `tools` list to `task` on each launch
to add capabilities to any local agent for that task, including `execute` for shell access. Supply context and skill
instructions in `description` or select
`read_file` to load them. `get_agent_tools` shows available attachments and inactive
edits; `list_subagents` shows launch-time additions.

`task` launches local subagents and `start_async_task` launches remote subagents.
In a chat conversation both return immediately. The user can continue chatting while
the main agent uses `list_subagents` to inspect work and `cancel_subagent` to cancel
it. When work finishes, its result is passed to the main agent for processing on the
next idle turn, then the main agent replies to the channel.

Workers and pending results live only in memory and are discarded on restart.
`/stop` and `/new` cancel all subagents belonging to that conversation; ordinary
messages interrupt only the main turn. Shutdown cancels all workers. Local tool
approval policy still applies; a child needing approval reports that it could not
complete the action. Remote runs cancel when their stream disconnects.

Talon allows four simultaneous subagents, retains at most 128 unprocessed jobs,
and limits each run to one hour. Completed results are capped at 64,000 characters.

### Scheduled runs

A scheduled run is already unattended, so it does not delegate in the background.
Both tools run the subagent to completion and return its result, and the run acts on
that result in the turn that asked for it; there is no follow-up turn and no separate
delivery. `list_subagents` and `cancel_subagent` are hidden from a scheduled run,
which owns no background work to inspect. Subagents launched in one assistant message
still run concurrently, and a scheduled run no longer competes with chat for the four
worker slots.

One delegation may take ten minutes, at most four run at once, and further ones queue
rather than being refused. Set `DEEPAGENTS_TALON_INLINE_SUBAGENT_TIMEOUT` to change
the per-delegation bound; because due jobs run one at a time, it caps how long one
stuck subagent holds up every other job. A delegation that overruns or fails reports
that to the run, which still writes and delivers its own reply. A whole run is bounded
at 30 minutes, after which its thread is repaired and the job is recorded as failed.

## Cron Schedules

`create_job` and `edit_job` accept five schedule forms:

| Form | Kind | Example |
| --- | --- | --- |
| `in <N>{m,h}` | one-shot | `in 30m` |
| `every <N>{m,h}` | recurring | `every 6h` |
| `at <YYYY-MM-DD> <HH:MM> <tz>` | one-shot | `at 2026-09-04 13:30 America/New_York` |
| `daily at <HH:MM> <tz>` | recurring | `daily at 08:00 America/New_York` |
| `cron <min> <hour> <dom> <month> <dow> <tz>` | recurring | `cron */15 * * jun mon-fri America/New_York` |

The `cron` form is a standard five-field crontab expression evaluated in the
given timezone's local time:

| Field | Values | Extensions |
| --- | --- | --- |
| minute | `0-59` | |
| hour | `0-23` | |
| day of month | `1-31` | `L` last day, `LW` last weekday, `15W` weekday nearest the 15th |
| month | `1-12`, `jan`-`dec` | |
| day of week | `0-7` (0 and 7 are Sunday), `sun`-`sat` | `5L` last Friday, `2#1` first Tuesday |

Each field accepts `*`, a value, a range (`1-5`), a step (`*/15`, `9-17/2`, or
`5/10` meaning 5 through the field maximum), and comma lists of those. `L`, `W`,
and `#` terms stand alone in their field. `@hourly`, `@daily`/`@midnight`,
`@weekly`, `@monthly`, and `@yearly`/`@annually` may replace the five fields, as
in `cron @daily UTC`; `@reboot` is not supported. As in Vixie cron, when both day
fields are restricted, a day matches if either one does; when either starts with
`*`, both must match. So `0 0 13 * 5` fires on the 13th and on every Friday,
while `0 0 */2 * 5` fires only on odd-numbered Fridays. An expression that can
never fire, such as `0 0 31 2 *`, is rejected at create time.

`create_job` and `edit_job` return an `upcoming` list with the next three run
times, so the agent can check a schedule against what the user asked for.

Recurring jobs accept an optional `until`, written `YYYY-MM-DD HH:MM <tz>`: the
last local time the job may run, inclusive. "Weekends at noon for the next three
months" is `cron 0 12 * * sat,sun <tz>` with `until` set three months out. A
`until` that falls before the first run is rejected. A run due inside the window
may start up to five minutes late to absorb scheduler latency; a run missed for
longer, such as across host downtime that spans `until`, is dropped rather than
delivered after the window closed. Pass `until=""` to `edit_job` to remove the
bound.

Jobs clean up after themselves. A job that will never run again (a one-shot that
ran, a recurring job whose `repeat_times` cap is used up, or one whose `until`
has passed) is deleted at the start of the next scheduler tick, whether or not
it ever ran. A job whose last run failed is kept, so `list_jobs` still shows the
error, until the retention window below removes it.

The wall-clock and cron forms require an explicit IANA timezone name; there is no default
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

`cron` schedules follow the same two rules. Every minute of a spring-forward gap
snaps to the same first valid minute, so `*/15 * * * *` fires once at 03:00
rather than four times. In a repeated fall-back hour, only the first pass fires.

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
- `cron.run_timeout`
- `cron.job_removed`

These logs complement the persisted `last_status` and `last_error` fields.

## Security and Data Lifecycle

Talon is single-operator by design. It does not provide multi-tenant isolation, production-grade HITL policy enforcement, or channel administrator boundaries. Any tool approval prompt surfaced through a channel is an experimental convenience feature, not a complete security boundary. Channel exposure should be treated as direct access to the operator's agent, model credentials, MCP tools, and local host resources. That includes senders admitted through [sender pairing](#sender-pairing).

`*_MENTION_PATTERNS` in `allowlist` mode admits any sender, in any chat, whose message text matches a pattern. Text says nothing about who sent it, so treat a mention pattern as opening that channel to anyone who can post there.

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
- `cron/jobs.json` stores cron prompts, origin conversation ids, message ids, run status, and errors. Active jobs are retained while enabled. Jobs that finish successfully or pass their `until` are deleted on the next scheduler tick. Jobs whose final run failed are deleted on startup after `DEEPAGENTS_TALON_CRON_RETENTION_DAYS`, default `30`.
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
