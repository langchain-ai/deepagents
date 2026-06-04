# Threat Model: DeepAgents Talon

> Generated: 2026-06-04 | Commit: 3ae9475a | Scope: `libs/talon` and `examples/talon-whatsapp`

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference. Findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. Suggestions and corrections are welcome.

## Scope

### In Scope

- `libs/talon/deepagents_talon/`: the `deepagents-talon` package, CLI host, WhatsApp channel adapter, loopback bridge client, cron scheduler/store/tools, MCP loader, voice transcription helper, tracing helper, and data-lifecycle cleanup.
- `libs/talon/deepagents_talon/channels/whatsapp_bridge/`: the packaged Node.js WhatsApp bridge used by the WhatsApp channel.
- `examples/talon-whatsapp/`: the runnable WhatsApp example and its environment defaults.
- The direct Talon call into `deepagents.create_deep_agent`, only where Talon controls the invocation shape.

### Out of Scope

- Other Deep Agents packages, except as dependencies invoked by Talon.
- Hosted Fleet, LangSmith control plane internals, and Fleet import/export behavior not implemented in this worktree.
- Remote sandbox backend selection for Talon. The current `DeepAgentRuntime` does not pass a backend to `create_deep_agent`; backend-specific threats are covered by the backend packages and the core Deep Agents threat models.
- User-authored agents, prompts, MCP servers, custom tools, custom channel adapters, and deployment infrastructure.
- Tests, benchmarks, and local ticket files.
- WhatsApp, model provider, OpenAI transcription, MCP server, and LangSmith provider-side infrastructure.

### Assumptions

1. Talon is an open source, single-operator local runtime. The project owns package code and defaults; operators own their environment, deployment, model choice, channel account, tool servers, and prompts.
2. `deepagents-talon` is a separate installable package. Users of the core `deepagents` package do not get Talon behavior unless they install and run this package.
3. The WhatsApp channel is not enabled unless the operator passes `--whatsapp` or sets `DEEPAGENTS_TALON_WHATSAPP_ENABLED`.
4. The current default `DeepAgentRuntime` uses `create_deep_agent` without a Talon-provided backend, so Talon does not currently expose a local shell backend or remote sandbox selector.
5. Conversation state is in-memory unless core Deep Agents or a future Talon feature adds durable thread persistence.
6. Published GitHub security advisories for `langchain-ai/deepagents` were checked during this assessment; none were returned for the public upstream repository.

## System Overview

DeepAgents Talon is a local host for a long-running Deep Agent. It receives channel messages, optionally transcribes voice messages, invokes a Deep Agents graph, delivers replies back to the channel, and can run persisted cron jobs that re-enter the same agent path. Talon stores assistant-scoped state under `~/.deepagents/<assistant_id>/` by default and integrates outbound with model providers, MCP servers, optional OpenAI transcription, optional LangSmith tracing, and WhatsApp through a local Node.js bridge.

### Architecture Diagram

```
Operator config/env/files
        |
        v
+------------------------------ local host / container ------------------------------+
|                                                                                     |
|  deepagents-talon CLI                                                               |
|        |                                                                            |
|        v                                                                            |
|  TalonConfig -----> assistant home, manifest, cron, channel, media dirs             |
|        |                                                                            |
|        v                                                                            |
|  TalonHost <---- loopback HTTP ---- WhatsAppChannel ---- Node WhatsApp bridge       |
|      |  ^                                 |                         |               |
|      |  |                                 |                         v               |
|      |  +------ PersistentCronScheduler <-+                  WhatsApp LocalAuth      |
|      |                 |                                                           |
|      |                 v                                                           |
|      |          CronJobStore/jobs.json                                              |
|      |                                                                             |
|      +----> OpenAIVoiceTranscriber ----> OpenAI transcription API                   |
|      |                                                                             |
|      +----> DeepAgentRuntime ----> create_deep_agent ----> model provider           |
|      |                                  ^                                           |
|      |                                  |                                           |
|      +----> load_mcp_tools -------------+----> MCP servers                          |
|      |                                                                             |
|      +----> langsmith_trace_context ----------> LangSmith tracing                   |
|                                                                                     |
+-------------------------------------------------------------------------------------+
                                    |
                                    v
                              WhatsApp service
```

## Components

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | Talon package and CLI | Separate package and console script that starts the host, channel adapters, cron scheduler, MCP commands, and runtime. | framework-controlled | Yes within `deepagents-talon`; No for core `deepagents` (separate install) | `libs/talon/pyproject.toml:project.scripts`, `libs/talon/deepagents_talon/__main__.py:main` |
| C2 | Runtime configuration and state directories | Reads selected environment variables, validates assistant IDs, and creates assistant-scoped local directories. | framework-controlled | Yes within `deepagents-talon` | `libs/talon/deepagents_talon/config.py:TalonConfig.from_env`, `libs/talon/deepagents_talon/config.py:TalonConfig.ensure_home` |
| C3 | Talon host | Owns lifecycle, channel registration, per-conversation serialization, `/stop` cancellation, scheduled job invocation, and trace wrapping. | framework-controlled | Yes when the CLI or embedding app starts a host | `libs/talon/deepagents_talon/host.py:TalonHost.start`, `libs/talon/deepagents_talon/host.py:TalonHost.receive_message`, `libs/talon/deepagents_talon/host.py:TalonHost.run_scheduled_job` |
| C4 | Deep Agent runtime wrapper | Constructs and invokes a Deep Agents graph from the configured model, MCP tools, and optional manifest system prompt. | framework-controlled wrapper around external dependency | No (requires `AGENT_MODEL` or `DEEPAGENTS_TALON_MODEL`; otherwise echo runtime) | `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.start`, `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.invoke`, `libs/deepagents/deepagents/graph.py:create_deep_agent` |
| C5 | WhatsApp channel adapter | Python adapter that starts or connects to the Node bridge, polls inbound messages, applies exposure policy, and sends/edit messages. | framework-controlled | No (explicit `--whatsapp` or env opt-in) | `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannelConfig.from_talon_config`, `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel.start`, `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel._poll_messages` |
| C6 | WhatsApp bridge | Packaged Node.js process using `whatsapp-web.js`, LocalAuth, Chromium, and loopback HTTP endpoints. | framework-controlled process over external WhatsApp service | No (started only when WhatsApp channel is enabled and bridge start is configured) | `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:handle`, `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:server.listen`, `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:client.on("message")` |
| C7 | Channel exposure and media helpers | Shared policy for `self`, `allowlist`, and `open`; output chunking/formatting; outbound media file validation. | framework-controlled | No (used by channel adapters) | `libs/talon/deepagents_talon/channels/base.py:ChannelExposure.allows`, `libs/talon/deepagents_talon/channels/base.py:validate_media`, `libs/talon/deepagents_talon/channels/whatsapp.py:_require_open_acknowledgement` |
| C8 | Cron scheduler and store | JSON-backed persistent cron jobs, due-job claiming, outcome recording, retention pruning, and optional agent-facing helper methods. | framework-controlled | No (scheduler is attached by CLI only when channels are configured; tools require explicit embedding) | `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.create_job`, `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.advance_next_run`, `libs/talon/deepagents_talon/cron/scheduler.py:PersistentCronScheduler.tick_once`, `libs/talon/deepagents_talon/cron/tools.py:CronTools.create_job` |
| C9 | MCP configuration and tool loader | Discovers `tools.json`/`.mcp.json` or env config, validates server entries, expands env references, loads tools, and applies allow/deny filters. | framework-controlled loader over user-controlled config and external servers | No (requires MCP config) | `libs/talon/deepagents_talon/mcp.py:load_mcp_tools`, `libs/talon/deepagents_talon/mcp.py:write_mcp_server_config`, `libs/talon/deepagents_talon/mcp.py:_connection` |
| C10 | Voice transcription helper | Optional OpenAI-backed transcription for channel messages carrying local voice/media paths. | framework-controlled wrapper over external provider | No (requires opt-in env and model) | `libs/talon/deepagents_talon/speech.py:build_voice_transcriber`, `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe`, `libs/talon/deepagents_talon/speech.py:transcribe_voice_message` |
| C11 | Observability and lifecycle cleanup | Optional LangSmith tracing context, structured cron logs, cron/media retention cleanup. | framework-controlled | Tracing: No; cleanup: Yes at CLI startup | `libs/talon/deepagents_talon/observability.py:langsmith_trace_context`, `libs/talon/deepagents_talon/observability.py:log_event`, `libs/talon/deepagents_talon/data_lifecycle.py:cleanup_sensitive_state` |
| C12 | External providers and operator services | WhatsApp, model provider, MCP servers, OpenAI transcription, LangSmith, and the operator filesystem/runtime. | external or user-controlled | No except where operator config enables them | Provider SDK/API calls from C4, C6, C9, C10, C11 |

## Data Classification

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | Channel session credentials | `channels/whatsapp/LocalAuth`, Chromium profile state, `WHATSAPP_SESSION_DIR` contents | Critical | Local session directory configured by `WhatsAppChannelConfig.session_dir`; Node bridge `LocalAuth` data path | No Talon-managed encryption; directory permissions set to `0700` | Until operator deletes the session directory | GDPR, CCPA, credential breach handling |
| DC2 | Runtime and provider secrets | `LANGSMITH_API_KEY`, `OPENAI_API_KEY`, MCP `headers`, MCP `env`, `${ENV_VAR}` expansions, provider credentials in process environment | Critical | Process environment; MCP config files may contain literal secrets or env references | Environment and config storage are operator-controlled; Talon does not encrypt them | Process lifetime for env; config files until operator deletes | Credential breach handling |
| DC3 | Channel identifiers and metadata | `ChannelMessage.conversation_id`, `sender_id`, `message_id`, WhatsApp `chat_id`, `user_id`, `user_name`, `chat_name`, `raw_message.from`, `raw_message.author` | High | In-memory request metadata; `cron/jobs.json` origin; LangSmith trace metadata when enabled; structured logs for cron delivery | No Talon-managed encryption; local files use `0600`/`0700`; external trace storage is provider-controlled | In-memory for conversation; cron retention for completed jobs defaults to 30 days; external provider-controlled for traces | GDPR, CCPA |
| DC4 | Conversation, prompt, and assistant content | `ChannelMessage.text`, voice transcript text, `CronJob.prompt`, `AGENTS.md`, assistant responses, selected tool outputs | High | In-memory agent request; `cron/jobs.json`; `agent/AGENTS.md`; model provider requests; LangSmith traces when enabled | No Talon-managed encryption; local files use filesystem permissions; external storage is provider-controlled | In-memory for live turns; active cron jobs while enabled; completed cron jobs default 30 days; manifests until operator deletes | GDPR, CCPA, contractual data-processing obligations |
| DC5 | Inbound media and audio | `voice_path`, `media_path`, files under `media/inbound/`, audio file bytes sent for transcription | High | Local inbound media directory; OpenAI transcription request when enabled | No Talon-managed encryption; directory permissions set to `0700`; provider storage is provider-controlled | Inbound media cleanup defaults to 24 hours; current WhatsApp bridge does not download inbound media | GDPR, CCPA |
| DC6 | MCP configuration, tool arguments, and tool results | `mcpServers.*.url`, `command`, `args`, `headers`, `env`, loaded tool schemas, model-chosen tool arguments, tool outputs | High | `~/.deepagents/.mcp.json`, `<assistant-home>/agent/tools.json`, process memory, external MCP servers | Config file encryption is operator-controlled; Talon writes config parent directories as `0700` | Config files until operator deletes; external server retention is provider-controlled | Depends on tool data; often GDPR, CCPA, credential handling |
| DC7 | Trace and lifecycle telemetry | Trace metadata `assistant_id`, `conversation_id`, `sender_id`, `message_id`, cron event fields `job_id`, `job_name`, `conversation_id`, error strings | Medium to High | Standard logs; LangSmith tracing context when enabled | Log sink and LangSmith storage are operator/provider-controlled | Log retention and LangSmith retention are operator/provider-controlled | GDPR, CCPA when identifiers or content are present |

### Data Classification Details

#### DC1: Channel session credentials

- **Fields**: WhatsApp `LocalAuth` data and Chromium profile state under the configured session directory.
- **Storage**: Created by `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:LocalAuth` and prepared by `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel.start`.
- **Access**: The local Talon process, the Node bridge process, and any local principal with filesystem access to the session directory.
- **Encryption**: Talon applies directory permissions through `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel.start`; it does not provide application-level encryption.
- **Retention**: Unbounded until the operator deletes the session directory.
- **Logging exposure**: The code does not intentionally log session material. Bridge status may expose `botId` through `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:handle`.
- **Cross-border**: WhatsApp provider-side session and message handling is external to Talon.
- **Gaps**: No Talon-managed encryption or automatic session credential expiry.

#### DC2: Runtime and provider secrets

- **Fields**: Environment credentials retained by `libs/talon/deepagents_talon/config.py:TalonConfig.from_env`; MCP literal headers/env values and `${ENV_VAR}` expansions resolved by `libs/talon/deepagents_talon/mcp.py:_expand_env`.
- **Storage**: Process environment and operator-managed MCP config files.
- **Access**: Talon process, MCP client library, subprocesses started by the operator-configured MCP transport, and any local principal with access to config files or process environment.
- **Encryption**: Talon does not encrypt environment variables or config files.
- **Retention**: Process lifetime for environment values; config lifetime for values written to disk.
- **Logging exposure**: MCP server load errors are logged by `libs/talon/deepagents_talon/mcp.py:load_mcp_tools_from_config`; secrets in exception messages are not redacted by Talon.
- **Cross-border**: Secrets may be sent to configured providers or MCP servers according to operator configuration.
- **Gaps**: No built-in secret redaction for logs or validation that MCP headers avoid literal secrets.

#### DC3: Channel identifiers and metadata

- **Fields**: `ChannelMessage` identifiers and WhatsApp message metadata parsed by `libs/talon/deepagents_talon/channels/whatsapp.py:_parse_message`.
- **Storage**: In-memory while handling messages; cron origin in `cron/jobs.json`; LangSmith trace metadata when enabled by `libs/talon/deepagents_talon/observability.py:langsmith_trace_context`.
- **Access**: Talon host, scheduler, channel adapter, optional tracing provider.
- **Encryption**: Local cron file permissions are enforced by `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore._write_jobs`; external trace encryption is provider-controlled.
- **Retention**: In-memory for normal message handling; completed cron jobs are pruned by `libs/talon/deepagents_talon/data_lifecycle.py:cleanup_sensitive_state`.
- **Logging exposure**: `libs/talon/deepagents_talon/cron/scheduler.py:PersistentCronScheduler._run_due_job` logs `conversation_id` for cron dispatch and delivery.
- **Cross-border**: Trace metadata may leave the host when LangSmith tracing is enabled.
- **Gaps**: Trace/log sinks are operator-controlled and Talon does not redact channel identifiers.

#### DC4: Conversation, prompt, and assistant content

- **Fields**: `AgentRequest.text`, `CronJob.prompt`, manifest `AGENTS.md`, voice transcript text, assistant output.
- **Storage**: In-memory request path through `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent`; `cron/jobs.json`; manifest directory; model provider and LangSmith when configured.
- **Access**: Talon host, Deep Agents graph, model provider, optional MCP tools and tracing provider.
- **Encryption**: Talon provides local filesystem permissions but no content encryption.
- **Retention**: Active cron prompts persist while jobs are enabled; completed jobs default to 30-day retention; manifest files persist until operator deletion.
- **Logging exposure**: Unhandled agent errors are logged by `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent`; cron `last_error` is persisted by `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.mark_job_run`.
- **Cross-border**: Model provider, MCP server, tracing provider, and WhatsApp channel delivery are external boundaries.
- **Gaps**: No content classification or redaction before model, MCP, channel, or tracing egress.

#### DC5: Inbound media and audio

- **Fields**: Local paths from channel metadata and file bytes opened by `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe`.
- **Storage**: Reserved inbound media directory under assistant home. The current WhatsApp bridge records media type metadata but does not download inbound media.
- **Access**: Talon host and optional transcription provider.
- **Encryption**: Local directory permissions only.
- **Retention**: Files older than the configured media retention window are deleted by `libs/talon/deepagents_talon/data_lifecycle.py:cleanup_sensitive_state`.
- **Logging exposure**: Missing media paths are logged by `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe`.
- **Cross-border**: Audio bytes leave the host when OpenAI transcription is enabled.
- **Gaps**: `voice_path`/`media_path` are trusted when supplied by a channel adapter; custom adapters must not derive those paths directly from attacker-controlled text.

#### DC6: MCP configuration, tool arguments, and tool results

- **Fields**: MCP server connection configuration and all tool schemas/results exposed to the model through `libs/talon/deepagents_talon/mcp.py:load_mcp_tools`.
- **Storage**: Config files, process memory, external MCP servers, and model context.
- **Access**: Talon, model provider, MCP server process or endpoint.
- **Encryption**: Talon writes config directories with restrictive permissions through `libs/talon/deepagents_talon/mcp.py:write_mcp_server_config`, but does not encrypt config contents.
- **Retention**: Operator-controlled config lifetime; provider/server-controlled for external MCP state.
- **Logging exposure**: Server load errors can be logged by `libs/talon/deepagents_talon/mcp.py:load_mcp_tools_from_config`.
- **Cross-border**: Remote MCP transports send data over HTTP/SSE; stdio MCP may invoke local commands configured by the operator.
- **Gaps**: No domain allowlist for remote MCP URLs and no sandboxing for operator-configured stdio MCP commands.

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | Operator configuration to Talon | Environment variables, CLI flags, assistant home files, manifest files, and MCP config enter framework code. | `TalonConfig.from_env`, `_validate_assistant_id`, `_exposure_from_env`, `_validate_server_config`, `write_mcp_server_config` | Who can set env vars, file contents, manifest prompt, provider credentials, and deployment filesystem permissions |
| TB2 | WhatsApp participants to Talon channel | WhatsApp messages enter the Node bridge, cross loopback HTTP, then reach the Python adapter and host. | `ChannelExposure.allows`, `_parse_message`, `_required_str`, `_require_open_acknowledgement`, `BridgeTransport._request` | WhatsApp sender authenticity, message content, group membership, WhatsApp delivery semantics, and the local bridge process if independently compromised |
| TB3 | Talon host to Deep Agents runtime and model provider | Talon passes untrusted text and metadata to a Deep Agents graph, which calls the selected chat model. | `DeepAgentRuntime.invoke`, `DeepAgentRuntime.start`, `TalonHost._invoke_agent` | Model behavior, provider retention, user-selected model, and core Deep Agents internal behavior beyond the invocation contract |
| TB4 | Talon to MCP servers and tools | Talon loads operator-configured MCP servers and exposes their tools to the agent. | `load_mcp_tools`, `_validate_config`, `_connection`, `_apply_tool_filter`, `_expand_env` | MCP server behavior, tool side effects, server authentication, remote endpoint security, and stdio command safety |
| TB5 | Talon to local sensitive state | Talon persists assistant-scoped cron, manifest, channel, and media state under a local home directory. | `TalonConfig.ensure_home`, `CronJobStore._ensure_store`, `CronJobStore._write_jobs`, `cleanup_sensitive_state` | Host-level disk encryption, backups, local OS users, filesystem ACLs outside the assistant home, and operator deletion policy |
| TB6 | Talon Python process to Node bridge loopback API | Python sends unauthenticated loopback HTTP requests to the packaged bridge; the bridge exposes `/health`, `/messages`, `/send`, `/send-media`, and `/edit`. | `_validate_loopback_url`, `WhatsAppChannel._post_result`, `bridge.js:handle`, `bridge.js:server.listen` | Other local processes that can reach the same loopback port, Node dependency behavior, Chromium behavior, and WhatsApp Web behavior |
| TB7 | Talon to observability and transcription providers | Optional tracing and transcription send metadata, prompts, outputs, or audio bytes to external providers. | `langsmith_tracing_enabled`, `langsmith_trace_context`, `build_voice_transcriber`, `OpenAIVoiceTranscriber.transcribe` | Provider-side retention, access controls, model/transcription behavior, and data residency |
| TB8 | Talon scheduler to future agent turns | Persisted cron prompts and origins later re-enter the same agent and channel delivery path. | `CronSchedule.parse`, `CronJobStore.advance_next_run`, `PersistentCronScheduler.tick_once`, `TalonHost.run_scheduled_job` | Whether the stored prompt was authored safely, whether a user or model was allowed to create/edit jobs, and external channel availability |

### Boundary Details

#### TB1: Operator configuration to Talon

- **Inside**: Talon validates assistant IDs in `libs/talon/deepagents_talon/config.py:_validate_assistant_id`, parses channel exposure in `libs/talon/deepagents_talon/channels/whatsapp.py:_exposure_from_env`, and validates MCP config shape in `libs/talon/deepagents_talon/mcp.py:_validate_server_config`.
- **Outside**: Operators control env values, filesystem paths, model IDs, manifest text, MCP server definitions, and provider credentials.
- **Crossing mechanism**: Process environment, CLI arguments, JSON config files, and local files.

#### TB2: WhatsApp participants to Talon channel

- **Inside**: The adapter parses required message fields in `libs/talon/deepagents_talon/channels/whatsapp.py:_parse_message` and gates triggers through `libs/talon/deepagents_talon/channels/base.py:ChannelExposure.allows`.
- **Outside**: WhatsApp participants control message content and metadata presented through WhatsApp Web. Talon does not cryptographically authenticate channel senders.
- **Crossing mechanism**: WhatsApp Web event -> Node in-memory queue -> loopback HTTP `/messages` -> Python `ChannelMessage`.

#### TB3: Talon host to Deep Agents runtime and model provider

- **Inside**: Talon builds a request containing `conversation_id`, text, and metadata in `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent`, then passes user content to `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.invoke`.
- **Outside**: Model providers and model outputs are external. Prompt injection and tool-choice behavior are not fully controlled by Talon.
- **Crossing mechanism**: Python function call into Deep Agents and provider SDK calls inside the graph.

#### TB4: Talon to MCP servers and tools

- **Inside**: Talon validates server names and transports, resolves env references, loads tools, and applies allow/deny filters.
- **Outside**: MCP servers define their own tools and side effects. Remote MCP endpoints and stdio command binaries are operator-controlled.
- **Crossing mechanism**: JSON config -> MCP client connection -> tool schemas/results exposed to model.

#### TB5: Talon to local sensitive state

- **Inside**: Talon creates assistant-scoped state directories with `0700`, writes cron files with `0600`, and prunes completed cron/media state.
- **Outside**: Disk encryption, local account compromise, backups, and manual deletion are operator responsibilities.
- **Crossing mechanism**: Local filesystem reads and writes.

#### TB6: Talon Python process to Node bridge loopback API

- **Inside**: Talon validates that the Python client uses HTTP loopback and packages the bridge script path.
- **Outside**: Any local process able to reach the same loopback port may interact with the bridge; the bridge itself has no request authentication.
- **Crossing mechanism**: HTTP on loopback.

#### TB7: Talon to observability and transcription providers

- **Inside**: Talon requires explicit tracing enablement plus an API key, and requires explicit voice transcription enablement plus a model.
- **Outside**: Provider retention, access controls, and content handling are external.
- **Crossing mechanism**: Provider SDK calls and tracing context.

#### TB8: Talon scheduler to future agent turns

- **Inside**: Talon validates schedule grammar, advances due jobs before invocation, scopes cron tools by conversation, and records outcomes.
- **Outside**: Whether persisted prompt content is trustworthy depends on who can create/edit jobs and what tools the operator exposes.
- **Crossing mechanism**: JSON `jobs.json` -> scheduler -> agent invocation -> channel delivery.

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | Operator environment and CLI | Talon configuration | Assistant ID, home path, model ID, tracing flags, channel exposure, provider credentials | DC2, DC7 | TB1 | Environment variables and argparse |
| DF2 | Talon configuration | Local assistant state | Manifest, cron, channel, media directories | DC1, DC3, DC4, DC5 | TB5 | Local filesystem |
| DF3 | WhatsApp service and participants | Node bridge queue | Message text, sender/chat identifiers, media metadata | DC3, DC4 | TB2 | WhatsApp Web callbacks |
| DF4 | Node bridge | WhatsApp channel adapter | Queued messages and bridge health | DC3, DC4 | TB2, TB6 | Loopback HTTP JSON |
| DF5 | WhatsApp channel adapter | Talon host | Accepted `ChannelMessage` values | DC3, DC4 | TB2 | Python callback |
| DF6 | Talon host | Voice transcription provider | Local audio file bytes and transcript text | DC4, DC5 | TB7 | OpenAI SDK |
| DF7 | Talon host | Deep Agents graph and model provider | User message text, cron prompt, manifest system prompt, metadata, selected tool outputs | DC3, DC4, DC6 | TB3 | Python call and model provider SDK |
| DF8 | Agent or embedding app | Cron job store | Cron prompt, schedule, repeat cap, origin conversation/message ids | DC3, DC4 | TB8, TB5 | Python API and JSON file |
| DF9 | Cron scheduler | Talon host and channel | Persisted cron prompt and result delivery target | DC3, DC4, DC7 | TB8, TB2 | Python callback and channel send |
| DF10 | Operator MCP config | MCP client and external MCP servers | Server URLs, stdio commands, headers/env values, tool filters | DC2, DC6 | TB1, TB4 | JSON config, stdio, HTTP/SSE |
| DF11 | External MCP servers | Deep Agents graph and model context | Tool schemas, tool results, errors | DC4, DC6 | TB4, TB3 | MCP protocol and model context |
| DF12 | Talon host and scheduler | LangSmith tracing and logs | Trace metadata, run context, cron event records, errors | DC3, DC4, DC7 | TB7 | Provider SDK and Python logging |
| DF13 | Talon channel adapter | Node bridge and WhatsApp service | Assistant replies, edits, outbound media paths | DC4, DC5 | TB6, TB2 | Loopback HTTP JSON and WhatsApp Web |
| DF14 | Data lifecycle cleanup | Local assistant state | Expired completed cron records and inbound media files | DC4, DC5 | TB5 | Local filesystem unlink |

### Flow Details

#### DF1: Operator environment and CLI -> Talon configuration

- **Data**: Assistant ID, home path, model ID, channel exposure, tracing flags, and credentials.
- **Validation**: `libs/talon/deepagents_talon/config.py:_validate_assistant_id`, `libs/talon/deepagents_talon/channels/whatsapp.py:_exposure_mode`, `libs/talon/deepagents_talon/channels/whatsapp.py:_require_open_acknowledgement`, `libs/talon/deepagents_talon/mcp.py:_validate_server_config`.
- **Trust assumption**: Operators can set safe environment values and protect local secrets.

#### DF2: Talon configuration -> Local assistant state

- **Data**: Assistant-scoped directories for manifests, cron jobs, channel sessions, and inbound media.
- **Validation**: `libs/talon/deepagents_talon/config.py:TalonConfig.ensure_home` creates directories and applies `0700`.
- **Trust assumption**: Local filesystem permissions are meaningful in the deployment environment.

#### DF3: WhatsApp service and participants -> Node bridge queue

- **Data**: WhatsApp message body, chat id, contact id, contact names, message id, media metadata, and raw message identifiers.
- **Validation**: The bridge maps WhatsApp Web events to a fixed JSON shape in `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:client.on("message")`.
- **Trust assumption**: Message content and identifiers are untrusted until the Python adapter applies exposure policy.

#### DF4: Node bridge -> WhatsApp channel adapter

- **Data**: Queued messages and health status.
- **Validation**: The Python transport requires loopback in `libs/talon/deepagents_talon/channels/whatsapp.py:_validate_loopback_url` and validates JSON shape in `libs/talon/deepagents_talon/channels/whatsapp.py:_parse_messages`.
- **Trust assumption**: The local loopback bridge is reachable only by trusted local processes.

#### DF5: WhatsApp channel adapter -> Talon host

- **Data**: Accepted `ChannelMessage` values.
- **Validation**: `libs/talon/deepagents_talon/channels/base.py:ChannelExposure.allows` gates `self`, `allowlist`, and `open` exposure modes.
- **Trust assumption**: Exposure policy is configured for the operator's intended blast radius.

#### DF6: Talon host -> Voice transcription provider

- **Data**: Local audio file bytes and returned transcript text.
- **Validation**: `libs/talon/deepagents_talon/speech.py:build_voice_transcriber` requires opt-in and a model; `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe` checks `Path.is_file`.
- **Trust assumption**: Channel adapters provide safe local media paths and operators accept provider-side processing.

#### DF7: Talon host -> Deep Agents graph and model provider

- **Data**: Channel text, cron prompts, voice transcripts, manifest prompt, and tool outputs.
- **Validation**: `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.invoke` passes text as a user message; `libs/deepagents/deepagents/graph.py:create_deep_agent` defaults to `StateBackend` when no backend is supplied.
- **Trust assumption**: The operator-selected model and configured tools are appropriate for the channel exposure level.

#### DF8: Agent or embedding app -> Cron job store

- **Data**: Scheduled prompt, schedule text, repeat cap, and origin identifiers.
- **Validation**: `libs/talon/deepagents_talon/cron/jobs.py:CronSchedule.parse`, `libs/talon/deepagents_talon/cron/jobs.py:CronRepeat.__post_init__`, `libs/talon/deepagents_talon/cron/tools.py:CronTools.create_job`.
- **Trust assumption**: Only authorized users or model/tool paths can create/edit jobs for a conversation.

#### DF9: Cron scheduler -> Talon host and channel

- **Data**: Persisted prompt and delivery target.
- **Validation**: `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.advance_next_run` claims before execution; `libs/talon/deepagents_talon/cron/scheduler.py:_is_silent` supports delivery suppression.
- **Trust assumption**: Stored prompts remain trusted enough to re-enter the agent context later.

#### DF10: Operator MCP config -> MCP client and external MCP servers

- **Data**: Server definitions, commands, URLs, headers, env references, and filters.
- **Validation**: `libs/talon/deepagents_talon/mcp.py:_validate_config`, `libs/talon/deepagents_talon/mcp.py:_resolve_transport`, `libs/talon/deepagents_talon/mcp.py:_expand_env`.
- **Trust assumption**: The operator controls MCP config and does not import untrusted `tools.json` without review.

#### DF11: External MCP servers -> Deep Agents graph and model context

- **Data**: Tool schemas, results, and errors.
- **Validation**: `libs/talon/deepagents_talon/mcp.py:_apply_tool_filter` constrains visible tools when configured.
- **Trust assumption**: MCP tool output is untrusted content and may influence subsequent model behavior.

#### DF12: Talon host and scheduler -> LangSmith tracing and logs

- **Data**: Assistant ID, conversation ID, trigger metadata, message IDs, cron job names, and errors.
- **Validation**: `libs/talon/deepagents_talon/observability.py:langsmith_tracing_enabled` requires explicit tracing plus API key; `libs/talon/deepagents_talon/observability.py:log_event` emits JSON.
- **Trust assumption**: Operators configure log and trace sinks with retention and access controls suitable for the data.

#### DF13: Talon channel adapter -> Node bridge and WhatsApp service

- **Data**: Assistant replies, edits, and outbound media file paths.
- **Validation**: `libs/talon/deepagents_talon/channels/base.py:chunk_text`, `libs/talon/deepagents_talon/channels/base.py:format_markdown_for_channel`, `libs/talon/deepagents_talon/channels/base.py:validate_media`, `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel._post_result`.
- **Trust assumption**: The target conversation is correct and any outbound media path was selected by trusted code.

#### DF14: Data lifecycle cleanup -> Local assistant state

- **Data**: Completed cron records and inbound media files.
- **Validation**: `libs/talon/deepagents_talon/data_lifecycle.py:_env_non_negative_int`, `libs/talon/deepagents_talon/data_lifecycle.py:_delete_old_files`.
- **Trust assumption**: Cleanup runs on startup often enough for the configured retention to be meaningful.

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|------------|----------------|
| T1 | DF5, DF7, DF13 | DC3, DC4 | `open` WhatsApp exposure lets arbitrary senders drive the operator-scoped agent and any configured tools. | TB2, TB3 | Medium | Verified | `libs/talon/deepagents_talon/channels/whatsapp.py:_require_open_acknowledgement`, `libs/talon/deepagents_talon/host.py:TalonHost.receive_message` |
| T2 | DF4, DF13 | DC3, DC4, DC5 | Unauthenticated loopback bridge endpoints can be abused by another local process to read queued messages or send/edit WhatsApp messages. | TB6 | Medium | Verified | `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:handle`, `libs/talon/deepagents_talon/channels/whatsapp.py:_validate_loopback_url` |
| T3 | DF2, DF3 | DC1 | WhatsApp session credentials persist locally without Talon-managed encryption or automatic expiry. | TB5, TB2 | High | Verified | `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:LocalAuth`, `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel.start` |
| T4 | DF7, DF10, DF11, DF12 | DC2, DC4, DC6, DC7 | Sensitive conversation content, cron prompts, tool data, identifiers, or secrets can leave the host through model, MCP, tracing, or tool integrations. | TB3, TB4, TB7 | Medium | Likely | `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.invoke`, `libs/talon/deepagents_talon/mcp.py:load_mcp_tools`, `libs/talon/deepagents_talon/observability.py:langsmith_trace_context` |
| T5 | DF8, DF9 | DC3, DC4 | Persisted cron prompts can become delayed context injection that re-enters the agent after the original conversation turn has ended. | TB8, TB3 | Medium | Verified | `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.create_job`, `libs/talon/deepagents_talon/cron/scheduler.py:PersistentCronScheduler.tick_once`, `libs/talon/deepagents_talon/host.py:TalonHost.run_scheduled_job` |
| T6 | DF10, DF11 | DC2, DC6 | Operator-configured MCP servers can expose powerful tools or run stdio commands that the model may invoke with conversation-derived arguments. | TB4, TB3 | Medium | Verified | `libs/talon/deepagents_talon/mcp.py:_connection`, `libs/talon/deepagents_talon/mcp.py:_apply_tool_filter`, `libs/talon/deepagents_talon/__main__.py:_server_config_from_args` |
| T7 | DF6 | DC5, DC4 | Voice transcription can exfiltrate any local file path that a channel adapter supplies as `voice_path` or `media_path`. | TB7, TB2 | Medium | Likely for custom adapters; not reachable through the current WhatsApp bridge | `libs/talon/deepagents_talon/speech.py:_voice_path`, `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe` |
| T8 | DF5, DF7, DF8, DF9 | DC4, DC7 | No built-in rate limits or quotas means open/allowlisted channels or aggressive cron schedules can drive model/tool cost and local resource exhaustion. | TB2, TB3, TB8 | Medium | Verified | `libs/talon/deepagents_talon/host.py:TalonHost.receive_message`, `libs/talon/deepagents_talon/cron/scheduler.py:PersistentCronScheduler.tick_once`, `libs/talon/deepagents_talon/cron/jobs.py:CronSchedule.parse` |
| T9 | DF12 | DC3, DC4, DC7 | Logs and trace metadata may contain personal identifiers, cron job names, or error strings derived from channel/tool content. | TB7 | Low | Verified | `libs/talon/deepagents_talon/observability.py:log_event`, `libs/talon/deepagents_talon/cron/scheduler.py:PersistentCronScheduler._run_due_job`, `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent` |

### Threat Details

#### T1: Arbitrary sender trigger in `open` exposure

- **Flow**: DF5 and DF7 (WhatsApp channel -> Talon host -> Deep Agents graph)
- **Description**: In `open` exposure, any WhatsApp sender whose message reaches the paired account can trigger the same agent instance that uses operator-configured model credentials, MCP tools, tracing configuration, and channel account.
- **Preconditions**: The operator enables WhatsApp and sets both `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open` and the explicit acknowledgement value. Impact depends on the tools/model/providers configured by the operator.
- **Validation**: `libs/talon/deepagents_talon/channels/whatsapp.py:_require_open_acknowledgement` blocks open exposure without acknowledgement, and `libs/talon/deepagents_talon/channels/base.py:ChannelExposure.allows` implements the exposure decision.

#### T2: Local loopback bridge abuse

- **Flow**: DF4 and DF13 (Node bridge <-> Python adapter)
- **Description**: The bridge endpoints are unauthenticated because they are intended to bind to loopback. A different local process that can reach the bridge port could call `/messages`, `/send`, `/send-media`, or `/edit`.
- **Preconditions**: The attacker can run code on the same host/container network namespace or otherwise reach the configured loopback port.
- **Validation**: `libs/talon/deepagents_talon/channels/whatsapp.py:_validate_loopback_url` prevents the Python client from targeting a non-loopback bridge URL. `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:server.listen` binds to the configured host.

#### T3: Persistent WhatsApp session credential exposure

- **Flow**: DF2 and DF3 (session directory -> bridge authentication)
- **Description**: WhatsApp pairing material persists on disk so the bridge can reconnect without requiring a new QR scan. Compromise of that local state may allow account reuse outside Talon.
- **Preconditions**: The attacker can read the local session directory, backups, mounted volume, or container state.
- **Validation**: `libs/talon/deepagents_talon/channels/whatsapp.py:WhatsAppChannel.start` creates the session directory and applies restrictive permissions, while `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:LocalAuth` stores provider authentication material there.

#### T4: Sensitive outbound data exposure through integrations

- **Flow**: DF7, DF10, DF11, and DF12 (host/model/MCP/tracing)
- **Description**: Conversation text, cron prompts, manifest instructions, tool outputs, channel identifiers, and error strings may leave the host through the selected model provider, MCP servers, or LangSmith tracing.
- **Preconditions**: The operator configures a real model, remote MCP server, tracing, or tools that receive conversation-derived data.
- **Validation**: `libs/talon/deepagents_talon/observability.py:langsmith_tracing_enabled` requires tracing opt-in, and `libs/talon/deepagents_talon/mcp.py:_validate_server_config` validates MCP config shape, but Talon does not classify or redact content before external egress.

#### T5: Persisted cron prompt context injection

- **Flow**: DF8 and DF9 (cron store -> scheduled agent turn)
- **Description**: A cron job stores free-form prompt text and later replays it into the agent. If an attacker can cause job creation or edit a job through an exposed tool path, the injected instruction persists beyond the original channel message.
- **Preconditions**: Cron jobs are exposed to an agent or embedding app and an attacker can influence the stored prompt. In this branch, `CronTools` is public API but not automatically passed to `DeepAgentRuntime` by the CLI.
- **Validation**: `libs/talon/deepagents_talon/cron/tools.py:CronTools.create_job` scopes tool operations to the current origin, and `libs/talon/deepagents_talon/cron/jobs.py:CronSchedule.parse` limits supported schedule syntax. Prompt content itself is not sanitized before later invocation.

#### T6: MCP tool and stdio command exposure

- **Flow**: DF10 and DF11 (MCP config -> agent tools)
- **Description**: MCP config can connect to remote endpoints or launch stdio commands. Loaded tools are exposed to the model, which can call them using arguments derived from channel or cron content.
- **Preconditions**: The operator supplies MCP config through environment, `~/.deepagents/.mcp.json`, `<assistant-home>/agent/tools.json`, or the `deepagents-talon mcp add` command.
- **Validation**: `libs/talon/deepagents_talon/mcp.py:_validate_server_config` validates shape and mutual exclusivity of allow/deny filters; `libs/talon/deepagents_talon/mcp.py:_apply_tool_filter` constrains visible tools when configured.

#### T7: Voice path exfiltration through custom adapters

- **Flow**: DF6 (channel metadata -> transcription provider)
- **Description**: The transcriber opens the path supplied in `voice_path` or `media_path` and sends bytes to OpenAI. If a channel adapter maps attacker-controlled content into those metadata fields, arbitrary readable local files could be sent to the provider.
- **Preconditions**: Voice transcription is enabled, a channel message is classified as voice, and the adapter provides an attacker-controlled local path. The current packaged WhatsApp bridge does not populate `voice_path` or `media_path`.
- **Validation**: `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe` checks only that the path is a file before opening it.

#### T8: Resource exhaustion and cost amplification

- **Flow**: DF5, DF7, DF8, and DF9 (channels/cron -> agent/model/tools)
- **Description**: Talon serializes work per conversation but does not enforce sender quotas, global concurrency limits, channel rate limits, or model cost limits. Open channels, large allowlists, or many due cron jobs can repeatedly invoke the model and tools.
- **Preconditions**: The operator enables a channel exposure mode that permits the attacker or configures enough cron jobs to create load.
- **Validation**: `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent` serializes by conversation ID, and `libs/talon/deepagents_talon/cron/jobs.py:CronSchedule.parse` enforces a one-minute minimum schedule granularity.

#### T9: Identifier and error leakage in logs/traces

- **Flow**: DF12 (host/scheduler -> logs/tracing)
- **Description**: Cron logs include conversation IDs and job names; trace metadata includes assistant ID, conversation ID, sender ID, and message ID. Errors from tools or providers may include sensitive values.
- **Preconditions**: The operator collects logs or enables LangSmith tracing; an attacker or tool can influence names, identifiers, or error messages.
- **Validation**: `libs/talon/deepagents_talon/observability.py:log_event` produces structured JSON but does not redact fields.

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input | DF3, DF4, DF5, DF13 | T1, T2, T8, T9 | `ChannelExposure.allows`, `_parse_message`, `_required_str`, `_require_open_acknowledgement`, `chunk_text`, `validate_media` | Shared: Talon owns defaults and parsing; operator owns exposure choice and channel account | No cryptographic sender auth; no quotas |
| LLM output | DF7, DF8, DF13 | T4, T5, T6, T8 | `DeepAgentRuntime.invoke`, `CronTools.create_job`, `validate_media`, `_apply_tool_filter` | Shared: Talon owns routing contracts; operator owns model and tool choice | No semantic content safety or redaction |
| Tool/function results | DF11, DF12 | T4, T6, T9 | `_apply_tool_filter`, `log_event`, `langsmith_trace_context` | Shared: Talon owns config validation; operator owns server trust | Tool results can poison future model context |
| URL-sourced content | DF11 | T4, T6 | `_validate_server_config`, `_connection` | Operator for configured MCP/search tools; Talon for config shape | No URL/domain allowlist for remote MCP endpoints |
| Configuration | DF1, DF2, DF10 | T1, T3, T4, T6, T7 | `_validate_assistant_id`, `_exposure_mode`, `_require_open_acknowledgement`, `_validate_server_config`, `_expand_env` | Shared: Talon owns validation; operator owns values | Literal secrets in config are not redacted or encrypted |
| Inter-service calls | DF6, DF7, DF10, DF11, DF12, DF13 | T4, T6, T7, T9 | `langsmith_tracing_enabled`, `build_voice_transcriber`, `_connection`, `_post_result` | Shared with providers and operator | Provider retention and data residency are external |
| Local filesystem state | DF2, DF8, DF14 | T3, T5, T7 | `TalonConfig.ensure_home`, `CronJobStore._write_jobs`, `cleanup_sensitive_state` | Shared: Talon owns file modes/cleanup; operator owns host disk and backups | No content encryption; WhatsApp session retention is unbounded |

## Out-of-Scope Threats

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection causing arbitrary code execution through user-selected tools or backends | Talon does not currently select a local shell backend and does not automatically expose cron tools to the default `DeepAgentRuntime`. Operators control custom MCP tools, model selection, and any future backend choice. | `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.start` calling `libs/deepagents/deepagents/graph.py:create_deep_agent` without a backend |
| Malicious MCP server intentionally configured by the operator | MCP server behavior, stdio binaries, remote endpoint control, and tool semantics are operator-controlled. Talon validates config shape and filtering, not server trustworthiness. | `libs/talon/deepagents_talon/mcp.py:_validate_server_config` and `libs/talon/deepagents_talon/mcp.py:_apply_tool_filter` |
| Compromise of WhatsApp, OpenAI, LangSmith, model provider, or MCP provider infrastructure | Provider infrastructure and account controls are outside Talon. Talon only decides when to send data to configured providers. | `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:client.initialize`, `libs/talon/deepagents_talon/speech.py:OpenAIVoiceTranscriber.transcribe`, `libs/talon/deepagents_talon/observability.py:langsmith_trace_context` |
| Multi-tenant isolation bypass | Talon is single-operator and single-assistant-per-process by design in this scope. It does not claim tenant isolation. | `libs/talon/deepagents_talon/config.py:TalonConfig.from_env` assistant namespacing and `libs/talon/deepagents_talon/host.py:TalonHost` single host instance |
| Fleet import trust and unsupported Fleet fields | Fleet import is not implemented in this worktree. Manifest import policy should be modeled when code lands. | No current Talon import entry point found under `libs/talon/deepagents_talon/` |
| Durable conversation persistence data exposure | Talon does not implement durable conversation persistence in this branch; normal conversation state is in-memory. | `libs/talon/deepagents_talon/host.py:TalonHost._invoke_agent` request construction |

### Rationale

Talon is a local runtime wrapper, not a managed multi-tenant service. Findings that require the operator to install a malicious MCP server, select an unsafe model/provider, expose dangerous tools, or run on a compromised host are primarily operator/deployer responsibility. Talon remains responsible for safe defaults, accurate opt-in friction, local parsing, restrictive local file modes, and clear boundaries where content leaves the process.

## Investigated and Dismissed

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|-----------------|---------------|----------|------------|
| D1 | Assistant ID path traversal can write state outside the assistant home | Traced environment loading and state directory creation. | `libs/talon/deepagents_talon/config.py:_validate_assistant_id`, `libs/talon/deepagents_talon/config.py:TalonConfig.from_env` | Disproven for `assistant_id`; only letters, numbers, underscore, hyphen, and dot are accepted. |
| D2 | Python bridge client can be configured to SSRF arbitrary HTTP endpoints | Traced `WhatsAppChannelConfig.base_url` into `BridgeTransport`. | `libs/talon/deepagents_talon/channels/whatsapp.py:_validate_loopback_url`, `libs/talon/deepagents_talon/channels/whatsapp.py:BridgeTransport.__init__` | Disproven for the Python client path; only HTTP loopback hostnames are accepted. |
| D3 | Default WhatsApp voice messages can cause local file exfiltration to OpenAI | Traced bridge message payload fields and voice transcription gate. | `libs/talon/deepagents_talon/channels/whatsapp_bridge/bridge.js:client.on("message")`, `libs/talon/deepagents_talon/speech.py:_is_voice_message`, `libs/talon/deepagents_talon/speech.py:_voice_path` | Disproven for the current packaged WhatsApp bridge because it does not set `voice_path` or `media_path`; remains relevant for custom adapters. |
| D4 | Cron tools allow cross-conversation edits or deletion | Traced cron tool operations into store scoping. | `libs/talon/deepagents_talon/cron/tools.py:CronTools.edit_job`, `libs/talon/deepagents_talon/cron/jobs.py:CronJobStore.edit_job`, `libs/talon/deepagents_talon/cron/jobs.py:_same_origin_scope` | Disproven for the provided `CronTools`; edits/removals require matching conversation and channel origin. |
| D5 | Talon currently exposes unrestricted host shell execution by default | Traced `DeepAgentRuntime.start` and core `create_deep_agent` default backend behavior. | `libs/talon/deepagents_talon/runtime.py:DeepAgentRuntime.start`, `libs/deepagents/deepagents/graph.py:create_deep_agent` | Disproven for this branch. Talon does not pass a backend, and core Deep Agents defaults to `StateBackend`. |

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-06-04 | generated by langster-threat-model | Initial scoped threat model for DeepAgents Talon |
