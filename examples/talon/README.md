# Talon Example

This example runs a Talon host process with one or more channel adapters in the same container. The host `~/talon-workspace/` directory is mounted at `/workspace`.

> **Experimental:** Talon is an experimental runtime and is subject to change or removal at any time.

## Run

```bash
cp .env.example .env
mkdir -p ~/talon-workspace ~/.deepagents
# Fill AGENT_MODEL provider credentials, then uncomment the channel you want to use.
# Build once and run:
docker compose build
docker compose up
```

### WhatsApp

Uncomment the WhatsApp env vars in `.env` and scan the QR code printed by the bridge. The default exposure mode is `self`, so only messages sent by the paired WhatsApp account trigger the agent. Use `allowlist` or `open` only when you intentionally want other chats to trigger the agent.

### Telegram

Uncomment the Telegram env vars in `.env` and set `DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN`. The default exposure mode is `self`, which requires `DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID` to identify your Telegram user ID. Use `allowlist` or `open` only when you intentionally want other chats to trigger the agent.

## Voice Transcription

Voice transcription is enabled by default in `.env.example`. The Docker example installs `ffmpeg` plus the Talon `media` extra, so inbound voice notes are transcribed locally with NVIDIA Parakeet through Transformers before reaching the agent. The first voice message can be slow because the ASR model is downloaded lazily. Set `DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE=cuda` when running on a GPU-enabled host.

Parakeet and Qwen embedding model downloads persist through the existing home bind mount. For `docker run`, add `-v "$HOME/.deepagents:/root/.deepagents"`.

Cron records, downloaded inbound media, and channel session state persist under `~/.deepagents/<assistant-id>/`. The agent's default working directory is `/workspace`, so files it creates are written into `~/talon-workspace/` on the host.

The image installs the Talon package at build time. Rebuild after changing the Dockerfile, system packages, Node dependencies, or Talon Python dependencies.

## Tool Approvals and Persistent Configuration

The Docker image and Compose keep the assistant home and MCP configuration outside
`/workspace`. `DEEPAGENTS_TALON_HOME=/root/.deepagents` is the base directory;
each assistant's fixed policy is `TalonConfig.home/tools.json`, or
`/root/.deepagents/<assistant-id>/tools.json` in this container. MCP configuration
is `/root/.deepagents/.mcp.json`. Compose fixes these paths even if `.env` supplies
host-local paths.

The existing `~/.deepagents:/root/.deepagents` bind mount persists the whole parent
directory, including each assistant's `tools.json`. Keep this directory mount:
do not mount a single `tools.json` or `.mcp.json`, because updates replace files
with an atomic rename. For `docker run`, use
`-v "$HOME/.deepagents:/root/.deepagents" -v "$HOME/talon-workspace:/workspace"`.

See the [tool approval policy](../../libs/talon/README.md#tool-approvals) for
defaults, operator authorization, and next-invocation activation, and
[MCP configuration guidance](../../libs/talon/README.md#mcp-tools) for credential
restrictions. Out-of-workspace placement is not a same-UID shell isolation boundary.

## Optional Persistent Browser

See [the browser deployment guide](browser/README.md) for the opt-in Linux rootful
Docker sidecar, persistent profile, private Node bridge and filtered egress. The
overlay enables native raw `browser_cdp` and `browser_request_handoff` tools and
requires explicit `TALON_BROWSER_OPERATOR_ID` plus single-quoted JSON
`TALON_BROWSER_IDENTITIES` configuration. The host owns identity mapping; one global
lease coordinates browser use. Handoff pauses automation but there is no viewer,
tunnel or interactive takeover yet.

This remains synthetic-only: public browsing is blocked in the current sandbox by
shared-range DNS answers, and end-to-end acceptance is incomplete. Raw CDP uploads
and downloads use browser-side paths/streams, not seamless local file transfer;
Steel and the bridge have no Talon workspace mounts. The normal channel-only
deployment is unchanged.

## Local Run Without Docker

```bash
cp .env.example .env
set -a
. ./.env
set +a

cd ../../libs/talon/deepagents_talon/channels/whatsapp_bridge
npm install

cd ../../../..
uv sync --directory libs/talon --extra media
mkdir -p ~/.deepagents/talon-local/agents
cp examples/talon/AGENTS.md ~/.deepagents/talon-local/AGENTS.md
export DEEPAGENTS_TALON_WORKSPACE=~/talon-workspace
uv run --directory libs/talon deepagents-talon --whatsapp
```

For Telegram, use `--telegram` instead of `--whatsapp`:

```bash
uv run --directory libs/talon deepagents-talon --telegram
```

## Environment Reference

`AGENT_ASSISTANT_ID` names the local state directory under `~/.deepagents/`. The materialized assistant lives at `~/.deepagents/<assistant-id>/AGENTS.md`, with custom subagents under `~/.deepagents/<assistant-id>/agents/`. `AGENT_MODEL` selects the Deep Agents chat model. If it is unset, Talon runs the echo runtime for smoke tests.

The Docker example mounts `~/.deepagents` to `/root/.deepagents`, so cron jobs are stored at `~/.deepagents/<assistant-id>/cron/jobs.json`. Assistant Markdown image/video attachments must use relative paths inside `DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR`, or inside `DEEPAGENTS_TALON_WORKSPACE` when no outbound media directory is configured.

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace each channel or cron-triggered run. `LANGSMITH_PROJECT` defaults to `deepagents-talon`.

WhatsApp exposure:

- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=self` allows only messages from the paired account.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` allows chats in `DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or messages matching `DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS`.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open` allows every inbound WhatsApp message.

Telegram exposure:

- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=self` allows only messages from the operator ID set in `DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID`.
- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist` allows chats in `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS`, users in `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS`, or messages matching `DEEPAGENTS_TALON_TELEGRAM_MENTION_PATTERNS`.
- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=open` allows every inbound Telegram message.

Cron jobs are stored in the assistant state directory at `cron/jobs.json`. Scheduler ticks, dispatch, success/failure, and delivery outcomes are logged as `talon_event` JSON records.

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
