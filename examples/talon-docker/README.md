# Talon Docker Example

This example runs one Deep Agents Talon host in Docker. It can run WhatsApp,
Telegram, or both channels against the same local assistant state and workspace.

> **Experimental:** Talon is an experimental runtime and is subject to change or
> removal at any time.

The host `~/agent-workspace/` directory is mounted at `/workspace`. Cron records,
downloaded inbound media, channel session state, and agent-created files persist
under that host directory.

## Run

```bash
cp .env.example .env
mkdir -p ~/agent-workspace

# Fill AGENT_MODEL and the required provider credentials in .env.
# Enable the channels you want, then build once and run:
docker compose build
docker compose up
```

The image installs the Talon package at build time. Rebuild after changing the
Dockerfile, system packages, Node dependencies, or Talon Python dependencies.

## Channel Setup

### WhatsApp

The sample `.env.example` enables WhatsApp by default:

```bash
DEEPAGENTS_TALON_WHATSAPP_ENABLED=true
DEEPAGENTS_TALON_WHATSAPP_START_BRIDGE=true
DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=self
```

On first run, scan the QR code printed by the bridge. With `self` exposure, only
messages from the paired WhatsApp account trigger the agent. Use `allowlist` or
`open` only when you intentionally want other chats to trigger the agent.

Voice transcription is enabled in the sample env. The image installs `ffmpeg` and
the Talon `speech` extra so inbound WhatsApp voice notes are transcribed locally
with NVIDIA Parakeet through Transformers before reaching the agent. The first
voice message can be slow because the ASR model is downloaded lazily. Set
`DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE=cuda` on GPU-enabled hosts.

### Telegram

Create a Telegram bot with BotFather, add its token to `.env`, and enable the
Telegram channel:

```bash
DEEPAGENTS_TALON_TELEGRAM_ENABLED=true
DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN=
DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS=123456789
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS=-1001234567890
```

Use `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS` for private bot DMs from specific
Telegram user IDs. Use `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS` for channel
posts from specific Telegram channel chat IDs. Channel chat IDs are usually
negative and often start with `-100`.

To process Telegram channel posts, add the bot to the channel with permission to
read posts. If you only want private DMs, leave `ALLOWLIST_CHATS` empty. If you
only want channel posts, leave `ALLOWLIST_USERS` empty.

## Example Env Values

The tracked `.env.example` uses placeholder-only values. Keep real provider keys
and bot tokens in your local `.env`, which is ignored by git.

```bash
AGENT_ASSISTANT_ID=talon-docker-local
AGENT_MODEL=

DEEPAGENTS_TALON_WHATSAPP_ENABLED=true
DEEPAGENTS_TALON_TELEGRAM_ENABLED=false

DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN=
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS=
DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS=

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

`AGENT_MODEL` selects the Deep Agents chat model. If it is unset, Talon runs the
echo runtime for smoke tests and replies with the inbound text unchanged.

The Docker example overrides `DEEPAGENTS_TALON_HOME` to
`/workspace/.deepagents`, so cron jobs are stored at
`~/agent-workspace/.deepagents/<assistant-id>/cron/jobs.json`. Assistant Markdown
image/video attachments must use relative paths inside
`DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR`, or inside `DEEPAGENTS_TALON_WORKSPACE`
when no outbound media directory is configured.

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace each channel or
cron-triggered run. `LANGSMITH_PROJECT` defaults to `deepagents-talon`.

## Local Run Without Docker

```bash
cp .env.example .env
set -a
. ./.env
set +a

cd ../../libs/talon/deepagents_talon/channels/whatsapp_bridge
npm install

cd ../../../..
uv sync --directory libs/talon --extra speech
cp examples/talon-docker/AGENTS.md ~/.deepagents/talon-docker-local/agent/AGENTS.md
export DEEPAGENTS_TALON_WORKSPACE=~/agent-workspace
uv run --directory libs/talon deepagents-talon
```

## Exposure Reference

WhatsApp exposure:

- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=self` allows only messages from the paired
  account.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` allows chats in
  `DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or messages matching
  `DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS`.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open` allows every inbound WhatsApp
  message and requires `DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK=allow-arbitrary-senders`.

Telegram exposure:

- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=self` allows only messages from
  `DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID`.
- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=allowlist` allows private DMs from
  `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS`, channel posts from
  `DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS`, or messages matching
  `DEEPAGENTS_TALON_TELEGRAM_MENTION_PATTERNS`.
- `DEEPAGENTS_TALON_TELEGRAM_EXPOSURE=open` allows every inbound Telegram
  message and requires `DEEPAGENTS_TALON_TELEGRAM_OPEN_ACK=allow-arbitrary-senders`.

Cron jobs are stored in the assistant state directory at `cron/jobs.json`.
Scheduler ticks, dispatch, success/failure, and delivery outcomes are logged as
`talon_event` JSON records.

## Resources

- [LangChain Academy](https://academy.langchain.com/) - Comprehensive, free
  courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file)
  - community guidelines and standards
