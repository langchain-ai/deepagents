# Talon WhatsApp Example

This example runs one Talon host process with the WhatsApp bridge subprocess in
the same container. The bridge binds to loopback inside the container and the
WhatsApp session plus Talon state are stored in a Docker volume.

## Run

```bash
cp .env.example .env
# Fill AGENT_MODEL provider credentials, then run:
docker compose up
```

Scan the QR code printed by the bridge. The default exposure mode is `self`, so
only messages sent by the paired WhatsApp account trigger the agent. Use
`allowlist` or `open` only when you intentionally want other chats to trigger
the agent.

## Local Run Without Docker

```bash
cp .env.example .env
set -a
. ./.env
set +a

cd ../../libs/talon/deepagents_talon/channels/whatsapp_bridge
npm install

cd ../../../..
uv sync --directory libs/talon
cp examples/talon-whatsapp/AGENTS.md ~/.deepagents/whatsapp-local/agent/AGENTS.md
uv run --directory libs/talon deepagents-talon --whatsapp
```

## Environment Reference

`AGENT_ASSISTANT_ID` names the local state directory under `~/.deepagents/`.
`AGENT_MODEL` selects the Deep Agents chat model. If it is unset, Talon runs the
echo runtime for smoke tests.

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace each channel or
cron-triggered run. `LANGSMITH_PROJECT` defaults to `deepagents-talon`.

WhatsApp exposure:

- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=self` allows only messages from the paired account.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=allowlist` allows chats in `DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS` or messages matching `DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS`.
- `DEEPAGENTS_TALON_WHATSAPP_EXPOSURE=open` allows every inbound WhatsApp message.

Cron jobs are stored in the assistant state directory at `cron/jobs.json`.
Scheduler ticks, dispatch, success/failure, and delivery outcomes are logged as
`talon_event` JSON records.
