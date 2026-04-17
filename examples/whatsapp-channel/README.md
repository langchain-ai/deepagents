# WhatsApp Channel for Deep Agents

Connect a [Deep Agent](https://github.com/langchain-ai/deepagents) to WhatsApp using a Node.js bridge powered by [whatsapp-web.js](https://github.com/pedroslopez/whatsapp-web.js).

## Prerequisites

- Python 3.11+
- Node.js 18+
- An LLM API key (Anthropic, OpenAI, or any provider supported by `langchain`)

## Setup

1. **Install Python dependencies:**

```bash
cd examples/whatsapp-channel
pip install -e .
```

2. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. **Run:**

```bash
python main.py
```

On first run, the bridge will display a QR code in your terminal. Scan it with WhatsApp on your phone to link the session.

Subsequent runs reuse the saved session (stored in `./session/`).

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `claude-sonnet-4-6` | Model for the agent (`provider:model` format) |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `OPENAI_API_KEY` | | OpenAI API key |
| `OPENAI_BASE_URL` | | Custom OpenAI-compatible endpoint |
| `WHATSAPP_BRIDGE_PORT` | `3000` | Bridge HTTP port |
| `WHATSAPP_SESSION_PATH` | `./session` | WhatsApp session storage |
| `WHATSAPP_REQUIRE_MENTION` | `false` | Require @mention in groups |
| `WHATSAPP_MENTION_PATTERNS` | | Comma-separated regex patterns |
| `WHATSAPP_FREE_RESPONSE_CHATS` | | Chat IDs that skip mention requirement |
| `WHATSAPP_REPLY_PREFIX` | | Prefix for bot responses |
| `LANGSMITH_API_KEY` | | LangSmith tracing key |
| `LANGSMITH_TRACING` | `false` | Enable tracing |

## Architecture

```
WhatsApp <-> whatsapp-web.js (bridge.js) <-> HTTP <-> WhatsAppAdapter (Python) <-> Deep Agent
```

The bridge runs as a Node.js subprocess managed by the Python adapter. Messages flow through HTTP endpoints on localhost.

## Features

- Text messaging with conversation history per chat
- Media support: images, video, voice notes, documents
- WhatsApp-compatible markdown formatting
- Group chat support with mention filtering
- Typing indicators
- Message chunking for long responses
- Automatic text extraction from document attachments

## Scheduled tasks (cron)

The agent can schedule background tasks whose results are delivered back to the chat where you scheduled them. Results come as regular WhatsApp messages.

Ask the agent to schedule things in plain language:

- *"Every 2 hours, summarize new posts on Hacker News."*
- *"Remind me in 45 minutes to take the bread out."*
- *"List my scheduled tasks."* / *"Cancel the Hacker News one."*

Supported schedules:

- Duration (one-shot): `30m`, `2h`, `1d`.
- Interval (recurring): `every 15m`, `every 2h`, `every 1d`.

Cron expressions (`0 9 * * *`) and absolute timestamps are not supported.

Configuration:

| Variable | Default | Description |
|---|---|---|
| `WHATSAPP_CRON_PATH` | `./cron/jobs.json` | Where job state is persisted |
| `WHATSAPP_CRON_TICK_SECONDS` | `60` | How often the scheduler checks for due jobs |

Limitations:

- Only one `main.py` process should run against a given `jobs.json`; there is no multi-process file lock.
- A failed delivery is logged and stored in `last_error` but not retried — the output is lost. (Interval jobs simply fire again on the next tick.)
- If the process is down when a recurring job was due, that window is skipped (no backfill).

## Troubleshooting

- **QR code not showing:** Check that Node.js 18+ is installed (`node --version`)
- **Session expired:** Delete the `session/` directory and re-scan
- **Bridge won't start:** Check if port 3000 is in use (`lsof -i :3000`)
- **Chrome not found:** The bridge auto-detects system Chrome/Chromium. If detection fails, set `CHROME_PATH=/path/to/chrome` in your `.env`
- **Puppeteer download fails:** Chrome download is skipped by default — the bridge uses your system Chrome. If you have no system Chrome, install one or set `PUPPETEER_SKIP_DOWNLOAD=false` and retry
