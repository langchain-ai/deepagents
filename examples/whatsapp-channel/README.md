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

On first run, the adapter runs `npm install` in `bridge/` to fetch the Node dependencies, then starts the bridge and displays a QR code in your terminal. Scan it with WhatsApp on your phone to link the session. Subsequent runs reuse `bridge/node_modules/` and the session stored in `./session/`.

By default `WHATSAPP_SELF_ONLY=true`, so the agent will **only** respond to messages you send to yourself in WhatsApp (your own chat with yourself). This is a safe default for trying the example — strangers and groups are ignored even if they DM you. To accept inbound messages from other contacts, set `WHATSAPP_SELF_ONLY=false` in your `.env`.

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
| `WHATSAPP_SELF_ONLY` | `true` | Only respond to messages you send to yourself. Safe default for a first-run example — the agent will not answer anyone else. Set to `false` to accept inbound messages from other contacts. |
| `WHATSAPP_REQUIRE_MENTION` | `false` | Require @mention in groups |
| `WHATSAPP_MENTION_PATTERNS` | | Comma-separated regex patterns |
| `WHATSAPP_FREE_RESPONSE_CHATS` | | Chat IDs that skip mention requirement |
| `LANGSMITH_API_KEY` | | LangSmith tracing key |
| `LANGSMITH_TRACING` | `false` | Enable tracing |
| `SKILLS_DIRS` | | Skill source dirs (see "Skills" below) |
| `MCP_CONFIG` | | Path to a Claude-Desktop-style MCP config JSON |

## Skills and MCP tools

The agent can optionally load [Agent Skills](https://agentskills.io/specification) and [MCP](https://modelcontextprotocol.io/) tools.

**Skills** — set `SKILLS_DIRS` to one or more directories containing skill sub-folders (each with a `SKILL.md`). Separate multiple paths with `;` (or `:` on Linux/macOS). Later entries override earlier ones:

```
SKILLS_DIRS=/home/me/.deepagents/skills;./skills
```

**MCP tools** — point `MCP_CONFIG` at a Claude-Desktop-style JSON config file. Both stdio and remote (`sse`, `http`) servers are supported. Follows the same loader pattern as the `deepagents` CLI.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/shared"]
    },
    "my-remote": {
      "type": "http",
      "url": "https://mcp.example.com/",
      "headers": {"Authorization": "Bearer ..."}
    }
  }
}
```

Stdio sessions are kept alive for the lifetime of the process and cleaned up on shutdown.

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
- **Image and video support (inbound images, outbound images and videos)** — photos sent by the user are forwarded to the LLM as multimodal content (up to 5 MB per image). The agent can attach images or videos in its replies by writing `![description](/absolute/path/to/file)` in its final message; the adapter strips those refs from the text, classifies each by extension, and sends it as a WhatsApp media attachment with the alt text as the caption. Supported outbound formats: images (PNG, JPEG, GIF, WebP) and videos (MP4, MOV, WebM, 3GP, M4V). MP4 with H.264/AAC is most reliable for video playback. Size limit 16 MB per outbound file.

## Scheduled tasks (cron)

The agent can schedule background tasks whose results are delivered back to the chat where you scheduled them. Results come as regular WhatsApp messages.

Ask the agent to schedule things in plain language:

- *"Every 2 hours, summarize new posts on Hacker News."*
- *"Remind me in 45 minutes to take the bread out."*
- *"List my scheduled tasks."* / *"Cancel the Hacker News one."*

Supported schedules:

- Duration (one-shot): any positive integer + `m`/`h`/`d` — e.g. `1m`, `30m`, `2h`, `1d`.
- Interval (recurring): `every <duration>` — e.g. `every 1m`, `every 15m`, `every 2h`, `every 1d`.

Minimum interval is `1m`. The scheduler polls every `WHATSAPP_CRON_TICK_SECONDS` (default 60s), so firing can be up to one tick late.

Cron expressions (`0 9 * * *`) and absolute timestamps are not supported.

Configuration:

| Variable | Default | Description |
|---|---|---|
| `WHATSAPP_CRON_PATH` | `~/.deepagents/<AGENT_ASSISTANT_ID>/cron/jobs.json` | Where job state is persisted |
| `WHATSAPP_CRON_TICK_SECONDS` | `60` | How often the scheduler checks for due jobs |

Limitations:

- Only one `main.py` process should run against a given `jobs.json`; there is no multi-process file lock.
- A failed delivery is logged and stored in `last_error` but not retried — the output is lost. (Interval jobs simply fire again on the next tick.)
- If the process is down when a recurring job was due, that window is skipped (no backfill).

## Development

Install the example with its test dependencies, then run the suite:

```bash
cd examples/whatsapp-channel
pip install -e ".[dev]"
pytest
```

The test suite is offline (no network, no WhatsApp bridge required) and covers the adapter helpers, the first-run `npm install` bootstrap, config defaults, and the cron scheduler.

## Troubleshooting

- **QR code not showing:** Check that Node.js 18+ is installed (`node --version`)
- **npm not found:** The adapter runs `npm install` in `bridge/` on first run. If `npm` is missing, install Node.js 18+ or run `npm install` in `bridge/` manually and re-run
- **Session expired:** Delete the `session/` directory and re-scan
- **Bridge won't start:** Check if port 3000 is in use (`lsof -i :3000`)
- **Chrome not found:** The bridge auto-detects system Chrome/Chromium. If detection fails, set `CHROME_PATH=/path/to/chrome` in your `.env`
- **Puppeteer download fails:** Chrome download is skipped by default — the bridge uses your system Chrome. If you have no system Chrome, install one or set `PUPPETEER_SKIP_DOWNLOAD=false` and retry
