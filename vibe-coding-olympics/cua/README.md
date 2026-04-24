# CUA demo

Drives a website with an OpenAI computer-use agent running on a Browserbase
cloud browser. Prints a Browserbase session replay URL you can share as
demo content.

Single file: `demo.py`. Roughly 200 lines, no framework layer.

## Prerequisites

- Python 3.11+
- `uv` installed
- API access to an OpenAI model with the `computer` tool (defaults to
  `gpt-5.4`; `computer-use-preview` also works if enabled on your org)

Required env vars:

```bash
export OPENAI_API_KEY=sk-...
export BROWSERBASE_API_KEY=bb_...
export BROWSERBASE_PROJECT_ID=...
```

Optional:

```bash
export CUA_MODEL=gpt-5.4        # or computer-use-preview
export CUA_MAX_TURNS=40
```

## Install

```bash
cd cua
uv sync
```

## Run

```bash
uv run python demo.py "scroll the homepage, open Docs, click Get started" \
  --url https://langchain.com
```

Flags:

- `task` (positional) — natural-language instruction for the agent
- `--url` — starting page (default: `https://www.google.com`)
- `--model` — override `CUA_MODEL`
- `--max-turns` — override `CUA_MAX_TURNS`

The Browserbase replay URL prints at start and end:

```
Live view / replay: https://browserbase.com/sessions/<id>
```

Open in a browser while the run is in progress to watch live, or revisit
later for the replay.

## How it works

1. Creates a Browserbase session, connects Playwright via CDP.
2. Navigates the initial URL before handing control to the model.
3. Calls the OpenAI Responses API with the `computer` tool.
4. Each `computer_call` response contains one or more actions
   (`click`, `scroll`, `type`, `keypress`, `drag`, `wait`, `screenshot`).
5. `dispatch()` maps each action to a Playwright call.
6. After the action batch, capture a fresh viewport screenshot and return
   it as the next `computer_call_output`. Repeat until the model emits a
   final message or the turn cap is hit.
7. Browserbase records the session; the replay URL is stable after close.

## Gotchas

- **Localhost is unreachable** from Browserbase — expose local sites with
  `cloudflared tunnel --url http://localhost:3000` and pass the public URL
  via `--url`.
- **Model access is gated.** If you see a generic
  `invalid_request_error` 400 with no `param`/`code`, your org probably
  does not have the requested model's CUA capability enabled. Switch
  models with `--model`.
- **The demo is intentionally simple** — no retry, no cost limits, no
  structured action log. Wrap or fork before using in anything that
  can't tolerate a stuck run.
