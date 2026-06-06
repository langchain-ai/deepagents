# Swarm Part 2

This example runs a Deep Agent with the `swarm` skill and a URL-based `researcher` subagent.
The default prompt fans out company research across rows loaded from `/companies.txt`.

## What It Does

- Loads the swarm skill from `skills/swarm`.
- Uses `create_swarm_task_tool` with two subagents:
  - `researcher` for web-page based company research (`fetch_webpage`).
  - `reviewer` for local file analysis.
- Reads and writes files under `output/` via `FilesystemBackend`.

## Prerequisites

- Python 3.11+
- `uv`
- `ANTHROPIC_API_KEY` set in your environment

## Companies File

`agent.py` expects `/companies.txt`, which maps to `output/companies.txt` in this example.

Required format:

```txt
Company Name,https://example.com
Another Company,https://another.example.com
```

One row per company. The current repository copy includes 500 entries.

## Run

```bash
cd examples/swarm-part-2
uv run python agent.py
```

Run with a custom model:

```bash
uv run python agent.py --model claude-sonnet-4-6
```

Run with a custom prompt:

```bash
uv run python agent.py "Use the swarm skill to research the first 50 companies from /companies.txt and return a compact table."
```

## Notes

- The default task asks the agent to parse `name,url` lines and run swarm with `subagentType: "researcher"`.
- `fetch_webpage` converts HTML to markdown and truncates long pages to keep each row manageable.
