# Ralph Mode for Deep Agents

<img src=".github/images/ralph_mode_diagram.png" alt="Ralph Mode Diagram" width="100%"/>

## What is Ralph?

Ralph is an autonomous looping pattern created by [Geoff Huntley](https://ghuntley.com) that went viral in late 2025. The original implementation is literally one line:

```bash
while :; do cat PROMPT.md | agent ; done
```

Each loop starts with **fresh context**—the simplest pattern for context management. No conversation history to manage, no token limits to worry about. Just start fresh every iteration.

The filesystem and git allow the agent to track progress over time. This serves as its memory and worklog.

## Quick Start

```bash
cd libs/deepagents-cli

# Run Ralph with a task (unlimited iterations, Ctrl+C to stop)
uv run deepagents --ralph "Build a Python programming course for beginners. Use git."

# Or with a specific iteration limit
uv run deepagents --ralph "Build a REST API" --ralph-iterations 5
```

## How It Works

1. **You provide a task** — declarative, what you want (not how)
2. **Agent runs** — creates files, makes progress
3. **Loop repeats** — same prompt, but files persist
4. **You stop it** — Ctrl+C when satisfied

## CLI Options

| Flag | Description |
|------|-------------|
| `--ralph "TASK"` | Enable Ralph mode with the given task |
| `--ralph-iterations N` | Max iterations (default: 0 = unlimited) |
| `--model MODEL` | Model to use (e.g., `claude-haiku-4-5-20251001`) |

## Credits

- Original Ralph concept by [Geoff Huntley](https://ghuntley.com)
- [Brief History of Ralph](https://www.humanlayer.dev/blog/brief-history-of-ralph) by HumanLayer
