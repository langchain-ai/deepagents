# Ralph Mode for Deep Agents

![Ralph Mode Diagram](ralph_mode_diagram.png)

## What is Ralph?

Ralph is an autonomous looping pattern created by [Geoff Huntley](https://ghuntley.com) that went viral in late 2025. The original implementation is literally one line:

```bash
while :; do cat PROMPT.md | agent ; done
```

Each loop starts with **fresh context**—the simplest pattern for context management. No conversation history to manage, no token limits to worry about. Just start fresh every iteration.

The filesystem and git allow the agent to track progress over time. This serves as its memory and worklog.

## Quick Start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv
source .venv/bin/activate

# Install the CLI
uv pip install deepagents-cli

# Download the script (or copy from examples/ralph_mode/ if you have the repo)
curl -O https://raw.githubusercontent.com/langchain-ai/deepagents/main/examples/ralph_mode/ralph_mode.py

# Run Ralph
python ralph_mode.py "Build a Python programming course for beginners. Use git."
```

## Usage

```bash
# Unlimited iterations (Ctrl+C to stop)
python ralph_mode.py "Build a Python course"

# With iteration limit
python ralph_mode.py "Build a REST API" --iterations 5

# With specific model
python ralph_mode.py "Create a CLI tool" --model claude-sonnet-4-6

# With a specific working directory
python ralph_mode.py "Build a web app" --work-dir ./my-project

# Run in a remote sandbox (Modal, Daytona, or Runloop)
python ralph_mode.py "Build an app" --sandbox modal
python ralph_mode.py "Build an app" --sandbox daytona --sandbox-setup ./setup.sh

# Reuse an existing sandbox instance
python ralph_mode.py "Build an app" --sandbox modal --sandbox-id my-sandbox

# Auto-approve specific shell commands (or "recommended" for safe defaults)
python ralph_mode.py "Build an app" --shell-allow-list recommended
python ralph_mode.py "Build an app" --shell-allow-list "ls,cat,grep,pwd"

# Pass model parameters
python ralph_mode.py "Build an app" --model-params '{"temperature": 0.5}'

# Disable streaming output
python ralph_mode.py "Build an app" --no-stream
```

### Remote sandboxes

Ralph supports running agent code in isolated remote environments via the
`--sandbox` flag. The agent runs locally but executes all code operations in the
remote sandbox. See the
[sandbox documentation](https://docs.langchain.com/oss/python/deepagents/cli/overview)
for provider setup (API keys, etc.) and the
[sandboxes concept guide](https://docs.langchain.com/oss/python/deepagents/sandboxes)
for architecture details.

Supported providers: **Modal**, **Daytona**, **Runloop**.

## How It Works

1. **You provide a task** — declarative, what you want (not how)
2. **Agent runs** — creates files, makes progress
3. **Loop repeats** — same prompt, but files persist
4. **You stop it** — Ctrl+C when satisfied

## Credits

- Original Ralph concept by [Geoff Huntley](https://ghuntley.com)
- [Brief History of Ralph](https://www.humanlayer.dev/blog/brief-history-of-ralph) by HumanLayer
