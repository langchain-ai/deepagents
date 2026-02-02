# Ralph Mode for DeepAgents

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
curl -O https://raw.githubusercontent.com/langchain-ai/deepagents/master/examples/ralph_mode/ralph_mode.py

```
# Run Ralph
![](image_1.png)

## Credits

- Original Ralph concept by [Geoff Huntley](https://ghuntley.com)
- [Brief History of Ralph](https://www.humanlayer.dev/blog/brief-history-of-ralph) by HumanLayer
