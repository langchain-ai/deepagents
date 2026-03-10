# deepagents-codex

Codex OAuth integration for [Deep Agents](https://github.com/langchain-ai/deepagents).

Enables authentication with ChatGPT Plus/Pro subscriptions via browser-based OAuth.

## Installation

```bash
pip install 'deepagents-cli[codex]'
```

## Usage

```bash
# Login via browser
deepagents auth login --provider codex

# Check status
deepagents auth status --provider codex

# Use a Codex model
deepagents --model codex:gpt-4o

# Logout
deepagents auth logout --provider codex
```
