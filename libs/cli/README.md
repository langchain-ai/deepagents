<p align="center">
  <img src="https://raw.githubusercontent.com/langchain-ai/deepagents/master/libs/cli/images/cli.png" alt="Deep Agents CLI" width="600"/>
</p>

<h1 align="center">Deep Agents CLI</h1>

<p align="center">
  The Deep Agents harness in your terminal.
</p>

## Quickstart

```bash
uv tool install deepagents-cli
deepagents
```

This gives you a fully-featured coding agent with file operations, shell commands, web search, planning, and sub-agent delegation in your terminal.

## Usage

```bash
# Use a specific model
deepagents --model claude-sonnet-4-5-20250929
deepagents --model gpt-4o

# Auto-approve tool usage (skip confirmation prompts)
deepagents --auto-approve

# Execute code in a remote sandbox
deepagents --sandbox modal

# Run non-interactively with safe commands
deepagents -n "what's your public IP? try using curl" --shell-allow-list "ls,cat,grep,pwd,echo,head,tail,find,wc"
Running task non-interactively...
Agent: agent | Thread: 0049b0e6

I'll use curl to check the public IP address.
🔧 Calling tool: shell

❌ Shell command rejected: curl -s ifconfig.me
Allowed commands: ls, cat, grep, pwd, echo, head, tail, find, wc
It looks like curl isn't in the allowed command list for the shell tool. Let me try using the http_request tool instead to check the public IP.
🔧 Calling tool: http_request
The public IP address is **X.X.X.X**.
```

## Model Configuration

The CLI auto-detects your provider based on available API keys:

| Priority | API Key | Default Model |
|----------|---------|---------------|
| 1st | `OPENAI_API_KEY` | `gpt-5.2` |
| 2nd | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5-20250929` |
| 3rd | `GOOGLE_API_KEY` | `gemini-3-pro-preview` |

## Customization

The CLI supports persistent memory, project-specific configurations, and custom skills. See the [documentation](https://docs.langchain.com/oss/python/deepagents/cli) for details on:

- **AGENTS.md** — Persistent memory for preferences and coding style
- **Skills** — Reusable workflows and domain knowledge
- **Project configs** — Per-project settings in `.deepagents/`

## Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents/cli)** — Full CLI reference
- **[Deep Agents](https://github.com/langchain-ai/deepagents)** — The underlying agent harness
