<p align="center">
  <img src="da-cli-cover-photo.png" alt="Deep Agents CLI" width="600"/>
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

# Load MCP tools from config file
deepagents --mcp-config mcp.json
```

## MCP Tools

Load additional tools from [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers using the `--mcp-config` flag:

```bash
deepagents --mcp-config path/to/mcp-config.json
```

The config file uses Claude Desktop format:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "your-token" }
    },
    "remote-api": {
      "type": "sse",
      "url": "https://api.example.com/mcp",
      "headers": { "Authorization": "Bearer your-token" }
    }
  }
}
```

**Server types:**

| Type | Required fields | Optional fields |
|------|-----------------|-----------------|
| stdio (default) | `command` | `args`, `env` |
| sse | `type: "sse"`, `url` | `headers` |
| http | `type: "http"`, `url` | `headers` |

**Requirements:** Install `langchain-mcp-adapters`:

```bash
pip install langchain-mcp-adapters
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
