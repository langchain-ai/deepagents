# Agents as Filesystem

This example demonstrates a key pattern in modern AI agents: **agents are defined by the files they have access to**.

## The Concept

Agent harnesses like Deep Agents abstract away complex context engineering and let you configure your agent through files on disk. This pattern has emerged across multiple tools:

| Pattern | Standard/Tool | Purpose |
|---------|--------------|---------|
| `AGENTS.md` | [agents.md](https://agents.md) | System prompt and instructions |
| `SKILL.md` | [Agent Skills](https://agentskills.io) | Specialized workflows |
| `/agents/*.md` | [Claude Code](https://claude.ai/code) | Subagent definitions |
| `mcp.json` | [MCP](https://modelcontextprotocol.io) | Tool configuration |

The core idea: **the agent harness stays the same, it's just configured by files in the filesystem**.

## This Example

A content writer agent configured entirely through files:

```
agents_as_filesystem/
├── AGENTS.md                    # Brand voice & style guide
├── skills/
│   ├── blog-post/SKILL.md      # Blog writing workflow
│   └── social-media/SKILL.md   # Social media workflow
└── content_writer.py            # Wires it together
```

## How It Maps to Deep Agents

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],      # ← AGENTS.md loaded into system prompt
    skills=["./skills/"],        # ← Skills available on-demand
    subagents=[{...}],           # ← Subagent definitions (inline)
    backend=FilesystemBackend(root_dir="./"),
)
```

| Filesystem | API Parameter | Behavior |
|------------|--------------|----------|
| `AGENTS.md` | `memory=[]` | Always loaded, persistent context |
| `skills/*.md` | `skills=[]` | Progressive disclosure, loaded when needed |
| (inline) | `subagents=[]` | Task delegation |

## Quick Start

```bash
# Install dependencies
uv pip install deepagents

# Run the example
cd examples/agents_as_filesystem
python content_writer.py "Write a blog post about prompt engineering"

# Or for social media
python content_writer.py "Create a LinkedIn post about AI agents"
```

## Customizing

### Modify the Style Guide

Edit `AGENTS.md` to change the brand voice, writing standards, or content pillars. Changes take effect immediately on the next run.

### Add a New Skill

Create a new directory under `skills/` with a `SKILL.md` file:

```yaml
---
name: newsletter
description: Use this skill when writing email newsletters
---

# Newsletter Writing Skill

[Your workflow here...]
```

### Add a Subagent

Add to the `subagents` list in `content_writer.py`:

```python
subagents=[
    {
        "name": "editor",
        "description": "Review and improve written content",
        "system_prompt": "You are an editor. Review content for...",
    }
]
```

## Limitations

**MCP Configuration**: Deep Agents doesn't auto-parse `mcp.json` files. To use MCP tools, integrate via [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters):

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient(...) as client:
    mcp_tools = await client.get_tools()
    agent = create_deep_agent(tools=mcp_tools, ...)
```

**Subagent Files**: Unlike Claude Code's `/agents/*.md` pattern, subagents are defined inline in Python rather than as separate files.

## Learn More

- [Agent Harnesses Blog Post](https://www.vtrivedy.com/posts/claude-code-sdk-haas-harness-as-a-service)
- [AGENTS.md Specification](https://agents.md)
- [Agent Skills Specification](https://agentskills.io)
- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
