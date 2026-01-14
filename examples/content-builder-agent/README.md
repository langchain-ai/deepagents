# Content Builder Agent

A content writing agent for writing blog posts, LinkedIn posts, and tweets with cover images included. This example also shows how an agent can be defined entirely through files on a filesystem!

## Quick Start

```bash
# Set API keys
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."      # For image generation
export TAVILY_API_KEY="..."      # For web search (optional)

# Run (uv automatically installs dependencies on first run)
cd examples/content-builder-agent
uv run python content_writer.py "Write a blog post about prompt engineering"
```

**More examples:**
```bash
uv run python content_writer.py "Create a LinkedIn post about AI agents"
uv run python content_writer.py "Write a Twitter thread about the future of coding"
```

## How It Works

The agent is configured by files on disk, not code:

```
content-builder-agent/
├── AGENTS.md              # Brand voice & style guide
├── subagents.yaml         # Subagent definitions
├── skills/
│   ├── blog-post/
│   │   └── SKILL.md       # Blog writing workflow
│   └── social-media/
│       └── SKILL.md       # Social media workflow
└── content_writer.py      # Wires it together
```

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Brand voice, tone, writing standards | Always (system prompt) |
| `subagents.yaml` | Research and other delegated tasks | Always (defines `task` tool) |
| `skills/*/SKILL.md` | Content-specific workflows | On demand |

## Architecture

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],                        # ← Always in context
    skills=["./skills/"],                          # ← Loaded when relevant
    tools=[generate_cover, generate_social_image], # ← Image generation
    subagents=load_subagents("./subagents.yaml"),  # ← See note below
    backend=FilesystemBackend(root_dir="./"),
)
```

**Note on subagents:** Unlike `memory` and `skills`, deepagents doesn't natively load subagents from files. We use a small `load_subagents()` helper to externalize config to YAML. You can also define them inline:

```python
subagents=[
    {
        "name": "researcher",
        "description": "Research topics before writing...",
        "model": "anthropic:claude-3-5-haiku-latest",
        "system_prompt": "You are a research assistant...",
        "tools": [web_search],
    }
],
```

**Flow:**
1. Agent receives task → loads relevant skill (blog-post or social-media)
2. Delegates research to `researcher` subagent → saves to `research/`
3. Writes content following skill workflow → saves to `blogs/` or `linkedin/`
4. Generates cover image with Gemini → saves alongside content

## Output

```
blogs/
└── prompt-engineering/
    ├── post.md       # Blog content
    └── hero.png      # Generated cover image

linkedin/
└── ai-agents/
    ├── post.md       # Post content
    └── image.png     # Generated image

research/
└── prompt-engineering.md   # Research notes
```

## Customizing

**Change the voice:** Edit `AGENTS.md` to modify brand tone and style.

**Add a content type:** Create `skills/<name>/SKILL.md` with YAML frontmatter:
```yaml
---
name: newsletter
description: Use this skill when writing email newsletters
---
# Newsletter Skill
...
```

**Add a subagent:** Add to `subagents.yaml`:
```yaml
editor:
  description: Review and improve drafted content
  model: anthropic:claude-3-5-haiku-latest
  system_prompt: |
    You are an editor. Review the content and suggest improvements...
  tools: []
```

**Add tools:** Define tools in `content_writer.py`, add to `tools=[]`, and register in `load_subagents()` if subagents need them.

## Security Note

This agent has filesystem access and can read/write files within its working directory. The `FilesystemBackend` scopes access to `root_dir`, but the agent can still create, modify, and delete files there. Review generated content before publishing and be mindful when running with sensitive data in the directory.

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` - For the main agent
- `GOOGLE_API_KEY` - For image generation (Gemini)
- `TAVILY_API_KEY` - For web search (optional, research still works without it)
