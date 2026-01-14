# Content Builder Agent

<img width="1255" height="756" alt="content-cover-image" src="https://github.com/user-attachments/assets/4ebe0aba-2780-4644-8a00-ed4b96680dc9" />

A content writing agent for writing blog posts, LinkedIn posts, and tweets with cover images included.

**This example demonstrates how to define an agent through three filesystem primitives:**
- **Memory** (`AGENTS.md`) – persistent context like brand voice and style guidelines
- **Skills** (`skills/*/SKILL.md`) – workflows for specific tasks, loaded on demand
- **Subagents** (`subagents.yaml`) – specialized agents for delegated tasks like research

The `content_writer.py` script shows how to combine these into a working agent.

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
├── AGENTS.md                    # Brand voice & style guide
├── subagents.yaml               # Subagent definitions
├── skills/
│   ├── blog-post/
│   │   └── SKILL.md             # Blog writing workflow
│   └── social-media/
│       └── SKILL.md             # Social media workflow
└── content_writer.py            # Wires it together (includes tools)
```

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Brand voice, tone, writing standards | Always (system prompt) |
| `subagents.yaml` | Research and other delegated tasks | Always (defines `task` tool) |
| `skills/*/SKILL.md` | Content-specific workflows | On demand |

**What's in the skills?** Each skill teaches the agent a specific workflow:
- **Blog posts:** Structure (hook → context → main content → CTA), SEO best practices, research-first approach
- **Social media:** Platform-specific formats (LinkedIn character limits, Twitter thread structure), hashtag usage
- **Image generation:** Detailed prompt engineering guides with examples for different content types (technical posts, announcements, thought leadership)

## Architecture

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],                        # ← Middleware loads into system prompt
    skills=["./skills/"],                          # ← Middleware loads on demand
    tools=[generate_cover, generate_social_image], # ← Image generation tools
    subagents=load_subagents("./subagents.yaml"),  # ← See note below
    backend=FilesystemBackend(root_dir="./"),
)
```

The `memory` and `skills` parameters are handled natively by deepagents middleware. Tools are defined in the script and passed directly.

**Note on subagents:** Unlike `memory` and `skills`, subagents must be defined in code. We use a small `load_subagents()` helper to externalize config to YAML. You can also define them inline:

```python
subagents=[
    {
        "name": "researcher",
        "description": "Research topics before writing...",
        "model": "anthropic:claude-haiku-4-5-20251001",
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
  model: anthropic:claude-haiku-4-5-20251001
  system_prompt: |
    You are an editor. Review the content and suggest improvements...
  tools: []
```

**Add a tool:** Define it in `content_writer.py` with the `@tool` decorator and add to `tools=[]`.

## Security Note

This agent has filesystem access and can read, write, and delete files on your machine. Review generated content before publishing and avoid running in directories with sensitive data.

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` - For the main agent
- `GOOGLE_API_KEY` - For image generation (uses Gemini's [Imagen / "nano banana"](https://ai.google.dev/gemini-api/docs/image-generation) via `gemini-2.5-flash-image`)
- `TAVILY_API_KEY` - For web search (optional, research still works without it)
