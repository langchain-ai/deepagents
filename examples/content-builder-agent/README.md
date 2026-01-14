# Content Builder Agent

A content writer agent configured entirely through files on disk.

## Structure

```
content-builder-agent/
├── AGENTS.md                    # Brand voice & style guide (always loaded)
├── skills/
│   ├── blog-post/SKILL.md      # Blog writing workflow
│   └── social-media/SKILL.md   # Social media workflow
└── content_writer.py            # Wires it together
```

## How It Works

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],      # ← Loaded into system prompt
    skills=["./skills/"],        # ← Available on-demand
    subagents=[{...}],           # ← Task delegation
    backend=FilesystemBackend(root_dir="./"),
)
```

| File | API Parameter | Behavior |
|------|--------------|----------|
| `AGENTS.md` | `memory=[]` | Always loaded, defines agent personality |
| `skills/*.md` | `skills=[]` | Loaded when needed (progressive disclosure) |
| — | `subagents=[]` | Inline definitions for task delegation |

## Quick Start

```bash
cd examples/content-builder-agent
uv run python content_writer.py "Write a blog post about prompt engineering"

# Or for social media
uv run python content_writer.py "Create a LinkedIn post about AI agents"
```

## Customizing

**Change the style**: Edit `AGENTS.md` to modify brand voice and writing standards.

**Add a skill**: Create `skills/<name>/SKILL.md` with YAML frontmatter:

```yaml
---
name: newsletter
description: Use this skill when writing email newsletters
---
# Newsletter Skill
[workflow here]
```

**Add a subagent**: Add to `subagents=[]` in `content_writer.py`.
