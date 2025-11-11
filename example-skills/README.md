## Example Skills for DeepAgents

This directory contains example skills demonstrating how to extend deepagents with specialized capabilities.

### What are Skills?

Skills are folders of instructions that deepagents loads dynamically to improve performance on specialized tasks. Each skill is self-contained with a `SKILL.md` file that teaches the agent how to complete specific tasks in a repeatable way.

### Available Example Skills

- **template-skill** - A basic template to use as a starting point for new skills
- **code-reviewer** - Review code for quality, security, and best practices
- **python-expert** - Expert Python development following PEP 8, type hints, and modern best practices

### Using Skills

To use skills with deepagents:

```python
from deepagents import create_deep_agent
from deepagents.middleware import SkillsMiddleware

# Create agent with skills
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[
        SkillsMiddleware(
            skills_dir="./example-skills",
            auto_activate=["template-skill"],  # Optional: auto-activate skills
        )
    ]
)

# Skills can be activated during conversation using the use_skill tool
```

### Creating Your Own Skills

Each skill is a folder containing a `SKILL.md` file with this structure:

```markdown
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---

# My Skill Name

[Add your instructions here that the agent will follow when this skill is active]

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
```

**Required frontmatter fields:**
- `name` - A unique identifier for your skill (lowercase, hyphens for spaces)
- `description` - A complete description of what the skill does and when to use it

### Skill Structure

```
example-skills/
├── template-skill/
│   └── SKILL.md
├── code-reviewer/
│   └── SKILL.md
└── python-expert/
    └── SKILL.md
```

### Best Practices

1. **Clear naming** - Use descriptive, lowercase names with hyphens
2. **Comprehensive descriptions** - Help the agent know when to use the skill
3. **Detailed instructions** - Include examples, guidelines, and expected outputs
4. **Focused scope** - Each skill should have a specific, well-defined purpose
5. **Self-contained** - Skills should work independently

### Learn More

For more information about skills, see:
- [Skills Middleware Documentation](../libs/deepagents/middleware/skills.py)
- [Creating Custom Skills Guide](../docs/skills.md) (coming soon)
