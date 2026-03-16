# Skills with MCP and Native Tools - Examples

This directory contains example skills that demonstrate MCP server integration and native tool registration.

## Available Examples

### GitHub Integration Skill

**Location:** [`github/`](./github)

A comprehensive example that shows:
- MCP server configuration for GitHub API
- Native Python tools complementing MCP tools
- Environment variable usage
- Complete documentation

## Running the Examples

### Prerequisites

1. **Install dependencies:**
   ```bash
   cd /root/deepagents
   pip install -e .
   ```

2. **Set up GitHub token (for GitHub integration):**
   ```bash
   export GITHUB_TOKEN="your-github-token-here"
   ```

   Get a token from: https://github.com/settings/tokens

### Using the Skills in an Agent

```python
from deepagents import create_deep_agent

# Create agent with the example skill
agent = create_deep_agent(
    model="claude-sonnet-4-6",
    skills=["/root/deepagents/examples/skills-with-mcp/github/"],
)

# Invoke the agent
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Use the GitHub integration to create a new repository called 'test-repo'"
    }]
})
```

## Skill Structure

Each example skill follows this structure:

```
skill-name/
├── SKILL.md          # Required: YAML frontmatter + markdown instructions
└── helper.py         # Optional: Native Python tools
```

## Creating Your Own Skills

### Step 1: Create Skill Directory

```bash
mkdir -p /path/to/skills/my-skill
cd /path/to/skills/my-skill
```

### Step 2: Create SKILL.md

```markdown
---
name: my-skill
description: A brief description of what this skill does
mcp-servers:
  - name: my-server
    transport: stdio
    command: npx
    args: [-y, @mcp-server-package]
---

# My Skill

Detailed instructions for when and how to use this skill.
```

### Step 3: (Optional) Create helper.py

```python
from deepagents.middleware.skill_tools import register_tool

@register_tool("my-skill")
def my_tool(param: str) -> str:
    """A native tool for my skill."""
    return f"Result: {param}"
```

### Step 4: Use Your Skill

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    skills=["/path/to/skills/"],
)
```

## MCP Server Reference

### stdio Servers

For process-based MCP servers:

```yaml
mcp-servers:
  - name: my-stdio-server
    transport: stdio
    command: npx  # or uvx, python, etc.
    args: [-y, @package/name]
    env:
      API_KEY: ${API_KEY}
```

### HTTP Servers

For HTTP-based MCP servers:

```yaml
mcp-servers:
  - name: my-http-server
    transport: http
    url: https://api.example.com/mcp
    headers:
      Authorization: Bearer ${TOKEN}
```

### SSE Servers

For Server-Sent Events MCP servers:

```yaml
mcp-servers:
  - name: my-sse-server
    transport: sse
    url: https://api.example.com/mcp
```

## Troubleshooting

### MCP Server Won't Connect

1. **Check the server command:**
   ```bash
   npx -y @package/name --help
   ```

2. **Verify environment variables:**
   ```bash
   echo $MY_TOKEN
   ```

3. **Check the agent logs for detailed errors**

### Native Tools Not Available

1. **Verify helper.py exists** in the skill directory
2. **Check the skill name matches**: `@register_tool("skill-name")`
3. **Look for import errors** in helper.py

## Further Reading

- [Skills with MCP Guide](../../docs/guides/skills-with-mcp.md)
- [Agent Skills Specification](https://agentskills.io/specification)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Contributing Examples

To contribute your own example:

1. Create a new directory following the structure above
2. Include comprehensive documentation in SKILL.md
3. Add comments in helper.py explaining the implementation
4. Test your example thoroughly
5. Submit a pull request

## License

These examples are provided as-is for educational purposes.
