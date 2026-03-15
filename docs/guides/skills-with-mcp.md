# Skills with MCP and Native Tools

This guide explains how to create skills that provide tools through MCP (Model Context Protocol) servers and native Python functions.

## Overview

Skills can now provide tools in two ways:

1. **MCP Servers**: Connect to external Model Context Protocol servers
2. **Native Tools**: Register Python functions as tools

When a skill becomes active, its tools are automatically made available to the agent through a single `skill_tool_call` dispatcher.

## MCP Server Configuration

### Adding MCP Servers to a Skill

Add the `mcp-servers` field to your skill's YAML frontmatter:

```yaml
---
name: github-integration
description: GitHub API integration for repository and issue management
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
---
```

### Supported Transport Types

#### stdio (Default)
For process-based MCP servers:

```yaml
mcp-servers:
  - name: my-stdio-server
    transport: stdio
    command: npx
    args: [-y, @organization/server-package]
    env:
      API_KEY: ${API_KEY}  # Environment variables are expanded
```

#### http
For HTTP-based MCP servers:

```yaml
mcp-servers:
  - name: my-http-server
    transport: http
    url: https://api.example.com/mcp
    headers:
      Authorization: Bearer ${TOKEN}
```

#### sse
For Server-Sent Events MCP servers:

```yaml
mcp-servers:
  - name: my-sse-server
    transport: sse
    url: https://api.example.com/mcp
    headers:
      Authorization: Bearer ${TOKEN}
```

### Multiple MCP Servers

You can specify multiple MCP servers per skill:

```yaml
---
name: multi-server-skill
description: Skill with multiple MCP servers
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]

  - name: fetch
    transport: stdio
    command: uvx
    args: [mcp-server-fetch]
---
```

### Environment Variables

Use `${VAR_NAME}` syntax for environment variable expansion:

```yaml
mcp-servers:
  - name: server-with-env
    transport: http
    url: ${API_URL}
    headers:
      Authorization: Bearer ${API_TOKEN}
```

## Native Tool Registration

### Creating Native Tools

Create a `helper.py` file in your skill directory and use the `@register_tool` decorator:

```python
from deepagents.middleware.skill_tools import register_tool

@register_tool("my-skill")
def my_native_tool(param: str) -> str:
    """A native Python tool for my skill.

    Args:
        param: The input parameter.

    Returns:
        Processed result as string.
    """
    return f"Processed: {param}"
```

### Multiple Native Tools

You can register multiple tools for the same skill:

```python
from deepagents.middleware.skill_tools import register_tool

@register_tool("github-integration")
def create_repository(repo_name: str, description: str) -> str:
    """Create a new GitHub repository."""
    # Implementation here
    return f"Created {repo_name}"

@register_tool("github-integration")
def list_issues(repo: str) -> str:
    """List issues in a repository."""
    # Implementation here
    return f"Issues in {repo}"
```

## Usage

### Creating an Agent with MCP-Enabled Skills

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    skills=["/path/to/skills/"],
    # The agent will automatically connect to MCP servers
    # when skills become active
)

# Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Use the github integration"}]
})
```

### How Tools Are Called

When a skill with MCP or native tools is active, the agent can call:

```python
skill_tool_call(
    skill_name="github-integration",
    tool_name="create_repository",
    arguments={
        "repo_name": "my-repo",
        "description": "My repository"
    }
)
```

The dispatcher automatically routes to:
- Native Python tools (via `@register_tool`)
- MCP tools (via connected MCP servers)

## Tool Discovery

Available skill tools are automatically injected into the agent's system prompt when skills are active. The documentation includes:

- Skill names and descriptions
- Tool names and descriptions
- Parameter information

Example system prompt injection:

```
## Available Skill Tools

Use `skill_tool_call(skill_name=..., tool_name=..., arguments={...})` to invoke these tools:

### github-integration
- **github_create_repository**: Create a new GitHub repository
- **github_list_issues**: List issues in a repository
- **github_create_issue**: Create an issue in a repository
```

## Example Skills

### Complete GitHub Integration Skill

**File: `/skills/github-integration/SKILL.md`**

```markdown
---
name: github-integration
description: GitHub API integration for repository and issue management
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
---

# GitHub Integration Skill

This skill provides tools for working with GitHub repositories and issues.

## When to Use

- User asks to create, list, or manage GitHub repositories
- User wants to create or update GitHub issues
- User needs to search repositories or code
- User wants to manage pull requests

## MCP Tools Available

The following MCP tools are available from the GitHub server:

- `github_create_repository`: Create a new repository
- `github_create_issue`: Create an issue in a repository
- `github_search_repositories`: Search for repositories
- `github_search_code`: Search code in repositories
- `github_list_issues`: List issues in a repository

## Native Tools

This skill also provides custom native Python tools for specialized operations.
```

**File: `/skills/github-integration/helper.py`**

```python
from deepagents.middleware.skill_tools import register_tool
import os

@register_tool("github-integration")
def get_my_repositories() -> str:
    """Get a list of my GitHub repositories.

    Returns:
        JSON string of repository information.
    """
    # Custom implementation using GitHub REST API
    # This complements the MCP tools
    token = os.getenv("GITHUB_TOKEN")
    # ... implementation ...
    return "[]"

@register_tool("github-integration")
def get_repository_stats(repo: str) -> str:
    """Get statistics for a repository.

    Args:
        repo: Repository name in format "owner/repo"

    Returns:
        Repository statistics as JSON string.
    """
    # ... implementation ...
    return '{"stars": 0, "forks": 0}'
```

## Best Practices

### 1. Tool Naming

- Use descriptive, action-oriented names: `create_repository`, not `repo`
- For MCP tools, the server name is automatically prefixed: `github_create_repository`
- For native tools, use simple, clear names

### 2. Error Handling

Native tools should handle errors gracefully:

```python
@register_tool("my-skill")
def safe_tool(value: str) -> str:
    """A tool that handles errors gracefully."""
    try:
        # Implementation
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### 3. Type Hints

Always include type hints for better documentation and type safety:

```python
@register_tool("my-skill")
def typed_tool(
    query: str,
    limit: int = 10,
    filters: dict[str, str] | None = None,
) -> str:
    """A tool with proper type hints."""
    # Implementation
    return "result"
```

### 4. Documentation

Write clear docstrings that describe:
- What the tool does
- Parameters (with types)
- Return values
- Any important usage notes

### 5. Environment Variables

- Use environment variables for secrets and configuration
- Document required environment variables in your skill's documentation
- Provide default values where appropriate

## Troubleshooting

### MCP Server Connection Fails

If an MCP server fails to connect:

1. Check the server command is installed and accessible
2. Verify environment variables are set correctly
3. Test the server manually: `npx -y @server/package --version`
4. Check the agent logs for detailed error messages

The agent will continue to function, but tools from that server will be unavailable.

### Native Tools Not Found

If native tools aren't available:

1. Verify `helper.py` is in the skill directory
2. Check the skill name matches: `@register_tool("skill-name")`
3. Ensure the skill is loaded correctly
4. Check for import errors in `helper.py`

### Tools Not in System Prompt

If tools don't appear in the system prompt:

1. Verify the skill is active (loaded in `skills_metadata`)
2. Check that tools are registered/connected successfully
3. Look for warnings in the agent logs
4. Ensure `SkillToolMiddleware` is in the middleware stack

## Migration Guide

### From Pure MCP Config

If you previously used global MCP configuration in `.mcp.json`:

**Before (`.mcp.json`):**

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

**After (Skill with MCP):**

```yaml
---
name: github-integration
description: GitHub API integration
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]
---
```

### From Pure Python Tools

If you previously registered tools globally:

**Before:**

```python
# Somewhere in your code
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    return f"Result: {param}"
```

**After (Skill-based):**

```python
# In skills/my-skill/helper.py
from deepagents.middleware.skill_tools import register_tool

@register_tool("my-skill")
def my_tool(param: str) -> str:
    return f"Result: {param}"
```

## Advanced Usage

### Conditional MCP Servers

You can make MCP server configuration conditional by using environment variables:

```yaml
---
name: conditional-skill
description: Skill with conditional MCP servers
mcp-servers:
  - name: optional-server
    transport: stdio
    command: echo
    args: ["test"]
    env:
      ENABLED: ${ENABLE_MCP:-false}  # Default to false
---
```

### Combining MCP and Native Tools

The most powerful pattern combines both:

- **MCP tools** for standard operations (provided by MCP servers)
- **Native tools** for custom logic or specialized operations

```python
# helper.py
from deepagents.middleware.skill_tools import register_tool

@register_tool("hybrid-skill")
def custom_operation(data: dict) -> str:
    """Custom operation that complements MCP tools."""
    # Complex logic that MCP server doesn't provide
    return "custom result"
```

## API Reference

### MCPServerConfig

Type definition for MCP server configuration:

```python
class MCPServerConfig(TypedDict):
    name: str                              # Required
    transport: NotRequired[str]             # "stdio", "sse", or "http"
    command: NotRequired[str]               # For stdio
    args: NotRequired[list[str]]            # For stdio
    env: NotRequired[dict[str, str]]        # For stdio
    url: NotRequired[str]                   # For sse/http
    headers: NotRequired[dict[str, str]]    # For sse/http
```

### register_tool

Decorator to register a native tool for a skill:

```python
def register_tool(skill_name: str) -> Callable[[Callable], Callable]:
    """Register a native tool for a skill.

    Args:
        skill_name: Name of the skill (must match SKILL.md frontmatter)
    """
```

### get_native_tools

Get all native tools for a skill:

```python
def get_native_tools(skill_name: str) -> dict[str, Callable]:
    """Get all native tools registered for a skill."""
```

### clear_native_tools

Clear native tools from the registry:

```python
def clear_native_tools(skill_name: str | None = None) -> None:
    """Clear native tools. If skill_name is None, clears all."""
```

## See Also

- [Agent Skills Specification](https://agentskills.io/specification)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Skills Documentation](../../deepagents/middleware/skills.py)
- [MCP Tools Documentation](../../deepagents/middleware/skill_tools.py)
