---
name: github-integration
description: GitHub API integration for repository and issue management with MCP servers and native tools
mcp-servers:
  - name: github
    transport: stdio
    command: npx
    args: [-y, @modelcontextprotocol/server-github]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
---

# GitHub Integration Skill

This skill provides comprehensive GitHub integration through both MCP servers and native Python tools.

## When to Use

Invoke this skill when the user request involves:
- Creating or managing GitHub repositories
- Creating, listing, or updating GitHub issues
- Searching repositories or code
- Managing pull requests
- Analyzing repository statistics
- Working with GitHub's REST API

## MCP Tools Available

The following MCP tools are available from the GitHub MCP server when `GITHUB_TOKEN` is set:

### Repository Operations
- `github_create_repository`: Create a new GitHub repository
- `github_push_files`: Push files to a repository
- `github_create_or_update_file`: Create or update a file in a repository
- `github_repository_info`: Get information about a repository

### Issue Operations
- `github_create_issue`: Create an issue in a repository
- `github_create_pull_request`: Create a pull request
- `github_list_issues`: List issues in a repository
- `github_list_pull_requests`: List pull requests in a repository

### Search Operations
- `github_search_repositories`: Search for GitHub repositories
- `github_search_code`: Search code across repositories
- `github_search_commits`: Search commits in a repository

## Native Tools

In addition to MCP tools, this skill provides custom native Python tools for specialized operations:

### `get_my_repositories`
Get a list of repositories owned by the authenticated user.

**Parameters:** None

**Returns:** JSON string containing repository information

### `get_repository_stats`
Get detailed statistics for a repository.

**Parameters:**
- `repo` (str): Repository name in format "owner/repo"

**Returns:** JSON string with repository statistics (stars, forks, issues, etc.)

### `get_user_profile`
Get GitHub user profile information.

**Parameters:**
- `username` (str): GitHub username (defaults to authenticated user if omitted)

**Returns:** JSON string with user profile data

## Setup

### Required Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token with appropriate permissions
  - Required for MCP server connection
  - Get one from: https://github.com/settings/tokens
  - Required scopes: `repo`, `read:org`, `user`

### Installation

Ensure the GitHub MCP server is available:

```bash
npx -y @modelcontextprotocol/server-github --help
```

## Usage Examples

### Creating a Repository

```python
# The agent will call:
skill_tool_call(
    skill_name="github-integration",
    tool_name="github_create_repository",
    arguments={
        "repository": {
            "name": "my-new-repo",
            "description": "A sample repository",
            "visibility": "public"
        }
    }
)
```

### Creating an Issue

```python
skill_tool_call(
    skill_name="github-integration",
    tool_name="github_create_issue",
    arguments={
        "repository": "owner/repo",
        "title": "Bug found in production",
        "body": "Detailed description of the issue..."
    }
)
```

### Using Native Tools

```python
skill_tool_call(
    skill_name="github-integration",
    tool_name="get_repository_stats",
    arguments={
        "repo": "langchain-ai/langchain"
    }
)
```

## Best Practices

1. **Check Repository Existence**: Before creating, use `github_repository_info` to check if a repository exists
2. **Handle Errors**: GitHub operations may fail due to permissions, rate limits, or invalid input
3. **Use Descriptive Names**: Repository and issue names should be clear and descriptive
4. **Set Appropriate Visibility**: Consider whether repositories should be public or private
5. **Respect Rate Limits**: GitHub has API rate limits - batch operations when possible

## Troubleshooting

### MCP Server Connection Fails

**Error:** "Failed to connect to MCP server 'github'"

**Solutions:**
1. Verify `GITHUB_TOKEN` is set correctly
2. Check the token has required permissions
3. Ensure `npx` and the GitHub MCP server are installed
4. Test manually: `npx -y @modelcontextprotocol/server-github`

### Repository Not Found

**Error:** "Repository 'owner/repo' not found"

**Solutions:**
1. Verify the repository name format: "owner/repo"
2. Check you have access to the repository
3. Ensure the repository exists

### Rate Limit Exceeded

**Error:** "GitHub API rate limit exceeded"

**Solutions:**
1. Wait for the rate limit to reset (typically 1 hour)
2. Use authentication to increase rate limits
3. Reduce the number of API calls
4. Consider using GraphQL API for more efficient queries

## Related Skills

- **web-research**: For finding information about GitHub features
- **file-system**: For working with cloned repositories locally
- **subagents**: For delegating complex GitHub workflows
