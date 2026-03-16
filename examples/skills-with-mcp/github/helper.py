"""Native Python tools for the GitHub integration skill.

This module provides custom tools that complement the MCP GitHub server
with specialized operations.
"""

from __future__ import annotations

import json
import os
from typing import Any

from deepagents.middleware.skill_tools import register_tool


@register_tool("github-integration")
def get_my_repositories() -> str:
    """Get a list of repositories owned by the authenticated user.

    This native tool uses the GitHub REST API to fetch repositories,
    complementing the MCP tools with additional filtering and formatting.

    Returns:
        JSON string containing array of repository information.
        Each repository includes: name, full_name, description, language,
        stars, forks, is_private, updated_at.

    Raises:
        RuntimeError: If GITHUB_TOKEN is not set or API call fails.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"error": "GITHUB_TOKEN environment variable not set"})

    # Implementation would use requests or httpx to call GitHub API
    # For this example, we return mock data
    # In production, you would make actual API calls like:
    #
    # import requests
    # headers = {
    #     "Authorization": f"Bearer {token}",
    #     "Accept": "application/vnd.github.v3+json"
    # }
    # response = requests.get("https://api.github.com/user/repos", headers=headers)
    # response.raise_for_status()
    # return response.text

    mock_repos = [
        {
            "name": "example-repo-1",
            "full_name": "user/example-repo-1",
            "description": "First example repository",
            "language": "Python",
            "stars": 42,
            "forks": 8,
            "is_private": False,
            "updated_at": "2026-03-16T00:00:00Z",
        },
        {
            "name": "example-repo-2",
            "full_name": "user/example-repo-2",
            "description": "Second example repository",
            "language": "TypeScript",
            "stars": 128,
            "forks": 32,
            "is_private": True,
            "updated_at": "2026-03-15T12:34:56Z",
        },
    ]

    return json.dumps(mock_repos, indent=2)


@register_tool("github-integration")
def get_repository_stats(repo: str) -> str:
    """Get detailed statistics for a repository.

    This native tool provides comprehensive statistics beyond what's available
    through MCP tools, including:
    - Stars, forks, and watchers counts
    - Open issues and pull requests counts
    - Latest release information
    - Primary language and language breakdown
    - Contributor count

    Args:
        repo: Repository name in format "owner/repo".

    Returns:
        JSON string with repository statistics.

    Raises:
        ValueError: If repo format is invalid.
        RuntimeError: If GITHUB_TOKEN is not set or API call fails.
    """
    if "/" not in repo:
        return json.dumps({"error": "Invalid repo format. Expected 'owner/repo'"})

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"error": "GITHUB_TOKEN environment variable not set"})

    # Implementation would call GitHub API:
    # GET /repos/{owner}/{repo}
    # GET /repos/{owner}/{repo}/languages
    # GET /repos/{owner}/{repo}/contributors
    # GET /repos/{owner}/{repo}/releases/latest

    owner, repo_name = repo.split("/", 1)

    # Mock response for demonstration
    mock_stats = {
        "repository": repo,
        "stars": 1234,
        "forks": 567,
        "watchers": 89,
        "open_issues": 12,
        "open_pull_requests": 3,
        "primary_language": "Python",
        "languages": {
            "Python": "75.2%",
            "TypeScript": "15.8%",
            "JavaScript": "5.3%",
            "HTML": "2.1%",
            "CSS": "1.6%",
        },
        "contributors": 42,
        "latest_release": {
            "tag_name": "v1.2.3",
            "name": "Latest Release",
            "published_at": "2026-03-10T15:30:00Z",
        },
        "created_at": "2023-01-15T10:20:30Z",
        "updated_at": "2026-03-16T08:45:12Z",
    }

    return json.dumps(mock_stats, indent=2)


@register_tool("github-integration")
def get_user_profile(username: str | None = None) -> str:
    """Get GitHub user profile information.

    This native tool fetches detailed user profile information including:
    - Name, login, and bio
    - Avatar URL
    - Follower and following counts
    - Public repository count
    - Location and company
    - Blog/website URL
    - Account creation date

    Args:
        username: GitHub username. If None, returns profile for the
            authenticated user (based on GITHUB_TOKEN).

    Returns:
        JSON string with user profile data.

    Raises:
        RuntimeError: If GITHUB_TOKEN is not set or user not found.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"error": "GITHUB_TOKEN environment variable not set"})

    # Implementation would call:
    # GET /users/{username} (if username provided)
    # GET /user (if username is None, for authenticated user)

    # Mock response for demonstration
    if username:
        mock_profile = {
            "login": username,
            "name": f"{username.capitalize()} Developer",
            "bio": f"Software developer and open source contributor",
            "avatar_url": f"https://github.com/{username}.png",
            "followers": 1234,
            "following": 567,
            "public_repos": 89,
            "location": "San Francisco, CA",
            "company": "Tech Corp",
            "blog": f"https://{username}.dev",
            "created_at": "2020-01-01T00:00:00Z",
        }
    else:
        mock_profile = {
            "login": "authenticated-user",
            "name": "Authenticated User",
            "bio": "GitHub user authenticated via GITHUB_TOKEN",
            "avatar_url": "https://github.com/placeholder.png",
            "followers": 456,
            "following": 123,
            "public_repos": 23,
            "location": None,
            "company": None,
            "blog": None,
            "created_at": "2022-06-15T10:30:00Z",
        }

    return json.dumps(mock_profile, indent=2)


@register_tool("github-integration")
def search_code_snippets(
    query: str,
    language: str | None = None,
    per_page: int = 10,
) -> str:
    """Search for code snippets across GitHub repositories.

    This native tool provides enhanced code search with:
    - Syntax highlighting information
    - File path and line numbers
    - Repository context
    - Filter by programming language

    Args:
        query: Search query (e.g., "function_name", "class MyClass").
        language: Filter by programming language (e.g., "Python", "JavaScript").
        per_page: Number of results to return (default: 10, max: 100).

    Returns:
        JSON string with search results including code snippets.

    Raises:
        ValueError: If per_page is out of range.
        RuntimeError: If GITHUB_TOKEN is not set or search fails.
    """
    if not 1 <= per_page <= 100:
        return json.dumps({"error": "per_page must be between 1 and 100"})

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"error": "GITHUB_TOKEN environment variable not set"})

    # Implementation would use GitHub Code Search API:
    # GET /search/code?q={query}+language:{language}

    # Mock response for demonstration
    mock_results = {
        "query": query,
        "language": language,
        "total_count": 1234,
        "items": [
            {
                "name": "example.py",
                "path": "src/example.py",
                "repository": "owner/repo",
                "language": language or "Python",
                "snippet": f"def {query}(...):\n    # Implementation",
                "line_number": 42,
            }
            for _ in range(min(per_page, 3))
        ],
    }

    return json.dumps(mock_results, indent=2)


# Example of how to add more tools:
#
# @register_tool("github-integration")
# def create_branch(repo: str, branch_name: str, from_branch: str = "main") -> str:
#     """Create a new branch in a repository.
#
#     Args:
#         repo: Repository name in format "owner/repo".
#         branch_name: Name for the new branch.
#         from_branch: Source branch to create from (default: "main").
#
#     Returns:
#         JSON string with branch creation result.
#     """
#     # Implementation here
#     return json.dumps({"success": True, "branch": branch_name})
