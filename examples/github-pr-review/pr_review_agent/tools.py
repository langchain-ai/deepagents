"""GitHub tools for PR review agent.

These tools provide access to PR data, repository information,
and the ability to post review comments and commits.

Note: Tools use a global _github_client that must be set before use via set_github_client().
"""

from langchain_core.tools import tool

from .github_client import GitHubClient


# Global GitHub client - set before running agent
_github_client: GitHubClient | None = None

# Global to capture submitted plans (set by webhook, read by agent)
_submitted_plan: list[str] | None = None


def set_github_client(client: GitHubClient) -> None:
    """Set the global GitHub client for tools to use."""
    global _github_client
    _github_client = client


def get_github_client() -> GitHubClient:
    """Get the global GitHub client."""
    if _github_client is None:
        raise RuntimeError("GitHub client not set. Call set_github_client() first.")
    return _github_client


def get_submitted_plan() -> list[str] | None:
    """Get the plan submitted by the agent."""
    return _submitted_plan


def clear_submitted_plan() -> None:
    """Clear the submitted plan."""
    global _submitted_plan
    _submitted_plan = None


@tool
def submit_plan(plan_items: list[str]) -> str:
    """Submit a plan for approval before making changes.

    Use this tool when in planning mode to propose changes before executing them.
    The plan will be posted as a comment and require approval (👍) before proceeding.

    Args:
        plan_items: List of planned changes, each describing one specific change.
                   Format: "File: <path> - <description of change>"

    Returns:
        Confirmation that the plan was submitted
    """
    global _submitted_plan

    if not plan_items:
        return "Error: Plan cannot be empty. Provide at least one planned change."

    _submitted_plan = plan_items
    return f"Plan submitted with {len(plan_items)} items. Waiting for approval..."


@tool
async def get_pr_diff(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get the diff for a pull request.

    Args:
        owner: Repository owner (username or org)
        repo: Repository name
        pr_number: Pull request number

    Returns:
        The PR diff in unified diff format
    """
    try:
        client = get_github_client()
        return await client.get_pr_diff(owner, repo, pr_number)
    except Exception as e:
        return f"Error fetching PR diff: {e}"


@tool
async def get_pr_files(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get the list of files changed in a pull request with patch info.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number

    Returns:
        Formatted list of changed files with status and patch
    """
    files = await get_github_client().get_pr_files(owner, repo, pr_number)

    result = []
    for f in files:
        entry = f"## {f['filename']} ({f['status']})\n"
        entry += f"Additions: {f['additions']}, Deletions: {f['deletions']}\n"
        if f.get("patch"):
            entry += f"```diff\n{f['patch']}\n```\n"
        result.append(entry)

    return "\n".join(result)


@tool
async def get_pr_details(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get pull request details including title, description, metadata, and merge status.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number

    Returns:
        Formatted PR details including mergeable status
    """
    pr = await get_github_client().get_pr_details(owner, repo, pr_number)

    mergeable = pr.get("mergeable")
    mergeable_state = pr.get("mergeable_state", "unknown")

    merge_status = "✅ Clean" if mergeable else "❌ Has conflicts" if mergeable is False else "⏳ Checking..."

    return f"""# PR #{pr['number']}: {pr['title']}

**Author:** {pr['user']}
**State:** {pr['state']}
**Mergeable:** {merge_status} ({mergeable_state})
**Base:** {pr['base']} ← **Head:** {pr['head']}
**Commits:** {pr['commits']}
**Changed files:** {pr['changed_files']}
**Additions:** {pr['additions']} | **Deletions:** {pr['deletions']}

## Description
{pr['body'] or '_No description provided_'}
"""


@tool
async def get_pr_checks(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get CI check status and workflow runs for a pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number

    Returns:
        Formatted CI status including check runs and their conclusions
    """
    check_runs = await get_github_client().get_pr_checks(owner, repo, pr_number)

    result = ["# CI Check Status\n"]

    if not check_runs:
        result.append("No CI checks found for this PR.")
    else:
        passed = []
        failed = []
        pending = []

        for check in check_runs:
            name = check["name"]
            status = check["status"]
            conclusion = check.get("conclusion")

            if status != "completed":
                pending.append(f"- ⏳ **{name}**: {status}")
            elif conclusion == "success":
                passed.append(f"- ✅ **{name}**: passed")
            elif conclusion in ("failure", "cancelled", "timed_out"):
                summary = check.get("output_summary") or ""
                if summary:
                    failed.append(f"- ❌ **{name}**: {conclusion}\n  > {summary[:200]}")
                else:
                    failed.append(f"- ❌ **{name}**: {conclusion}")
            else:
                passed.append(f"- ⚪ **{name}**: {conclusion or 'neutral'}")

        if failed:
            result.append("## ❌ Failed Checks\n")
            result.extend(failed)
            result.append("")

        if pending:
            result.append("## ⏳ In Progress\n")
            result.extend(pending)
            result.append("")

        if passed:
            result.append("## ✅ Passed Checks\n")
            result.extend(passed)

    return "\n".join(result)


@tool
async def list_workflow_files(
    owner: str,
    repo: str,
) -> str:
    """List GitHub Actions workflow files in the repository.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        List of workflow files with their paths
    """
    try:
        contents = await get_github_client().get_directory_contents(
            owner, repo, ".github/workflows"
        )

        if not contents:
            return "No workflow files found in .github/workflows/"

        result = ["# GitHub Actions Workflows\n"]
        for item in contents:
            if item["type"] == "file" and item["name"].endswith((".yml", ".yaml")):
                result.append(f"- `{item['path']}`")

        result.append("\nUse get_file_content to read specific workflow files.")
        return "\n".join(result)

    except Exception:
        return "No .github/workflows/ directory found in this repository."


@tool
async def get_pr_commits(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get the commit history for a pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number

    Returns:
        Formatted list of commits with messages
    """
    commits = await get_github_client().get_pr_commits(owner, repo, pr_number)

    result = ["# Commits\n"]
    for c in commits:
        sha_short = c["sha"][:7]
        message = c["message"].split("\n")[0]  # First line only
        author = c["author"]
        result.append(f"- `{sha_short}` {message} ({author})")

    return "\n".join(result)


@tool
async def get_pr_comments(
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    """Get all comments on a pull request (issue comments + review comments).

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number

    Returns:
        Formatted list of comments
    """
    comments = await get_github_client().get_pr_comments(owner, repo, pr_number)

    result = ["# PR Comments\n"]

    if comments["issue_comments"]:
        result.append("## General Comments\n")
        for c in comments["issue_comments"]:
            result.append(f"**{c['user']}:** {c['body']}\n")

    if comments["review_comments"]:
        result.append("## Review Comments\n")
        for c in comments["review_comments"]:
            path = c.get("path", "unknown")
            line = c.get("line", "?")
            result.append(f"**{c['user']}** on `{path}:{line}`:\n{c['body']}\n")

    if len(result) == 1:
        return "No comments on this PR yet."

    return "\n".join(result)


@tool
async def get_repo_code_style(
    owner: str,
    repo: str,
) -> str:
    """Get repository code style configuration and guidelines.

    Looks for common style config files like .editorconfig, prettier, eslint,
    ruff, pyproject.toml, CONTRIBUTING.md, etc.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Formatted code style information
    """
    style_files = [
        ".editorconfig",
        ".prettierrc",
        ".prettierrc.json",
        ".eslintrc",
        ".eslintrc.json",
        "ruff.toml",
        "pyproject.toml",
        "setup.cfg",
        ".rubocop.yml",
        ".clang-format",
        "CONTRIBUTING.md",
        "STYLE.md",
        ".github/CONTRIBUTING.md",
    ]

    result = ["# Repository Code Style\n"]
    client = get_github_client()

    for filepath in style_files:
        try:
            content = await client.get_file_content(owner, repo, filepath, ref="HEAD")
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            result.append(f"## {filepath}\n```\n{content}\n```\n")
        except Exception:
            continue

    if len(result) == 1:
        return "No style configuration files found in repository."

    return "\n".join(result)


@tool
async def get_security_alerts(
    owner: str,
    repo: str,
) -> str:
    """Get security vulnerability alerts for the repository.

    Requires the GitHub App to have security_events permission.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Formatted list of security alerts
    """
    try:
        alerts = await get_github_client().get_dependabot_alerts(owner, repo)

        if not alerts:
            return "No open Dependabot security alerts."

        result = ["# Security Alerts\n"]
        for alert in alerts:
            severity = alert.get("severity", "unknown")
            summary = alert.get("summary", "No summary")
            package = alert.get("package", "unknown")
            result.append(f"- **{severity.upper()}** [{package}]: {summary}")

        return "\n".join(result)

    except Exception as e:
        return f"Unable to fetch security alerts: {e}"


@tool
async def get_code_scanning_alerts(
    owner: str,
    repo: str,
) -> str:
    """Get code scanning (CodeQL) alerts for the repository.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Formatted list of code scanning alerts
    """
    try:
        alerts = await get_github_client().get_code_scanning_alerts(owner, repo)

        if not alerts:
            return "No open code scanning alerts."

        result = ["# Code Scanning Alerts\n"]
        for alert in alerts:
            severity = alert.get("severity", "unknown")
            desc = alert.get("description", "No description")
            path = alert.get("path", "unknown")
            line = alert.get("line", "?")
            result.append(f"- **{severity.upper()}** `{path}:{line}`: {desc}")

        return "\n".join(result)

    except Exception as e:
        return f"Unable to fetch code scanning alerts: {e}"


@tool
async def search_repo_issues(
    owner: str,
    repo: str,
    query: str,
) -> str:
    """Search for related issues and PRs in the repository.

    Args:
        owner: Repository owner
        repo: Repository name
        query: Search query (keywords)

    Returns:
        Formatted list of related issues/PRs
    """
    items = await get_github_client().search_issues(owner, repo, query)

    if not items:
        return f"No related issues found for: {query}"

    result = [f"# Related Issues/PRs for '{query}'\n"]
    for item in items:
        item_type = "PR" if item["is_pr"] else "Issue"
        state = item["state"]
        result.append(f"- [{item_type} #{item['number']}]({item['html_url']}) {item['title']} ({state})")

    return "\n".join(result)


@tool
async def get_file_content(
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> str:
    """Get the content of a specific file from the repository.

    Args:
        owner: Repository owner
        repo: Repository name
        path: File path in the repository
        ref: Git ref (branch, tag, or commit SHA)

    Returns:
        File content
    """
    try:
        content = await get_github_client().get_file_content(owner, repo, path, ref)
        if len(content) > 10000:
            content = content[:10000] + "\n\n... (truncated, file too large)"
        return f"# {path}\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
async def post_pr_comment(
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
) -> str:
    """Post a comment on a pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        body: Comment body (markdown supported)

    Returns:
        Confirmation message
    """
    await get_github_client().post_comment(owner, repo, pr_number, body)
    return f"Comment posted to PR #{pr_number}"


@tool
async def create_pr_review(
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
    event: str,
) -> str:
    """Create a pull request review.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        body: Review body (markdown supported)
        event: Review event - "APPROVE", "REQUEST_CHANGES", or "COMMENT"

    Returns:
        Confirmation message
    """
    if event not in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
        return f"Invalid event: {event}. Must be APPROVE, REQUEST_CHANGES, or COMMENT"

    await get_github_client().create_review(owner, repo, pr_number, body, event)
    return f"Review ({event}) submitted to PR #{pr_number}"


@tool
async def create_or_update_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str,
) -> str:
    """Create or update a file in the repository.

    This commits a file directly to the specified branch. Use this when:
    - The requesting user has write access: commit to the PR branch
    - The requesting user has read-only access: commit to a new branch

    Args:
        owner: Repository owner
        repo: Repository name
        path: File path in the repository
        content: File content
        message: Commit message
        branch: Branch to commit to

    Returns:
        Confirmation with commit SHA
    """
    commit_sha = await get_github_client().create_or_update_file(
        owner, repo, path, content, message, branch
    )
    return f"Committed '{path}' on branch '{branch}' (commit: {commit_sha})"


@tool
async def create_branch(
    owner: str,
    repo: str,
    new_branch: str,
    from_ref: str,
) -> str:
    """Create a new branch from an existing ref.

    Args:
        owner: Repository owner
        repo: Repository name
        new_branch: Name for the new branch
        from_ref: Source ref (branch name or SHA) to branch from

    Returns:
        Confirmation message
    """
    await get_github_client().create_branch(owner, repo, new_branch, from_ref)
    return f"Created branch '{new_branch}' from '{from_ref}'"


@tool
async def create_pr_from_branch(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
) -> str:
    """Create a new pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        title: PR title
        body: PR description
        head: Head branch (the branch with changes)
        base: Base branch (the branch to merge into)

    Returns:
        PR URL and number
    """
    result = await get_github_client().create_pull_request(
        owner, repo, title, body, head, base
    )
    return f"Created PR #{result['number']}: {result['html_url']}"
