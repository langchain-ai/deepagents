"""GitHub API client with App authentication.

Handles JWT generation for GitHub App authentication and provides
an async wrapper around PyGithub for API operations.
"""

import asyncio
import base64
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

from github import Auth, Github, GithubIntegration
from github.PullRequest import PullRequest
from github.Repository import Repository


def get_private_key() -> str:
    """Load GitHub App private key from file or environment variable."""
    key_b64 = os.environ.get("GITHUB_PRIVATE_KEY_BASE64")
    if key_b64:
        return base64.b64decode(key_b64).decode("utf-8")

    key_path = os.environ.get("GITHUB_PRIVATE_KEY_PATH", "./private-key.pem")
    path = Path(key_path)
    if path.exists():
        return path.read_text()

    raise ValueError(
        "GitHub private key not found. Set GITHUB_PRIVATE_KEY_BASE64 or GITHUB_PRIVATE_KEY_PATH"
    )


def get_app_id() -> int:
    """Get GitHub App ID from environment."""
    app_id = os.environ.get("GITHUB_APP_ID")
    if not app_id:
        raise ValueError("GITHUB_APP_ID environment variable not set")
    return int(app_id)


def get_installation_client(installation_id: int) -> Github:
    """Get an authenticated GitHub client for a specific installation.

    Args:
        installation_id: The GitHub App installation ID

    Returns:
        Authenticated PyGithub client
    """
    app_id = get_app_id()
    private_key = get_private_key()

    auth = Auth.AppAuth(app_id, private_key)
    gi = GithubIntegration(auth=auth)
    installation_auth = gi.get_access_token(installation_id)

    return Github(auth=Auth.Token(installation_auth.token))


async def run_sync(func, *args, **kwargs) -> Any:
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


class GitHubClient:
    """Async wrapper around PyGithub for GitHub App installations."""

    def __init__(self, installation_id: int):
        """Initialize with installation ID."""
        self.installation_id = installation_id
        self._client: Github | None = None
        self._client_created: float = 0

    def _get_client(self) -> Github:
        """Get or refresh the PyGithub client (tokens valid for 1 hour)."""
        if self._client and time.time() < self._client_created + 3500:
            return self._client

        self._client = get_installation_client(self.installation_id)
        self._client_created = time.time()
        return self._client

    async def get_repo(self, owner: str, repo: str) -> Repository:
        """Get a repository object."""
        client = self._get_client()
        return await run_sync(client.get_repo, f"{owner}/{repo}")

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequest:
        """Get a pull request object."""
        repo_obj = await self.get_repo(owner, repo)
        return await run_sync(repo_obj.get_pull, pr_number)

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Get the diff for a pull request."""
        import urllib.request

        token = self._get_token()

        def _fetch_diff():
            req = urllib.request.Request(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3.diff",
                }
            )
            with urllib.request.urlopen(req) as resp:
                return resp.read().decode("utf-8")

        return await run_sync(_fetch_diff)

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """Get files changed in a pull request."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        files = await run_sync(lambda: list(pr.get_files()))
        return [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch": f.patch,
            }
            for f in files
        ]

    async def get_pr_details(self, owner: str, repo: str, pr_number: int) -> dict:
        """Get pull request details."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        return {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "user": pr.user.login,
            "base": pr.base.ref,
            "head": pr.head.ref,
            "commits": pr.commits,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "mergeable": pr.mergeable,
            "mergeable_state": pr.mergeable_state,
        }

    async def get_pr_commits(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """Get commits in a pull request."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        commits = await run_sync(lambda: list(pr.get_commits()))
        return [
            {
                "sha": c.sha,
                "message": c.commit.message,
                "author": c.commit.author.name,
            }
            for c in commits
        ]

    async def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> dict:
        """Get comments on a pull request."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        repo_obj = await self.get_repo(owner, repo)
        issue = await run_sync(repo_obj.get_issue, pr_number)

        issue_comments = await run_sync(lambda: list(issue.get_comments()))
        review_comments = await run_sync(lambda: list(pr.get_review_comments()))

        return {
            "issue_comments": [
                {"user": c.user.login, "body": c.body}
                for c in issue_comments
            ],
            "review_comments": [
                {
                    "user": c.user.login,
                    "body": c.body,
                    "path": c.path,
                    "line": c.line or c.original_line,
                }
                for c in review_comments
            ],
        }

    async def get_pr_checks(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """Get CI check runs for a pull request."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        repo_obj = await self.get_repo(owner, repo)
        commit = await run_sync(repo_obj.get_commit, pr.head.sha)
        check_runs = await run_sync(lambda: list(commit.get_check_runs()))

        return [
            {
                "name": cr.name,
                "status": cr.status,
                "conclusion": cr.conclusion,
                "output_summary": cr.output.summary if cr.output else None,
            }
            for cr in check_runs
        ]

    async def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> str:
        """Get content of a file from the repository."""
        repo_obj = await self.get_repo(owner, repo)
        content = await run_sync(repo_obj.get_contents, path, ref=ref)
        if isinstance(content, list):
            raise ValueError(f"{path} is a directory, not a file")
        return content.decoded_content.decode("utf-8")

    async def get_directory_contents(self, owner: str, repo: str, path: str, ref: str | None = None) -> list[dict]:
        """Get contents of a directory."""
        repo_obj = await self.get_repo(owner, repo)
        kwargs = {"ref": ref} if ref else {}
        contents = await run_sync(repo_obj.get_contents, path, **kwargs)
        if not isinstance(contents, list):
            contents = [contents]
        return [
            {"name": c.name, "path": c.path, "type": c.type}
            for c in contents
        ]

    async def post_comment(self, owner: str, repo: str, pr_number: int, body: str) -> None:
        """Post a comment on a pull request."""
        repo_obj = await self.get_repo(owner, repo)
        issue = await run_sync(repo_obj.get_issue, pr_number)
        await run_sync(issue.create_comment, body)

    async def create_review(
        self, owner: str, repo: str, pr_number: int, body: str, event: str
    ) -> None:
        """Create a pull request review."""
        pr = await self.get_pull_request(owner, repo, pr_number)
        await run_sync(pr.create_review, body=body, event=event)

    async def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
    ) -> str:
        """Create or update a file in the repository."""
        repo_obj = await self.get_repo(owner, repo)

        sha = None
        try:
            existing = await run_sync(repo_obj.get_contents, path, ref=branch)
            if not isinstance(existing, list):
                sha = existing.sha
        except Exception:
            pass

        if sha:
            result = await run_sync(
                repo_obj.update_file, path, message, content, sha, branch=branch
            )
        else:
            result = await run_sync(
                repo_obj.create_file, path, message, content, branch=branch
            )

        return result["commit"].sha[:7]

    async def create_branch(self, owner: str, repo: str, new_branch: str, from_ref: str) -> None:
        """Create a new branch from an existing ref."""
        repo_obj = await self.get_repo(owner, repo)

        try:
            source = await run_sync(repo_obj.get_branch, from_ref)
            sha = source.commit.sha
        except Exception:
            sha = from_ref

        await run_sync(repo_obj.create_git_ref, f"refs/heads/{new_branch}", sha)

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict:
        """Create a new pull request."""
        repo_obj = await self.get_repo(owner, repo)
        pr = await run_sync(repo_obj.create_pull, title=title, body=body, head=head, base=base)
        return {"number": pr.number, "html_url": pr.html_url}

    async def search_issues(self, owner: str, repo: str, query: str) -> list[dict]:
        """Search for issues and PRs in a repository."""
        client = self._get_client()
        search_query = f"repo:{owner}/{repo} {query}"
        results = await run_sync(lambda: list(client.search_issues(search_query))[:10])
        return [
            {
                "number": i.number,
                "title": i.title,
                "state": i.state,
                "html_url": i.html_url,
                "is_pr": i.pull_request is not None,
            }
            for i in results
        ]

    async def get_dependabot_alerts(self, owner: str, repo: str) -> list[dict]:
        """Get open Dependabot alerts."""
        repo_obj = await self.get_repo(owner, repo)
        try:
            alerts = await run_sync(lambda: list(repo_obj.get_dependabot_alerts(state="open"))[:20])
            return [
                {
                    "severity": a.security_advisory.severity if a.security_advisory else "unknown",
                    "summary": a.security_advisory.summary if a.security_advisory else "No summary",
                    "package": a.dependency.package.name if a.dependency and a.dependency.package else "unknown",
                }
                for a in alerts
            ]
        except Exception as e:
            return []

    async def get_code_scanning_alerts(self, owner: str, repo: str) -> list[dict]:
        """Get open code scanning alerts."""
        repo_obj = await self.get_repo(owner, repo)
        try:
            alerts = await run_sync(lambda: list(repo_obj.get_codescan_alerts())[:20])
            return [
                {
                    "severity": a.rule.severity if a.rule else "unknown",
                    "description": a.rule.description if a.rule else "No description",
                    "path": a.most_recent_instance.location.path if a.most_recent_instance and a.most_recent_instance.location else "unknown",
                    "line": a.most_recent_instance.location.start_line if a.most_recent_instance and a.most_recent_instance.location else None,
                }
                for a in alerts
                if a.state == "open"
            ]
        except Exception:
            return []

    # --- Webhook-specific operations using PyGithub ---

    def _get_token(self) -> str:
        """Get the current installation token (for raw diff requests only)."""
        client = self._get_client()
        return client._Github__requester._Requester__auth.token

    async def get_user_permission(self, owner: str, repo: str, username: str) -> str:
        """Get a user's permission level on a repository."""
        try:
            repo_obj = await self.get_repo(owner, repo)
            return await run_sync(repo_obj.get_collaborator_permission, username)
        except Exception:
            return "read"

    async def get_issue_comments(self, owner: str, repo: str, issue_number: int) -> list[dict]:
        """Get comments on an issue/PR."""
        repo_obj = await self.get_repo(owner, repo)
        issue = await run_sync(repo_obj.get_issue, issue_number)
        comments = await run_sync(lambda: list(issue.get_comments()))
        return [
            {"id": c.id, "user": {"login": c.user.login, "type": c.user.type}, "body": c.body}
            for c in comments
        ]

    async def get_comment_reactions(self, owner: str, repo: str, comment_id: int) -> list[dict]:
        """Get reactions on a comment."""
        repo_obj = await self.get_repo(owner, repo)
        comment = await run_sync(repo_obj.get_issue_comment, comment_id)
        reactions = await run_sync(lambda: list(comment.get_reactions()))
        return [
            {"id": r.id, "content": r.content, "user": {"login": r.user.login}}
            for r in reactions
        ]

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict:
        """Create a comment on an issue/PR."""
        repo_obj = await self.get_repo(owner, repo)
        issue = await run_sync(repo_obj.get_issue, issue_number)
        comment = await run_sync(issue.create_comment, body)
        return {"id": comment.id, "body": comment.body}

    async def update_issue_comment(self, owner: str, repo: str, comment_id: int, body: str) -> dict:
        """Update an existing comment."""
        repo_obj = await self.get_repo(owner, repo)
        comment = await run_sync(repo_obj.get_issue_comment, comment_id)
        await run_sync(comment.edit, body)
        return {"id": comment.id, "body": body}
