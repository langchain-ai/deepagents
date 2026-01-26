"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: E402

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
# ruff: noqa: E402
import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

from langchain.agents.middleware import AgentState, after_agent
from langgraph.config import get_config
from langgraph.graph.state import RunnableConfig
from langgraph.runtime import Runtime

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import asyncio

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import create_server_agent

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import settings
from deepagents_cli.encryption import decrypt_token
from deepagents_cli.integrations.sandbox_factory import create_sandbox_async
from deepagents_cli.tools import fetch_url, http_request, web_search

tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)

from langgraph_sdk import get_client

client = get_client()

SANDBOX_CREATING = "__creating__"
SANDBOX_CREATION_TIMEOUT = 180
SANDBOX_POLL_INTERVAL = 1.0

_SANDBOX_BACKENDS: dict[str, Any] = {}

LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")


async def comment_on_linear_issue(issue_id: str, comment_body: str) -> bool:
    """Add a comment to a Linear issue.

    Args:
        issue_id: The Linear issue ID
        comment_body: The comment text

    Returns:
        True if successful, False otherwise
    """
    if not LINEAR_API_KEY:
        return False

    import httpx

    # Linear uses GraphQL API
    url = "https://api.linear.app/graphql"

    mutation = """
    mutation CommentCreate($issueId: String!, $body: String!) {
        commentCreate(input: { issueId: $issueId, body: $body }) {
            success
            comment {
                id
            }
        }
    }
    """

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(
                url,
                headers={
                    "Authorization": LINEAR_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "query": mutation,
                    "variables": {"issueId": issue_id, "body": comment_body},
                },
            )
            response.raise_for_status()
            result = response.json()

            if result.get("data", {}).get("commentCreate", {}).get("success"):
                return True
            return False
        except Exception:
            return False


@after_agent
async def open_pr_if_needed(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Middleware that commits/pushes changes and comments on Linear after agent runs."""
    logger.info("After-agent middleware started")
    pr_url = None
    pr_number = None
    pr_title = "feat: Open SWE PR"

    try:
        # Get config using langgraph's get_config
        config = get_config()
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        logger.debug("Middleware running for thread %s", thread_id)

        # Get the last message content from state
        last_message_content = ""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Handle both dict and object message formats
            if isinstance(last_message, dict):
                last_message_content = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                last_message_content = last_message.content

        # Get Linear issue info for commenting
        linear_issue = configurable.get("linear_issue", {})
        linear_issue_id = linear_issue.get("id")

        if not thread_id:
            # Still comment on Linear if we have the issue ID
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        repo_config = configurable.get("repo", {})
        repo_owner = repo_config.get("owner")
        repo_name = repo_config.get("name")

        sandbox_backend = _SANDBOX_BACKENDS.get(thread_id)

        repo_dir = f"/workspace/{repo_name}"

        if not sandbox_backend or not repo_dir:
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        result = await asyncio.to_thread(
            sandbox_backend.execute, f"cd {repo_dir} && git status --porcelain"
        )

        has_uncommitted_changes = result.exit_code == 0 and result.output.strip()

        await asyncio.to_thread(
            sandbox_backend.execute, f"cd {repo_dir} && git fetch origin 2>/dev/null || true"
        )
        unpushed_result = await asyncio.to_thread(
            sandbox_backend.execute,
            f"cd {repo_dir} && git log --oneline @{{upstream}}..HEAD 2>/dev/null || git log --oneline origin/HEAD..HEAD 2>/dev/null || echo ''",
        )
        has_unpushed_commits = unpushed_result.exit_code == 0 and unpushed_result.output.strip()

        has_changes = has_uncommitted_changes or has_unpushed_commits

        if not has_changes:
            logger.info("No changes detected, skipping PR creation")
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        logger.info("Changes detected, preparing PR for thread %s", thread_id)

        branch_result = await asyncio.to_thread(
            sandbox_backend.execute, f"cd {repo_dir} && git rev-parse --abbrev-ref HEAD"
        )
        current_branch = branch_result.output.strip() if branch_result.exit_code == 0 else ""

        target_branch = f"open-swe/{thread_id}"

        if current_branch != target_branch:
            checkout_result = await asyncio.to_thread(
                sandbox_backend.execute, f"cd {repo_dir} && git checkout -b {target_branch}"
            )
            if checkout_result.exit_code != 0:
                await asyncio.to_thread(
                    sandbox_backend.execute, f"cd {repo_dir} && git checkout {target_branch}"
                )

        await asyncio.to_thread(
            sandbox_backend.execute, f"cd {repo_dir} && git config user.name 'Open SWE[bot]'"
        )
        await asyncio.to_thread(
            sandbox_backend.execute,
            f"cd {repo_dir} && git config user.email 'Open SWE@users.noreply.github.com'",
        )

        await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git add -A")

        await asyncio.to_thread(
            sandbox_backend.execute, f'cd {repo_dir} && git commit -m "feat: Open SWE PR"'
        )

        encrypted_token = configurable.get("github_token_encrypted")
        if encrypted_token:
            github_token = decrypt_token(encrypted_token)

        if github_token:
            remote_result = await asyncio.to_thread(
                sandbox_backend.execute, f"cd {repo_dir} && git remote get-url origin"
            )
            if remote_result.exit_code == 0:
                remote_url = remote_result.output.strip()
                if "github.com" in remote_url and "@" not in remote_url:
                    # Convert https://github.com/owner/repo.git to https://git:token@github.com/owner/repo.git
                    auth_url = remote_url.replace("https://", f"https://git:{github_token}@")
                    await asyncio.to_thread(
                        sandbox_backend.execute,
                        f"cd {repo_dir} && git push {auth_url} {target_branch}",
                    )
                else:
                    await asyncio.to_thread(
                        sandbox_backend.execute, f"cd {repo_dir} && git push origin {target_branch}"
                    )

            default_branch_result = await asyncio.to_thread(
                sandbox_backend.execute,
                f"cd {repo_dir} && git remote show origin | grep 'HEAD branch' | cut -d' ' -f5",
            )
            base_branch = (
                default_branch_result.output.strip()
                if default_branch_result.exit_code == 0
                else "main"
            )

            pr_title = "feat: Open SWE PR"
            pr_body = "Automated PR created by Open SWE agent."

            create_pr_cmd = f"""curl -s -X POST \\
                -H "Authorization: Bearer {github_token}" \\
                -H "Accept: application/vnd.github+json" \\
                -H "X-GitHub-Api-Version: 2022-11-28" \\
                https://api.github.com/repos/{repo_owner}/{repo_name}/pulls \\
                -d '{{"title":"{pr_title}","head":"{target_branch}","base":"{base_branch}","body":"{pr_body}"}}'
            """

            pr_result = await asyncio.to_thread(sandbox_backend.execute, create_pr_cmd)

            linear_issue = configurable.get("linear_issue", {})
            linear_issue_id = linear_issue.get("id")

            if pr_result.exit_code == 0:
                import json

                pr_response = json.loads(pr_result.output)
                pr_url = pr_response.get("html_url")
                pr_number = pr_response.get("number")
                if pr_url:
                    logger.info("PR created successfully: %s", pr_url)
                else:
                    logger.warning(
                        "PR creation response did not contain html_url: %s", pr_result.output[:200]
                    )

        if linear_issue_id and last_message_content:
            if pr_url:
                comment = f"""✅ **Pull Request Created**

I've created a pull request to address this issue:

**[PR #{pr_number}: {pr_title}]({pr_url})**

---

🤖 **Agent Response**

{last_message_content}"""
            else:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
            await comment_on_linear_issue(linear_issue_id, comment)

        logger.info("After-agent middleware completed successfully")
        return None

    except Exception as e:
        logger.error("Error in after-agent middleware: %s: %s", type(e).__name__, e)
        try:
            config = get_config()
            configurable = config.get("configurable", {})
            linear_issue = configurable.get("linear_issue", {})
            linear_issue_id = linear_issue.get("id")
            if linear_issue_id:
                error_comment = f"""❌ **Agent Error**

An error occurred while processing this issue:

```
{type(e).__name__}: {e}
```"""
                await comment_on_linear_issue(linear_issue_id, error_comment)
        except Exception as inner_e:
            logger.error("Failed to post error comment to Linear: %s", inner_e)
        return None


async def _clone_or_pull_repo_in_sandbox(
    sandbox_backend: Any, owner: str, repo: str, github_token: str | None = None
) -> str:
    """Clone a GitHub repo into the sandbox, or pull if it already exists.

    Args:
        sandbox_backend: The sandbox backend to execute commands in (LangSmithBackend)
        owner: GitHub repo owner
        repo: GitHub repo name
        github_token: GitHub access token (from agent auth or env var)

    Returns:
        Path to the cloned/updated repo directory
    """
    loop = asyncio.get_event_loop()

    token = github_token
    if not token:
        msg = "No GitHub token provided"
        raise ValueError(msg)

    repo_dir = f"/workspace/{repo}"

    check_result = await loop.run_in_executor(
        None, sandbox_backend.execute, f"test -d {repo_dir}/.git && echo exists"
    )

    if check_result.exit_code == 0 and "exists" in check_result.output:
        status_result = await loop.run_in_executor(
            None, sandbox_backend.execute, f"cd {repo_dir} && git status --porcelain"
        )

        # CRITICAL: Ensure remote URL doesn't contain token (clean up from previous runs)
        clean_url = f"https://github.com/{owner}/{repo}.git"
        await loop.run_in_executor(
            None, sandbox_backend.execute, f"cd {repo_dir} && git remote set-url origin {clean_url}"
        )

        if status_result.exit_code == 0 and not status_result.output.strip():
            auth_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
            pull_result = await loop.run_in_executor(
                None, sandbox_backend.execute, f"cd {repo_dir} && git pull {auth_url}"
            )

            if pull_result.exit_code != 0:
                pass
    else:
        clone_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
        result = await loop.run_in_executor(
            None, sandbox_backend.execute, f"git clone {clone_url} {repo_dir}"
        )

        if result.exit_code != 0:
            msg = f"Failed to clone repo {owner}/{repo}: {result.output}"
            raise RuntimeError(msg)

        clean_url = f"https://github.com/{owner}/{repo}.git"
        await loop.run_in_executor(
            None, sandbox_backend.execute, f"cd {repo_dir} && git remote set-url origin {clean_url}"
        )

    return repo_dir


async def _get_sandbox_id_from_metadata(thread_id: str) -> str | None:
    """Get sandbox_id from thread metadata."""
    thread = await client.threads.get(thread_id=thread_id)
    return thread.get("metadata", {}).get("sandbox_id")


async def _wait_for_sandbox_id(thread_id: str) -> str:
    """Wait for sandbox_id to be set in thread metadata.

    Polls thread metadata until sandbox_id is set to a real value
    (not the creating sentinel).

    Raises:
        TimeoutError: If sandbox creation takes too long
    """
    elapsed = 0.0
    while elapsed < SANDBOX_CREATION_TIMEOUT:
        sandbox_id = await _get_sandbox_id_from_metadata(thread_id)
        if sandbox_id is not None and sandbox_id != SANDBOX_CREATING:
            return sandbox_id
        await asyncio.sleep(SANDBOX_POLL_INTERVAL)
        elapsed += SANDBOX_POLL_INTERVAL

    msg = f"Timeout waiting for sandbox creation for thread {thread_id}"
    raise TimeoutError(msg)


async def get_agent(config: RunnableConfig):
    """Get or create an agent with a sandbox for the given thread."""
    thread_id = config["configurable"].get("thread_id", None)
    logger.info("get_agent called for thread %s", thread_id)

    repo_config = config["configurable"].get("repo", {})
    repo_owner = repo_config.get("owner")
    repo_name = repo_config.get("name")

    encrypted_token = config["configurable"].get("github_token_encrypted")
    if encrypted_token:
        github_token = decrypt_token(encrypted_token)
        logger.debug("Decrypted GitHub token")

    if thread_id is None:
        logger.info("No thread_id, returning agent without sandbox")
        return create_server_agent(
            model=None,
            assistant_id="agent",
            tools=tools,
            sandbox=None,
            sandbox_type=None,
            auto_approve=True,
        )

    sandbox_id = await _get_sandbox_id_from_metadata(thread_id)

    if sandbox_id == SANDBOX_CREATING:
        logger.info("Sandbox creation in progress, waiting...")
        sandbox_id = await _wait_for_sandbox_id(thread_id)

    if sandbox_id is None:
        logger.info("Creating new sandbox for thread %s", thread_id)
        await client.threads.update(thread_id=thread_id, metadata={"sandbox_id": SANDBOX_CREATING})

        try:
            sandbox_cm = create_sandbox_async("langsmith", cleanup=False)
            sandbox_backend = await sandbox_cm.__aenter__()
            logger.info("Sandbox created: %s", sandbox_backend.id)

            repo_dir = None
            if repo_owner and repo_name:
                logger.info("Cloning repo %s/%s into sandbox", repo_owner, repo_name)
                repo_dir = await _clone_or_pull_repo_in_sandbox(
                    sandbox_backend, repo_owner, repo_name, github_token
                )
                logger.info("Repo cloned to %s", repo_dir)

            await client.threads.update(
                thread_id=thread_id,
                metadata={
                    "sandbox_id": sandbox_backend.id,
                    "repo_dir": repo_dir,
                },
            )
        except Exception as e:
            logger.error("Failed to create sandbox: %s", e)
            await client.threads.update(thread_id=thread_id, metadata={"sandbox_id": None})
            raise
    else:
        logger.info("Connecting to existing sandbox %s", sandbox_id)
        sandbox_cm = create_sandbox_async("langsmith", sandbox_id=sandbox_id, cleanup=False)
        sandbox_backend = await sandbox_cm.__aenter__()

        thread = await client.threads.get(thread_id=thread_id)
        repo_dir = thread.get("metadata", {}).get("repo_dir")

        if repo_owner and repo_name:
            logger.info("Pulling latest changes for repo %s/%s", repo_owner, repo_name)
            repo_dir = await _clone_or_pull_repo_in_sandbox(
                sandbox_backend, repo_owner, repo_name, github_token
            )

    _SANDBOX_BACKENDS[thread_id] = sandbox_backend

    logger.info("Returning agent with sandbox for thread %s", thread_id)
    return create_server_agent(
        model=None,
        assistant_id="agent",
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type="langsmith",
        auto_approve=True,
        working_dir=repo_dir,
        middleware=[open_pr_if_needed],
    )
