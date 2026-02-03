"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: E402

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
# ruff: noqa: E402
import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

from langchain.agents.middleware import AgentState, after_agent, after_model, before_model
from langgraph.config import get_config, get_store
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

import httpx

LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")


async def create_github_pr(
    repo_owner: str,
    repo_name: str,
    github_token: str,
    title: str,
    head_branch: str,
    base_branch: str,
    body: str,
) -> tuple[str | None, int | None]:
    """Create a GitHub pull request via the API.

    Args:
        repo_owner: Repository owner (e.g., "langchain-ai")
        repo_name: Repository name (e.g., "deepagents")
        github_token: GitHub access token
        title: PR title
        head_branch: Source branch name
        base_branch: Target branch name
        body: PR description

    Returns:
        Tuple of (pr_url, pr_number) if successful, (None, None) otherwise
    """
    pr_payload = {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body,
    }

    logger.info(
        "Creating PR: head=%s, base=%s, repo=%s/%s",
        head_branch,
        base_branch,
        repo_owner,
        repo_name,
    )

    try:
        async with httpx.AsyncClient() as http_client:
            pr_response = await http_client.post(
                f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json=pr_payload,
            )

            pr_data = pr_response.json()

            if pr_response.status_code == 201:
                pr_url = pr_data.get("html_url")
                pr_number = pr_data.get("number")
                logger.info("PR created successfully: %s", pr_url)
                return pr_url, pr_number

            if pr_response.status_code == 422:
                logger.error("GitHub API validation error (422): %s", pr_data.get("message"))
            else:
                logger.error(
                    "GitHub API error (%s): %s",
                    pr_response.status_code,
                    pr_data.get("message"),
                )

            if "errors" in pr_data:
                logger.error("GitHub API errors detail: %s", pr_data.get("errors"))

            return None, None

    except httpx.HTTPError as e:
        logger.error("Failed to create PR via GitHub API: %s", e)
        return None, None


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


class LinearNotifyState(AgentState):
    """Extended agent state for tracking Linear notifications."""

    linear_messages_sent_count: int


@before_model(state_schema=LinearNotifyState)
async def check_message_queue_before_model(
    state: LinearNotifyState,  # noqa: ARG001
    runtime: Runtime,  # noqa: ARG001
) -> dict[str, Any] | None:
    """Middleware that checks for queued messages before each model call.

    If messages are found in the queue for this thread, it extracts all messages,
    adds them to the conversation state as new human messages, and clears the queue.
    Messages are processed in FIFO order (oldest first).

    This enables handling of follow-up comments that arrive while the agent is busy.
    The agent will see the new messages and can incorporate them into its response.
    """
    try:
        config = get_config()
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if not thread_id:
            return None

        try:
            store = get_store()
        except Exception as e:
            logger.debug("Could not get store from context: %s", e)
            return None

        if store is None:
            return None

        namespace = ("queue", thread_id)

        try:
            queued_item = await store.aget(namespace, "pending_messages")
        except Exception as e:
            logger.warning("Failed to get queued item: %s", e)
            return None

        if queued_item is None:
            return None

        queued_value = queued_item.value
        queued_messages = queued_value.get("messages", [])

        # Delete early to prevent duplicate processing if middleware runs again
        await store.adelete(namespace, "pending_messages")

        if not queued_messages:
            return None

        logger.info(
            "Found %d queued message(s) for thread %s, injecting into state",
            len(queued_messages),
            thread_id,
        )

        content_blocks = [
            {"type": "text", "text": msg.get("content", "")}
            for msg in queued_messages
            if msg.get("content")
        ]

        if not content_blocks:
            return None

        new_message = {
            "role": "user",
            "content": content_blocks,
        }

        logger.info(
            "Injected %d queued message(s) into state for thread %s",
            len(content_blocks),
            thread_id,
        )

        return {"messages": [new_message]}

    except Exception as e:
        logger.error("Error in check_message_queue_before_model: %s", e)
        return None


@after_model(state_schema=LinearNotifyState)
async def post_to_linear_after_model(
    state: LinearNotifyState, runtime: Runtime
) -> dict[str, Any] | None:
    """Middleware that posts AI responses to Linear after each model call.

    Only posts if:
    - This is a Linear-triggered conversation (has linear_issue in config)
    - There's exactly 1 human message (initial request)
    - The previous message was from human (not a tool result)
    - The AI response has text content (not just tool calls)
    - The message hasn't already been sent (tracked via linear_messages_sent_count)
    """
    try:
        config = get_config()
        configurable = config.get("configurable", {})

        linear_issue = configurable.get("linear_issue", {})
        linear_issue_id = linear_issue.get("id")

        if not linear_issue_id:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        sent_count = state.get("linear_messages_sent_count", 0)

        human_message_count = 0
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role in ("human", "user"):
                human_message_count += 1

        if human_message_count != 1:
            return None

        last_message = messages[-1]
        if isinstance(last_message, dict):
            role = last_message.get("role", "")
            content = last_message.get("content", "")
        else:
            role = getattr(last_message, "type", "") or getattr(last_message, "role", "")
            content = getattr(last_message, "content", "")

        if role not in ("ai", "assistant"):
            return None

        ai_message_count = 0
        for msg in messages:
            if isinstance(msg, dict):
                r = msg.get("role", "")
            else:
                r = getattr(msg, "type", "") or getattr(msg, "role", "")
            if r in ("ai", "assistant"):
                ai_message_count += 1

        if ai_message_count <= sent_count:
            return None

        if len(messages) >= 2:
            prev_message = messages[-2]
            if isinstance(prev_message, dict):
                prev_role = prev_message.get("role", "")
            else:
                prev_role = getattr(prev_message, "type", "") or getattr(prev_message, "role", "")

            if prev_role not in ("human", "user"):
                return None

        if not content or not isinstance(content, str):
            return None

        comment = f"""🤖 **Agent Response**

{content}"""
        logger.info("Posting AI response to Linear issue %s", linear_issue_id)
        success = await comment_on_linear_issue(linear_issue_id, comment)

        if success:
            logger.info("Successfully posted to Linear")
            return {"linear_messages_sent_count": ai_message_count}
        logger.warning("Failed to post to Linear")
        return None

    except Exception as e:
        logger.error("Error in post_to_linear_after_model: %s", e)
        return None


@after_agent
async def open_pr_if_needed(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Middleware that commits/pushes changes and comments on Linear after agent runs."""
    logger.info("After-agent middleware started")
    pr_url = None
    pr_number = None
    pr_title = "feat: Open SWE PR"

    try:
        config = get_config()
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        logger.debug("Middleware running for thread %s", thread_id)

        last_message_content = ""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                last_message_content = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                last_message_content = last_message.content

        linear_issue = configurable.get("linear_issue", {})
        linear_issue_id = linear_issue.get("id")

        if not thread_id:
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
                f"cd {repo_dir} && git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@'",
            )
            base_branch = (
                default_branch_result.output.strip() if default_branch_result.exit_code == 0 else ""
            )

            # Fallback to main if we couldn't determine the default branch
            if not base_branch:
                base_branch = "main"

            logger.info("Using base branch: %s", base_branch)

            pr_title = "feat: Open SWE PR"
            pr_body = "Automated PR created by Open SWE agent."

            pr_url, pr_number = await create_github_pr(
                repo_owner=repo_owner,
                repo_name=repo_name,
                github_token=github_token,
                title=pr_title,
                head_branch=target_branch,
                base_branch=base_branch,
                body=pr_body,
            )

            linear_issue = configurable.get("linear_issue", {})
            linear_issue_id = linear_issue.get("id")

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
    logger.info("_clone_or_pull_repo_in_sandbox called for %s/%s", owner, repo)
    loop = asyncio.get_event_loop()

    token = github_token
    if not token:
        msg = "No GitHub token provided"
        logger.error(msg)
        raise ValueError(msg)

    repo_dir = f"/workspace/{repo}"

    logger.debug("Checking if repo already exists at %s", repo_dir)
    try:
        check_result = await loop.run_in_executor(
            None, sandbox_backend.execute, f"test -d {repo_dir}/.git && echo exists"
        )
        logger.debug(
            "Check result: exit_code=%s, output=%s",
            check_result.exit_code,
            check_result.output[:200] if check_result.output else "",
        )
    except Exception as e:
        logger.error("Failed to execute check command in sandbox: %s", e, exc_info=True)
        raise

    if check_result.exit_code == 0 and "exists" in check_result.output:
        logger.info("Repo already exists at %s, pulling latest changes", repo_dir)
        try:
            status_result = await loop.run_in_executor(
                None, sandbox_backend.execute, f"cd {repo_dir} && git status --porcelain"
            )
            logger.debug("Git status result: exit_code=%s", status_result.exit_code)
        except Exception as e:
            logger.error("Failed to get git status: %s", e, exc_info=True)
            raise

        # CRITICAL: Ensure remote URL doesn't contain token (clean up from previous runs)
        clean_url = f"https://github.com/{owner}/{repo}.git"
        try:
            await loop.run_in_executor(
                None,
                sandbox_backend.execute,
                f"cd {repo_dir} && git remote set-url origin {clean_url}",
            )
        except Exception as e:
            logger.error("Failed to set remote URL: %s", e, exc_info=True)
            raise

        if status_result.exit_code == 0 and not status_result.output.strip():
            auth_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
            try:
                pull_result = await loop.run_in_executor(
                    None, sandbox_backend.execute, f"cd {repo_dir} && git pull {auth_url}"
                )
                logger.debug("Git pull result: exit_code=%s", pull_result.exit_code)
                if pull_result.exit_code != 0:
                    logger.warning(
                        "Git pull failed with exit code %s: %s",
                        pull_result.exit_code,
                        pull_result.output[:200] if pull_result.output else "",
                    )
            except Exception as e:
                logger.error("Failed to execute git pull: %s", e, exc_info=True)
                raise
    else:
        logger.info("Cloning repo %s/%s to %s", owner, repo, repo_dir)
        clone_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
        try:
            result = await loop.run_in_executor(
                None, sandbox_backend.execute, f"git clone {clone_url} {repo_dir}"
            )
            logger.debug("Git clone result: exit_code=%s", result.exit_code)
        except Exception as e:
            logger.error("Failed to execute git clone: %s", e, exc_info=True)
            raise

        if result.exit_code != 0:
            msg = f"Failed to clone repo {owner}/{repo}: {result.output}"
            logger.error(msg)
            raise RuntimeError(msg)

        clean_url = f"https://github.com/{owner}/{repo}.git"
        try:
            await loop.run_in_executor(
                None,
                sandbox_backend.execute,
                f"cd {repo_dir} && git remote set-url origin {clean_url}",
            )
        except Exception as e:
            logger.error("Failed to set remote URL after clone: %s", e, exc_info=True)
            raise

    logger.info("Repo setup complete at %s", repo_dir)
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


def graph_loaded_for_execution(config: RunnableConfig) -> bool:
    return (
        config["configurable"].get("__is_for_execution__", False)
        if "configurable" in config
        else False
    )


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

    if thread_id is None or not graph_loaded_for_execution(config):
        logger.info("No thread_id or not for execution, returning agent without sandbox")
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

            # Update metadata immediately after sandbox creation so other callers
            # can connect to the sandbox while we clone the repo
            await client.threads.update(
                thread_id=thread_id,
                metadata={"sandbox_id": sandbox_backend.id},
            )

            repo_dir = None
            if repo_owner and repo_name:
                logger.info("Cloning repo %s/%s into sandbox", repo_owner, repo_name)
                repo_dir = await _clone_or_pull_repo_in_sandbox(
                    sandbox_backend, repo_owner, repo_name, github_token
                )
                logger.info("Repo cloned to %s", repo_dir)

                await client.threads.update(
                    thread_id=thread_id,
                    metadata={"repo_dir": repo_dir},
                )
        except Exception as e:
            logger.error("Failed to create sandbox or clone repo: %s", e, exc_info=True)
            try:
                await client.threads.update(thread_id=thread_id, metadata={"sandbox_id": None})
                logger.info("Reset sandbox_id to None for thread %s", thread_id)
            except Exception as reset_error:
                logger.error("Failed to reset sandbox_id metadata: %s", reset_error)
            raise
    else:
        logger.info("Connecting to existing sandbox %s", sandbox_id)
        try:
            sandbox_cm = create_sandbox_async("langsmith", sandbox_id=sandbox_id, cleanup=False)
            sandbox_backend = await sandbox_cm.__aenter__()
            logger.info("Connected to existing sandbox %s", sandbox_id)
        except Exception as e:
            logger.error(
                "Failed to connect to existing sandbox %s: %s", sandbox_id, e, exc_info=True
            )
            raise

        thread = await client.threads.get(thread_id=thread_id)
        repo_dir = thread.get("metadata", {}).get("repo_dir")

        if repo_owner and repo_name:
            logger.info("Pulling latest changes for repo %s/%s", repo_owner, repo_name)
            try:
                repo_dir = await _clone_or_pull_repo_in_sandbox(
                    sandbox_backend, repo_owner, repo_name, github_token
                )
            except Exception as e:
                logger.error("Failed to pull repo in existing sandbox: %s", e, exc_info=True)
                raise

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
        middleware=[
            check_message_queue_before_model,
            post_to_linear_after_model,
            open_pr_if_needed,
        ],
    )
