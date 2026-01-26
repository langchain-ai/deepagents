"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: E402

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
# ruff: noqa: E402
import os
import warnings
from typing import Any

from langgraph.graph.state import RunnableConfig
from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langgraph.config import get_config

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

# Sentinel value to indicate sandbox creation is in progress
SANDBOX_CREATING = "__creating__"
# How long to wait for sandbox creation (seconds)
SANDBOX_CREATION_TIMEOUT = 180
# How often to poll for sandbox_id (seconds)
SANDBOX_POLL_INTERVAL = 1.0

# Store sandbox backends by thread_id for middleware access
_SANDBOX_BACKENDS: dict[str, Any] = {}
_REPO_DIRS: dict[str, str] = {}
_GITHUB_TOKENS: dict[str, str] = {}

# Linear API key for commenting on issues
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
        print("\n\n[Linear API] LINEAR_API_KEY not set, cannot comment\n\n")
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
                print(f"\n\n[Linear API] Successfully commented on issue {issue_id}\n\n")
                return True
            else:
                print(f"\n\n[Linear API] Failed to comment: {result}\n\n")
                return False
        except Exception as e:
            print(f"\n\n[Linear API] Error commenting: {e}\n\n")
            return False


@after_agent
async def open_pr_if_needed(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Middleware that commits/pushes changes and comments on Linear after agent runs."""
    print("\n\n[open_pr_if_needed] Middleware started\n\n")

    pr_url = None
    pr_number = None
    pr_title = "feat: Open SWE PR"

    try:
        # Get config using langgraph's get_config
        config = get_config()
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        print(f"\n\n[open_pr_if_needed] thread_id: {thread_id}\n\n")

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
        print(f"\n\n[open_pr_if_needed] Last message content length: {len(last_message_content)}\n\n")

        # Get Linear issue info for commenting
        linear_issue = configurable.get("linear_issue", {})
        linear_issue_id = linear_issue.get("id")

        if not thread_id:
            print("\n\n[open_pr_if_needed] No thread_id, skipping PR but will comment\n\n")
            # Still comment on Linear if we have the issue ID
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        # Get repo info from config
        repo_config = configurable.get("repo", {})
        repo_owner = repo_config.get("owner")
        repo_name = repo_config.get("name")

        print(f"\n\n[open_pr_if_needed] repo_owner: {repo_owner}, repo_name: {repo_name}\n\n")

        # Get the sandbox backend for this thread
        sandbox_backend = _SANDBOX_BACKENDS.get(thread_id)

        print(f"\n\n[open_pr_if_needed] sandbox_backend found: {sandbox_backend is not None}\n\n")

        # Get repo_dir from stored value or construct from config
        repo_dir = _REPO_DIRS.get(thread_id)
        if not repo_dir and repo_name:
            repo_dir = f"/workspace/{repo_name}"

        print(f"\n\n[open_pr_if_needed] repo_dir: {repo_dir}\n\n")

        if not sandbox_backend or not repo_dir:
            print("\n\n[open_pr_if_needed] Missing sandbox_backend or repo_dir, skipping PR but will comment\n\n")
            # Still comment on Linear if we have the issue ID
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        # Check for uncommitted changes (run blocking call in thread)
        print(f"\n\n[open_pr_if_needed] Running git status --porcelain in {repo_dir}\n\n")
        result = await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git status --porcelain")
        print(f"\n\n[open_pr_if_needed] git status exit_code: {result.exit_code}, output: '{result.output}'\n\n")

        has_uncommitted_changes = result.exit_code == 0 and result.output.strip()

        # Also check for unpushed commits (agent may have committed but failed to push)
        print(f"\n\n[open_pr_if_needed] Checking for unpushed commits\n\n")
        # Fetch to ensure we have latest remote refs
        await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git fetch origin 2>/dev/null || true")
        # Check if there are commits not on origin (comparing to default branch)
        unpushed_result = await asyncio.to_thread(
            sandbox_backend.execute,
            f"cd {repo_dir} && git log --oneline @{{upstream}}..HEAD 2>/dev/null || git log --oneline origin/HEAD..HEAD 2>/dev/null || echo ''"
        )
        has_unpushed_commits = unpushed_result.exit_code == 0 and unpushed_result.output.strip()
        print(f"\n\n[open_pr_if_needed] Unpushed commits check: '{unpushed_result.output.strip()}'\n\n")

        has_changes = has_uncommitted_changes or has_unpushed_commits

        if not has_changes:
            print("\n\n[open_pr_if_needed] No changes detected, skipping PR but will comment\n\n")
            # Still comment on Linear with the agent's response
            if linear_issue_id and last_message_content:
                comment = f"""🤖 **Agent Response**

{last_message_content}"""
                await comment_on_linear_issue(linear_issue_id, comment)
            return None

        # Get current branch
        branch_result = await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git rev-parse --abbrev-ref HEAD")
        current_branch = branch_result.output.strip() if branch_result.exit_code == 0 else ""

        print(f"\n\n[open_pr_if_needed] Current branch: '{current_branch}'\n\n")

        target_branch = f"open-swe/{thread_id}"

        # Checkout the branch if not already on it
        if current_branch != target_branch:
            print(f"\n\n[open_pr_if_needed] Checking out new branch: {target_branch}\n\n")
            checkout_result = await asyncio.to_thread(
                sandbox_backend.execute,
                f"cd {repo_dir} && git checkout -b {target_branch}"
            )
            if checkout_result.exit_code != 0:
                print(f"\n\n[open_pr_if_needed] Branch create failed, trying to switch: {checkout_result.output}\n\n")
                # Branch might already exist, try switching to it
                await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git checkout {target_branch}")

        # Configure git user identity
        print("\n\n[open_pr_if_needed] Configuring git user identity\n\n")
        await asyncio.to_thread(
            sandbox_backend.execute,
            f"cd {repo_dir} && git config user.name 'Open SWE[bot]'"
        )
        await asyncio.to_thread(
            sandbox_backend.execute,
            f"cd {repo_dir} && git config user.email 'Open SWE@users.noreply.github.com'"
        )

        # Stage all changes and commit
        print("\n\n[open_pr_if_needed] Staging all changes with git add -A\n\n")
        add_result = await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git add -A")
        print(f"\n\n[open_pr_if_needed] git add result: exit_code={add_result.exit_code}, output='{add_result.output}'\n\n")

        print("\n\n[open_pr_if_needed] Committing changes\n\n")
        commit_result = await asyncio.to_thread(
            sandbox_backend.execute,
            f'cd {repo_dir} && git commit -m "feat: Open SWE PR"'
        )
        print(f"\n\n[open_pr_if_needed] git commit result: exit_code={commit_result.exit_code}, output='{commit_result.output}'\n\n")

        # Push the branch (using the encrypted token from configurable, stored token, or env var)
        # Try encrypted token first, then stored token, then env var
        encrypted_token = configurable.get("github_token_encrypted")
        if encrypted_token:
            github_token = decrypt_token(encrypted_token)
        else:
            github_token = _GITHUB_TOKENS.get(thread_id) or os.environ.get("GITHUB_ACCESS_TOKEN", "")
        if github_token:
            print("\n\n[open_pr_if_needed] Pushing branch to remote\n\n")
            # Get the remote URL and inject token for push
            remote_result = await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git remote get-url origin")
            if remote_result.exit_code == 0:
                remote_url = remote_result.output.strip()
                print(f"\n\n[open_pr_if_needed] Remote URL: {remote_url}\n\n")
                # Convert to authenticated URL if needed
                if "github.com" in remote_url and "@" not in remote_url:
                    # Convert https://github.com/owner/repo.git to https://git:token@github.com/owner/repo.git
                    auth_url = remote_url.replace("https://", f"https://git:{github_token}@")
                    push_result = await asyncio.to_thread(
                        sandbox_backend.execute,
                        f"cd {repo_dir} && git push {auth_url} {target_branch}"
                    )
                    print(f"\n\n[open_pr_if_needed] git push result: exit_code={push_result.exit_code}, output='{push_result.output}'\n\n")
                else:
                    push_result = await asyncio.to_thread(sandbox_backend.execute, f"cd {repo_dir} && git push origin {target_branch}")
                    print(f"\n\n[open_pr_if_needed] git push result: exit_code={push_result.exit_code}, output='{push_result.output}'\n\n")
            # Create pull request via GitHub API
            print("\n\n[open_pr_if_needed] Creating pull request via GitHub API\n\n")

            # Get the default branch to use as base
            default_branch_result = await asyncio.to_thread(
                sandbox_backend.execute,
                f"cd {repo_dir} && git remote show origin | grep 'HEAD branch' | cut -d' ' -f5"
            )
            base_branch = default_branch_result.output.strip() if default_branch_result.exit_code == 0 else "main"
            print(f"\n\n[open_pr_if_needed] Base branch: {base_branch}\n\n")

            # Create PR using GitHub API via curl
            pr_title = "feat: Open SWE PR"
            pr_body = "Automated PR created by Open SWE agent."

            create_pr_cmd = f'''curl -s -X POST \\
                -H "Authorization: Bearer {github_token}" \\
                -H "Accept: application/vnd.github+json" \\
                -H "X-GitHub-Api-Version: 2022-11-28" \\
                https://api.github.com/repos/{repo_owner}/{repo_name}/pulls \\
                -d '{{"title":"{pr_title}","head":"{target_branch}","base":"{base_branch}","body":"{pr_body}"}}'
            '''

            pr_result = await asyncio.to_thread(sandbox_backend.execute, create_pr_cmd)
            print(f"\n\n[open_pr_if_needed] Create PR result: exit_code={pr_result.exit_code}, output='{pr_result.output}'\n\n")

            # Parse PR response and comment on Linear issue if configured
            linear_issue = configurable.get("linear_issue", {})
            linear_issue_id = linear_issue.get("id")

            if pr_result.exit_code == 0:
                try:
                    import json
                    pr_response = json.loads(pr_result.output)
                    pr_url = pr_response.get("html_url")
                    pr_number = pr_response.get("number")

                    if pr_url:
                        print(f"\n\n[open_pr_if_needed] PR created: {pr_url}\n\n")
                except json.JSONDecodeError as e:
                    print(f"\n\n[open_pr_if_needed] Failed to parse PR response as JSON: {e}\n\n")

        else:
            print("\n\n[open_pr_if_needed] No GITHUB_ACCESS_TOKEN, skipping push\n\n")

        # Always comment on Linear with the agent's response (and PR link if created)
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

        print("\n\n[open_pr_if_needed] Middleware completed successfully\n\n")
        return None

    except Exception as e:
        print(f"\n\n[open_pr_if_needed] ERROR: {type(e).__name__}: {e}\n\n")
        import traceback
        print(f"\n\n[open_pr_if_needed] Traceback:\n{traceback.format_exc()}\n\n")
        # Try to comment on Linear even if there was an error
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
        except Exception:
            pass
        return None

async def _clone_or_pull_repo_in_sandbox(sandbox_backend: Any, owner: str, repo: str, github_token: str | None = None) -> str:
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

    # Use provided token or fall back to env var
    token = github_token or os.environ.get("GITHUB_ACCESS_TOKEN")
    if not token:
        msg = "No GitHub token provided and GITHUB_ACCESS_TOKEN environment variable not set"
        raise ValueError(msg)

    repo_dir = f"/workspace/{repo}"

    # Check if repo directory already exists
    check_result = await loop.run_in_executor(
        None, sandbox_backend.execute, f"test -d {repo_dir}/.git && echo exists"
    )

    if check_result.exit_code == 0 and "exists" in check_result.output:
        # Repo exists - check if branch is clean before pulling
        status_result = await loop.run_in_executor(
            None, sandbox_backend.execute, f"cd {repo_dir} && git status --porcelain"
        )

        # CRITICAL: Ensure remote URL doesn't contain token (clean up from previous runs)
        clean_url = f"https://github.com/{owner}/{repo}.git"
        await loop.run_in_executor(
            None, sandbox_backend.execute, f"cd {repo_dir} && git remote set-url origin {clean_url}"
        )

        if status_result.exit_code == 0 and not status_result.output.strip():
            # Branch is clean, safe to pull
            # Use authenticated URL only for this pull command (not stored in remote)
            auth_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
            pull_result = await loop.run_in_executor(
                None, sandbox_backend.execute, f"cd {repo_dir} && git pull {auth_url}"
            )

            if pull_result.exit_code != 0:
                # Pull failed, but repo exists - continue anyway
                pass
        # If branch has changes, don't pull - just use existing state
    else:
        # Repo doesn't exist - clone it
        clone_url = f"https://git:{token}@github.com/{owner}/{repo}.git"
        result = await loop.run_in_executor(
            None, sandbox_backend.execute, f"git clone {clone_url} {repo_dir}"
        )

        if result.exit_code != 0:
            msg = f"Failed to clone repo {owner}/{repo}: {result.output}"
            raise RuntimeError(msg)

        # CRITICAL: Remove the token from the remote URL to prevent agent from pushing directly
        # Reset remote to use HTTPS without token (agent cannot push, only middleware can)
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

    # Extract repo config
    repo_config = config["configurable"].get("repo", {})
    repo_owner = repo_config.get("owner")
    repo_name = repo_config.get("name")

    # Extract GitHub token from configurable (encrypted from agent auth) or fall back to env var
    encrypted_token = config["configurable"].get("github_token_encrypted")
    if encrypted_token:
        github_token = decrypt_token(encrypted_token)
    else:
        github_token = os.environ.get("GITHUB_ACCESS_TOKEN")

    if thread_id is None:
        # No thread_id means no sandbox
        return create_server_agent(
            model=None,
            assistant_id="agent",
            tools=tools,
            sandbox=None,
            sandbox_type=None,
            auto_approve=True,
        )

    # Check if sandbox already exists or is being created
    sandbox_id = await _get_sandbox_id_from_metadata(thread_id)

    if sandbox_id == SANDBOX_CREATING:
        # Another call is creating the sandbox, wait for it
        sandbox_id = await _wait_for_sandbox_id(thread_id)

    if sandbox_id is None:
        # No sandbox yet - we need to create one
        # First, set sentinel to prevent other callers from also creating
        await client.threads.update(thread_id=thread_id, metadata={"sandbox_id": SANDBOX_CREATING})

        try:
            # Create the sandbox
            sandbox_cm = create_sandbox_async("langsmith", cleanup=False)
            sandbox_backend = await sandbox_cm.__aenter__()

            # Clone or pull the repo if configured
            repo_dir = None
            if repo_owner and repo_name:
                repo_dir = await _clone_or_pull_repo_in_sandbox(sandbox_backend, repo_owner, repo_name, github_token)

            # Update metadata with real sandbox_id and repo_dir
            await client.threads.update(
                thread_id=thread_id,
                metadata={
                    "sandbox_id": sandbox_backend.id,
                    "repo_dir": repo_dir,
                },
            )
        except Exception:
            # Clear sentinel on failure so others can retry
            await client.threads.update(thread_id=thread_id, metadata={"sandbox_id": None})
            raise
    else:
        # Connect to existing sandbox (repo already cloned)
        sandbox_cm = create_sandbox_async("langsmith", sandbox_id=sandbox_id, cleanup=False)
        sandbox_backend = await sandbox_cm.__aenter__()

        # Get repo_dir from metadata
        thread = await client.threads.get(thread_id=thread_id)
        repo_dir = thread.get("metadata", {}).get("repo_dir")

        # Pull latest changes if repo exists and branch is clean
        if repo_owner and repo_name:
            repo_dir = await _clone_or_pull_repo_in_sandbox(sandbox_backend, repo_owner, repo_name, github_token)

    # Store for middleware access
    _SANDBOX_BACKENDS[thread_id] = sandbox_backend
    if repo_dir:
        _REPO_DIRS[thread_id] = repo_dir
    if github_token:
        _GITHUB_TOKENS[thread_id] = github_token

    return create_server_agent(
        model=None,
        assistant_id="agent",
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type="langsmith",
        auto_approve=True,
        working_dir=repo_dir,
        middleware=[open_pr_if_needed]
    )
