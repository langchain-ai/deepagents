"""Custom FastAPI routes for LangGraph server."""

import hashlib
import hmac
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from langgraph_sdk import get_client

from deepagents_cli.encryption import encrypt_token

app = FastAPI()

LINEAR_WEBHOOK_SECRET = os.environ.get("LINEAR_WEBHOOK_SECRET", "")

LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL") or os.environ.get(
    "LANGGRAPH_URL_PROD", "http://localhost:2024"
)

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
    "LANGSMITH_API_KEY_PROD", ""
)
LANGSMITH_API_URL = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

GITHUB_OAUTH_PROVIDER_ID = os.environ.get("GITHUB_OAUTH_PROVIDER_ID", "")

LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")

X_SERVICE_AUTH_JWT_SECRET = os.environ.get("X_SERVICE_AUTH_JWT_SECRET", "")


def get_service_jwt_token_for_user(
    user_id: str, tenant_id: str, expiration_seconds: int = 300
) -> str:
    """Create a short-lived service JWT for authenticating as a specific user.

    Args:
        tenant_id: The LangSmith tenant ID to associate with the token
        user_id: The LangSmith user ID to associate with the token
        expiration_seconds: Token expiration time in seconds (default: 5 minutes)

    Returns:
        JWT token string

    Raises:
        ValueError: If X_SERVICE_AUTH_JWT_SECRET is not configured
    """
    if not X_SERVICE_AUTH_JWT_SECRET:
        raise ValueError(
            "X_SERVICE_AUTH_JWT_SECRET is not configured. Cannot generate service keys."
        )

    exp_datetime = datetime.now(tz=UTC) + timedelta(seconds=expiration_seconds)
    exp = int(exp_datetime.timestamp())

    payload = {
        "sub": "unspecified",
        "exp": exp,
        "tenant_id": tenant_id,
        "user_id": user_id,
    }

    return jwt.encode(payload, X_SERVICE_AUTH_JWT_SECRET, algorithm="HS256")


LINEAR_TEAM_TO_REPO: dict[str, dict[str, str]] = {
    "Brace's test workspace": {"owner": "langchain-ai", "name": "open-swe"},
    "Yogesh-dev": {"owner": "aran-yogesh", "name": "nimedge"},
}


async def get_ls_user_id_from_email(email: str) -> dict[str, str | None]:
    """Get the LangSmith user ID and tenant ID from a user's email.

    Args:
        email: The user's email address

    Returns:
        Dict with 'ls_user_id' and 'tenant_id' keys (values may be None if not found)
    """
    if not LANGSMITH_API_KEY:
        return {"ls_user_id": None, "tenant_id": None}

    url = f"{LANGSMITH_API_URL}/api/v1/workspaces/current/members/active"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers={"X-API-Key": LANGSMITH_API_KEY},
                params={"emails": [email]},
            )
            response.raise_for_status()
            members = response.json()

            if members and len(members) > 0:
                member = members[0]
                return {
                    "ls_user_id": member.get("ls_user_id"),
                    "tenant_id": member.get("tenant_id"),
                }
            return {"ls_user_id": None, "tenant_id": None}
        except Exception:
            return {"ls_user_id": None, "tenant_id": None}


LANGSMITH_HOST_API_URL = os.environ.get("LANGSMITH_HOST_API_URL", "https://api.host.langchain.com")


async def get_github_token_for_user(ls_user_id: str, tenant_id: str) -> dict[str, Any]:
    """Get GitHub OAuth token for a user via LangSmith agent auth.

    Args:
        ls_user_id: The LangSmith user ID
        tenant_id: The LangSmith tenant ID

    Returns:
        Dict with either 'token' key or 'auth_url' key
    """
    if not GITHUB_OAUTH_PROVIDER_ID:
        return {"error": "GITHUB_OAUTH_PROVIDER_ID not configured"}

    try:
        service_token = get_service_jwt_token_for_user(ls_user_id, tenant_id)

        headers = {
            "X-Service-Key": service_token,
            "X-Tenant-Id": tenant_id,
            "X-User-Id": ls_user_id,
        }

        payload = {
            "provider": GITHUB_OAUTH_PROVIDER_ID,
            "scopes": ["repo"],
            "user_id": ls_user_id,
            "ls_user_id": ls_user_id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LANGSMITH_HOST_API_URL}/v2/auth/authenticate",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            response_data = response.json()

            token = response_data.get("token")
            auth_url = response_data.get("url")

            if token:
                return {"token": token}
            if auth_url:
                return {"auth_url": auth_url}
            return {"error": f"Unexpected auth result: {response_data}"}

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


async def react_to_linear_comment(comment_id: str, emoji: str = "👀") -> bool:
    """Add an emoji reaction to a Linear comment.

    Args:
        comment_id: The Linear comment ID
        emoji: The emoji to react with (default: eyes 👀)

    Returns:
        True if successful, False otherwise
    """
    if not LINEAR_API_KEY:
        return False

    url = "https://api.linear.app/graphql"

    mutation = """
    mutation ReactionCreate($commentId: String!, $emoji: String!) {
        reactionCreate(input: { commentId: $commentId, emoji: $emoji }) {
            success
        }
    }
    """

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers={
                    "Authorization": LINEAR_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "query": mutation,
                    "variables": {"commentId": comment_id, "emoji": emoji},
                },
            )
            response.raise_for_status()
            result = response.json()

            if result.get("data", {}).get("reactionCreate", {}).get("success"):
                return True
            return False
        except Exception:
            return False


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

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
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


async def fetch_linear_issue_details(issue_id: str) -> dict[str, Any] | None:
    """Fetch full issue details from Linear API including description and comments.

    Args:
        issue_id: The Linear issue ID

    Returns:
        Full issue data dict, or None if fetch failed
    """
    if not LINEAR_API_KEY:
        return None

    url = "https://api.linear.app/graphql"

    query = """
    query GetIssue($issueId: String!) {
        issue(id: $issueId) {
            id
            identifier
            title
            description
            url
            comments {
                nodes {
                    id
                    body
                    createdAt
                    user {
                        id
                        name
                        email
                    }
                }
            }
        }
    }
    """

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers={
                    "Authorization": LINEAR_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "variables": {"issueId": issue_id},
                },
            )
            response.raise_for_status()
            result = response.json()

            issue = result.get("data", {}).get("issue")
            if issue:
                return issue
            return None
        except Exception:
            return None


def generate_thread_id_from_issue(issue_id: str) -> str:
    """Generate a deterministic thread ID from a Linear issue ID.

    Args:
        issue_id: The Linear issue ID

    Returns:
        A UUID-formatted thread ID derived from the issue ID
    """
    hash_bytes = hashlib.sha256(f"linear-issue:{issue_id}".encode()).hexdigest()
    return f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-{hash_bytes[16:20]}-{hash_bytes[20:32]}"


async def is_thread_active(thread_id: str) -> bool:
    """Check if a thread is currently active (has a running run).

    Args:
        thread_id: The LangGraph thread ID

    Returns:
        True if the thread status is "busy", False otherwise
    """
    langgraph_client = get_client(url=LANGGRAPH_URL)
    try:
        logger.debug("Fetching thread status for %s from %s", thread_id, LANGGRAPH_URL)
        thread = await langgraph_client.threads.get(thread_id)
        status = thread.get("status", "idle")
        logger.info(
            "Thread %s status check: status=%s, is_busy=%s",
            thread_id,
            status,
            status == "busy",
        )
        return status == "busy"
    except Exception as e:
        logger.warning(
            "Failed to get thread status for %s: %s (type: %s) - assuming not active",
            thread_id,
            e,
            type(e).__name__,
        )
        # If we can't get the thread, assume it doesn't exist yet (not active)
        return False


async def queue_message_for_thread(thread_id: str, message_content: str) -> bool:
    """Queue a message for a thread that is currently active.

    Stores the message in the langgraph store, namespaced to the thread.
    Supports multiple queued messages by storing them as a list (FIFO order).
    The before_model middleware will pick them up and inject them into state.

    Args:
        thread_id: The LangGraph thread ID
        message_content: The message content to queue

    Returns:
        True if successfully queued, False otherwise
    """
    langgraph_client = get_client(url=LANGGRAPH_URL)
    try:
        namespace = ("queue", thread_id)
        key = "pending_messages"

        new_message = {"content": message_content}

        existing_messages: list[dict[str, Any]] = []
        try:
            existing_item = await langgraph_client.store.get_item(namespace, key)
            if existing_item and existing_item.get("value"):
                existing_messages = existing_item["value"].get("messages", [])
        except Exception:
            logger.debug("No existing queued messages for thread %s", thread_id)

        existing_messages.append(new_message)
        value = {"messages": existing_messages}

        logger.info(
            "Attempting to queue message for thread %s (total queued: %d)",
            thread_id,
            len(existing_messages),
        )
        await langgraph_client.store.put_item(namespace, key, value)
        logger.info("Successfully queued message for thread %s", thread_id)
        return True
    except Exception as e:
        logger.error(
            "Failed to queue message for thread %s: %s (type: %s)",
            thread_id,
            e,
            type(e).__name__,
        )
        return False


async def process_linear_issue(issue_data: dict[str, Any], repo_config: dict[str, str]) -> None:
    """Process a Linear issue by creating a new LangGraph thread and run.

    Args:
        issue_data: The Linear issue data from webhook (basic info only).
        repo_config: The repo configuration with owner and name.
    """
    issue_id = issue_data.get("id", "")
    logger.info(
        "Processing Linear issue %s for repo %s/%s",
        issue_id,
        repo_config.get("owner"),
        repo_config.get("name"),
    )

    triggering_comment_id = issue_data.get("triggering_comment_id", "")
    if triggering_comment_id:
        await react_to_linear_comment(triggering_comment_id, "👀")

    thread_id = generate_thread_id_from_issue(issue_id)

    full_issue = await fetch_linear_issue_details(issue_id)
    if not full_issue:
        full_issue = issue_data

    user_email = None
    user_name = None
    comment_author = issue_data.get("comment_author", {})
    if comment_author:
        user_email = comment_author.get("email")
        user_name = comment_author.get("name")
    if not user_email:
        creator = full_issue.get("creator", {})
        if creator:
            user_email = creator.get("email")
            user_name = user_name or creator.get("name")
    if not user_email:
        assignee = full_issue.get("assignee", {})
        if assignee:
            user_email = assignee.get("email")
            user_name = user_name or assignee.get("name")

    user_mention = f"@{user_name}" if user_name else ""

    logger.info(
        "User email: %s, GITHUB_OAUTH_PROVIDER_ID set: %s",
        user_email,
        bool(GITHUB_OAUTH_PROVIDER_ID),
    )

    github_token = None
    if user_email and GITHUB_OAUTH_PROVIDER_ID:
        user_info = await get_ls_user_id_from_email(user_email)
        ls_user_id = user_info.get("ls_user_id")
        tenant_id = user_info.get("tenant_id")
        logger.info(
            "LangSmith user ID for %s: %s, tenant_id: %s", user_email, ls_user_id, tenant_id
        )

        if ls_user_id and tenant_id:
            auth_result = await get_github_token_for_user(ls_user_id, tenant_id)
            logger.info("Auth result keys: %s", list(auth_result.keys()))

            if "token" in auth_result:
                github_token = auth_result["token"]
                logger.info("GitHub token obtained for user %s", user_email)
            elif "auth_url" in auth_result:
                auth_url = auth_result["auth_url"]
                logger.info("GitHub auth required for user %s, sending auth URL", user_email)
                comment = f"""🔐 **GitHub Authentication Required** {user_mention}

To allow the Open SWE agent to work on this issue, please authenticate with GitHub by clicking the link below:

[Authenticate with GitHub]({auth_url})

Once authenticated, reply to this issue mentioning @openswe to retry."""

                await comment_on_linear_issue(issue_id, comment)
                return
            else:
                logger.warning("Auth result has neither token nor auth_url: %s", auth_result)
        else:
            logger.warning("User %s not found in LangSmith workspace", user_email)
            comment = f"""🔐 **GitHub Authentication Required** {user_mention}

Could not find a LangSmith account for **{user_email}**.

Please ensure this email is invited to the main LangSmith organization. If your Linear account uses a different email than your LangSmith account, you may need to update one of them to match.

Once your email is added to LangSmith, reply to this issue mentioning @openswe to retry."""

            await comment_on_linear_issue(issue_id, comment)
            return

    title = full_issue.get("title", "No title")
    description = full_issue.get("description") or "No description"

    comments = full_issue.get("comments", {}).get("nodes", [])
    comments_text = ""

    bot_message_prefixes = (
        "🔐 **GitHub Authentication Required**",
        "✅ **Pull Request Created**",
        "🤖 **Agent Response**",
        "❌ **Agent Error**",
    )

    if comments:
        last_bot_comment_idx = -1
        for i, comment in enumerate(comments):
            body = comment.get("body", "")
            if any(body.startswith(prefix) for prefix in bot_message_prefixes):
                last_bot_comment_idx = i

        relevant_comments = []
        for i, comment in enumerate(comments):
            if i <= last_bot_comment_idx:
                continue
            body = comment.get("body", "")
            if "@openswe" in body.lower():
                relevant_comments.append(comment)
                relevant_comments.extend(comments[i + 1 :])
                break

        if relevant_comments:
            comments_text = "\n\n## Comments:\n"
            for comment in relevant_comments:
                author = comment.get("user", {}).get("name", "Unknown")
                body = comment.get("body", "")
                if any(body.startswith(prefix) for prefix in bot_message_prefixes):
                    continue
                comments_text += f"\n**{author}:** {body}\n"

    prompt = f"""Please work on the following issue:

## Title: {title}

## Description:
{description}
{comments_text}

Please analyze this issue and implement the necessary changes. When you're done, commit and push your changes.
"""

    configurable: dict[str, Any] = {
        "repo": repo_config,
        "linear_issue": {
            "id": issue_id,
            "title": title,
            "url": full_issue.get("url", "") or issue_data.get("url", ""),
            "identifier": full_issue.get("identifier", "") or issue_data.get("identifier", ""),
        },
    }
    if github_token:
        configurable["github_token_encrypted"] = encrypt_token(github_token)

        logger.info("Checking if thread %s is active before creating run", thread_id)
        thread_active = await is_thread_active(thread_id)
        logger.info("Thread %s active status: %s", thread_id, thread_active)

        if thread_active:
            logger.info(
                "Thread %s is active (busy), will queue message instead of creating run",
                thread_id,
            )

            queued = await queue_message_for_thread(
                thread_id=thread_id,
                message_content=prompt,
            )

            if queued:
                logger.info(
                    "Message queued for thread %s, will be processed by middleware", thread_id
                )
            else:
                logger.error("Failed to queue message for thread %s", thread_id)
        else:
            logger.info("Creating LangGraph run for thread %s", thread_id)
            langgraph_client = get_client(url=LANGGRAPH_URL)
            await langgraph_client.runs.create(
                thread_id,
                "agent",
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": configurable},
                if_not_exists="create",
            )
            logger.info("LangGraph run created successfully for thread %s", thread_id)
    else:
        logger.warning("No GitHub token available, cannot create run for issue %s", issue_id)
        if not user_email:
            comment = f"""🔐 **GitHub Authentication Required** {user_mention}

Could not determine the user email from this issue. Please ensure your Linear account has an email address configured.

Reply to this issue mentioning @openswe to retry."""
        elif not GITHUB_OAUTH_PROVIDER_ID:
            comment = f"""❌ **Configuration Error** {user_mention}

The Open SWE agent is not properly configured (missing GitHub OAuth provider).

Please contact your administrator."""
        else:
            comment = f"""🔐 **GitHub Authentication Required** {user_mention}

Unable to authenticate with GitHub. Please ensure you have connected your GitHub account in LangSmith.

Reply to this issue mentioning @openswe to retry."""

        await comment_on_linear_issue(issue_id, comment)


def verify_linear_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify the Linear webhook signature.

    Args:
        body: Raw request body bytes
        signature: The Linear-Signature header value
        secret: The webhook signing secret

    Returns:
        True if signature is valid, False otherwise
    """
    if not secret:
        return True

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected, signature)


@app.post("/webhooks/linear")
async def linear_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Linear webhooks.

    Triggers a new LangGraph run when an issue gets the 'open-swe' label added.
    """
    logger.info("Received Linear webhook")
    body = await request.body()

    signature = request.headers.get("Linear-Signature", "")
    if LINEAR_WEBHOOK_SECRET and not verify_linear_signature(
        body, signature, LINEAR_WEBHOOK_SECRET
    ):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        import json

        payload = json.loads(body)
    except Exception:
        logger.error("Failed to parse webhook JSON")
        return {"status": "error", "message": "Invalid JSON"}

    if payload.get("type") != "Comment":
        logger.debug("Ignoring webhook: not a Comment event")
        return {"status": "ignored", "reason": "Not a Comment event"}

    action = payload.get("action")
    if action != "create":
        logger.debug("Ignoring webhook: action is %s, not create", action)
        return {
            "status": "ignored",
            "reason": f"Comment action is '{action}', only processing 'create'",
        }

    data = payload.get("data", {})

    if data.get("botActor"):
        logger.debug("Ignoring webhook: comment is from a bot")
        return {"status": "ignored", "reason": "Comment is from a bot"}

    comment_body = data.get("body", "")
    bot_message_prefixes = [
        "🔐 **GitHub Authentication Required**",
        "✅ **Pull Request Created**",
        "🤖 **Agent Response**",
        "❌ **Agent Error**",
    ]
    for prefix in bot_message_prefixes:
        if comment_body.startswith(prefix):
            logger.debug("Ignoring webhook: comment is our own bot message")
            return {"status": "ignored", "reason": "Comment is our own bot message"}
    if "@openswe" not in comment_body.lower():
        logger.debug("Ignoring webhook: comment doesn't mention @openswe")
        return {"status": "ignored", "reason": "Comment doesn't mention @openswe"}

    issue = data.get("issue", {})
    if not issue:
        logger.debug("Ignoring webhook: no issue data in comment")
        return {"status": "ignored", "reason": "No issue data in comment"}

    team = issue.get("team", {})
    team_id = team.get("id", "") if team else ""
    team_name = team.get("name", "") if team else ""

    repo_config = None
    if team_id and team_id in LINEAR_TEAM_TO_REPO:
        repo_config = LINEAR_TEAM_TO_REPO[team_id]
    elif team_name and team_name in LINEAR_TEAM_TO_REPO:
        repo_config = LINEAR_TEAM_TO_REPO[team_name]

    if not repo_config:
        for label in issue.get("labels", []):
            label_name = label.get("name", "")
            if label_name.startswith("repo:"):
                repo_ref = label_name[5:]  # Remove "repo:" prefix
                if "/" in repo_ref:
                    owner, name = repo_ref.split("/", 1)
                    repo_config = {"owner": owner, "name": name}
                    break

    if not repo_config:
        repo_config = {"owner": "langchain-ai", "name": "langchainplus"}

    repo_owner = repo_config["owner"]
    repo_name = repo_config["name"]

    issue["triggering_comment"] = comment_body
    issue["triggering_comment_id"] = data.get("id", "")
    comment_user = data.get("user", {})
    if comment_user:
        issue["comment_author"] = comment_user

    logger.info(
        "Accepted webhook for issue '%s' (%s), scheduling background task",
        issue.get("title"),
        issue.get("id"),
    )
    background_tasks.add_task(process_linear_issue, issue, repo_config)

    return {
        "status": "accepted",
        "message": f"Processing issue '{issue.get('title')}' for repo {repo_owner}/{repo_name}",
    }


@app.get("/webhooks/linear")
async def linear_webhook_verify():
    """Verify endpoint for Linear webhook setup."""
    return {"status": "ok", "message": "Linear webhook endpoint is active"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
