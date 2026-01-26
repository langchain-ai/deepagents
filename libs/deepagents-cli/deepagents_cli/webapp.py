"""Custom FastAPI routes for LangGraph server."""

import hashlib
import hmac
import os
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from langgraph_sdk import get_client

from deepagents_cli.encryption import encrypt_token

app = FastAPI()

LINEAR_WEBHOOK_SECRET = os.environ.get("LINEAR_WEBHOOK_SECRET", "")

LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL", "http://localhost:2024")

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_API_KEY_PROD = os.environ.get("LANGSMITH_API_KEY_PROD", "")
LANGSMITH_API_URL = os.environ.get("LANGSMITH_API_URL", "https://api.smith.langchain.com")

GITHUB_OAUTH_PROVIDER_ID = os.environ.get("GITHUB_OAUTH_PROVIDER_ID", "")

LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")

LINEAR_TEAM_TO_REPO: dict[str, dict[str, str]] = {
    "Brace's test workspace": {"owner": "langchain-ai", "name": "open-swe"},
}


async def get_ls_user_id_from_email(email: str) -> str | None:
    """Get the LangSmith user ID (ls_user_id) from a user's email.

    Args:
        email: The user's email address

    Returns:
        The ls_user_id if found, None otherwise
    """
    # TODO: Update this once sandbox can run on prod
    # if not LANGSMITH_API_KEY:
    if not LANGSMITH_API_KEY_PROD:
        return None

    url = f"{LANGSMITH_API_URL}/api/v1/workspaces/current/members/active"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                headers={"X-API-Key": LANGSMITH_API_KEY_PROD},
                params={"emails": [email]},
            )
            response.raise_for_status()
            members = response.json()

            if members and len(members) > 0:
                ls_user_id = members[0].get("ls_user_id")
                return ls_user_id
            return None
        except Exception:
            return None


async def get_github_token_for_user(ls_user_id: str) -> dict[str, Any]:
    """Get GitHub OAuth token for a user via LangSmith agent auth.

    Args:
        ls_user_id: The LangSmith user ID

    Returns:
        Dict with either 'token' key or 'auth_url' key
    """
    if not GITHUB_OAUTH_PROVIDER_ID:
        return {"error": "GITHUB_OAUTH_PROVIDER_ID not configured"}

    try:
        from langchain_auth import Client

        # TODO: Update this once sandbox can run on prod
        client = Client(api_key=LANGSMITH_API_KEY_PROD)

        auth_result = await client.authenticate(
            provider=GITHUB_OAUTH_PROVIDER_ID,
            scopes=["repo"],
            user_id=ls_user_id,
        )

        if hasattr(auth_result, "token") and auth_result.token:
            return {"token": auth_result.token}
        if hasattr(auth_result, "url") and auth_result.url:
            return {"auth_url": auth_result.url}
        return {"error": "Unexpected auth result"}

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


async def process_linear_issue(issue_data: dict[str, Any], repo_config: dict[str, str]) -> None:
    """Process a Linear issue by creating a new LangGraph thread and run.

    Args:
        issue_data: The Linear issue data from webhook (basic info only).
        repo_config: The repo configuration with owner and name.
    """
    triggering_comment_id = issue_data.get("triggering_comment_id", "")
    if triggering_comment_id:
        await react_to_linear_comment(triggering_comment_id, "👀")

    issue_id = issue_data.get("id", "")
    thread_id = generate_thread_id_from_issue(issue_id)

    full_issue = await fetch_linear_issue_details(issue_id)
    if not full_issue:
        full_issue = issue_data

    user_email = None
    comment_author = issue_data.get("comment_author", {})
    if comment_author:
        user_email = comment_author.get("email")
    if not user_email:
        creator = full_issue.get("creator", {})
        if creator:
            user_email = creator.get("email")
    if not user_email:
        assignee = full_issue.get("assignee", {})
        if assignee:
            user_email = assignee.get("email")

    github_token = None
    if user_email and GITHUB_OAUTH_PROVIDER_ID:
        ls_user_id = await get_ls_user_id_from_email(user_email)

        if ls_user_id:
            auth_result = await get_github_token_for_user(ls_user_id)

            if "token" in auth_result:
                github_token = auth_result["token"]
            elif "auth_url" in auth_result:
                auth_url = auth_result["auth_url"]
                comment = f"""🔐 **GitHub Authentication Required**

To allow the Open SWE agent to work on this issue, please authenticate with GitHub by clicking the link below:

[Authenticate with GitHub]({auth_url})

Once authenticated, reply to this issue mentioning @openswe to retry."""

                await comment_on_linear_issue(issue_id, comment)
                return

    title = full_issue.get("title", "No title")
    description = full_issue.get("description") or "No description"

    comments = full_issue.get("comments", {}).get("nodes", [])
    comments_text = ""

    if comments:
        found_trigger = False
        relevant_comments = []

        for comment in comments:
            body = comment.get("body", "")
            if "@openswe" in body.lower():
                found_trigger = True
            if found_trigger:
                relevant_comments.append(comment)

        if relevant_comments:
            comments_text = "\n\n## Comments:\n"
            for comment in relevant_comments:
                author = comment.get("user", {}).get("name", "Unknown")
                body = comment.get("body", "")
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

        langgraph_client = get_client(url=LANGGRAPH_URL)
        await langgraph_client.runs.create(
            thread_id,
            "agent",
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": configurable},
            if_not_exists="create",
        )


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
    body = await request.body()

    signature = request.headers.get("Linear-Signature", "")
    if LINEAR_WEBHOOK_SECRET and not verify_linear_signature(
        body, signature, LINEAR_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        import json

        payload = json.loads(body)
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    if payload.get("type") != "Comment":
        return {"status": "ignored", "reason": "Not a Comment event"}

    action = payload.get("action")
    if action != "create":
        return {
            "status": "ignored",
            "reason": f"Comment action is '{action}', only processing 'create'",
        }

    data = payload.get("data", {})

    if data.get("botActor"):
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
            return {"status": "ignored", "reason": "Comment is our own bot message"}
    if "@openswe" not in comment_body.lower():
        return {"status": "ignored", "reason": "Comment doesn't mention @openswe"}

    issue = data.get("issue", {})
    if not issue:
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
