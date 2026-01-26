"""Custom FastAPI routes for LangGraph server."""

import hashlib
import hmac
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from langgraph_sdk import get_client

from deepagents_cli.encryption import encrypt_token

app = FastAPI()

# Linear webhook signing secret - set this in your environment
LINEAR_WEBHOOK_SECRET = os.environ.get("LINEAR_WEBHOOK_SECRET", "")

# LangGraph client - connects to the same server
LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL", "http://localhost:2024")

# LangSmith API for agent auth
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_API_KEY_PROD = os.environ.get("LANGSMITH_API_KEY_PROD", "")
LANGSMITH_API_URL = os.environ.get("LANGSMITH_API_URL", "https://api.smith.langchain.com")

# OAuth provider ID for GitHub (configured in LangSmith)
GITHUB_OAUTH_PROVIDER_ID = os.environ.get("GITHUB_OAUTH_PROVIDER_ID", "")

# Linear API key for commenting on issues
LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")

# Mapping of Linear team IDs/names to GitHub repos
# Format: "linear_team_id_or_name": {"owner": "github_owner", "name": "repo_name"}
LINEAR_TEAM_TO_REPO: dict[str, dict[str, str]] = {
    # Example mappings - add your Linear team IDs or names here
    # "bfc05139-75db-4f3c-8ba0-9a95af6a493d": {"owner": "langchain-ai", "name": "langchain"},
    # "My Team": {"owner": "langchain-ai", "name": "langgraph"},
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
        print("\n\n[Agent Auth] LANGSMITH_API_KEY not set, skipping user lookup\n\n")
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
                print(f"\n\n[Agent Auth] Found ls_user_id: {ls_user_id} for email: {email}\n\n")
                return ls_user_id
            else:
                print(f"\n\n[Agent Auth] No user found for email: {email}\n\n")
                return None
        except Exception as e:
            print(f"\n\n[Agent Auth] Error fetching user: {e}\n\n")
            return None


async def get_github_token_for_user(ls_user_id: str) -> dict[str, Any]:
    """Get GitHub OAuth token for a user via LangSmith agent auth.

    Args:
        ls_user_id: The LangSmith user ID

    Returns:
        Dict with either 'token' key or 'auth_url' key
    """
    if not GITHUB_OAUTH_PROVIDER_ID:
        print("\n\n[Agent Auth] GITHUB_OAUTH_PROVIDER_ID not set\n\n")
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

        if hasattr(auth_result, 'token') and auth_result.token:
            print("\n\n[Agent Auth] Successfully retrieved GitHub token\n\n")
            return {"token": auth_result.token}
        elif hasattr(auth_result, 'auth_url') and auth_result.auth_url:
            print(f"\n\n[Agent Auth] Auth URL returned: {auth_result.auth_url}\n\n")
            return {"auth_url": auth_result.auth_url}
        elif hasattr(auth_result, 'url') and auth_result.url:
            # Some versions use 'url' instead of 'auth_url'
            print(f"\n\n[Agent Auth] Auth URL returned: {auth_result.url}\n\n")
            return {"auth_url": auth_result.url}
        else:
            print(f"\n\n[Agent Auth] Unexpected auth result: {auth_result}\n\n")
            return {"error": "Unexpected auth result"}

    except Exception as e:
        print(f"\n\n[Agent Auth] Error authenticating: {e}\n\n")
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
        print("\n\n[Linear API] LINEAR_API_KEY not set, cannot react\n\n")
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
                print(f"\n\n[Linear API] Successfully reacted to comment {comment_id} with {emoji}\n\n")
                return True
            else:
                print(f"\n\n[Linear API] Failed to react: {result}\n\n")
                return False
        except Exception as e:
            print(f"\n\n[Linear API] Error reacting: {e}\n\n")
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
        print("\n\n[Linear API] LINEAR_API_KEY not set, cannot comment\n\n")
        return False

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
                print(f"\n\n[Linear API] Successfully commented on issue {issue_id}\n\n")
                return True
            else:
                print(f"\n\n[Linear API] Failed to comment: {result}\n\n")
                return False
        except Exception as e:
            print(f"\n\n[Linear API] Error commenting: {e}\n\n")
            return False


async def fetch_linear_issue_details(issue_id: str) -> dict[str, Any] | None:
    """Fetch full issue details from Linear API including description and comments.

    Args:
        issue_id: The Linear issue ID

    Returns:
        Full issue data dict, or None if fetch failed
    """
    if not LINEAR_API_KEY:
        print("\n\n[Linear API] LINEAR_API_KEY not set, cannot fetch issue\n\n")
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
                print(f"\n\n[Linear API] Successfully fetched issue {issue_id}\n\n")
                return issue
            else:
                print(f"\n\n[Linear API] Failed to fetch issue: {result}\n\n")
                return None
        except Exception as e:
            print(f"\n\n[Linear API] Error fetching issue: {e}\n\n")
            return None


def generate_thread_id_from_issue(issue_id: str) -> str:
    """Generate a deterministic thread ID from a Linear issue ID.

    Args:
        issue_id: The Linear issue ID

    Returns:
        A UUID-formatted thread ID derived from the issue ID
    """
    # Create a hash of the issue ID
    hash_bytes = hashlib.sha256(f"linear-issue:{issue_id}".encode()).hexdigest()
    # Format as UUID (8-4-4-4-12)
    return f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-{hash_bytes[16:20]}-{hash_bytes[20:32]}"


async def process_linear_issue(issue_data: dict[str, Any], repo_config: dict[str, str]) -> None:
    """Process a Linear issue by creating a new LangGraph thread and run.

    Args:
        issue_data: The Linear issue data from webhook (basic info only).
        repo_config: The repo configuration with owner and name.
    """
    print(f"\n\n[Linear Webhook] Processing issue: {issue_data.get('title')}\n\n")

    # React to the triggering comment with eyes emoji to acknowledge receipt
    triggering_comment_id = issue_data.get("triggering_comment_id", "")
    if triggering_comment_id:
        await react_to_linear_comment(triggering_comment_id, "👀")

    # Generate deterministic thread ID from issue ID
    issue_id = issue_data.get("id", "")
    thread_id = generate_thread_id_from_issue(issue_id)

    print(f"\n\n[Linear Webhook] Using thread ID: {thread_id} (from issue {issue_id})\n\n")

    # Fetch full issue details from Linear API (webhook only has basic info)
    full_issue = await fetch_linear_issue_details(issue_id)
    if full_issue:
        print(f"\n\n[Linear Webhook] Fetched full issue details\n\n")
    else:
        print(f"\n\n[Linear Webhook] Could not fetch full issue, using webhook data\n\n")
        full_issue = issue_data

    # Get user email - prefer comment author, then fall back to issue creator/assignee
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

    print(f"\n\n[Linear Webhook] User email: {user_email}\n\n")

    # Try to get GitHub token via agent auth
    github_token = None
    if user_email and GITHUB_OAUTH_PROVIDER_ID:
        ls_user_id = await get_ls_user_id_from_email(user_email)

        if ls_user_id:
            auth_result = await get_github_token_for_user(ls_user_id)

            if "token" in auth_result:
                github_token = auth_result["token"]
                print("\n\n[Linear Webhook] Got GitHub token from agent auth\n\n")
            elif "auth_url" in auth_result:
                # User needs to authenticate - comment on the issue with the auth URL
                auth_url = auth_result["auth_url"]
                comment = f"""🔐 **GitHub Authentication Required**

To allow the Open SWE agent to work on this issue, please authenticate with GitHub by clicking the link below:

[Authenticate with GitHub]({auth_url})

Once authenticated, reply to this issue mentioning @openswe to retry."""

                await comment_on_linear_issue(issue_id, comment)
                print("\n\n[Linear Webhook] Posted auth URL comment, aborting run\n\n")
                return

    # Build the prompt from full issue data
    title = full_issue.get("title", "No title")
    description = full_issue.get("description") or "No description"

    # Get comments from full issue data
    # Include the triggering comment and any comments after it
    triggering_comment_body = issue_data.get("triggering_comment", "")
    comments = full_issue.get("comments", {}).get("nodes", [])
    comments_text = ""

    if comments:
        # Find the triggering comment and include it plus any after
        found_trigger = False
        relevant_comments = []

        for comment in comments:
            body = comment.get("body", "")
            # Check if this is the triggering comment (contains @openswe)
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

    print(f"\n\n[Linear Webhook] Starting run with prompt:\n{prompt}\n\n")

    # Build config with repo info, Linear issue info, and optional GitHub token (encrypted)
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
        # Encrypt the token before storing in configurable to avoid plain text storage
        configurable["github_token_encrypted"] = encrypt_token(github_token)

    # Start the run (fire and forget - it will run in background)
    # Use if_not_exists="create" to create thread only if it doesn't exist
    try:
        langgraph_client = get_client(url=LANGGRAPH_URL)
        await langgraph_client.runs.create(
            thread_id,
            "agent",
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": configurable},
            if_not_exists="create",
        )
        print(f"\n\n[Linear Webhook] Run started for thread {thread_id}\n\n")
    except Exception as e:
        print(f"\n\n[Linear Webhook] Error starting run: {e}\n\n")


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
        # No secret configured, skip verification (not recommended for production)
        return True

    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


@app.post("/webhooks/linear")
async def linear_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Linear webhooks.

    Triggers a new LangGraph run when an issue gets the 'open-swe' label added.
    """
    print("\n\n[Linear Webhook] Received webhook\n\n")

    # Get raw body for signature verification
    body = await request.body()

    # Verify Linear signature
    signature = request.headers.get("Linear-Signature", "")
    if LINEAR_WEBHOOK_SECRET and not verify_linear_signature(body, signature, LINEAR_WEBHOOK_SECRET):
        print("\n\n[Linear Webhook] Invalid signature\n\n")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        import json
        payload = json.loads(body)
    except Exception as e:
        print(f"\n\n[Linear Webhook] Error parsing JSON: {e}\n\n")
        return {"status": "error", "message": "Invalid JSON"}

    print(f"\n\n[Linear Webhook] Payload type: {payload.get('type')}, action: {payload.get('action')}\n\n")
    print(f"\n\n[Linear Webhook] Full payload:\n{json.dumps(payload, indent=2)}\n\n")

    # Check if this is a Comment event
    if payload.get("type") != "Comment":
        print("\n\n[Linear Webhook] Not a Comment event, ignoring\n\n")
        return {"status": "ignored", "reason": "Not a Comment event"}

    # Only process new comments, not edits or deletes
    action = payload.get("action")
    if action != "create":
        print(f"\n\n[Linear Webhook] Comment action is '{action}', only processing 'create'\n\n")
        return {"status": "ignored", "reason": f"Comment action is '{action}', only processing 'create'"}

    data = payload.get("data", {})

    # Ignore bot comments to prevent infinite loops
    if data.get("botActor"):
        print("\n\n[Linear Webhook] Comment is from a bot, ignoring\n\n")
        return {"status": "ignored", "reason": "Comment is from a bot"}

    # Also ignore comments that look like our own bot messages
    comment_body = data.get("body", "")
    bot_message_prefixes = [
        "🔐 **GitHub Authentication Required**",
        "✅ **Pull Request Created**",
        "🤖 **Agent Response**",
        "❌ **Agent Error**",
    ]
    for prefix in bot_message_prefixes:
        if comment_body.startswith(prefix):
            print(f"\n\n[Linear Webhook] Comment is our own bot message (prefix: {prefix[:20]}...), ignoring\n\n")
            return {"status": "ignored", "reason": "Comment is our own bot message"}
    if "@openswe" not in comment_body.lower():
        print("\n\n[Linear Webhook] Comment doesn't mention @openswe, ignoring\n\n")
        return {"status": "ignored", "reason": "Comment doesn't mention @openswe"}

    print("\n\n[Linear Webhook] Found @openswe mention in comment\n\n")

    # Get the issue data from the comment
    issue = data.get("issue", {})
    if not issue:
        print("\n\n[Linear Webhook] No issue data in comment, ignoring\n\n")
        return {"status": "ignored", "reason": "No issue data in comment"}

    # Extract repo info from Linear team mapping
    team = issue.get("team", {})
    team_id = team.get("id", "") if team else ""
    team_name = team.get("name", "") if team else ""

    # Try to find repo by team ID first, then by team name
    repo_config = None
    if team_id and team_id in LINEAR_TEAM_TO_REPO:
        repo_config = LINEAR_TEAM_TO_REPO[team_id]
        print(f"\n\n[Linear Webhook] Found repo mapping by team ID: {team_id}\n\n")
    elif team_name and team_name in LINEAR_TEAM_TO_REPO:
        repo_config = LINEAR_TEAM_TO_REPO[team_name]
        print(f"\n\n[Linear Webhook] Found repo mapping by team name: {team_name}\n\n")

    # Fallback: Try to extract repo from issue labels (e.g., "repo:owner/name")
    if not repo_config:
        for label in issue.get("labels", []):
            label_name = label.get("name", "")
            if label_name.startswith("repo:"):
                repo_ref = label_name[5:]  # Remove "repo:" prefix
                if "/" in repo_ref:
                    owner, name = repo_ref.split("/", 1)
                    repo_config = {"owner": owner, "name": name}
                    print(f"\n\n[Linear Webhook] Found repo from label: {owner}/{name}\n\n")
                    break

    if not repo_config:
        # Default to langchain-ai/langchainplus for unknown teams
        print(f"\n\n[Linear Webhook] No repo mapping found for team '{team_name}' (ID: {team_id}), using default: langchain-ai/langchainplus\n\n")
        repo_config = {"owner": "langchain-ai", "name": "langchainplus"}

    repo_owner = repo_config["owner"]
    repo_name = repo_config["name"]

    print(f"\n\n[Linear Webhook] Processing issue for repo: {repo_owner}/{repo_name}\n\n")

    # Include the triggering comment and comment author in the issue data for context
    issue["triggering_comment"] = comment_body
    issue["triggering_comment_id"] = data.get("id", "")
    # Add comment author info for agent auth (email lookup)
    comment_user = data.get("user", {})
    if comment_user:
        issue["comment_author"] = comment_user

    # Process in background to return quickly
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
