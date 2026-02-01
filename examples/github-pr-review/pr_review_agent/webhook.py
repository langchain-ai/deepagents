"""GitHub webhook handler for PR review bot.

FastAPI application that receives GitHub webhook events and triggers
the PR review agent when @deepagents-bot is mentioned.

Security model:
- Permission checks are done in CODE, not by the LLM
- Users with write access: can commit to PR branch
- Users with read-only access: can only commit to a new branch (bot creates PR)
- Commit rules are enforced by providing specific instructions, not by trusting LLM
"""

import hashlib
import hmac
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .agent import create_pr_review_agent
from .github_client import GitHubClient
from .state import StateManager
from .tools import set_github_client

# Bot username to listen for mentions
BOT_USERNAME = os.environ.get("BOT_USERNAME", "deepagents-bot")
BOT_MENTION_PATTERN = re.compile(rf"@{BOT_USERNAME}\b", re.IGNORECASE)

# Global agent instance (created on startup)
_agent = None


class Permission(Enum):
    """User permission levels."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class PRContext:
    """Context about the PR and user permissions."""

    owner: str
    repo: str
    pr_number: int
    requester: str
    permission: Permission
    head_branch: str
    head_sha: str
    head_repo_owner: str
    head_repo_name: str
    base_branch: str
    is_fork: bool


async def get_user_permission(
    github_client: GitHubClient, owner: str, repo: str, username: str
) -> Permission:
    """Check user's permission level on a repository.

    This is a SECURITY-CRITICAL function - do not delegate to LLM.
    """
    try:
        perm = await github_client.get_user_permission(owner, repo, username)
        return Permission(perm)
    except Exception:
        return Permission.READ


async def get_pr_context(
    github_client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    requester: str,
) -> PRContext:
    """Gather all context about a PR including permissions.

    This is done in code, not by the LLM, for security.
    """
    permission = await get_user_permission(github_client, owner, repo, requester)

    pr = await github_client.get_pull_request(owner, repo, pr_number)

    head_repo = pr.head.repo
    base_repo = pr.base.repo

    head_repo_dict = {"full_name": head_repo.full_name if head_repo else None,
                      "owner": {"login": head_repo.owner.login if head_repo else owner},
                      "name": head_repo.name if head_repo else repo} if head_repo else {}
    base_repo_dict = {"full_name": base_repo.full_name if base_repo else None} if base_repo else {}
    is_fork = head_repo_dict.get("full_name") != base_repo_dict.get("full_name")

    return PRContext(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        requester=requester,
        permission=permission,
        head_branch=pr.head.ref,
        head_sha=pr.head.sha,
        head_repo_owner=head_repo_dict.get("owner", {}).get("login", owner),
        head_repo_name=head_repo_dict.get("name", repo),
        base_branch=pr.base.ref,
        is_fork=is_fork,
    )


def build_commit_instructions(ctx: PRContext) -> str:
    """Build instructions for the agent about how it can make commits.

    The branch/repo to commit to is determined by code based on permissions.
    """
    can_write = ctx.permission in (Permission.WRITE, Permission.ADMIN)

    if can_write and not ctx.is_fork:
        return f"""## Making Code Changes (WRITE ACCESS GRANTED)

{ctx.requester} has write access. You MAY commit directly to the PR branch.

When asked to fix or change code:
1. Use `create_or_update_file` with these EXACT parameters:
   - owner: "{ctx.head_repo_owner}"
   - repo: "{ctx.head_repo_name}"
   - branch: "{ctx.head_branch}"
2. Add a comment explaining what you changed

Only make changes when explicitly asked (e.g., "fix this", "apply suggestions").
Do NOT make changes for review-only requests.
"""
    else:
        # Fork PR or read-only user - must create new branch
        new_branch = f"{BOT_USERNAME}/pr-{ctx.pr_number}-fixes"
        return f"""## Making Code Changes (NEW BRANCH REQUIRED)

{"This PR is from a fork." if ctx.is_fork else f"{ctx.requester} has read-only access."}
You MUST create a new branch for any changes.

When asked to fix or change code:
1. First, create a new branch using `create_branch`:
   - owner: "{ctx.owner}"
   - repo: "{ctx.repo}"
   - new_branch: "{new_branch}"
   - from_ref: "{ctx.head_sha}"
2. Commit changes using `create_or_update_file`:
   - owner: "{ctx.owner}"
   - repo: "{ctx.repo}"
   - branch: "{new_branch}"
3. Create a PR using `create_pr_from_branch`:
   - owner: "{ctx.owner}"
   - repo: "{ctx.repo}"
   - head: "{new_branch}"
   - base: "{ctx.base_branch}"
4. Comment on PR #{ctx.pr_number} with a link to your new PR

Only make changes when explicitly asked.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent on startup."""
    global _agent
    model = os.environ.get("MODEL", "anthropic:claude-sonnet-4-5")
    _agent = create_pr_review_agent(model_name=model)
    yield


app = FastAPI(
    title="GitHub PR Review Bot",
    description="AI-powered PR review bot using deepagents",
    lifespan=lifespan,
)


def verify_webhook_signature(payload: bytes, signature: str | None) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body
        signature: X-Hub-Signature-256 header value

    Returns:
        True if signature is valid
    """
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
    if not secret:
        # No secret configured, skip verification (not recommended for production)
        return True

    if not signature:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


class WebhookPayload(BaseModel):
    """Minimal webhook payload structure."""

    action: str
    installation: dict | None = None
    repository: dict | None = None
    issue: dict | None = None
    pull_request: dict | None = None
    comment: dict | None = None
    sender: dict | None = None


@dataclass
class ParsedCommand:
    """Parsed user command from a comment."""

    command: str  # "review", "help", "security", "style", "feedback", "conflict"
    message: str = ""  # Optional message/instructions after the command


# Available slash commands
COMMANDS = {
    "help": "Show available commands",
    "review": "Full code review",
    "security": "Security-focused review (reads security docs first)",
    "style": "Code style review (reads style guides first)",
    "feedback": "Incorporate reviewer feedback into a new commit",
    "conflict": "Resolve merge conflicts for this PR",
    "remember": "Save a convention or preference for this repository",
    "memories": "List saved conventions for this repository",
}

# Special command for freeform requests (not in COMMANDS, handled separately)
FREEFORM_COMMAND = "chat"

# Status emojis (Dependabot-style)
STATUS_EMOJIS = {
    "pending": "🔄",
    "running": "⏳",
    "reading": "📖",
    "analyzing": "🔍",
    "reviewing": "📝",
    "planning": "📋",
    "waiting": "⏸️",
    "committing": "💾",
    "success": "✅",
    "error": "❌",
    "cancelled": "🚫",
}

# Command descriptions for status messages
COMMAND_DESCRIPTIONS = {
    "chat": "Responding",
    "review": "Reviewing PR",
    "security": "Security review",
    "style": "Style review", 
    "feedback": "Applying feedback",
    "conflict": "Resolving conflicts",
    "remember": "Saving memory",
    "memories": "Listing memories",
}

# Commands that require approval before making changes
COMMANDS_REQUIRING_APPROVAL = {"feedback", "conflict"}

# How long to wait for approval (in seconds)
APPROVAL_TIMEOUT = 3600  # 1 hour


def get_help_text() -> str:
    """Generate help text for available commands."""
    lines = [
        f"## 🤖 {BOT_USERNAME}\n",
        "Hi! I'm a PR review bot. Mention me to get started:\n",
        "| Command | Description |",
        "|---------|-------------|",
        f"| `@{BOT_USERNAME}` | Full code review |",
        f"| `@{BOT_USERNAME} <question>` | Ask me anything about this PR |",
    ]
    for cmd, desc in COMMANDS.items():
        if cmd not in ("help", "review"):  # Skip these, shown separately
            lines.append(f"| `@{BOT_USERNAME} /{cmd}` | {desc} |")
    lines.append(f"| `@{BOT_USERNAME} /help` | Show this help |")
    
    lines.extend([
        "",
        "### Examples",
        "```",
        f"@{BOT_USERNAME}                        # Full code review",
        f"@{BOT_USERNAME} what does this PR do?  # Ask a question",
        f"@{BOT_USERNAME} explain the changes    # Get an explanation",
        f"@{BOT_USERNAME} /review Focus on error handling",
        f"@{BOT_USERNAME} /security Check for SQL injection",
        f"@{BOT_USERNAME} /feedback              # Apply reviewer feedback",
        "```",
    ])
    return "\n".join(lines)


class StatusComment:
    """Manages a status comment that gets updated as work progresses."""

    def __init__(
        self,
        github_client: "GitHubClient",
        owner: str,
        repo: str,
        pr_number: int,
        command: str,
    ):
        self.github_client = github_client
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number
        self.command = command
        self.comment_id: int | None = None
        self.steps: list[tuple[str, str]] = []  # (status, message)

    def _build_body(self, current_status: str, current_message: str) -> str:
        """Build the comment body with status history."""
        desc = COMMAND_DESCRIPTIONS.get(self.command, "Working")
        emoji = STATUS_EMOJIS.get(current_status, "🔄")
        
        lines = [f"## {emoji} {desc}\n"]
        
        # Show completed steps
        for status, message in self.steps:
            step_emoji = STATUS_EMOJIS.get(status, "✓")
            lines.append(f"{step_emoji} {message}")
        
        # Show current step if not final
        if current_status not in ("success", "error"):
            lines.append(f"{emoji} {current_message}...")
        else:
            lines.append(f"{emoji} {current_message}")
        
        return "\n".join(lines)

    async def create(self, message: str = "Starting") -> int:
        """Create the initial status comment."""
        body = self._build_body("pending", message)
        result = await self.github_client.create_issue_comment(
            self.owner, self.repo, self.pr_number, body
        )
        self.comment_id = result["id"]
        return self.comment_id

    async def update(self, status: str, message: str) -> None:
        """Update the status comment with new progress."""
        if not self.comment_id:
            await self.create(message)
            return

        body = self._build_body(status, message)
        await self.github_client.update_issue_comment(
            self.owner, self.repo, self.comment_id, body
        )

    async def complete_step(self, message: str) -> None:
        """Mark a step as complete and update the comment."""
        self.steps.append(("success", message))
        await self.update("running", "Continuing")

    async def finish(self, success: bool, message: str) -> None:
        """Mark the work as complete."""
        status = "success" if success else "error"
        body = self._build_body(status, message)

        if self.comment_id:
            await self.github_client.update_issue_comment(
                self.owner, self.repo, self.comment_id, body
            )
        else:
            await self.github_client.create_issue_comment(
                self.owner, self.repo, self.pr_number, body
            )

    def _build_plan_body(
        self, 
        plan_items: list[str], 
        state: str = "waiting",
        revision: int = 1,
        feedback_from: str | None = None,
    ) -> str:
        """Build the plan comment body.
        
        Args:
            plan_items: List of planned changes
            state: "waiting", "approved", "feedback_requested", "timed_out"
            revision: Plan revision number
            feedback_from: Username who requested changes (for feedback_requested state)
        """
        desc = COMMAND_DESCRIPTIONS.get(self.command, "Working")
        revision_text = f" (Revision {revision})" if revision > 1 else ""
        
        if state == "waiting":
            header = f"## 📋 {desc} - Plan{revision_text}\n"
            footer = [
                "---",
                "**React with 👍 to approve, or 👎 to request changes.**",
            ]
        elif state == "approved":
            header = f"## ✅ {desc} - Plan Approved{revision_text}\n"
            footer = ["---", "_Executing approved plan..._"]
        elif state == "feedback_requested":
            header = f"## 💬 {desc} - Changes Requested{revision_text}\n"
            footer = [
                "---",
                f"@{feedback_from} Please reply with how you'd like me to modify the plan.",
                "",
                "_I'll revise the plan based on your feedback and ask for approval again._",
            ]
        elif state == "timed_out":
            header = f"## ⏸️ {desc} - Timed Out{revision_text}\n"
            footer = ["---", f"_Mention @{BOT_USERNAME} again to retry._"]
        else:
            header = f"## 📋 {desc}\n"
            footer = []

        lines = [header, "**Planned changes:**\n"]
        for i, item in enumerate(plan_items, 1):
            lines.append(f"{i}. {item}")
        lines.append("")
        lines.extend(footer)
        
        return "\n".join(lines)

    async def post_plan_and_wait_for_approval(
        self, 
        plan_items: list[str],
        requester: str,
        replan_callback=None,
        timeout: int = APPROVAL_TIMEOUT,
        poll_interval: int = 10,
    ) -> tuple[bool, list[str]]:
        """Post a plan and wait for approval. On rejection, ask for feedback and revise.

        Args:
            plan_items: List of planned actions
            requester: GitHub username who requested the action
            replan_callback: Async function(feedback: str) -> list[str] to generate new plan
            timeout: How long to wait for approval (seconds)
            poll_interval: How often to check for reactions (seconds)

        Returns:
            Tuple of (approved: bool, final_plan: list[str])
        """
        import asyncio
        import time

        current_plan = list(plan_items)
        revision = 1
        processed_reaction_ids: set[int] = set()
        last_comment_check_id = 0
        
        # Post initial plan
        body = self._build_plan_body(current_plan, "waiting", revision)
        if self.comment_id:
            await self.github_client.update_issue_comment(
                self.owner, self.repo, self.comment_id, body
            )
        else:
            result = await self.github_client.create_issue_comment(
                self.owner, self.repo, self.pr_number, body
            )
            self.comment_id = result["id"]
        
        last_comment_check_id = self.comment_id
        start_time = time.time()
        waiting_for_feedback = False
        feedback_requester: str | None = None

        while time.time() - start_time < timeout:
            try:
                # Check reactions on our plan comment
                reactions = await self.github_client.get_comment_reactions(
                    self.owner, self.repo, self.comment_id
                )

                for reaction in reactions:
                    reaction_id = reaction.get("id")
                    if reaction_id in processed_reaction_ids:
                        continue
                    
                    content = reaction.get("content")
                    reactor = reaction.get("user", {}).get("login", "")
                    
                    if content == "+1":  # 👍 Approved!
                        processed_reaction_ids.add(reaction_id)
                        body = self._build_plan_body(current_plan, "approved", revision)
                        await self.github_client.update_issue_comment(
                            self.owner, self.repo, self.comment_id, body
                        )
                        self.steps.append(("success", f"Plan approved by @{reactor}"))
                        return True, current_plan

                    elif content == "-1" and not waiting_for_feedback:  # 👎 Request changes
                        processed_reaction_ids.add(reaction_id)
                        waiting_for_feedback = True
                        feedback_requester = reactor

                        # Update plan comment to ask for feedback
                        body = self._build_plan_body(current_plan, "feedback_requested", revision, reactor)
                        await self.github_client.update_issue_comment(
                            self.owner, self.repo, self.comment_id, body
                        )

                        # Remember when we asked for feedback to find replies
                        last_comment_check_id = self.comment_id

                # If waiting for feedback, look for reply comments
                if waiting_for_feedback and feedback_requester:
                    comments = await self.github_client.get_issue_comments(
                        self.owner, self.repo, self.pr_number
                    )
                    
                    for c in comments:
                        comment_id = c.get("id", 0)
                        if comment_id <= last_comment_check_id:
                            continue
                        
                        commenter = c.get("user", {}).get("login", "")
                        # Accept feedback from the person who thumbs-downed or anyone with write access
                        if commenter.lower() == feedback_requester.lower():
                            feedback_text = c.get("body", "").strip()
                            if not feedback_text:
                                continue
                            
                            # Generate revised plan
                            revision += 1
                            if replan_callback:
                                try:
                                    current_plan = await replan_callback(feedback_text)
                                except Exception:
                                    # Fallback: append feedback as note
                                    current_plan = list(plan_items) + [
                                        f"_(Revised based on feedback from @{feedback_requester})_"
                                    ]
                            else:
                                # No callback - just note the feedback
                                current_plan = list(plan_items) + [
                                    f"_(Feedback from @{feedback_requester}: {feedback_text[:100]}...)_"
                                ]
                            
                            # Reset state and post revised plan
                            waiting_for_feedback = False
                            feedback_requester = None
                            body = self._build_plan_body(current_plan, "waiting", revision)
                            await self.github_client.update_issue_comment(
                                self.owner, self.repo, self.comment_id, body
                            )
                            last_comment_check_id = comment_id
                            # Reset timeout for new plan
                            start_time = time.time()
                            break

            except Exception:
                pass  # Ignore polling errors
            
            await asyncio.sleep(poll_interval)

        # Timed out
        body = self._build_plan_body(current_plan, "timed_out", revision)
        await self.github_client.update_issue_comment(
            self.owner, self.repo, self.comment_id, body
        )
        return False, current_plan


async def is_first_interaction(
    github_client: "GitHubClient",
    owner: str,
    repo: str,
    pr_number: int,
) -> bool:
    """Check if this is the first time the bot has been invoked on this PR."""
    comments = await github_client.get_issue_comments(owner, repo, pr_number)
    
    for comment in comments:
        user = comment.get("user", {})
        if user.get("login", "").lower() == BOT_USERNAME.lower():
            return False
        # Also check if comment body contains our bot name (in case of app comments)
        if user.get("type") == "Bot" and BOT_USERNAME.lower() in comment.get("body", "").lower():
            return False
    
    return True


def parse_command(request_text: str) -> ParsedCommand:
    """Parse a command from the request text.

    Args:
        request_text: Text after the @mention

    Returns:
        ParsedCommand with command and message.
        - If slash command: use that command
        - If text but no command: use "chat" (freeform response)
        - If no text: use "review" (default full review)
    """
    text = request_text.strip()
    
    # Check for slash command
    if text.startswith("/"):
        parts = text[1:].split(None, 1)  # Split on first whitespace
        if parts:
            cmd = parts[0].lower()
            message = parts[1].strip() if len(parts) > 1 else ""
            
            if cmd in COMMANDS:
                return ParsedCommand(command=cmd, message=message)
    
    # No slash command
    if text:
        # User provided text without a command - treat as freeform chat/question
        return ParsedCommand(command=FREEFORM_COMMAND, message=text)
    else:
        # Just @mentioned with nothing else - do a full review
        return ParsedCommand(command="review", message="")


def extract_user_request(comment_body: str) -> ParsedCommand | None:
    """Extract the user's request from a comment mentioning the bot.

    Args:
        comment_body: Full comment text

    Returns:
        ParsedCommand if bot was mentioned, None otherwise
    """
    match = BOT_MENTION_PATTERN.search(comment_body)
    if not match:
        return None

    # Get everything after the mention
    request_text = comment_body[match.end():].strip()
    return parse_command(request_text)


BUILD_AND_CI_DOCS = """
## Build & CI Context (read these first)
Before analyzing code, gather context about how this repo builds and tests:

1. **GitHub Actions workflows** (check .github/workflows/):
   - Look for CI/test workflows to understand how code is validated
   - Check for build commands, test commands, lint commands
   - Note any required checks that must pass

2. **Build configuration** (based on languages in the PR):
   - Python: pyproject.toml, setup.py, setup.cfg, requirements.txt, Makefile
   - JavaScript/TypeScript: package.json, tsconfig.json
   - Go: go.mod, Makefile
   - Rust: Cargo.toml
   - Java: pom.xml, build.gradle

3. **PR CI status**:
   - Use get_pr_details to check if CI checks are passing/failing
   - If there are failures, understand what's broken before reviewing

4. **Documentation**:
   - README.md for project overview and setup instructions
   - CONTRIBUTING.md for contribution guidelines
   - Any docs/ folder content relevant to changed files
"""


def get_task_instructions(command: str, ctx: "PRContext", message: str = "") -> str:
    """Get task-specific instructions for each command.

    Args:
        command: The parsed command
        ctx: PR context
        message: Optional user message/instructions

    Returns:
        Detailed instructions for the agent
    """
    # Format user instructions if provided
    user_instructions = ""
    if message:
        user_instructions = f"""
## User Instructions
The user provided the following additional instructions:
> {message}

Pay special attention to these instructions when performing your task.
"""

    instructions = {
        "chat": f"""## Task: Respond to User Request

The user mentioned you with a question or request:
> {message}

Use the available tools to help answer their question or fulfill their request.
You have access to PR information, file contents, and can post comments.

**Guidelines:**
- Be helpful and conversational
- If they ask about the PR, get the diff and relevant files
- If they ask to explain something, read the code and explain it
- If they ask a question you can answer from context, answer it
- Post your response as a PR comment using post_pr_comment

**Available context:**
- PR diff and changed files
- File contents from the repository
- PR details, commits, and comments
- CI/check status""",

        "review": f"""{BUILD_AND_CI_DOCS}
{user_instructions}
## Task: Comprehensive Code Review

1. **Gather context** (do these IN PARALLEL):
   - Get the PR diff and list of changed files
   - Read relevant workflow files from .github/workflows/
   - Read build configs (pyproject.toml, package.json, etc.) for changed file types
   - Check PR CI status for any build/test failures
   - Read CONTRIBUTING.md and any style guides

2. **Analyze**:
   - Delegate to code-review and security-review subagents IN PARALLEL
   - Pass them the context you gathered (style guides, build info)

3. **Synthesize and post**:
   - Combine feedback from both subagents
   - If CI is failing, prioritize feedback related to the failures
   - Post using create_pr_review (COMMENT for suggestions, REQUEST_CHANGES for blocking issues)""",

        "security": f"""{BUILD_AND_CI_DOCS}
{user_instructions}
## Task: Security-Focused Review

1. **Read security documentation FIRST**:
   - SECURITY.md, .github/SECURITY.md
   - Security policies in CONTRIBUTING.md
   - .github/workflows/ - look for security scanning workflows (CodeQL, Snyk, etc.)
   - Any security-related configs (.snyk, .trivyignore, etc.)

2. **Get security context**:
   - Fetch Dependabot alerts for known vulnerable dependencies
   - Fetch CodeQL/code-scanning alerts
   - Check PR CI status for security scan results

3. **Analyze the PR**:
   - Get the diff and changed files
   - Delegate to security-review subagent with all the security context
   - Focus on: injection, auth issues, data exposure, crypto, dependencies

4. **Post findings** as a review comment with severity levels""",

        "style": f"""{BUILD_AND_CI_DOCS}
{user_instructions}
## Task: Code Style Review

1. **Read ALL style configuration FIRST**:
   - pyproject.toml - look for [tool.ruff], [tool.black], [tool.isort], [tool.mypy]
   - .editorconfig, .prettierrc, .prettierrc.json, .eslintrc, .eslintrc.json
   - tslint.json, .stylelintrc
   - CONTRIBUTING.md, STYLE.md, CODE_STYLE.md, docs/style-guide.md
   - .github/workflows/ - look for lint/format workflows to see what tools are used

2. **Understand the lint/format commands**:
   - Check Makefile, package.json scripts, or CI workflows for lint commands
   - Note any auto-formatting tools (black, prettier, gofmt)

3. **Analyze the PR**:
   - Get the diff and changed files
   - Check if CI lint checks are passing or failing
   - Delegate to code-review subagent with style configs
   - Focus ONLY on style issues - skip security and logic

4. **Post findings** - if auto-formatters exist, suggest running them""",

        "feedback_plan": f"""{BUILD_AND_CI_DOCS}
{user_instructions}
## Task: Plan Feedback Changes (Planning Phase)

You are in PLANNING MODE. Do NOT make any changes yet.

1. **Gather feedback**:
   - Get all review comments and PR comments
   - Filter for actionable feedback from HUMAN reviewers (ignore bot comments)
   - List each piece of feedback and which file it affects

2. **Understand the codebase**:
   - Read build configs to understand the project structure
   - Read the files that need to be changed

3. **Create a plan** by calling submit_plan with a list of planned changes:
   - Each item should describe ONE specific change
   - Format: "File: <path> - <what will be changed>"
   - Be specific about what you will modify

Example plan:
- "File: src/utils.py - Add docstring to calculate_total function"
- "File: src/utils.py - Fix typo 'recieve' -> 'receive' on line 42"
- "File: tests/test_utils.py - Add test case for edge case mentioned in review"

DO NOT make any changes. Only call submit_plan with your planned changes.""",

        "feedback_execute": f"""{BUILD_AND_CI_DOCS}

## Task: Execute Approved Feedback Changes

The plan has been approved. Now execute the changes.

1. **Get current file contents** for each file you need to modify

2. **Make the approved changes** using create_or_update_file:
   - Only make the changes that were in the approved plan
   - Commit to: {ctx.head_repo_owner}/{ctx.head_repo_name} branch {ctx.head_branch}

3. **Post a summary comment** explaining what you changed

Approved plan:
{{approved_plan}}""",

        "conflict_plan": f"""{BUILD_AND_CI_DOCS}
{user_instructions}
## Task: Plan Conflict Resolution (Planning Phase)

You are in PLANNING MODE. Do NOT make any changes yet.

1. **Understand the conflict**:
   - Get PR details to confirm there are merge conflicts
   - Identify which files have conflicts

2. **Gather file versions**:
   - Get the base branch ({ctx.base_branch}) version of conflicting files
   - Get the head branch ({ctx.head_branch}) version of conflicting files
   - Understand what each side was trying to accomplish

3. **Create a plan** by calling submit_plan with your resolution strategy:
   - Each item should describe how you'll resolve ONE file
   - Format: "File: <path> - <resolution strategy>"
   - Be specific about how you'll merge the changes

Example plan:
- "File: src/config.py - Keep both new config options from base and head"
- "File: src/api.py - Merge the two new endpoints, update imports"
- "File: package.json - Combine dependencies from both branches, use higher versions"

DO NOT make any changes. Only call submit_plan with your resolution strategy.""",

        "conflict_execute": f"""{BUILD_AND_CI_DOCS}

## Task: Execute Approved Conflict Resolution

The plan has been approved. Now resolve the conflicts.

1. **Get current file contents** for base and head versions of each conflicting file

2. **Resolve and commit** using create_or_update_file:
   - Follow the approved resolution strategy
   - Commit to: {ctx.head_repo_owner}/{ctx.head_repo_name} branch {ctx.head_branch}

3. **Post a summary comment** explaining how you resolved each conflict

Approved plan:
{{approved_plan}}""",

        "memories": f"""## Task: List and Explain Repository Memories
{user_instructions}
You have been asked to show what conventions and preferences you remember for this repository.

The repository memories are shown in the "Repository Conventions" section of this prompt (if any exist).

**Your task:**
1. If there are memories/conventions listed above, present them clearly to the user:
   - List each memory with context about when/why it was added if available
   - Explain how these memories affect your reviews
   
2. If there are NO memories listed, explain:
   - The repository doesn't have any saved conventions yet
   - How to add memories using `/remember <convention>`
   - Give examples of useful things to remember (coding style, testing requirements, etc.)

3. Post your response using post_pr_comment

Be helpful and conversational. If the user provided additional instructions, address those too.""",
    }
    return instructions.get(command, instructions["review"])


async def handle_pr_comment(payload: WebhookPayload) -> dict:
    """Handle a comment on a PR that mentions the bot.

    Args:
        payload: Webhook payload

    Returns:
        Response dict with status
    """
    comment = payload.comment
    if not comment:
        print("[handle_pr_comment] No comment in payload")
        return {"status": "ignored", "reason": "No comment in payload"}

    # Check for bot mention - this is the ONLY way to invoke the bot
    comment_body = comment.get("body", "")
    print(f"[handle_pr_comment] Comment body: {comment_body!r}")
    print(f"[handle_pr_comment] BOT_USERNAME: {BOT_USERNAME}")
    print(f"[handle_pr_comment] Pattern: {BOT_MENTION_PATTERN.pattern}")
    
    parsed = extract_user_request(comment_body)
    print(f"[handle_pr_comment] Parsed: {parsed}")

    if not parsed:
        return {"status": "ignored", "reason": "No bot mention"}

    # Don't respond to our own comments
    sender = payload.sender or {}
    sender_login = sender.get("login", "")
    if sender_login.lower() == BOT_USERNAME.lower():
        return {"status": "ignored", "reason": "Ignoring own comment"}

    # Also ignore other bots to prevent loops
    if sender.get("type") == "Bot":
        return {"status": "ignored", "reason": "Ignoring bot comment"}

    # Extract repo/PR info
    repo = payload.repository or {}
    owner = repo.get("owner", {}).get("login")
    repo_name = repo.get("name")

    pr_number = None
    if payload.pull_request:
        pr_number = payload.pull_request.get("number")
    elif payload.issue:
        if payload.issue.get("pull_request"):
            pr_number = payload.issue.get("number")

    print(f"[handle_pr_comment] owner={owner}, repo={repo_name}, pr={pr_number}")

    if not all([owner, repo_name, pr_number]):
        return {"status": "error", "message": "Missing repo/PR info"}

    installation_id = payload.installation.get("id") if payload.installation else None
    if not installation_id:
        return {"status": "error", "message": "No installation ID"}
    
    print(f"[handle_pr_comment] Processing /{parsed.command} for {owner}/{repo_name}#{pr_number}")

    # Create GitHub client for this installation
    github_client = GitHubClient(installation_id)

    # Set global client for tools to use
    set_github_client(github_client)

    # Check if this is the first interaction - show help first
    first_time = await is_first_interaction(github_client, owner, repo_name, pr_number)
    print(f"[handle_pr_comment] first_time={first_time}")
    
    if first_time and parsed.command != "help":
        print("[handle_pr_comment] Posting first-time help")
        try:
            result = await github_client.create_issue_comment(
                owner, repo_name, pr_number, get_help_text()
            )
            print(f"[handle_pr_comment] Posted help: {result.get('id', 'no id')}")
        except Exception as e:
            print(f"[handle_pr_comment] Failed to post help: {e}")

    # Handle /help command - always post help when explicitly requested
    if parsed.command == "help":
        print("[handle_pr_comment] Handling /help command")
        try:
            result = await github_client.create_issue_comment(
                owner, repo_name, pr_number, get_help_text()
            )
            print(f"[handle_pr_comment] Posted help: {result.get('id', 'no id')}")
            return {"status": "success", "command": "help"}
        except Exception as e:
            print(f"[handle_pr_comment] Failed to post help: {e}")
            return {"status": "error", "message": f"Failed to post help: {e}"}

    # Handle /remember - saves to repo-specific memory (no agent needed)
    if parsed.command == "remember":
        if not parsed.message or not parsed.message.strip():
            await github_client.create_issue_comment(
                owner, repo_name, pr_number,
                "❌ Please provide something to remember. Example:\n\n"
                "`@bot /remember We use 4-space indentation in Python files`"
            )
            return {"status": "error", "message": "No memory content provided"}
        
        try:
            state_mgr = StateManager(owner, repo_name, pr_number)
            state_mgr.repo_memory.add_user_memory(
                memory=parsed.message.strip(),
                added_by=sender_login,
                pr_number=pr_number,
            )
            
            # Confirm the memory was saved
            await github_client.create_issue_comment(
                owner, repo_name, pr_number,
                f"✅ **Remembered for {owner}/{repo_name}:**\n\n"
                f"> {parsed.message.strip()}\n\n"
                f"I'll follow this convention in future reviews of this repository."
            )
            return {"status": "success", "command": "remember"}
        except Exception as e:
            print(f"[handle_pr_comment] Failed to save memory: {e}")
            await github_client.create_issue_comment(
                owner, repo_name, pr_number,
                f"❌ Failed to save memory: {e}"
            )
            return {"status": "error", "message": f"Failed to save memory: {e}"}

    # For all other commands, we need the agent
    global _agent
    if not _agent:
        return {"status": "error", "message": "Agent not initialized"}

    # Create status comment to show progress
    status = StatusComment(github_client, owner, repo_name, pr_number, parsed.command)
    await status.create("Starting")

    # Gather PR context with permissions (checked in CODE, not by LLM)
    try:
        await status.update("reading", "Gathering PR context")
        ctx = await get_pr_context(
            github_client, owner, repo_name, pr_number, sender_login
        )
        await status.complete_step("Gathered PR context")
    except Exception as e:
        await status.finish(False, f"Failed to get PR context: {e}")
        return {"status": "error", "message": f"Failed to get PR context: {e}"}

    # Build commit instructions based on permissions (determined by CODE)
    commit_instructions = build_commit_instructions(ctx)

    # Base context for all prompts
    context_block = f"""## Context (verified by system)
- Repository: {owner}/{repo_name}
- PR Number: {pr_number}
- Requester: {sender_login}
- Requester Permission: {ctx.permission.value}
- Head Branch: {ctx.head_branch} (SHA: {ctx.head_sha[:7]})
- Base Branch: {ctx.base_branch}
- Is Fork: {ctx.is_fork}

{commit_instructions}"""

    # Initialize state manager for this repo/PR
    state_mgr = StateManager(owner, repo_name, pr_number)
    config = state_mgr.get_agent_config()

    # Add repository memory context if available
    memory_context = state_mgr.get_memory_context()

    # Commands that require approval use a two-phase flow
    if parsed.command in COMMANDS_REQUIRING_APPROVAL:
        return await _handle_approval_flow(
            agent=_agent,
            config=config,
            status=status,
            ctx=ctx,
            parsed=parsed,
            context_block=context_block,
            owner=owner,
            repo_name=repo_name,
            pr_number=pr_number,
            sender_login=sender_login,
            state_mgr=state_mgr,
        )

    # Standard flow for review/security/style commands
    task_instructions = get_task_instructions(parsed.command, ctx, parsed.message)

    await status.update("analyzing", "Reading repository docs and configuration")

    prompt = f"""PR #{pr_number} in {owner}/{repo_name}

## Task: /{parsed.command}
{task_instructions}

{context_block}

{memory_context}

## Important
- Do NOT post status updates - the system handles that
- Post your final review/results using create_pr_review or post_pr_comment
"""

    try:
        await status.update("reviewing", "Analyzing code")

        async for _ in _agent.astream(
            {"messages": [("user", prompt)]},
            config=config,
            stream_mode="values",
        ):
            pass

        await status.finish(True, "Complete!")

        # Record the review in PR state
        state_mgr.pr_state.record_review(
            command=parsed.command,
            summary=f"Completed /{parsed.command} review",
        )

        return {
            "status": "success",
            "message": f"Processed /{parsed.command} for PR #{pr_number}",
            "command": parsed.command,
            "permission": ctx.permission.value,
        }

    except Exception as e:
        await status.finish(False, f"Error: {e}")
        return {"status": "error", "message": str(e)}


async def _handle_approval_flow(
    agent,
    config: dict,
    status: StatusComment,
    ctx: "PRContext",
    parsed,
    context_block: str,
    owner: str,
    repo_name: str,
    pr_number: int,
    sender_login: str,
    state_mgr: "StateManager",
) -> dict:
    """Handle two-phase approval flow for /feedback and /conflict commands.

    Phase 1: Agent creates a plan using submit_plan tool
    Phase 2: Wait for approval, then execute the plan
    """
    from .tools import clear_submitted_plan, get_submitted_plan

    # Phase 1: Planning
    await status.update("planning", "Analyzing and creating plan")

    plan_instructions = get_task_instructions(f"{parsed.command}_plan", ctx, parsed.message)
    plan_prompt = f"""PR #{pr_number} in {owner}/{repo_name}

## Task: /{parsed.command} (Planning Phase)
{plan_instructions}

{context_block}

## Important
- You are in PLANNING MODE - do NOT make any changes
- Analyze what needs to be done and call submit_plan with your planned changes
- Each plan item should describe ONE specific change
"""

    try:
        clear_submitted_plan()
        
        async for _ in agent.astream(
            {"messages": [("user", plan_prompt)]},
            config=config,
            stream_mode="values",
        ):
            pass

        plan = get_submitted_plan()
        if not plan:
            await status.finish(False, "Agent did not submit a plan")
            return {"status": "error", "message": "No plan submitted"}

        await status.complete_step(f"Created plan with {len(plan)} items")

    except Exception as e:
        await status.finish(False, f"Planning failed: {e}")
        return {"status": "error", "message": f"Planning failed: {e}"}

    # Phase 2: Wait for approval
    async def replan_with_feedback(feedback: str) -> list[str]:
        """Re-run planning phase with user feedback."""
        clear_submitted_plan()
        
        replan_prompt = f"""PR #{pr_number} in {owner}/{repo_name}

## Task: /{parsed.command} (Revised Planning)

The previous plan was rejected. User feedback:
> {feedback}

Please revise the plan based on this feedback.

{plan_instructions}

{context_block}

## Important
- You are in PLANNING MODE - do NOT make any changes
- Call submit_plan with your REVISED planned changes
"""
        async for _ in agent.astream(
            {"messages": [("user", replan_prompt)]},
            config=config,
            stream_mode="values",
        ):
            pass
        
        return get_submitted_plan() or plan  # Fallback to original if replan fails

    approved, final_plan = await status.post_plan_and_wait_for_approval(
        plan_items=plan,
        requester=sender_login,
        replan_callback=replan_with_feedback,
    )

    if not approved:
        return {
            "status": "cancelled",
            "message": "Plan not approved",
            "command": parsed.command,
        }

    # Phase 3: Execute approved plan
    await status.update("committing", "Executing approved changes")
    
    exec_instructions = get_task_instructions(f"{parsed.command}_execute", ctx)
    # Insert the approved plan into the execute instructions
    exec_instructions = exec_instructions.replace(
        "{approved_plan}",
        "\n".join(f"- {item}" for item in final_plan)
    )
    
    exec_prompt = f"""PR #{pr_number} in {owner}/{repo_name}

## Task: /{parsed.command} (Execution Phase)
{exec_instructions}

{context_block}

## Important
- The plan has been approved - execute ONLY the approved changes
- Post a summary comment when done
"""

    try:
        async for _ in agent.astream(
            {"messages": [("user", exec_prompt)]},
            config=config,
            stream_mode="values",
        ):
            pass

        await status.finish(True, "Changes applied!")

        # Record the successful execution in PR state
        state_mgr.pr_state.record_review(
            command=parsed.command,
            summary=f"Executed approved plan with {len(final_plan)} changes",
        )

        return {
            "status": "success",
            "message": f"Executed approved /{parsed.command} plan",
            "command": parsed.command,
            "permission": ctx.permission.value,
            "plan_items": len(final_plan),
        }

    except Exception as e:
        await status.finish(False, f"Execution failed: {e}")
        return {"status": "error", "message": f"Execution failed: {e}"}


@app.post("/webhook")
async def webhook_handler(
    request: Request,
    x_hub_signature_256: str | None = Header(None),
    x_github_event: str | None = Header(None),
):
    """Handle incoming GitHub webhooks.

    Args:
        request: FastAPI request
        x_hub_signature_256: GitHub signature header
        x_github_event: GitHub event type header

    Returns:
        Response dict
    """
    # Read raw body for signature verification
    body = await request.body()

    # Verify signature
    if not verify_webhook_signature(body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    try:
        payload = WebhookPayload.model_validate_json(body)
    except Exception as e:
        print(f"[webhook] Failed to parse payload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    print(f"[webhook] Event: {x_github_event}, Action: {payload.action}")
    
    # Route based on event type
    if x_github_event == "issue_comment":
        # Comment on an issue (which could be a PR)
        if payload.action == "created":
            print(f"[webhook] Processing issue_comment from {payload.sender}")
            result = await handle_pr_comment(payload)
            print(f"[webhook] Result: {result}")
            return result

    elif x_github_event == "pull_request_review_comment":
        # Comment on a PR review
        if payload.action == "created":
            print(f"[webhook] Processing pull_request_review_comment")
            result = await handle_pr_comment(payload)
            print(f"[webhook] Result: {result}")
            return result

    print(f"[webhook] Ignoring event: {x_github_event} / {payload.action}")
    return {"status": "ignored", "event": x_github_event, "action": payload.action}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent_ready": _agent is not None}
