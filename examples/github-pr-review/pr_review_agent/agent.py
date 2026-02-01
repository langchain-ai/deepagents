"""PR Review Agent - Main orchestrator with specialized subagents.

Creates a deep agent that coordinates code review and security review
subagents to provide comprehensive PR feedback.
"""

from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent

from .prompts import CODE_REVIEW_PROMPT, ORCHESTRATOR_PROMPT, SECURITY_REVIEW_PROMPT
from .tools import (
    create_branch,
    create_or_update_file,
    create_pr_from_branch,
    create_pr_review,
    get_code_scanning_alerts,
    get_file_content,
    get_pr_checks,
    get_pr_comments,
    get_pr_commits,
    get_pr_details,
    get_pr_diff,
    get_pr_files,
    get_repo_code_style,
    get_security_alerts,
    list_workflow_files,
    post_pr_comment,
    search_repo_issues,
    submit_plan,
)

# All tools available to the orchestrator
ORCHESTRATOR_TOOLS = [
    # PR information
    get_pr_diff,
    get_pr_files,
    get_pr_details,
    get_pr_commits,
    get_pr_comments,
    get_pr_checks,
    # Repository information
    get_repo_code_style,
    get_security_alerts,
    get_code_scanning_alerts,
    search_repo_issues,
    get_file_content,
    list_workflow_files,
    # Write operations (permission rules enforced via prompt instructions from webhook)
    post_pr_comment,
    create_pr_review,
    create_or_update_file,
    create_branch,
    create_pr_from_branch,
    # Planning (for commands that require approval)
    submit_plan,
]

# Tools for the code review subagent (read-only, focused on code)
CODE_REVIEW_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_file_content,
    get_repo_code_style,
    get_pr_checks,
]

# Tools for the security review subagent (read-only, focused on security)
SECURITY_REVIEW_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_file_content,
    get_security_alerts,
    get_code_scanning_alerts,
    list_workflow_files,
]


def create_pr_review_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create the PR review agent with subagents.

    Args:
        model_name: Model identifier in provider:model format

    Returns:
        Configured deep agent
    """
    model = init_chat_model(model=model_name, temperature=0.0)

    # Define subagents
    code_review_subagent = {
        "name": "code-review",
        "description": (
            "Specialized code review agent. Delegate to this agent to review code quality, "
            "style compliance, best practices, and documentation. Provide it with the PR diff "
            "and any style guidelines from the repository."
        ),
        "system_prompt": CODE_REVIEW_PROMPT,
        "tools": CODE_REVIEW_TOOLS,
        "model": "anthropic:claude-sonnet-4-5",
    }

    security_review_subagent = {
        "name": "security-review",
        "description": (
            "Specialized security review agent. Delegate to this agent to identify potential "
            "security vulnerabilities, unsafe patterns, and security best practice violations. "
            "Provide it with the PR diff and any existing security alerts."
        ),
        "system_prompt": SECURITY_REVIEW_PROMPT,
        "tools": SECURITY_REVIEW_TOOLS,
        "model": "anthropic:claude-sonnet-4-5",
    }

    return create_deep_agent(
        model=model,
        tools=ORCHESTRATOR_TOOLS,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[code_review_subagent, security_review_subagent],
    )
