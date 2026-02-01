"""PR Review Agents - Specialized agents for different review types.

Creates separate agents for:
- Security review
- Code quality review  
- General chat/questions
- Feedback execution (with approval flow)
"""

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from .prompts import CODE_REVIEW_PROMPT, SECURITY_REVIEW_PROMPT, CHAT_PROMPT
from .tools import (
    create_branch,
    create_or_update_file,
    create_pr_from_branch,
    create_pr_review,
    get_file_content,
    get_pr_checks,
    get_pr_comments,
    get_pr_commits,
    get_pr_details,
    get_pr_diff,
    get_pr_files,
    get_repo_code_style,
    list_workflow_files,
    post_pr_comment,
    search_repo_issues,
    submit_plan,
)

# Tools for security review (read-only + post review)
SECURITY_REVIEW_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_file_content,
    list_workflow_files,
    get_pr_checks,
    create_pr_review,
]

# Tools for code quality review (read-only + post review)
CODE_REVIEW_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_file_content,
    get_repo_code_style,
    get_pr_checks,
    create_pr_review,
]

# Tools for general chat (read-only + post comment)
CHAT_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_pr_details,
    get_pr_commits,
    get_pr_comments,
    get_pr_checks,
    get_file_content,
    get_repo_code_style,
    list_workflow_files,
    search_repo_issues,
    post_pr_comment,
]

# Tools for feedback/conflict resolution (full access)
FEEDBACK_TOOLS = [
    get_pr_diff,
    get_pr_files,
    get_pr_details,
    get_pr_commits,
    get_pr_comments,
    get_pr_checks,
    get_file_content,
    get_repo_code_style,
    list_workflow_files,
    post_pr_comment,
    create_pr_review,
    create_or_update_file,
    create_branch,
    create_pr_from_branch,
    submit_plan,
]


def create_security_review_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create an agent for security-focused PR reviews."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_react_agent(
        model=model,
        tools=SECURITY_REVIEW_TOOLS,
        prompt=SECURITY_REVIEW_PROMPT,
    )


def create_quality_review_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create an agent for code quality PR reviews."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_react_agent(
        model=model,
        tools=CODE_REVIEW_TOOLS,
        prompt=CODE_REVIEW_PROMPT,
    )


def create_chat_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create an agent for general PR questions and chat."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_react_agent(
        model=model,
        tools=CHAT_TOOLS,
        prompt=CHAT_PROMPT,
    )


def create_feedback_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create an agent for applying feedback and resolving conflicts."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_react_agent(
        model=model,
        tools=FEEDBACK_TOOLS,
        prompt=None,  # Prompt provided dynamically based on task
    )
