"""PR Review Agents - Specialized deep agents for different review types.

Creates separate deep agents for:
- Security review
- Code quality review
- General chat/questions (with memory support)
- Feedback execution (with approval flow)

Each agent is specialized and does NOT invoke other agents.
"""

from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent, MemoryMiddleware, FilesystemMiddleware

from .prompts import CODE_REVIEW_PROMPT, SECURITY_REVIEW_PROMPT, CHAT_PROMPT
from .storage import DeepAgentsBackend, get_memory_path
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
    """Create a deep agent for security-focused PR reviews."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_deep_agent(
        model=model,
        tools=SECURITY_REVIEW_TOOLS,
        system_prompt=SECURITY_REVIEW_PROMPT,
    )


def create_quality_review_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create a deep agent for code quality PR reviews."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_deep_agent(
        model=model,
        tools=CODE_REVIEW_TOOLS,
        system_prompt=CODE_REVIEW_PROMPT,
    )


def create_chat_agent(
    owner: str | None = None,
    repo: str | None = None,
    model_name: str = "anthropic:claude-sonnet-4-5",
):
    """Create a deep agent for general PR questions and chat.

    Uses deepagents for complex reasoning and multi-step tasks.
    When owner/repo are provided, includes MemoryMiddleware to load and update
    repository conventions from AGENTS.md.

    Args:
        owner: Repository owner (e.g., 'langchain-ai'). If None, no memory support.
        repo: Repository name (e.g., 'langchain'). If None, no memory support.
        model_name: Model to use for the agent.
    """
    model = init_chat_model(model=model_name, temperature=0.0)

    middleware = []

    # Add memory support if owner/repo provided
    if owner and repo:
        backend = DeepAgentsBackend(owner, repo)
        memory_path = get_memory_path(owner, repo)

        # MemoryMiddleware loads AGENTS.md and injects learning guidelines
        middleware.append(
            MemoryMiddleware(
                backend=backend,
                sources=[memory_path],
            )
        )

        # FilesystemMiddleware provides edit_file, read_file, write_file tools
        middleware.append(
            FilesystemMiddleware(
                backend=backend,
            )
        )

    return create_deep_agent(
        model=model,
        tools=CHAT_TOOLS,
        middleware=middleware if middleware else None,
        system_prompt=CHAT_PROMPT,
    )


def create_feedback_agent(model_name: str = "anthropic:claude-sonnet-4-5"):
    """Create a deep agent for applying feedback and resolving conflicts."""
    model = init_chat_model(model=model_name, temperature=0.0)
    return create_deep_agent(
        model=model,
        tools=FEEDBACK_TOOLS,
        system_prompt=None,  # Prompt provided dynamically based on task
    )


# Prompt for /remember command - guides the chat agent to update AGENTS.md
# Follows the pattern from libs/cli/deepagents_cli/app.py
REMEMBER_PROMPT = """A repository maintainer wants to save a convention/preference to this repository's memory.

**Memory file location:** {memory_path}

**Maintainer:** @{username}

**Convention to remember:**
{memory_content}

## Instructions

1. First, read the current memory file using `read_file` to understand its structure
2. Add the new convention to the appropriate section
3. Use `edit_file` to update the file (or `write_file` if it doesn't exist)
4. Organize content logically - group related items together
5. Preserve existing content - don't remove anything unless clearly obsolete
6. Use clear markdown formatting

## Memory File Format

If creating a new file, use this structure:
```markdown
# Repository Conventions for {owner}/{repo}

## Code Style
- [conventions about code formatting, naming, etc.]

## Review Guidelines
- [conventions about how reviews should be done]

## Project-Specific
- [project-specific conventions]
```

When adding to an existing file, find the most appropriate section or create a new one.

After updating, briefly confirm what was saved."""
