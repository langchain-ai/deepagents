"""Utilities for accurate token counting using LangChain models."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from deepagents_cli.config import console, settings


def calculate_baseline_tokens(model, agent_dir: Path, system_prompt: str, assistant_id: str) -> int:
    """Calculate baseline context tokens using the model's official tokenizer.

    This uses the model's get_num_tokens_from_messages() method to get
    accurate token counts for the initial context (system prompt + agent.md).

    Note: Tool definitions cannot be accurately counted before the first API call
    due to LangChain limitations. They will be included in the total after the
    first message is sent (~5,000 tokens).

    Args:
        model: LangChain model instance (ChatAnthropic or ChatOpenAI)
        agent_dir: Path to agent directory containing agent.md
        system_prompt: The base system prompt string
        assistant_id: The agent identifier for path references

    Returns:
        Token count for system prompt + agent.md (tools not included)
    """
    # Load user agent.md content
    agent_md_path = agent_dir / "agent.md"
    user_memory = ""
    if agent_md_path.exists():
        user_memory = agent_md_path.read_text()

    # Load project agent.md content
    from .config import _find_project_agent_md, _find_project_root

    project_memory = ""
    project_root = _find_project_root()
    if project_root:
        project_md_paths = _find_project_agent_md(project_root)
        if project_md_paths:
            try:
                # Combine all project agent.md files (if multiple exist)
                contents = []
                for path in project_md_paths:
                    contents.append(path.read_text())
                project_memory = "\n\n".join(contents)
            except Exception:
                pass

    # Build the complete system prompt as it will be sent
    # This mimics what AgentMemoryMiddleware.wrap_model_call() does
    memory_section = (
        f"<user_memory>\n{user_memory or '(No user agent.md)'}\n</user_memory>\n\n"
        f"<project_memory>\n{project_memory or '(No project agent.md)'}\n</project_memory>"
    )

    # Get the long-term memory system prompt
    memory_system_prompt = get_memory_system_prompt(
        assistant_id, project_root, bool(project_memory)
    )

    # Combine all parts in the same order as the middleware
    full_system_prompt = memory_section + "\n\n" + system_prompt + "\n\n" + memory_system_prompt

    # Count tokens using the model's official method
    messages = [SystemMessage(content=full_system_prompt)]

    try:
        # Note: tools parameter is not supported by LangChain's token counting
        # Tool tokens will be included in the API response after first message
        return model.get_num_tokens_from_messages(messages)
    except Exception as e:
        # Fallback if token counting fails
        console.print(f"[yellow]Warning: Could not calculate baseline tokens: {e}[/yellow]")
        return 0


def get_memory_system_prompt(
    assistant_id: str, project_root: Path | None = None, has_project_memory: bool = False
) -> str:
    """Get the long-term memory system prompt text.

    Args:
        assistant_id: The agent identifier for path references
        project_root: Path to the detected project root (if any)
        has_project_memory: Whether project memory was loaded
    """
    # Import from agent_memory middleware
    from .agent_memory import LONGTERM_MEMORY_SYSTEM_PROMPT

    agent_dir = settings.get_agent_dir(assistant_id)
    agent_dir_absolute = str(agent_dir)
    agent_dir_display = f"~/.deepagents/{assistant_id}"

    # Build project memory info
    if project_root and has_project_memory:
        project_memory_info = f"`{project_root}` (detected)"
    elif project_root:
        project_memory_info = f"`{project_root}` (no agent.md found)"
    else:
        project_memory_info = "None (not in a git project)"

    # Build project deepagents directory path
    if project_root:
        project_deepagents_dir = f"{project_root}/.deepagents"
    else:
        project_deepagents_dir = "[project-root]/.deepagents (not in a project)"

    return LONGTERM_MEMORY_SYSTEM_PROMPT.format(
        agent_dir_absolute=agent_dir_absolute,
        agent_dir_display=agent_dir_display,
        project_memory_info=project_memory_info,
        project_deepagents_dir=project_deepagents_dir,
    )
