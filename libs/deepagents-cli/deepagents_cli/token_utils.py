"""Utilities for accurate token counting using LangChain models."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from .config import console


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
    # Load agent.md content
    agent_md_path = agent_dir / "agent.md"
    agent_memory = ""
    if agent_md_path.exists():
        agent_memory = agent_md_path.read_text()

    # Build the complete system prompt as it will be sent
    # This mimics what AgentMemoryMiddleware.wrap_model_call() does
    memory_section = f"<agent_memory>\n{agent_memory}\n</agent_memory>"

    # Get the long-term memory system prompt
    memory_system_prompt = get_memory_system_prompt(assistant_id)

    # Combine all parts in the same order as the middleware
    full_system_prompt = memory_section + "\n\n" + system_prompt + "\n\n" + memory_system_prompt

    # Count tokens using the model's official method
    messages = [SystemMessage(content=full_system_prompt)]

    try:
        # Note: tools parameter is not supported by LangChain's token counting
        # Tool tokens will be included in the API response after first message
        token_count = model.get_num_tokens_from_messages(messages)
        return token_count
    except Exception as e:
        # Fallback if token counting fails
        console.print(f"[yellow]Warning: Could not calculate baseline tokens: {e}[/yellow]")
        return 0


def get_memory_system_prompt(assistant_id: str) -> str:
    """Get the long-term memory system prompt text.

    Args:
        assistant_id: The agent identifier for path references
    """
    # Import from agent_memory middleware
    from pathlib import Path

    from .agent_memory import LONGTERM_MEMORY_SYSTEM_PROMPT

    agent_dir = Path.home() / ".deepagents" / assistant_id
    agent_dir_absolute = str(agent_dir)
    agent_dir_display = f"~/.deepagents/{assistant_id}"
    return LONGTERM_MEMORY_SYSTEM_PROMPT.format(
        agent_dir_absolute=agent_dir_absolute, agent_dir_display=agent_dir_display
    )
