"""ACP-specific prompt fragments."""

from __future__ import annotations

_MODE_PROMPTS: dict[str, tuple[str, str]] = {
    "plan": (
        "ACP Planning Mode",
        (
            "You are in planning mode. Inspect and reason as needed, then present "
            "a concrete plan and wait for approval before editing files, running "
            "shell commands, or making other changes."
        ),
    ),
    "ask": (
        "ACP Ask Mode",
        (
            "You are in ask mode. Answer questions and discuss the codebase, but "
            "do not edit files, run shell commands, or make other workspace "
            "changes."
        ),
    ),
}


def append_acp_mode_prompt(base_prompt: str, mode: str) -> str:
    """Append ACP mode-specific instructions to a base system prompt.

    Args:
        base_prompt: Existing system prompt generated for the agent.
        mode: ACP session mode identifier.

    Returns:
        System prompt with ACP mode instructions appended.

    Raises:
        ValueError: If `mode` has no ACP-specific prompt fragment.
    """
    fragment = _MODE_PROMPTS.get(mode)
    if fragment is None:
        msg = f"ACP mode {mode!r} has no prompt fragment"
        raise ValueError(msg)
    heading, instruction = fragment
    return f"{base_prompt}\n\n### {heading}\n\n{instruction}"
