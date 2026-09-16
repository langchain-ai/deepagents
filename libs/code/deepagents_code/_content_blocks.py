"""Shared reading of provider content blocks.

The three surfaces that render a streamed or replayed `AIMessage` -- the
transcript projection in `deepagents_code.app`, the interactive TUI
(`deepagents_code.tui.textual_adapter`), and the headless runner
(`deepagents_code.client.non_interactive`) -- all have to decide which blocks
carry reasoning. Keeping that decision here stops the three from drifting when
the block schema grows a new shape.
"""

from __future__ import annotations


def reasoning_text(block: object) -> str | None:
    """Extract renderable reasoning text from a content block.

    Args:
        block: One entry of `AIMessage.content_blocks`. Typed loosely because
            the three surfaces reach it through differently typed streams.

    Returns:
        The block's reasoning text, or `None` when the block is not reasoning
        or carries nothing worth rendering.
    """
    if not isinstance(block, dict) or block.get("type") != "reasoning":
        return None
    text = block.get("reasoning")
    if not isinstance(text, str) or not text:
        return None
    return text
