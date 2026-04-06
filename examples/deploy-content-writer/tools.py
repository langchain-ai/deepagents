"""Custom tools for the content writer agent."""

from langchain_core.tools import tool


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text.

    Args:
        text: The text to count words in.

    Returns:
        Number of words.
    """
    return len(text.split())


@tool
def reading_time(text: str) -> str:
    """Estimate the reading time for a piece of text.

    Assumes an average reading speed of 200 words per minute.

    Args:
        text: The text to estimate reading time for.

    Returns:
        Human-readable reading time estimate.
    """
    words = len(text.split())
    minutes = max(1, round(words / 200))
    return f"{minutes} min read ({words} words)"
