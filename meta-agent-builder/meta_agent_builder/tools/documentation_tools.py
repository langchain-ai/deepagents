"""Tools for documentation research and analysis."""

import os
from typing import Literal

from langchain_core.tools import tool
from tavily import TavilyClient

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for documentation, articles, and information.

    Use this tool to find:
    - Official documentation
    - Technical articles
    - Code examples
    - Best practices
    - Framework capabilities

    Args:
        query: Search query (be specific for better results)
        max_results: Number of results to return (1-10)
        topic: Type of search ('general' or 'news')

    Returns:
        Dictionary containing search results with URLs, titles, and content

    Example:
        >>> results = internet_search("LangChain SubAgentMiddleware documentation")
        >>> for result in results['results']:
        ...     print(result['title'], result['url'])
    """
    try:
        results = tavily_client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
        )
        return results
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "message": "Check TAVILY_API_KEY environment variable",
        }


@tool
def extract_code_examples(text: str, language: str = "python") -> list[str]:
    """Extract code blocks from markdown or documentation text.

    Args:
        text: Text containing code blocks
        language: Programming language to extract (default: python)

    Returns:
        List of code blocks found in the text

    Example:
        >>> doc_text = "```python\\nprint('hello')\\n```"
        >>> examples = extract_code_examples(doc_text)
        >>> print(examples[0])
        print('hello')
    """
    import re

    # Pattern for markdown code blocks
    pattern = rf"```{language}\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    # Also try without language specifier
    if not matches:
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

    return [match.strip() for match in matches]


@tool
def summarize_documentation(text: str, max_length: int = 500) -> str:
    """Create a concise summary of documentation text.

    Args:
        text: Documentation text to summarize
        max_length: Maximum length of summary in characters

    Returns:
        Summarized text focusing on key points

    Example:
        >>> long_doc = "..."  # Long documentation text
        >>> summary = summarize_documentation(long_doc, max_length=200)
    """
    # Simple summarization: extract first paragraph and key points
    paragraphs = text.split("\n\n")

    # Get first substantial paragraph
    summary_parts = []
    for para in paragraphs[:3]:
        if len(para.strip()) > 50:
            summary_parts.append(para.strip())
            if len(" ".join(summary_parts)) > max_length:
                break

    summary = " ".join(summary_parts)

    # Truncate if still too long
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

    return summary
