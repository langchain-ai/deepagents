"""arXiv Paper List tool.

This tool lists all downloaded arXiv papers.
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import ARXIV_STORAGE_PATH, extract_payload, run_async


async def _arxiv_list_papers_async() -> dict[str, Any]:
    """Async implementation of listing downloaded arXiv papers."""
    try:
        client = MultiServerMCPClient({
            "arxiv": {
                "transport": "stdio",
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", ARXIV_STORAGE_PATH],
            }
        })

        async with client.session("arxiv") as session:
            tools = await load_mcp_tools(session)

            list_tool = next(
                (t for t in tools if "list" in t.name.lower()),
                None
            )

            if list_tool is None:
                return {
                    "success": False,
                    "error": f"No list tool found. Available tools: {[t.name for t in tools]}",
                    "papers": [],
                }

            raw = await list_tool.ainvoke({})
            payload = extract_payload(raw)

        papers = payload.get("papers", payload.get("items", []))

        return {
            "success": True,
            "num_papers": len(papers) if isinstance(papers, list) else 0,
            "papers": papers,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"List failed: {type(e).__name__}: {str(e)}",
            "papers": [],
        }


def arxiv_list_papers() -> dict[str, Any]:
    """List all downloaded arXiv papers.

    Returns a list of papers that have been downloaded and are available
    for reading locally.

    Returns:
        Dictionary containing:
        - success: Whether the operation succeeded
        - num_papers: Number of downloaded papers
        - papers: List of paper info (id, title, etc.)

    Example:
        result = arxiv_list_papers()
        for paper in result["papers"]:
            print(f"- {paper['paper_id']}: {paper['title']}")
    """
    return run_async(_arxiv_list_papers_async())
