"""arXiv Paper Read tool.

This tool reads the content of downloaded arXiv papers.
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import ARXIV_STORAGE_PATH, extract_payload, run_async


async def _arxiv_read_paper_async(paper_id: str) -> dict[str, Any]:
    """Async implementation of reading downloaded arXiv paper."""
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

            read_tool = next(
                (t for t in tools if "read" in t.name.lower()),
                None
            )

            if read_tool is None:
                return {
                    "success": False,
                    "paper_id": paper_id,
                    "error": f"No read tool found. Available tools: {[t.name for t in tools]}",
                    "content": "",
                }

            raw = await read_tool.ainvoke({"paper_id": paper_id})
            payload = extract_payload(raw)

        content = payload.get("content", payload.get("text", ""))
        if isinstance(content, str):
            pass
        elif isinstance(content, dict):
            content = content.get("text", str(content))
        else:
            content = str(content)

        return {
            "success": True,
            "paper_id": paper_id,
            "title": payload.get("title", ""),
            "content": content,
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id,
            "error": f"Read failed: {type(e).__name__}: {str(e)}",
            "content": "",
        }


def arxiv_read_paper(paper_id: str) -> dict[str, Any]:
    """Read the content of a downloaded arXiv paper.

    Returns the extracted text content of a previously downloaded paper.
    You must download the paper first using arxiv_download_paper().

    Args:
        paper_id: arXiv paper ID (e.g., "2301.00001")

    Returns:
        Dictionary containing:
        - success: Whether reading succeeded
        - paper_id: The paper ID
        - title: Paper title
        - content: Extracted text content of the paper

    Example:
        # First download
        arxiv_download_paper("2301.00001")

        # Then read
        result = arxiv_read_paper("2301.00001")
        if result["success"]:
            print(f"Title: {result['title']}")
            print(f"Content: {result['content'][:1000]}...")
    """
    return run_async(_arxiv_read_paper_async(paper_id))
