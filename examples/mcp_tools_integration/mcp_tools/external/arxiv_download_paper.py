"""arXiv Paper Download tool.

This tool downloads papers from arXiv.
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import ARXIV_STORAGE_PATH, extract_payload, run_async


async def _arxiv_download_paper_async(paper_id: str) -> dict[str, Any]:
    """Async implementation of arXiv paper download."""
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

            download_tool = next(
                (t for t in tools if "download" in t.name.lower()),
                None
            )

            if download_tool is None:
                return {
                    "success": False,
                    "paper_id": paper_id,
                    "error": f"No download tool found. Available tools: {[t.name for t in tools]}",
                }

            raw = await download_tool.ainvoke({"paper_id": paper_id})
            payload = extract_payload(raw)

        return {
            "success": True,
            "paper_id": paper_id,
            "message": payload.get("message", "Paper downloaded successfully"),
            "storage_path": payload.get("path", ARXIV_STORAGE_PATH),
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id,
            "error": f"Download failed: {type(e).__name__}: {str(e)}",
        }


def arxiv_download_paper(paper_id: str) -> dict[str, Any]:
    """Download an arXiv paper by its ID.

    Downloads the PDF and extracts text content for later reading.

    Args:
        paper_id: arXiv paper ID (e.g., "2301.00001" or "2301.00001v2")

    Returns:
        Dictionary containing:
        - success: Whether the download succeeded
        - paper_id: The paper ID
        - message: Status message
        - storage_path: Where the paper is stored

    Example:
        result = arxiv_download_paper("2301.00001")
        if result["success"]:
            print(f"Downloaded to: {result['storage_path']}")
    """
    return run_async(_arxiv_download_paper_async(paper_id))
