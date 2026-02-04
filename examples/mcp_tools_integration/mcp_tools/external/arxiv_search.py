"""arXiv Paper Search tool.

This tool searches for papers on arXiv preprint repository.
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import ARXIV_STORAGE_PATH, extract_payload, run_async


async def _arxiv_search_async(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Async implementation of arXiv paper search."""
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

            search_tool = next(
                (t for t in tools if "search" in t.name.lower()),
                None
            )

            if search_tool is None:
                return {
                    "success": False,
                    "query": query,
                    "error": f"No search tool found. Available tools: {[t.name for t in tools]}",
                    "papers": [],
                }

            params = {
                "query": query,
                "max_results": max_results,
            }
            if date_from:
                params["date_from"] = date_from
            if categories:
                params["categories"] = categories

            raw = await search_tool.ainvoke(params)
            payload = extract_payload(raw)

        papers = payload.get("papers", payload.get("results", []))
        if isinstance(papers, list):
            paper_list = []
            for p in papers:
                if isinstance(p, dict):
                    paper_list.append({
                        "paper_id": p.get("id", p.get("paper_id", "")),
                        "title": p.get("title", ""),
                        "authors": p.get("authors", []),
                        "summary": p.get("summary", p.get("abstract", "")),
                        "published": p.get("published", ""),
                        "updated": p.get("updated", ""),
                        "categories": p.get("categories", []),
                        "pdf_url": p.get("pdf_url", ""),
                    })
            papers = paper_list

        return {
            "success": True,
            "query": query,
            "num_papers": len(papers) if isinstance(papers, list) else 0,
            "papers": papers,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "query": query,
            "error": "arXiv MCP server not installed. Install with: uv tool install arxiv-mcp-server",
            "papers": [],
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"arXiv search failed: {type(e).__name__}: {str(e)}",
            "papers": [],
        }


def arxiv_search(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Search for papers on arXiv.

    This tool searches the arXiv preprint repository for academic papers.
    Use this for finding research papers, preprints, and scientific literature.

    Args:
        query: Search query - can include:
               - Keywords: "transformer attention mechanism"
               - Author: "au:Vaswani"
               - Title: "ti:attention is all you need"
               - Abstract: "abs:deep learning"
               - arXiv ID: "2301.00001"
        max_results: Maximum number of papers to return (default: 10, max: 100)
        date_from: Filter papers from this date (format: "YYYY-MM-DD")
        categories: List of arXiv categories to filter by, e.g.:
                   - "cs.AI" (Artificial Intelligence)
                   - "cs.LG" (Machine Learning)
                   - "cs.CL" (Computation and Language)
                   - "cs.CV" (Computer Vision)
                   - "stat.ML" (Statistics - Machine Learning)
                   - "physics.space-ph" (Space Physics)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - num_papers: Number of papers found
        - papers: List of paper dictionaries, each with:
            - paper_id: arXiv paper ID (e.g., "2301.00001")
            - title: Paper title
            - authors: List of author names
            - summary: Paper abstract
            - published: Publication date
            - updated: Last update date
            - categories: List of arXiv categories
            - pdf_url: URL to download PDF

    Example:
        # Search for transformer papers in AI category
        result = arxiv_search(
            query="transformer neural network",
            max_results=5,
            categories=["cs.AI", "cs.LG"]
        )
        for paper in result["papers"]:
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Abstract: {paper['summary'][:200]}...")

    Note:
        Requires arxiv-mcp-server to be installed:
        $ uv tool install arxiv-mcp-server
    """
    return run_async(_arxiv_search_async(
        query=query,
        max_results=max_results,
        date_from=date_from,
        categories=categories,
    ))
