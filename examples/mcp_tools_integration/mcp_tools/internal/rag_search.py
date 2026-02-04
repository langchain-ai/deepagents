"""RAG (Retrieval-Augmented Generation) search tool.

This tool searches a local vector store for relevant documents.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import RAG_SERVER, extract_payload, run_async


async def _rag_search_async(
    query: str,
    k: int = 4,
    vectorstore_dir: str = "vectorstore.db",
) -> dict[str, Any]:
    """Async implementation of RAG search."""
    if not os.path.exists(RAG_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"RAG MCP server not found: {RAG_SERVER}. "
                     f"Set MCP_RAG_SERVER environment variable.",
            "documents": [],
        }

    try:
        client = MultiServerMCPClient({
            "local_rag": {
                "transport": "stdio",
                "command": "python",
                "args": [RAG_SERVER],
                "env": {"RAG_VECTORSTORE_DIR": vectorstore_dir},
            }
        })

        async with client.session("local_rag") as session:
            tools = await load_mcp_tools(session)
            retrieve_tool = next(
                (t for t in tools if "rag" in t.name.lower() or "retrieve" in t.name.lower()),
                tools[0] if tools else None
            )

            if retrieve_tool is None:
                return {
                    "success": False,
                    "query": query,
                    "error": "No RAG retrieve tool found",
                    "documents": [],
                }

            raw = await retrieve_tool.ainvoke({"query": query, "k": k})
            payload = extract_payload(raw)

        sources = payload.get("sources", [])
        documents = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            documents.append({
                "content": s.get("content", ""),
                "metadata": s.get("metadata", {}),
                "rank": s.get("rank", 0),
            })

        return {
            "success": True,
            "query": query,
            "k": k,
            "num_documents": len(documents),
            "documents": documents,
        }

    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "documents": [],
        }


def rag_search(
    query: str,
    k: int = 4,
    vectorstore_dir: str = "vectorstore.db",
) -> dict[str, Any]:
    """Search documents using RAG (Retrieval-Augmented Generation).

    This tool searches a local vector store for relevant documents based on
    semantic similarity. Use this for searching papers, research documents,
    technical documentation, or any pre-indexed content.

    Args:
        query: The search query (natural language question or keywords)
        k: Number of top documents to retrieve (default: 4)
        vectorstore_dir: Path to the vector store directory (default: "vectorstore.db")

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - k: Number of documents requested
        - num_documents: Number of documents actually retrieved
        - documents: List of document dictionaries, each with:
            - content: Document text content
            - metadata: Document metadata (source, page, etc.)
            - rank: Relevance ranking

    Example:
        result = rag_search("SpaceOps 논문 중 SCIENCE GOAL DRIVEN OBSERVING")
        for doc in result["documents"]:
            print(f"Content: {doc['content'][:300]}...")
            print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
    """
    return run_async(_rag_search_async(
        query=query,
        k=k,
        vectorstore_dir=vectorstore_dir,
    ))
