#!/usr/bin/env python3
"""Test script for arXiv tools.

Usage:
    python -m mcp_tools.examples.external.test_arxiv

Or directly:
    python test_arxiv.py

Note:
    Requires arxiv-mcp-server to be installed:
    $ uv tool install arxiv-mcp-server
"""

from __future__ import annotations

import sys


def test_arxiv_tools():
    """Test arXiv tools functionality."""
    from mcp_tools import (
        arxiv_download_paper,
        arxiv_list_papers,
        arxiv_read_paper,
        arxiv_search,
    )

    print("=" * 60)
    print("Testing arXiv Tools")
    print("=" * 60)

    # Test 1: Search
    print("\n[Test 1] arXiv search")
    print("-" * 40)

    search_result = arxiv_search(
        query="transformer attention mechanism",
        max_results=3,
        categories=["cs.AI", "cs.LG"]
    )

    print(f"Success: {search_result['success']}")

    if search_result["success"]:
        print(f"Papers found: {search_result['num_papers']}")
        for i, paper in enumerate(search_result["papers"], 1):
            print(f"\n--- Paper {i} ---")
            print(f"ID: {paper.get('paper_id', 'N/A')}")
            print(f"Title: {paper.get('title', 'N/A')}")
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                authors = ", ".join(authors[:3])
                if len(paper.get('authors', [])) > 3:
                    authors += " et al."
            print(f"Authors: {authors}")
            summary = paper.get('summary', '')
            print(f"Summary: {summary[:200]}..." if len(summary) > 200 else f"Summary: {summary}")
    else:
        print(f"Error: {search_result.get('error', 'Unknown error')}")

    # Test 2: List downloaded papers
    print("\n[Test 2] List downloaded papers")
    print("-" * 40)

    list_result = arxiv_list_papers()

    print(f"Success: {list_result['success']}")

    if list_result["success"]:
        print(f"Downloaded papers: {list_result['num_papers']}")
        for paper in list_result.get("papers", []):
            if isinstance(paper, dict):
                print(f"  - {paper.get('paper_id', 'N/A')}: {paper.get('title', 'N/A')}")
            else:
                print(f"  - {paper}")
    else:
        print(f"Error: {list_result.get('error', 'Unknown error')}")

    # Test 3: Download and read (if search found papers)
    if search_result["success"] and search_result["papers"]:
        paper_id = search_result["papers"][0].get("paper_id", "")

        if paper_id:
            print(f"\n[Test 3] Download paper: {paper_id}")
            print("-" * 40)

            download_result = arxiv_download_paper(paper_id)

            print(f"Success: {download_result['success']}")
            if download_result["success"]:
                print(f"Message: {download_result.get('message', '')}")
                print(f"Storage: {download_result.get('storage_path', '')}")

                # Test 4: Read the paper
                print(f"\n[Test 4] Read paper: {paper_id}")
                print("-" * 40)

                read_result = arxiv_read_paper(paper_id)

                print(f"Success: {read_result['success']}")
                if read_result["success"]:
                    print(f"Title: {read_result.get('title', 'N/A')}")
                    content = read_result.get("content", "")
                    print(f"Content: {content[:500]}..." if len(content) > 500 else f"Content: {content}")
                else:
                    print(f"Error: {read_result.get('error', 'Unknown error')}")
            else:
                print(f"Error: {download_result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("arXiv Tools Test Complete")
    print("=" * 60)

    return search_result["success"]


if __name__ == "__main__":
    success = test_arxiv_tools()
    sys.exit(0 if success else 1)
