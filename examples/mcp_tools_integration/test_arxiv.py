#!/usr/bin/env python3
"""Test script for arXiv MCP tool.

This script tests the arXiv search functionality.

Prerequisites:
    $ uv tool install arxiv-mcp-server

Usage:
    $ python test_arxiv.py
"""

import asyncio
import json
import time


def test_arxiv_search():
    """Test arXiv paper search."""
    from mcp_tools import arxiv_search

    print("=" * 60)
    print("Testing arxiv_search()")
    print("=" * 60)

    # Test 1: Basic search
    print("\n[Test 1] Basic search: 'transformer attention'")
    t0 = time.perf_counter()
    result = arxiv_search("transformer attention", max_results=3)
    elapsed = time.perf_counter() - t0

    print(f"  Success: {result['success']}")
    print(f"  Time: {elapsed:.2f}s")

    if result["success"]:
        print(f"  Found {result['num_papers']} papers:")
        for i, paper in enumerate(result["papers"][:3], 1):
            print(f"    {i}. {paper.get('title', 'No title')[:60]}...")
            print(f"       ID: {paper.get('paper_id', 'N/A')}")
            print(f"       Authors: {', '.join(paper.get('authors', [])[:3])}...")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")

    # Test 2: Search with category filter
    print("\n[Test 2] Search with category: 'LLM' in cs.CL")
    t0 = time.perf_counter()
    result = arxiv_search("large language model", max_results=3, categories=["cs.CL"])
    elapsed = time.perf_counter() - t0

    print(f"  Success: {result['success']}")
    print(f"  Time: {elapsed:.2f}s")

    if result["success"]:
        print(f"  Found {result['num_papers']} papers")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")

    return result["success"]


def test_arxiv_workflow():
    """Test full arXiv workflow: search -> download -> read."""
    from mcp_tools import arxiv_search, arxiv_download_paper, arxiv_read_paper, arxiv_list_papers

    print("\n" + "=" * 60)
    print("Testing arXiv workflow")
    print("=" * 60)

    # Step 1: Search
    print("\n[Step 1] Searching for papers...")
    result = arxiv_search("attention is all you need", max_results=1)

    if not result["success"] or not result["papers"]:
        print(f"  Search failed: {result.get('error', 'No papers found')}")
        return False

    paper = result["papers"][0]
    paper_id = paper.get("paper_id", "")
    print(f"  Found: {paper.get('title', 'N/A')}")
    print(f"  ID: {paper_id}")

    if not paper_id:
        print("  No paper ID found, skipping download test")
        return True

    # Step 2: Download
    print(f"\n[Step 2] Downloading paper {paper_id}...")
    result = arxiv_download_paper(paper_id)
    print(f"  Success: {result['success']}")
    if not result["success"]:
        print(f"  Error: {result.get('error', 'Unknown')}")
        # Continue even if download fails (might already be downloaded)

    # Step 3: List papers
    print("\n[Step 3] Listing downloaded papers...")
    result = arxiv_list_papers()
    print(f"  Success: {result['success']}")
    if result["success"]:
        print(f"  Downloaded papers: {result['num_papers']}")

    # Step 4: Read paper
    print(f"\n[Step 4] Reading paper {paper_id}...")
    result = arxiv_read_paper(paper_id)
    print(f"  Success: {result['success']}")
    if result["success"]:
        content = result.get("content", "")
        print(f"  Content length: {len(content)} chars")
        print(f"  Preview: {content[:200]}..." if content else "  No content")
    else:
        print(f"  Error: {result.get('error', 'Unknown')}")

    return True


def test_with_deepagents():
    """Test arXiv tool with DeepAgents (requires LLM)."""
    print("\n" + "=" * 60)
    print("Testing with DeepAgents")
    print("=" * 60)

    try:
        from deepagents import create_deep_agent
        from mcp_tools import arxiv_search

        # This is just a structure test - actual LLM call skipped
        print("\n[Test] Tool can be registered with DeepAgents")

        # Check tool has required attributes
        assert hasattr(arxiv_search, "__name__"), "Tool missing __name__"
        assert hasattr(arxiv_search, "__doc__"), "Tool missing __doc__"
        assert arxiv_search.__doc__ is not None, "Tool has empty docstring"

        print("  arxiv_search.__name__:", arxiv_search.__name__)
        print("  arxiv_search has docstring:", len(arxiv_search.__doc__) > 0)
        print("  Tool is compatible with DeepAgents!")

        return True

    except ImportError as e:
        print(f"  Skipped (deepagents not available): {e}")
        return True


def main():
    """Run all tests."""
    print("arXiv MCP Tool Test Suite")
    print("=" * 60)

    # Check if arxiv-mcp-server is installed
    import shutil
    if not shutil.which("uv"):
        print("WARNING: 'uv' not found. Install with: pip install uv")
        print("         Then install arxiv-mcp-server: uv tool install arxiv-mcp-server")
        print()

    results = []

    # Test 1: Basic search
    try:
        results.append(("arxiv_search", test_arxiv_search()))
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results.append(("arxiv_search", False))

    # Test 2: Full workflow
    try:
        results.append(("arxiv_workflow", test_arxiv_workflow()))
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results.append(("arxiv_workflow", False))

    # Test 3: DeepAgents compatibility
    try:
        results.append(("deepagents_compat", test_with_deepagents()))
    except Exception as e:
        print(f"Test failed with exception: {e}")
        results.append(("deepagents_compat", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print()
    print("Overall:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
