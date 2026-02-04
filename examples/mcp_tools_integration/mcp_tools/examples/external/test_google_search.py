#!/usr/bin/env python3
"""Test script for Google Search tool.

Usage:
    python -m mcp_tools.examples.external.test_google_search

Or directly:
    python test_google_search.py
"""

from __future__ import annotations

import sys


def test_google_search():
    """Test Google search functionality."""
    from mcp_tools import google_search_and_summarize

    print("=" * 60)
    print("Testing Google Search Tool")
    print("=" * 60)

    # Test 1: Basic search
    print("\n[Test 1] Basic Google search")
    print("-" * 40)

    query = "Python asyncio tutorial"
    print(f"Query: {query}")

    result = google_search_and_summarize(
        query=query,
        num_results=3,
        fetch_top_n=2,
        max_chars_per_page=1000
    )

    print(f"\nSuccess: {result['success']}")

    if result["success"]:
        print(f"Sources found: {result['num_sources']}")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n--- Source {i} ---")
            print(f"Title: {source.get('title', 'N/A')}")
            print(f"URL: {source.get('url', 'N/A')}")
            print(f"Snippet: {source.get('snippet', 'N/A')[:200]}...")
            text = source.get("text", "")
            if text:
                print(f"Content: {text[:300]}...")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("Google Search Test Complete")
    print("=" * 60)

    return result["success"]


if __name__ == "__main__":
    success = test_google_search()
    sys.exit(0 if success else 1)
