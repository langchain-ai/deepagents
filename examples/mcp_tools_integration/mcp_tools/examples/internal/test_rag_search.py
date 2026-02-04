#!/usr/bin/env python3
"""Test script for RAG search tool.

Usage:
    python -m mcp_tools.examples.internal.test_rag_search

Or directly:
    python test_rag_search.py
"""

from __future__ import annotations

import json
import sys


def test_rag_search():
    """Test RAG search functionality."""
    from mcp_tools import rag_search

    print("=" * 60)
    print("Testing RAG Search Tool")
    print("=" * 60)

    # Test 1: Basic search
    print("\n[Test 1] Basic RAG search")
    print("-" * 40)

    query = "SpaceOps 논문 중 SCIENCE GOAL DRIVEN OBSERVING"
    print(f"Query: {query}")

    result = rag_search(query, k=3)

    print(f"\nSuccess: {result['success']}")

    if result["success"]:
        print(f"Documents found: {result['num_documents']}")
        for i, doc in enumerate(result["documents"], 1):
            print(f"\n--- Document {i} ---")
            content = doc.get("content", "")
            print(f"Content: {content[:300]}..." if len(content) > 300 else f"Content: {content}")
            print(f"Metadata: {doc.get('metadata', {})}")
            print(f"Rank: {doc.get('rank', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("RAG Search Test Complete")
    print("=" * 60)

    return result["success"]


if __name__ == "__main__":
    success = test_rag_search()
    sys.exit(0 if success else 1)
