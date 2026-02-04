#!/usr/bin/env python3
"""Test script for Sentinel Search tool.

Usage:
    python -m mcp_tools.examples.external.test_sentinel

Or directly:
    python test_sentinel.py
"""

from __future__ import annotations

import sys


def test_sentinel_search():
    """Test Sentinel satellite search functionality."""
    from mcp_tools import sentinel_search

    print("=" * 60)
    print("Testing Sentinel Search Tool")
    print("=" * 60)

    # Test 1: Basic search
    print("\n[Test 1] Sentinel-2 search for Daejeon")
    print("-" * 40)

    result = sentinel_search(
        query="대전 유성구 위성 영상",
        sensor="S2",
        aoi="Daejeon",
        date_range="2024-01-01/2024-01-31",
        cloud_cover_max=30
    )

    print(f"Success: {result['success']}")
    print(f"Query: {result['query']}")

    if result["success"]:
        print(f"Results: {result['results']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # Test 2: SAR search (no cloud cover constraint)
    print("\n[Test 2] Sentinel-1 SAR search")
    print("-" * 40)

    result2 = sentinel_search(
        query="Seoul SAR imagery",
        sensor="S1",
        aoi="Seoul",
        date_range="2024-01-01/2024-01-15"
    )

    print(f"Success: {result2['success']}")

    if result2["success"]:
        print(f"Results: {result2['results']}")
    else:
        print(f"Error: {result2.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("Sentinel Search Test Complete")
    print("=" * 60)

    return result["success"] or result2["success"]


if __name__ == "__main__":
    success = test_sentinel_search()
    sys.exit(0 if success else 1)
