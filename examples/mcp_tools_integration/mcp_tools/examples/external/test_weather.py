#!/usr/bin/env python3
"""Test script for Weather Forecast tool.

Usage:
    python -m mcp_tools.examples.external.test_weather

Or directly:
    python test_weather.py

Note:
    Requires OPENWEATHER_API_KEY environment variable to be set.
"""

from __future__ import annotations

import os
import sys


def test_weather_forecast():
    """Test weather forecast functionality."""
    from mcp_tools import weather_forecast

    print("=" * 60)
    print("Testing Weather Forecast Tool")
    print("=" * 60)

    # Check API key
    if not os.environ.get("OPENWEATHER_API_KEY"):
        print("\n[WARNING] OPENWEATHER_API_KEY not set")
        print("Set the environment variable and try again.")

    # Test 1: Korean city (Korean name)
    print("\n[Test 1] Weather for 서울 (Korean name)")
    print("-" * 40)

    result = weather_forecast("서울", units="c", lang="kr")

    print(f"Success: {result['success']}")
    print(f"City: {result['city']}")

    if result["success"]:
        print(f"Forecast: {result['forecast']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # Test 2: English city name
    print("\n[Test 2] Weather for Seoul (English name)")
    print("-" * 40)

    result2 = weather_forecast("Seoul", units="c", lang="en")

    print(f"Success: {result2['success']}")
    print(f"City: {result2['city']}")

    if result2["success"]:
        print(f"Forecast: {result2['forecast']}")
    else:
        print(f"Error: {result2.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("Weather Forecast Test Complete")
    print("=" * 60)

    return result["success"] or result2["success"]


if __name__ == "__main__":
    success = test_weather_forecast()
    sys.exit(0 if success else 1)
