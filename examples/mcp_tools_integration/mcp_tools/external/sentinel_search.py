"""Sentinel Satellite Search tool.

This tool searches for Sentinel satellite imagery from ESA/Copernicus.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import SENTINEL_SERVER, extract_payload, run_async


async def _sentinel_search_async(
    query: str,
    sensor: str | None = None,
    aoi: str | None = None,
    date_range: str | None = None,
    cloud_cover_max: int | None = None,
) -> dict[str, Any]:
    """Async implementation of Sentinel satellite search."""
    if not os.path.exists(SENTINEL_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"Sentinel MCP server not found: {SENTINEL_SERVER}. "
                     f"Set MCP_SENTINEL_SERVER environment variable.",
            "results": {},
        }

    try:
        client = MultiServerMCPClient({
            "sentinel": {
                "transport": "stdio",
                "command": "python",
                "args": [SENTINEL_SERVER],
            }
        })

        async with client.session("sentinel") as session:
            tools = await load_mcp_tools(session)

            search_tool = None
            for t in tools:
                if "sentinel" in t.name.lower() or "search" in t.name.lower() or "query" in t.name.lower():
                    search_tool = t
                    break

            if search_tool is None:
                search_tool = tools[0] if tools else None

            if search_tool is None:
                return {
                    "success": False,
                    "error": "No Sentinel search tool available",
                    "query": query,
                    "results": {},
                }

            params = {"query": query}
            if sensor:
                params["sensor"] = sensor
            if aoi:
                params["aoi"] = aoi
            if date_range:
                params["date_range"] = date_range
            if cloud_cover_max is not None:
                params["cloud_cover_max"] = cloud_cover_max

            raw = await search_tool.ainvoke(params)
            payload = extract_payload(raw)

        return {
            "success": True,
            "query": query,
            "results": payload,
        }

    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "results": {},
        }


def sentinel_search(
    query: str,
    sensor: str | None = None,
    aoi: str | None = None,
    date_range: str | None = None,
    cloud_cover_max: int | None = None,
) -> dict[str, Any]:
    """Search for Sentinel satellite imagery.

    This tool searches for Sentinel-1, Sentinel-2, or Sentinel-3 satellite
    imagery from ESA/Copernicus. Use this for finding satellite scenes,
    checking acquisition dates, or planning satellite observations.

    Args:
        query: Natural language search query describing what you're looking for
        sensor: Sensor type - "S1" (Sentinel-1 SAR), "S2" (Sentinel-2 optical),
                "S3" (Sentinel-3)
        aoi: Area of interest - can be:
             - City/region name: "Daejeon", "Seoul"
             - Bounding box: "126.5,36.0,127.5,37.0" (min_lon,min_lat,max_lon,max_lat)
        date_range: Date range in format "YYYY-MM-DD/YYYY-MM-DD"
                    e.g., "2024-01-01/2024-01-31"
        cloud_cover_max: Maximum cloud coverage percentage (0-100),
                         only applicable to optical sensors (S2, S3)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - results: Search results including:
            - scenes: List of available satellite scenes
            - acquisition_dates: Acquisition timestamps
            - orbits: Orbit information
            - footprints: Geographic coverage
            - cloud_coverage: Cloud percentage (for optical)
            - polarization: Polarization mode (for SAR)

    Example:
        # Search for Sentinel-2 imagery of Daejeon with low cloud cover
        result = sentinel_search(
            query="대전 유성구 위성 영상",
            sensor="S2",
            aoi="Daejeon",
            date_range="2024-01-01/2024-01-31",
            cloud_cover_max=20
        )

    Note:
        - Sentinel-1 (SAR) works regardless of weather/clouds
        - Sentinel-2 (optical) is affected by cloud cover
        - Use weather_forecast to check cloud conditions before optical imaging
    """
    return run_async(_sentinel_search_async(
        query=query,
        sensor=sensor,
        aoi=aoi,
        date_range=date_range,
        cloud_cover_max=cloud_cover_max,
    ))
