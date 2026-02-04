"""Weather Forecast tool.

This tool retrieves weather forecast data from OpenWeatherMap API.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from .._utils import WEATHER_SERVER, extract_payload, run_async


async def _weather_forecast_async(
    city: str,
    units: str = "c",
    lang: str = "kr",
) -> dict[str, Any]:
    """Async implementation of weather forecast."""
    if not city.strip():
        return {
            "success": False,
            "error": "City name is required",
            "city": city,
            "forecast": {},
        }

    if not os.path.exists(WEATHER_SERVER):
        return {
            "success": False,
            "city": city,
            "error": f"Weather MCP server not found: {WEATHER_SERVER}. "
                     f"Set MCP_WEATHER_SERVER environment variable.",
            "forecast": {},
        }

    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "city": city,
            "error": "OPENWEATHER_API_KEY environment variable not set.",
            "forecast": {},
        }

    try:
        client = MultiServerMCPClient({
            "openweather": {
                "transport": "stdio",
                "command": "python",
                "args": [WEATHER_SERVER],
                "env": {"OWM_API_KEY": api_key},
            }
        })

        tools = await client.get_tools()
        weather_tool = next(
            (t for t in tools if "weather" in t.name.lower()),
            tools[0] if tools else None
        )

        if weather_tool is None:
            return {
                "success": False,
                "city": city,
                "error": "No weather tool found in MCP server",
                "forecast": {},
            }

        result = await weather_tool.ainvoke({
            "city": city,
            "units": units,
            "lang": lang,
        })

        payload = extract_payload(result)

        return {
            "success": True,
            "city": city,
            "units": units,
            "forecast": payload,
        }

    except Exception as e:
        return {
            "success": False,
            "city": city,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "forecast": {},
        }


# City name translations (Korean to English)
CITY_TRANSLATIONS = {
    "서울": "Seoul",
    "부산": "Busan",
    "대전": "Daejeon",
    "인천": "Incheon",
    "대구": "Daegu",
    "광주": "Gwangju",
    "울산": "Ulsan",
    "세종": "Sejong",
    "제주": "Jeju",
    "경기": "Gyeonggi",
    "수원": "Suwon",
    "의왕": "Uiwang",
    "유성": "Yuseong",
}


def weather_forecast(
    city: str,
    units: str = "c",
    lang: str = "kr",
) -> dict[str, Any]:
    """Get 5-day weather forecast for a city.

    This tool retrieves weather forecast data from OpenWeatherMap API.
    Use this for weather-related queries, outdoor activity planning,
    or any task requiring weather information.

    Args:
        city: City name in English (e.g., "Seoul", "Busan", "Daejeon")
              Korean city names will be automatically translated:
              - 서울 -> Seoul
              - 부산 -> Busan
              - 대전 -> Daejeon
              - 인천 -> Incheon
              - 대구 -> Daegu
        units: Temperature units - "c" (Celsius), "f" (Fahrenheit), "k" (Kelvin)
        lang: Language for descriptions - "kr" (Korean), "en" (English)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - city: The city queried
        - units: Temperature units used
        - forecast: Weather forecast data including:
            - Temperature
            - Weather conditions
            - Humidity
            - Wind speed
            - Cloud coverage
            - Precipitation probability

    Example:
        result = weather_forecast("Seoul", units="c", lang="kr")
        if result["success"]:
            print(f"Weather in {result['city']}: {result['forecast']}")

    Note:
        Requires OPENWEATHER_API_KEY environment variable to be set.
    """
    city_normalized = city.strip()
    for kr, en in CITY_TRANSLATIONS.items():
        if kr in city_normalized:
            city_normalized = city_normalized.replace(kr, en)
            break

    return run_async(_weather_forecast_async(
        city=city_normalized,
        units=units,
        lang=lang,
    ))
