"""Mock city-data tools for the langchain TodoListMiddleware eval suite.

Lightweight lookup tools returning fixed city facts, used to exercise
``langchain``'s ``TodoListMiddleware`` on a bare ``create_agent``.

Extracted from ``tests/evals/test_langchain_middleware_todo.py`` so both the
pytest suite and the Harbor sandbox dispatcher share the same definitions.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def lookup_population(city: str) -> str:
    """Return the population of a city as a string."""
    data = {
        "tokyo": "13,960,000",
        "delhi": "32,900,000",
        "shanghai": "29,200,000",
        "cairo": "21,800,000",
    }
    return data.get(city.lower(), "unknown")


@tool
def lookup_area_km2(city: str) -> str:
    """Return the area of a city in square kilometers as a string."""
    data = {
        "tokyo": "2,194",
        "delhi": "1,484",
        "shanghai": "6,341",
        "cairo": "606",
    }
    return data.get(city.lower(), "unknown")
