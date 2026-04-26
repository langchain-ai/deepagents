from __future__ import annotations

from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_monty.middleware import MontyMiddleware


class UserLookup(TypedDict):
    id: int
    name: str


@tool
def find_users_by_name(name: str) -> list[UserLookup]:
    """Find users with the given name.

    Args:
        name: The user name to search for.
    """
    return [{"id": 1, "name": name}]


@tool
def get_user_location(user_id: int) -> int:
    """Get the location id for a user.

    Args:
        user_id: The user identifier.
    """
    return user_id


@tool
def get_city_for_location(location_id: int) -> str:
    """Get the city for a location.

    Args:
        location_id: The location identifier.
    """
    return f"City {location_id}"


def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""
    return name.strip().lower()


async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"Weather for {city}"


def test_system_prompt_includes_rendered_foreign_function_docs() -> None:
    middleware = MontyMiddleware(
        ptc=[
            find_users_by_name,
            get_user_location,
            get_city_for_location,
            normalize_name,
            fetch_weather,
        ],
        add_ptc_docs=True,
    )

    prompt = middleware._format_repl_system_prompt()
    assert "Available foreign functions:" in prompt
    assert "```python" in prompt
    assert "Prefer solving the task in a single `repl` call" in prompt
    assert "Do as much useful work as possible in one program" in prompt
    assert (
        "When several awaited calls are independent, prefer running them in parallel"
        in prompt
    )
    assert "prefer writing one complete Python program" in prompt
    assert "trust it and chain the calls" in prompt
    assert "print it inside the same REPL program" in prompt
    assert "If you can compute an intermediate value and immediately use it" in prompt
    assert "def find_users_by_name(name: str) -> list[UserLookup]" in prompt
    assert "async def fetch_weather(city: str) -> str" in prompt
    assert "Referenced types:" in prompt
    assert "class UserLookup(TypedDict):" in prompt
    assert "    id: int" in prompt
