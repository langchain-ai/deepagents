from __future__ import annotations

from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_quickjs._foreign_function_docs import render_foreign_function_section


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


def test_render_foreign_function_section() -> None:
    actual = render_foreign_function_section(
        {
            "find_users_by_name": find_users_by_name,
            "get_user_location": get_user_location,
            "get_city_for_location": get_city_for_location,
            "normalize_name": normalize_name,
            "fetch_weather": fetch_weather,
        }
    )

    assert actual == """Available foreign functions:

```python
def find_users_by_name(name: str) -> list[UserLookup]:
    \"\"\"Find users with the given name.

    Args:
        name: The user name to search for.
    \"\"\"

def get_user_location(user_id: int) -> int:
    \"\"\"Get the location id for a user.

    Args:
        user_id: The user identifier.
    \"\"\"

def get_city_for_location(location_id: int) -> str:
    \"\"\"Get the city for a location.

    Args:
        location_id: The location identifier.
    \"\"\"

def normalize_name(name: str) -> str:
    \"\"\"Normalize a user name for matching.\"\"\"

async def fetch_weather(city: str) -> str:
    \"\"\"Fetch the current weather for a city.\"\"\"
```

Referenced types:
```python
class UserLookup(TypedDict):
    id: int
    name: str
```"""
