from __future__ import annotations

from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_repl._foreign_function_docs import (
    format_foreign_function_docs,
    render_foreign_function_section,
)


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


@tool
def combine_user_details(name: str, city: str, active: bool) -> str:
    """Combine user details into a summary string.

    Args:
        name: The user name.
        city: The user's city.
        active: Whether the user is active.
    """
    return f"{name} in {city} active={active}"


@tool
def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""
    return name.strip().lower()


async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"Weather for {city}"


def test_format_foreign_function_docs_for_plain_function() -> None:
    assert (
        format_foreign_function_docs("normalize_name", normalize_name)
        == """normalize_name
  args: (name string)
  returns: string
  summary: Normalize a user name for matching."""
    )


def test_format_foreign_function_docs_for_tool_with_args_and_return_type() -> None:
    assert (
        format_foreign_function_docs("get_user_location", get_user_location)
        == """get_user_location
  args: (user_id number)
  returns: number
  summary: Get the location id for a user."""
    )


def test_format_foreign_function_docs_for_async_function() -> None:
    assert (
        format_foreign_function_docs("fetch_weather", fetch_weather)
        == """fetch_weather
  args: (city string)
  returns: string
  summary: Fetch the current weather for a city.
  note: async function"""
    )


def test_format_foreign_function_docs_for_three_arg_tool() -> None:
    assert (
        format_foreign_function_docs("combine_user_details", combine_user_details)
        == """combine_user_details
  args: (name string) (city string) (active boolean)
  returns: string
  summary: Combine user details into a summary string."""
    )


def test_format_foreign_function_docs_for_single_line_tool_docstring() -> None:
    assert (
        format_foreign_function_docs("greet_user", greet_user)
        == """greet_user
  args: (name string)
  returns: string
  summary: Greet a user by name."""
    )


def test_render_foreign_function_section() -> None:
    actual = render_foreign_function_section(
        {
            "find_users_by_name": find_users_by_name,
            "get_user_location": get_user_location,
            "get_city_for_location": get_city_for_location,
            "combine_user_details": combine_user_details,
            "greet_user": greet_user,
            "normalize_name": normalize_name,
            "fetch_weather": fetch_weather,
        }
    )

    assert (  # keep snapshot-like multiline string content byte-exact
        actual
        == """Available foreign functions:

These functions are callable from the REPL. Argument and return types use the same simple value shapes as the language: strings, numbers, booleans, None, lists, and dict-like records.

```text
find_users_by_name
  args: (name string)
  returns: UserLookup[]
  summary: Find users with the given name.

get_user_location
  args: (user_id number)
  returns: number
  summary: Get the location id for a user.

get_city_for_location
  args: (location_id number)
  returns: string
  summary: Get the city for a location.

combine_user_details
  args: (name string) (city string) (active boolean)
  returns: string
  summary: Combine user details into a summary string.

greet_user
  args: (name string)
  returns: string
  summary: Greet a user by name.

normalize_name
  args: (name string)
  returns: string
  summary: Normalize a user name for matching.

fetch_weather
  args: (city string)
  returns: string
  summary: Fetch the current weather for a city.
  note: async function
```

Referenced types:
```text
UserLookup
  fields:
    (id number)
    (name string)
```"""  # noqa: E501
    )
