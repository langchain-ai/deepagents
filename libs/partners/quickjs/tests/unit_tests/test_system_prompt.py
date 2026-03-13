from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import SystemMessage

from langchain_quickjs.middleware import QuickJSMiddleware


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


def _system_message_as_text(message: SystemMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part)
        for part in content
    )


def test_system_prompt_includes_rendered_foreign_function_docs() -> None:
    middleware = QuickJSMiddleware(
        external_functions=[
            "find_users_by_name",
            "get_user_location",
            "get_city_for_location",
            "normalize_name",
            "fetch_weather",
        ],
        external_function_implementations={
            "find_users_by_name": find_users_by_name,
            "get_user_location": get_user_location,
            "get_city_for_location": get_city_for_location,
            "normalize_name": normalize_name,
            "fetch_weather": fetch_weather,
        },
        auto_include=True,
    )

    prompt = middleware._format_repl_system_prompt()
    assert "Available foreign functions:" in prompt
    assert "```ts" in prompt
    assert "function find_users_by_name(name: string): UserLookup[]" in prompt
    assert "async function fetch_weather(city: string): Promise<string>" in prompt
    assert "Referenced types:" in prompt
    assert "type UserLookup = {" in prompt
