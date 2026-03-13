from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware._utils import append_to_system_message
from langchain_core.tools import tool
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.messages import SystemMessage

from langchain_lisp.middleware import LispMiddleware


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


def _assert_snapshot(
    snapshot_path: Path, actual: str, *, update_snapshots: bool
) -> None:
    if update_snapshots or not snapshot_path.exists():
        snapshot_path.write_text(actual)
        if update_snapshots:
            return
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)

    expected = snapshot_path.read_text()
    assert actual == expected


def _capture_system_prompt(middleware: LispMiddleware) -> str:
    system_message = append_to_system_message(None, BASE_AGENT_PROMPT)
    system_message = append_to_system_message(
        system_message, middleware._format_repl_system_prompt()
    )
    return _system_message_as_text(system_message)


def test_system_prompt_snapshot_no_tools(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    prompt = _capture_system_prompt(LispMiddleware())
    snapshot_path = snapshots_dir / "langchain_lisp_system_prompt_no_tools.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)


def test_system_prompt_snapshot_with_mixed_foreign_functions(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    prompt = _capture_system_prompt(
        LispMiddleware(
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
    )
    snapshot_path = snapshots_dir / "langchain_lisp_system_prompt_mixed_foreign_functions.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)
