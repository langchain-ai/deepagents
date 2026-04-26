from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.graph import create_deep_agent

if TYPE_CHECKING:
    from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langchain_monty.middleware import MontyMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


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


def _capture_system_prompt(middleware: MontyMiddleware) -> str:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))
    agent = create_deep_agent(model=model, middleware=[middleware])
    agent.invoke({"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    return _system_message_as_text(system_messages[0])


def test_system_prompt_snapshot_basic(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    prompt = _capture_system_prompt(MontyMiddleware())
    snapshot_path = snapshots_dir / "monty_system_prompt_basic.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)


def test_system_prompt_snapshot_with_mixed_foreign_functions(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    prompt = _capture_system_prompt(
        MontyMiddleware(
            ptc=[
                find_users_by_name,
                get_user_location,
                get_city_for_location,
                normalize_name,
                fetch_weather,
            ],
            add_ptc_docs=True,
        )
    )
    snapshot_path = snapshots_dir / "monty_system_prompt_mixed_foreign_functions.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)
