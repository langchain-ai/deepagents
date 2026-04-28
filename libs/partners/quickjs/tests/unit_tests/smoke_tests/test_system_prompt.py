from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from pydantic import Field
from typing_extensions import TypedDict

from langchain_quickjs import REPLMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


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
def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""
    return name.strip().lower()


@tool
async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"Weather for {city}"


class _SmokeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with call-history capture and stable tool binding."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)
    call_history: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _SmokeChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_history.append({"messages": messages, "kwargs": kwargs})
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def _smoke_model() -> _SmokeChatModel:
    """Return a fake model with enough canned responses for prompt snapshot tests."""
    return _SmokeChatModel(
        messages=iter([AIMessage(content="hello!") for _ in range(4)])
    )


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
        snapshot_path.write_text(actual, encoding="utf-8")
        if update_snapshots:
            return
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)

    expected = snapshot_path.read_text(encoding="utf-8")
    assert actual == expected


def _invoke_for_snapshot(agent: object, payload: dict[str, Any]) -> None:
    """Invoke the agent and tolerate fake-model exhaustion after the first call."""
    try:
        if not hasattr(agent, "invoke"):
            msg = f"Expected compiled agent with invoke(), got {type(agent)!r}"
            raise TypeError(msg)
        agent.invoke(payload)
    except RuntimeError as exc:
        if "StopIteration" not in str(exc):
            raise


def _capture_system_prompt(model: _SmokeChatModel) -> str:
    history = model.call_history
    assert len(history) >= 1

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    return _system_message_as_text(system_messages[0])


def test_system_prompt_snapshot_no_tools(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    model = _smoke_model()
    agent = create_deep_agent(
        model=model,
        middleware=[REPLMiddleware()],
    )
    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})
    prompt = _capture_system_prompt(model)

    snapshot_path = snapshots_dir / "quickjs_system_prompt_no_tools.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)


def test_system_prompt_snapshot_with_mixed_foreign_functions(
    snapshots_dir: Path, *, update_snapshots: bool
) -> None:
    mixed_tools = [
        find_users_by_name,
        get_user_location,
        get_city_for_location,
        normalize_name,
        fetch_weather,
    ]
    model = _smoke_model()
    agent = create_deep_agent(
        model=model,
        middleware=[REPLMiddleware(ptc=mixed_tools)],
        tools=mixed_tools,
    )
    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})
    prompt = _capture_system_prompt(model)

    snapshot_path = snapshots_dir / "quickjs_system_prompt_mixed_foreign_functions.md"
    _assert_snapshot(snapshot_path, prompt, update_snapshots=update_snapshots)
