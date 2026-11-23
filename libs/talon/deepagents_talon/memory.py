"""Memory loading for long-lived Talon conversations."""

from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING

from deepagents.middleware.memory import MemoryMiddleware
from langchain.agents.middleware import hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.middleware.memory import MemoryState, MemoryStateUpdate
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime


_CORRECTION = re.compile(
    r"\b(you(?:'|\u2019)re right|you are right|i should have|i shouldn(?:'|\u2019)t have|"
    r"you (?:stopped|stop|keep|missed|forgot|failed)|stop being|too (?:early|soon|verbose)|"
    r"that(?:'|\u2019)s (?:wrong|incorrect)|not what i asked)\b",
    re.IGNORECASE,
)
_REMINDER_NAME = "talon_memory_correction"
_REMINDER = (
    "Before ending, consider the operator's correction and record the underlying durable "
    "lesson in a configured memory source using edit_file or write_file if appropriate. "
    "Follow the existing memory guidelines and exclusions, and the user's instructions "
    "about what must not be saved. Do not save temporary details, one-time requests, "
    "or credentials. If no durable lesson should be saved, finish normally."
)


def _memory_updated(messages: Sequence[BaseMessage], sources: list[str]) -> bool:
    paths = {posixpath.normpath(source) for source in sources}
    calls = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] in {"edit_file", "write_file"}
        and isinstance(path := call["args"].get("file_path"), str)
        and posixpath.normpath(path) in paths
    }
    return any(
        isinstance(message, ToolMessage)
        and message.tool_call_id in calls
        and message.status == "success"
        for message in messages
    )


class TalonMemoryMiddleware(MemoryMiddleware):
    """Reload memory each turn and remind the model to persist missed corrections."""

    @property
    def name(self) -> str:
        """Replace the SDK memory middleware in its existing stack position."""
        return "MemoryMiddleware"

    def before_agent(
        self, state: MemoryState, runtime: Runtime, config: RunnableConfig
    ) -> MemoryStateUpdate | None:
        """Read current sources without changing the incoming checkpoint state."""
        state = state.copy()
        state.pop("memory_contents", None)
        return super().before_agent(state, runtime, config)

    async def abefore_agent(
        self, state: MemoryState, runtime: Runtime, config: RunnableConfig
    ) -> MemoryStateUpdate | None:
        """Read current sources without changing the incoming checkpoint state."""
        state = state.copy()
        state.pop("memory_contents", None)
        return await super().abefore_agent(state, runtime, config)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: MemoryState, runtime: Runtime) -> dict[str, object] | None:
        """Give missed operator corrections one opportunity to reach memory."""
        del runtime
        messages = state["messages"]
        latest = messages[-1]
        if not self.sources or not isinstance(latest, AIMessage) or latest.tool_calls:
            return None
        start = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
                and messages[index].name != _REMINDER_NAME
            ),
            None,
        )
        if start is None:
            return None
        turn = messages[start:]
        if any(message.name == _REMINDER_NAME for message in turn):
            return None
        if not _CORRECTION.search(turn[0].text + "\n" + latest.text):
            return None
        if _memory_updated(turn, self.sources):
            return None
        return {
            "messages": [HumanMessage(content=_REMINDER, name=_REMINDER_NAME)],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: MemoryState, runtime: Runtime) -> dict[str, object] | None:
        """Give missed operator corrections one opportunity to reach memory."""
        return self.after_model(state, runtime)
