"""Memory loading for long-lived Talon conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict, cast

from deepagents.middleware.memory import MemoryMiddleware, MemoryState
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.middleware.memory import MemoryStateUpdate
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime


class _MemoryPrompt(TypedDict):
    contents: dict[str, str]
    delivered: dict[str, str]
    history_start: str | None
    updates: dict[str, HumanMessage]


class TalonMemoryState(MemoryState):
    """Keep the prompt snapshot and updates private to the conversation."""

    memory_prompt: NotRequired[Annotated[_MemoryPrompt, PrivateStateAttr]]


class TalonMemoryMiddleware(MemoryMiddleware):
    """Reload memory each turn while preserving the cached system prefix."""

    state_schema = TalonMemoryState

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

    def _prepare_request(self, request: ModelRequest) -> tuple[ModelRequest, _MemoryPrompt]:
        state = cast("TalonMemoryState", request.state)
        contents = state.get("memory_contents", {})
        first = request.messages[0] if request.messages else None
        start = first.id if first is not None else None
        if first is not None and start is None:
            tail = request.messages[1].id if len(request.messages) > 1 else None
            start = f"{first.model_dump_json()}:{tail}"
        prompt = state.get("memory_prompt")
        # Summarization replaces the first message, including within the current model call.
        if prompt is None or prompt["history_start"] != start:
            prompt = _MemoryPrompt(
                contents=contents, delivered=contents, history_start=start, updates={}
            )
        else:
            prompt = prompt.copy()
            prompt["updates"] = dict(prompt["updates"])
        self._add_update(request, prompt, contents)
        messages = []
        for message in request.messages:
            if message.id is not None and message.id in prompt["updates"]:
                messages.append(prompt["updates"][message.id])
            messages.append(message)
        snapshot = state.copy()
        snapshot["memory_contents"] = prompt["contents"]
        modified = super().modify_request(request.override(state=snapshot))
        return modified.override(messages=messages, state=request.state), prompt

    def _add_update(
        self, request: ModelRequest, prompt: _MemoryPrompt, contents: dict[str, str]
    ) -> None:
        if contents == prompt["delivered"] or self.system_prompt is None:
            return
        human = next(
            (
                message
                for message in reversed(request.messages)
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if human is not None and human.id is not None:
            prompt["updates"][human.id] = HumanMessage(
                content="Updated memory files (replaces previous memory):\n"
                + self._format_agent_memory(contents, self.system_prompt)
            )
            prompt["delivered"] = contents

    def wrap_model_call(  # ty: ignore[invalid-method-override]  # AgentMiddleware supports command responses.
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ExtendedModelResponse:
        """Persist the prompt snapshot alongside the model response."""
        modified, prompt = self._prepare_request(request)
        return ExtendedModelResponse(
            model_response=handler(modified), command=Command(update={"memory_prompt": prompt})
        )

    async def awrap_model_call(  # ty: ignore[invalid-method-override]  # AgentMiddleware supports command responses.
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ExtendedModelResponse:
        """Persist the prompt snapshot alongside the asynchronous response."""
        modified, prompt = self._prepare_request(request)
        return ExtendedModelResponse(
            model_response=await handler(modified),
            command=Command(update={"memory_prompt": prompt}),
        )
