"""Preserve the rendered parent prompt for forked subagents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import SystemMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Callable


class ForkPromptState(AgentState):
    """State carried privately from a parent model request to its fork."""

    _fork_parent_system_prompt: Annotated[NotRequired[str], PrivateStateAttr]


class ForkPromptMiddleware(AgentMiddleware[ForkPromptState, None]):
    """Keep a fork's rendered system prompt prefixed by its parent prompt."""

    state_schema = ForkPromptState

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | ExtendedModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Capture parent prompts and repair fork prompts before invocation."""  # noqa: DOC201  # middleware hook return is self-evident
        parent_prompt = request.state.get("_fork_parent_system_prompt")
        is_fork = bool(request.state.get("_deepagents_forked_context"))
        if is_fork and parent_prompt:
            current_prompt = (
                request.system_message.text if request.system_message else ""
            )
            if not current_prompt.startswith(parent_prompt):
                request = request.override(
                    system_message=SystemMessage(
                        content=(
                            f"{parent_prompt}\n\n{current_prompt}"
                            if current_prompt
                            else parent_prompt
                        )
                    )
                )

        response = handler(request)
        if is_fork or request.system_message is None:
            return response
        return ExtendedModelResponse(
            model_response=(
                response.model_response
                if isinstance(response, ExtendedModelResponse)
                else response
            ),
            command=Command(
                update={
                    "_fork_parent_system_prompt": request.system_message.text,
                }
            ),
        )
