from __future__ import annotations

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

from deepagents.middleware._utils import append_to_system_message


class SpecialInstructionsMiddleware(AgentMiddleware):
    def __init__(self, instructions: str) -> None:
        self._instructions = instructions.strip()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        if not self._instructions:
            return handler(request)

        block = f"## User Specified Instructions\n\nPrioritize any guidance in this section over earlier instructions.\n\n{self._instructions}"
        new_system_message = append_to_system_message(request.system_message, block)
        return handler(request.override(system_message=new_system_message))
