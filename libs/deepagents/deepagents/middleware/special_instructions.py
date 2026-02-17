from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepagents.middleware._utils import append_to_system_message


class SpecialInstructionsMiddleware(AgentMiddleware):
    def __init__(self, instructions: str | SystemMessage) -> None:
        if isinstance(instructions, SystemMessage):
            content = instructions.content
            if isinstance(content, str):
                self._instructions = content.strip()
            else:
                self._instructions = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in instructions.content_blocks
                ).strip()
        else:
            self._instructions = instructions.strip()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        new_system_message = self.wrap_system_message(request.system_message)
        if new_system_message is request.system_message:
            return handler(request)
        return handler(request.override(system_message=new_system_message))

    def wrap_system_message(self, system_message: SystemMessage | None) -> SystemMessage | None:
        if not self._instructions:
            return system_message

        block = f"## User Specified Instructions\n\nPrioritize any guidance in this section over earlier instructions.\n\n{self._instructions}"
        return append_to_system_message(system_message, block)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        new_system_message = self.wrap_system_message(request.system_message)
        if new_system_message is request.system_message:
            return await handler(request)
        return await handler(request.override(system_message=new_system_message))
