"""Middleware implementations for deep agents."""

from typing import Dict, Any, Optional, Callable
from langchain.agents.middleware import AgentMiddleware, AgentState
from deepagents.interrupt import create_interrupt_hook

def create_custom_state_middleware(agent_state) -> AgentMiddleware:
    class CustomStateMiddleware(AgentMiddleware[agent_state]):
        state_schema: agent_state
    return CustomStateMiddleware()


class PostModelHookMiddleware(AgentMiddleware):
    def __init__(self, hook: Callable):
        self.hook = hook

    def after_model(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.hook:
            return self.hook(state)
        return None


class InterruptMiddleware(AgentMiddleware):
    def __init__(
        self,
        tool_configs: Dict[str, Any],
    ):
        self.tool_configs = tool_configs
        self.interrupt_hook = create_interrupt_hook(tool_configs)

    def after_model(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.interrupt_hook(state)

class OffloadFileReadContextMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        # TODO: Offload file read context to a separate thread
        pass

