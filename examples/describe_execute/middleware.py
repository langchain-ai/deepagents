"""Add model-generated annotations without changing the execute implementation."""

from collections.abc import Awaitable, Callable
from copy import deepcopy

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command


def _describe_execute(tool: BaseTool | dict[str, object]) -> BaseTool | dict[str, object]:
    schema = convert_to_openai_tool(tool)
    if schema.get("function", {}).get("name") != "execute":
        return tool
    schema = deepcopy(schema)
    parameters = schema["function"]["parameters"]
    parameters.setdefault("properties", {})["description"] = {
        "type": "string",
        "description": "Briefly explain what this command does and why you are running it.",
    }
    required = parameters.setdefault("required", [])
    if "description" not in required:
        required.append("description")
    return schema


def _strip_description(request: ToolCallRequest) -> ToolCallRequest:
    if request.tool_call["name"] != "execute":
        return request
    args = dict(request.tool_call["args"])
    args.pop("description", None)
    return request.override(tool_call={**request.tool_call, "args": args})


class DescribeExecuteMiddleware(AgentMiddleware):
    """Expose an execute annotation to the model, but not to the backend."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Extend the execute schema for a synchronous model call."""
        return handler(request.override(tools=[_describe_execute(tool) for tool in request.tools]))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        """Extend the execute schema for an asynchronous model call."""
        return await handler(request.override(tools=[_describe_execute(tool) for tool in request.tools]))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        """Remove the annotation before invoking the original execute tool."""
        return handler(_strip_description(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        """Remove the annotation before asynchronously invoking execute."""
        return await handler(_strip_description(request))
