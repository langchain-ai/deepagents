"""Interrupt configuration functionality for deep agents using LangGraph prebuilts."""

from typing import Dict, Any, List, Optional, Union
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    ActionRequest,
    HumanInterrupt,
    HumanResponse,
)

ToolInterruptConfig = Dict[str, HumanInterruptConfig]

def create_interrupt_hook(
    tool_configs: ToolInterruptConfig,
    message_prefix: str = "Tool execution requires approval",
) -> callable:
    """Create a post model hook that handles interrupts using native LangGraph schemas.
    
    Args:
        tool_configs: Dict mapping tool names to HumanInterruptConfig objects
        message_prefix: Optional message prefix for interrupt descriptions
    """
    
    def interrupt_hook(state: Dict[str, Any]) -> Dict[str, Any]:
        """Post model hook that checks for tool calls and triggers interrupts if needed."""
        messages = state.get("messages", [])
        if not messages:
            return

        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls = []
        auto_approved_tool_calls = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in tool_configs:
                interrupt_tool_calls.append(tool_call)
            else:
                auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return

        approved_tool_calls = auto_approved_tool_calls.copy()

        # Process all tool calls that need interrupts in parallel
        requests = []
        
        for tool_call in interrupt_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            description = f"{message_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
            tool_config = tool_configs[tool_name]

            request: HumanInterrupt = {
                "action_request": ActionRequest(
                    action=tool_name,
                    args=tool_args,
                ),
                "config": tool_config,
                "description": description,
            }
            requests.append(request)

        responses: List[HumanResponse] = interrupt(requests)
        
        for i, response in enumerate(responses):
            tool_call = interrupt_tool_calls[i]
            
            if response["type"] == "accept":
                approved_tool_calls.append(tool_call)
            elif response["type"] == "edit":
                edited: ActionRequest = response["args"]
                new_tool_call = {
                    "name": tool_call["name"],
                    "args": edited["args"],
                    "id": tool_call["id"],
                }
                approved_tool_calls.append(new_tool_call)
            else:
                raise ValueError(f"Unknown response type: {response['type']}")

        last_message.tool_calls = approved_tool_calls

        return {"messages": [last_message]}

    return interrupt_hook


