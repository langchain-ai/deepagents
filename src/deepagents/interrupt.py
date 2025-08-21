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

# Common prebuilt configurations
STANDARD_CONFIGS = {
    "approve_only": HumanInterruptConfig(
        allow_ignore=False,
        allow_respond=False,
        allow_edit=False,
        allow_accept=True,
    ),
    "approve_or_skip": HumanInterruptConfig(
        allow_ignore=True,
        allow_respond=False,
        allow_edit=False,
        allow_accept=True,
    ),
    "full_control": HumanInterruptConfig(
        allow_ignore=True,
        allow_respond=True,
        allow_edit=True,
        allow_accept=True,
    ),
    "review_and_edit": HumanInterruptConfig(
        allow_ignore=False,
        allow_respond=True,
        allow_edit=True,
        allow_accept=True,
    ),
}


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
            return state

        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state

        approved_tool_calls = []

        # Check each tool call for approval
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Check if this tool should trigger an interrupt
            if tool_name in tool_configs:
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

                responses: List[HumanResponse] = interrupt([request])
                response = responses[0]

                if response["type"] == "accept":
                    approved_tool_calls.append(tool_call)
                elif response["type"] == "edit":
                    edited: ActionRequest = response["args"] 
                    # Replace args with edited args
                    new_tool_call = {
                        "name": tool_name,
                        "args": edited["args"],
                        "id": tool_call.get("id", ""),
                    }
                    approved_tool_calls.append(new_tool_call)
                else:
                    continue
            else:
                approved_tool_calls.append(tool_call)

        last_message.tool_calls = approved_tool_calls

        return state

    return interrupt_hook


