"""Interrupt configuration functionality for deep agents using LangGraph prebuilts."""

from typing import Dict, Any, List, Optional, Union
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    ActionRequest,
    HumanInterrupt,
    HumanResponse,
)

ToolInterruptConfig = Dict[str, Union[HumanInterruptConfig, bool]]

def create_interrupt_hook(
    tool_configs: ToolInterruptConfig,
    message_prefix: str = "Tool execution requires approval",
) -> callable:
    """Create a post model hook that handles interrupts using native LangGraph schemas.
    
    Args:
        tool_configs: Dict mapping tool names to HumanInterruptConfig objects
        message_prefix: Optional message prefix for interrupt descriptions
    """
    # Right now we don't properly handle `ignore`
    for tool, interrupt_config in tool_configs.items():
        if isinstance(interrupt_config, dict):
            if 'allow_ignore' in interrupt_config and interrupt_config['allow_ignore']:
                raise ValueError(
                    f"For {tool} we get `allow_ignore = True` - we currently don't support `ignore`."
                )
    
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

        # Right now, no easy handling for when multiple tools need interrupt
        if len(interrupt_tool_calls) > 1:
            raise ValueError(
                "Right now, interrupt hook only works when one tool requires interrupts"
            )
        tool_call = interrupt_tool_calls[0]

        approved_tool_calls = auto_approved_tool_calls.copy()

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        description = f"{message_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
        tool_config = tool_configs[tool_name]
        default_tool_config: HumanInterruptConfig = {"allow_accept": True, "allow_edit": True, "allow_respond": True, "allow_ignore": False}

        request: HumanInterrupt = {
            "action_request": ActionRequest(
                action=tool_name,
                args=tool_args,
            ),
            "config": tool_config if isinstance(tool_config, dict) else default_tool_config,
            "description": description,
        }

        responses: List[HumanResponse] = interrupt([request])

        if len(responses) != 1:
            raise ValueError(
                f"Expected a list of one response, got {responses}"
            )
        response = responses[0]
            
        if response["type"] == "accept":
            approved_tool_calls.append(tool_call)
        elif response["type"] == "edit":
            edited: ActionRequest = response["args"]
            new_tool_call = {
                "type": "tool_call",
                "name": edited["action"],
                "args": edited["args"],
                "id": tool_call["id"],
            }
            approved_tool_calls.append(new_tool_call)
        elif response["type"] == "response":
            response_message = {
                "type": "tool",
                "tool_call_id": tool_call['id'],
                "content": response['args']
            }
            return {"messages": [response_message]}
        else:
            raise ValueError(f"Unknown response type: {response['type']}")

        last_message.tool_calls = approved_tool_calls

        return {"messages": [last_message]}

    return interrupt_hook


