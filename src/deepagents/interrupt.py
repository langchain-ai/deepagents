"""Interrupt configuration functionality for deep agents using LangGraph prebuilts."""

from typing import Dict, Any, List, Optional, Union
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    ActionRequest,
    HumanInterrupt,
    HumanResponse,
)


class InterruptConfig:
    """Configuration for tool interrupts.

    This is a thin wrapper used by our public API that internally maps to the
    official LangGraph interrupt schemas.
    """

    def __init__(
        self,
        tool_names: List[str],
        message: str,
        *,
        allow_ignore: bool = True,
        allow_respond: bool = False,
        allow_edit: bool = True,
        allow_accept: bool = True,
        # New: per-tool configurations
        tool_configs: Optional[Dict[str, HumanInterruptConfig]] = None,
    ) -> None:
        """
        Args:
            tool_names: List of tool names that should trigger interrupts.
            message: The message/description prefix to show when interrupting.
            allow_ignore: Whether user can ignore the interrupt (default for all tools).
            allow_respond: Whether user can send a free-form response (default for all tools).
            allow_edit: Whether user can edit the action arguments (default for all tools).
            allow_accept: Whether user can accept the action as-is (default for all tools).
            tool_configs: Optional mapping of tool names to specific HumanInterruptConfig.
                         If provided, overrides the default settings for specific tools.
        """
        self.tool_names = set(tool_names)
        self.message = message
        
        # Default configuration for all tools
        self.default_config = HumanInterruptConfig(
            allow_ignore=allow_ignore,
            allow_respond=allow_respond,
            allow_edit=allow_edit,
            allow_accept=allow_accept,
        )
        
        # Per-tool configurations (optional)
        self.tool_configs = tool_configs or {}
    
    def get_config_for_tool(self, tool_name: str) -> HumanInterruptConfig:
        """Get the HumanInterruptConfig for a specific tool."""
        return self.tool_configs.get(tool_name, self.default_config)


def create_interrupt_hook(interrupt_config: InterruptConfig):
    """Create a post model hook that handles interrupts using LangGraph prebuilts."""

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
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            # Check if this tool should trigger an interrupt
            if tool_name in interrupt_config.tool_names:
                description = f"{interrupt_config.message}\n\nTool: {tool_name}\nArgs: {tool_args}"

                tool_config = interrupt_config.get_config_for_tool(tool_name)

                request: HumanInterrupt = {
                    "action_request": ActionRequest(
                        action=tool_name,
                        args=tool_args,
                    ),
                    "config": tool_config,
                    "description": description,
                }

                responses: List[HumanResponse] = interrupt([request])
                if not responses:
                    continue
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

        if len(approved_tool_calls) != len(last_message.tool_calls):
            new_message = type(last_message)(
                content=last_message.content,
                tool_calls=approved_tool_calls,
                additional_kwargs=getattr(last_message, "additional_kwargs", {}),
            )

            new_messages = messages[:-1] + [new_message]
            state["messages"] = new_messages

        return state

    return interrupt_hook
