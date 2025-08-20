"""Interrupt configuration functionality for deep agents using LangGraph prebuilts."""

from typing import Dict, Any, List, Optional
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
        include_tool_args: bool = True,
        *,
        allow_ignore: bool = True,
        allow_respond: bool = False,
        allow_edit: bool = True,
        allow_accept: bool = True,
    ) -> None:
        """
        Args:
            tool_names: List of tool names that should trigger interrupts.
            message: The message/description prefix to show when interrupting.
            include_tool_args: Whether to include tool arguments in the description.
            allow_ignore: Whether user can ignore the interrupt.
            allow_respond: Whether user can send a free-form response.
            allow_edit: Whether user can edit the action arguments.
            allow_accept: Whether user can accept the action as-is.
        """
        self.tool_names = set(tool_names)
        self.message = message
        self.include_tool_args = include_tool_args
        self.human_config = HumanInterruptConfig(
            allow_ignore=allow_ignore,
            allow_respond=allow_respond,
            allow_edit=allow_edit,
            allow_accept=allow_accept,
        )


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
                description = interrupt_config.message
                if interrupt_config.include_tool_args:
                    description += f"\n\nTool: {tool_name}\nArgs: {tool_args}"

                # Build HumanInterrupt request using official schema
                request: HumanInterrupt = {
                    "action_request": ActionRequest(
                        action=tool_name,
                        args=tool_args,
                    ),
                    "config": interrupt_config.human_config,
                    "description": description,
                }

                # Trigger interrupt; Agent Inbox returns a list, use the first response
                responses: List[HumanResponse] = interrupt([request])
                if not responses:
                    # No response; default to ignore
                    continue
                response = responses[0]

                if response["type"] == "accept":
                    approved_tool_calls.append(tool_call)
                elif response["type"] == "edit":
                    edited: ActionRequest = response["args"]  # type: ignore[assignment]
                    # Replace args with edited args
                    new_tool_call = {
                        "name": tool_name,
                        "args": edited["args"],
                        "id": tool_call.get("id", ""),
                    }
                    approved_tool_calls.append(new_tool_call)
                # 'ignore' and 'response' both result in not executing the tool
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

            # Update the state with the new message
            new_messages = messages[:-1] + [new_message]
            state["messages"] = new_messages

        return state

    return interrupt_hook
