"""Interrupt configuration functionality for deep agents."""

from typing import Dict, Any, List
from langgraph.types import interrupt


class InterruptConfig:
    """Configuration for tool interrupts."""
    
    def __init__(
        self,
        tool_names: List[str],
        message: str,
        include_tool_args: bool = True
    ):
        """
        Args:
            tool_names: List of tool names that should trigger interrupts
            message: The message to show when interrupting
            include_tool_args: Whether to include tool arguments in the interrupt message
        """
        self.tool_names = set(tool_names)
        self.message = message
        self.include_tool_args = include_tool_args


def create_interrupt_hook(interrupt_config: InterruptConfig):
    """Create a post model hook that handles interrupts based on configuration."""
    
    def interrupt_hook(state: Dict[str, Any]) -> Dict[str, Any]:
        """Post model hook that checks for tool calls and triggers interrupts if needed."""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        approved_tool_calls = []
        
        # Check each tool call for approval
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            # Check if this tool should trigger an interrupt
            if tool_name in interrupt_config.tool_names:
                if interrupt_config.include_tool_args:
                    question = f"{interrupt_config.message}\n\nTool: {tool_name}\nArgs: {tool_args}\n\nRespond with True to approve or False to reject."
                else:
                    question = f"{interrupt_config.message}\n\nTool: {tool_name}\n\nRespond with True to approve or False to reject."
                
                # Trigger interrupt for approval
                is_approved = interrupt({
                    "question": question,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                })
                
                if is_approved:
                    approved_tool_calls.append(tool_call)
                else:
                    continue
            else:
                approved_tool_calls.append(tool_call)
        
        if len(approved_tool_calls) != len(last_message.tool_calls):
            new_message = type(last_message)(
                content=last_message.content,
                tool_calls=approved_tool_calls,
                additional_kwargs=getattr(last_message, 'additional_kwargs', {})
            )
            
            # Update the state with the new message
            new_messages = messages[:-1] + [new_message]
            state["messages"] = new_messages
        
        return state
    
    return interrupt_hook
