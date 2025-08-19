#!/usr/bin/env python3
"""
Example demonstrating the interrupt configuration functionality in deepagents.
"""

from deepagents import create_deep_agent, InterruptConfig

# Define tools that should trigger interrupts
def dangerous_file_operation(file_path: str, operation: str):
    """A potentially dangerous file operation that should require approval."""
    return f"Would perform {operation} on {file_path}"

def safe_operation(data: str):
    """A safe operation that doesn't need approval."""
    return f"Processed: {data}"

# Create interrupt configuration
interrupt_config = InterruptConfig(
    tool_names=["dangerous_file_operation"],  # Only this tool will trigger interrupts
    message="This operation could modify your files. Do you want to proceed?",
    include_tool_args=True  # Show the tool arguments in the interrupt message
)

# Create the agent with interrupt configuration
agent = create_deep_agent(
    tools=[dangerous_file_operation, safe_operation],
    instructions="You are a helpful assistant that can perform various operations.",
    interrupt_config=interrupt_config
)

# Example usage:
# When the agent tries to call dangerous_file_operation, it will trigger an interrupt
# asking for user approval. The safe_operation will execute without interruption.

if __name__ == "__main__":
    print("Agent created with interrupt configuration!")
    print("The 'dangerous_file_operation' tool will require user approval.")
    print("The 'safe_operation' tool will execute without interruption.")
