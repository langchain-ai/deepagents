"""
Simple example demonstrating deepagents' mock filesystem functionality.
This example shows how to mount files, edit them using the agent, and persist changes.

NOTE: You need to set the ANTHROPIC_API_KEY environment variable before running this example.
"""

import os
from deepagents import create_deep_agent

# Export the graph for LangGraph deployment
graph = create_deep_agent(
    model="claude-3-5-sonnet-20241022",
    tools=[],  # File system tools are included by default
    instructions="You are a helpful assistant that can edit files.",
)


def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Step 1: Create some sample files to work with
    sample_files = {
        "hello.py": """def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
""",
        "math_utils.py": """def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""",
        "README.md": """# My Project

This is a simple project to demonstrate file editing.

## Features
- Basic math operations
- Greeting functionality
""",
    }

    # Step 2: Use the exported graph (for consistency with deployment)
    print("Creating agent...")
    agent = graph

    # Step 3: Invoke agent with files and editing instructions
    print("\nSending request to agent...")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """Please make the following changes to the files:
 In hello.py, add a new function called 'farewell' that returns "Goodbye, {name}!""",
                }
            ],
            "files": sample_files,
        }
    )

    # Step 4: Display the results
    print("\n" + "=" * 50)
    print("AGENT MESSAGES:")
    print("=" * 50)
    for msg in result["messages"]:
        if msg.type == "human":
            print(f"\nUser: {msg.content}")
        elif msg.type == "ai":
            print(f"\nAssistant: {msg.content}")
        elif msg.type == "tool":
            print(f"\nTool result: {msg.content}")

    # Step 5: Show modified files
    print("\n" + "=" * 50)
    print("MODIFIED FILES:")
    print("=" * 50)

    modified_files = result.get("files", {})
    for filename, content in modified_files.items():
        print(f"\n--- {filename} ---")
        print(content)
        print("--- end ---")

    # Step 6: Optionally save to actual filesystem
    save_to_disk = input("\nWould you like to save these changes to disk? (y/n): ")
    if save_to_disk.lower() == "y":
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        for filename, content in modified_files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Saved: {filepath}")

        print(f"\nFiles saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()
