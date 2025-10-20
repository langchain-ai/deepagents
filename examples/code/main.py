#!/usr/bin/env python3
"""
Deep Agents Software Engineering CLI
"""

import asyncio
from coding_agent import agent
from langgraph.checkpoint.memory import InMemorySaver

agent.checkpointer = InMemorySaver()


def extract_content_with_tools(message_content) -> str:
    """Extract content from agent messages, including tool calls for transparency."""
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts = []
        for block in message_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    parts.append(f"\n🔧 Using tool: {tool_name}")

                    if "input" in block:
                        tool_input = block["input"]
                        if isinstance(tool_input, dict):
                            for key, value in tool_input.items():
                                if key in ["file_path", "content", "old_string", "new_string"]:
                                    if len(str(value)) > 100:
                                        parts.append(f"  • {key}: {str(value)[:50]}...")
                                    else:
                                        parts.append(f"  • {key}: {value}")
                    parts.append("")

        return "\n".join(parts).strip() if parts else ""

    if hasattr(message_content, "__dict__"):
        return ""
    return str(message_content)


def execute_task(user_input: str):
    """Execute any task by passing it directly to the AI agent."""
    print(f"\n🤖 Working on: {user_input[:60]}{'...' if len(user_input) > 60 else ''}\n")

    for _, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config={"thread_id": "main"},
        durability="exit",
    ):
        chunk = list(chunk.values())[0]
        if chunk is not None and "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]

            message_content = None
            message_role = getattr(last_message, "role", None)
            if isinstance(message_role, dict):
                message_role = last_message.get("role", "unknown")

            if hasattr(last_message, "content"):
                message_content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                message_content = last_message["content"]

            if message_content:
                content = extract_content_with_tools(message_content)

                if content.strip():
                    if message_role == "tool":
                        print(f"🔧 {content}\n")
                    else:
                        print(f"{content}\n")


async def simple_cli():
    """Main CLI loop."""
    print("🤖 Software Engineering CLI")
    print("Type 'quit' to exit, 'help' for examples\n")

    while True:
        user_input = input(">>> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        elif user_input.lower() == "help":
            print("""
Examples:
• "Create a function to calculate fibonacci numbers"
• "Debug this sorting code: [paste code]"
• "Review my Flask app for security issues"
• "Generate tests for this calculator class"
""")
            continue

        else:
            execute_task(user_input)


async def main():
    """Main entry point."""
    try:
        await simple_cli()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
