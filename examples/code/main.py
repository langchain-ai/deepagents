#!/usr/bin/env python3
"""
Deep Agents Software Engineering CLI - Simplified

A clean, honest CLI that passes your requests directly to the AI agent
without fake command parsing. Much simpler and more transparent!
"""

import asyncio
import sys
import time
from datetime import datetime
from coding_agent import agent
from langgraph.checkpoint.memory import InMemorySaver

agent.checkpointer = InMemorySaver()


class ProgressTracker:
    """Simple progress tracking with timing."""

    def __init__(self):
        self.start_time = None

    def start_task(self, description: str):
        """Start tracking a task."""
        self.start_time = time.time()
        print(
            f"\n🚀 Starting: {description[:60]}{'...' if len(description) > 60 else ''}"
        )
        print("─" * 60)

    def complete_task(self):
        """Mark task as completed."""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n✅ Completed in {total_time:.1f}s!")
            print("─" * 60)


def print_banner():
    """Print a nice banner for the CLI."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   🤖 SOFTWARE ENGINEERING CLI                ║
║                    Powered by Deep Agents                    ║
║                                                              ║
║       Just tell me what you want to build or work on!       ║
║         No fake commands, just direct AI conversation       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
💬 HOW TO USE:
Just describe what you want to do in natural language! Examples:

🔹 "Create a function to calculate fibonacci numbers"
🔹 "Debug this sorting code: [paste your code here]"  
🔹 "Review my Flask app for security issues"
🔹 "Generate tests for this calculator class"
🔹 "Explain how this algorithm works: [paste code]"
🔹 "Build a REST API for a todo app"
🔹 "Optimize this slow database query"
🔹 "Create a web scraper for news articles"

💡 TIPS:
• Be specific about what you want
• Paste code directly if you need help with existing code
• The AI will automatically use specialized sub-agents when needed
• All code is tested and validated

🛠️  UTILITY COMMANDS:
• help - Show this help
• quit/exit/q - Exit the CLI
• clear - Clear screen

That's it! No fake commands, just honest AI assistance.
"""
    print(help_text)


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
                    # Show tool usage transparently
                    tool_name = block.get("name", "unknown_tool")
                    parts.append(f"\n🔧 Using tool: {tool_name}")

                    # Show key parameters if they exist
                    if "input" in block:
                        tool_input = block["input"]
                        if isinstance(tool_input, dict):
                            # Show relevant parameters
                            for key, value in tool_input.items():
                                if key in [
                                    "file_path",
                                    "content",
                                    "old_string",
                                    "new_string",
                                ]:
                                    if len(str(value)) > 100:
                                        parts.append(f"  • {key}: {str(value)[:50]}...")
                                    else:
                                        parts.append(f"  • {key}: {value}")
                    parts.append("")  # Add blank line after tool usage

        return "\n".join(parts).strip() if parts else ""

    # Avoid printing complex objects
    if hasattr(message_content, "__dict__"):
        return ""
    return str(message_content)


async def execute_task(user_input: str, progress: ProgressTracker):
    """Execute any task by passing it directly to the AI agent."""
    try:
        progress.start_task(user_input)

        print(f"\n🤖 AI Agent is working on your request...")

        # Stream the agent's response with interrupt handling
        async for _, chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",
            subgraphs=True,
            config={"thread_id": "main"},
        ):
            chunk = list(chunk.values())[0]
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                # Handle different message types
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
                        # Show tool results with a different icon
                        if message_role == "tool":
                            print(f"\n🔧 Tool result: {content}")
                        else:
                            print(f"\n🤖 {content}")
                        print("─" * 40)

        progress.complete_task()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try rephrasing your request.")


async def simple_cli():
    """Main CLI loop - much simpler!"""

    print_banner()

    print("\n🎯 Welcome! Just tell me what you want to build or work on.")
    print("Type 'help' for examples, or 'quit' to exit.\n")

    progress = ProgressTracker()
    session_start = datetime.now()
    tasks_completed = 0

    while True:
        try:
            # Show simple session info
            session_time = datetime.now() - session_start
            print(
                f"\n┌─ Session: {session_time.seconds // 60}m {session_time.seconds % 60}s | Tasks: {tasks_completed} ─┐"
            )

            user_input = input("🛠️  What can I help you build? ").strip()

            if not user_input:
                continue

            # Handle only essential utility commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print(f"\n👋 Thanks for using the CLI!")
                print(f"📊 Completed {tasks_completed} tasks in {session_time}")
                break

            elif user_input.lower() == "help":
                print_help()
                continue

            elif user_input.lower() == "clear":
                print("\033[2J\033[H", end="")  # Clear screen
                print_banner()
                continue

            # Everything else goes directly to the agent
            else:
                await execute_task(user_input, progress)
                tasks_completed += 1

        except KeyboardInterrupt:
            print(f"\n\n👋 Thanks for using the CLI!")
            print(f"📊 Completed {tasks_completed} tasks")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please try again or type 'help'.")


async def main():
    """Main entry point."""
    try:
        await simple_cli()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
