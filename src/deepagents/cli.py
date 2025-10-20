#!/usr/bin/env python3
import argparse
import asyncio
import os
import subprocess
import platform
import requests
from typing import Dict, Any, Union, Literal
from pathlib import Path

from tavily import TavilyClient
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt
import shutil
from rich import box

import dotenv

dotenv.load_dotenv()

console = Console()

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY")) if os.environ.get("TAVILY_API_KEY") else None


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Union[str, Dict] = None,
    params: Dict[str, str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {str(e)}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for programming-related information."""
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query
        }
    
    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {
            "error": f"Web search error: {str(e)}",
            "query": query
        }


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.
    
    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (agent.md) is handled separately by the middleware.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


config = {"recursion_limit": 1000}

# Constants for display truncation
MAX_ARG_LENGTH = 200
MAX_RESULT_LENGTH = 200


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_args(tool_input: dict) -> str:
    """Format tool arguments for display, truncating long values."""
    if not tool_input:
        return ""
    
    args_parts = []
    for key, value in tool_input.items():
        value_str = str(value)
        value_str = truncate_value(value_str)
        args_parts.append(f"{key}={value_str}")
    
    return ", ".join(args_parts)


def display_tool_call(tool_name: str, tool_input: dict):
    """Display a tool call with arguments, truncating long values."""
    
    tool_icons = {
        "read_file": "📖",
        "write_file": "✏️",
        "edit_file": "✂️",
        "ls": "📁",
        "glob": "🔍",
        "grep": "🔎",
        "shell": "⚡",
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "write_todos": "📋",
    }
    
    icon = tool_icons.get(tool_name, "🔧")
    args_str = format_tool_args(tool_input)
    
    # Display: icon tool_name(args)
    if args_str:
        console.print(f"[dim]{icon} {tool_name}({args_str})[/dim]")
    else:
        console.print(f"[dim]{icon} {tool_name}()[/dim]")


def display_text_content(text: str):
    """Display text content with markdown rendering."""
    if text.strip():
        if "```" in text or "#" in text or "**" in text:
            md = Markdown(text)
            console.print(md)
        else:
            console.print(text, style="white")


def extract_and_display_content(message_content):
    """Extract content from agent messages and display with rich formatting."""
    if isinstance(message_content, str):
        display_text_content(message_content)
        return

    if isinstance(message_content, list):
        for block in message_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    display_text_content(block["text"])
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    tool_input = block.get("input", {})
                    display_tool_call(tool_name, tool_input)
                # Skip tool_result blocks - they're just noise
                elif block.get("type") == "tool_result":
                    pass  # Don't display tool results


def execute_task(user_input: str, agent, assistant_id: str | None):
    """Execute any task by passing it directly to the AI agent."""
    console.print()
    
    config = {
        "configurable": {"thread_id": "main"},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {}
    }
    
    for _, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        chunk = list(chunk.values())[0]
        if chunk is not None:
            # Check for interrupts
            if "__interrupt__" in chunk:
                result = chunk
                break
            
            # Normal message processing
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                message_content = None
                message_role = getattr(last_message, "type", None)
                if isinstance(message_role, dict):
                    message_role = last_message.get("role", "unknown")

                if hasattr(last_message, "content"):
                    message_content = last_message.content
                elif isinstance(last_message, dict) and "content" in last_message:
                    message_content = last_message["content"]

                if message_content:
                    # Show tool calls with truncated args
                    if message_role != "tool":
                        if isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown_tool")
                                    tool_input = block.get("input", {})
                                    
                                    tool_icons = {
                                        "read_file": "📖",
                                        "write_file": "✏️",
                                        "edit_file": "✂️",
                                        "ls": "📁",
                                        "glob": "🔍",
                                        "grep": "🔎",
                                        "shell": "⚡",
                                        "web_search": "🌐",
                                        "http_request": "🌍",
                                        "task": "🤖",
                                        "write_todos": "📋",
                                    }
                                    
                                    icon = tool_icons.get(tool_name, "🔧")
                                    args_str = format_tool_args(tool_input)
                                    
                                    if args_str:
                                        console.print(f"[dim]{icon} {tool_name}({args_str})[/dim]")
                                    else:
                                        console.print(f"[dim]{icon} {tool_name}()[/dim]")
                    
                    # Show tool results with truncation
                    if message_role == "tool":
                        result_str = str(message_content)
                        result_str = truncate_value(result_str, MAX_RESULT_LENGTH)
                        console.print(f"[dim]  → {result_str}[/dim]")
                    
                    # Show text content
                    if message_role != "tool":
                        has_text_content = False
                        if isinstance(message_content, str):
                            has_text_content = True
                        elif isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text" and block.get("text", "").strip():
                                        has_text_content = True
                                        break
                        
                        if has_text_content:
                            extract_and_display_content(message_content)
                            console.print()
    
    console.print()


async def simple_cli(agent, assistant_id: str | None):
    """Main CLI loop."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DeepAgents[/bold cyan] [dim]|[/dim] AI Coding Assistant",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print("[dim]Type 'quit' to exit[/dim]")
    
    if tavily_client is None:
        console.print("[yellow]⚠ Web search disabled:[/yellow] TAVILY_API_KEY not found.")
        console.print("[dim]  To enable web search, set your Tavily API key:[/dim]")
        console.print("[dim]    export TAVILY_API_KEY=your_api_key_here[/dim]")
        console.print("[dim]  Or add it to your .env file. Get your key at: https://tavily.com[/dim]")
    
    console.print()

    while True:
        try:
            console.print("[bold green]❯[/bold green] ", end="")
            user_input = input().strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n[bold cyan]👋 Goodbye![/bold cyan]\n")
            break

        else:
            console.print("[dim]✓ Command sent[/dim]")
            execute_task(user_input, agent, assistant_id)


def list_agents():
    """List all available agents."""
    agents_dir = Path.home() / ".deepagents"
    
    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print("[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]")
        return
    
    console.print("\n[bold cyan]Available Agents:[/bold cyan]\n")
    
    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"
            
            if agent_md.exists():
                console.print(f"  [green]•[/green] [bold]{agent_name}[/bold]")
                console.print(f"    [dim]{agent_path}[/dim]")
            else:
                console.print(f"  [yellow]•[/yellow] [bold]{agent_name}[/bold] [dim](incomplete)[/dim]")
                console.print(f"    [dim]{agent_path}[/dim]")
    
    console.print()


def reset_agent(agent_name: str, source_agent: str = None):
    """Reset an agent to default or copy from another agent."""
    agents_dir = Path.home() / ".deepagents"
    agent_dir = agents_dir / agent_name
    
    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"
        
        if not source_md.exists():
            console.print(f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found or has no agent.md")
            return
        
        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        # Reset to empty - agent builds their own memory
        source_content = ""
        action_desc = "empty (agent will build their own memory)"
    
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"[yellow]Removed existing agent directory:[/yellow] {agent_dir}")
    
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)
    
    console.print(f"[bold green]✓[/bold green] Agent '{agent_name}' reset to {action_desc}")
    console.print(f"[dim]Location: {agent_dir}[/dim]\n")


def show_help():
    """Show help information."""
    help_text = """
[bold cyan]DeepAgents - AI Coding Assistant[/bold cyan]

[bold]Usage:[/bold]
  deepagents [--agent NAME] [--no-memory]    Start interactive session
  deepagents list                            List all available agents
  deepagents reset --agent AGENT             Reset agent to default prompt
  deepagents reset --agent AGENT --target SOURCE   Reset agent to copy of another agent
  deepagents help                            Show this help message

[bold]Examples:[/bold]
  deepagents                          # Start with default agent (long-term memory enabled)
  deepagents --agent mybot            # Start with agent named 'mybot'
  deepagents --no-memory              # Start without long-term memory
  deepagents list                     # List all agents
  deepagents reset --agent mybot      # Reset mybot to default
  deepagents reset --agent mybot --target other   # Reset mybot to copy of 'other' agent

[bold]Long-term Memory:[/bold]
  By default, long-term memory is ENABLED using agent name 'agent'.
  Memory includes:
  - Persistent agent.md file with your instructions
  - /memories/ folder for storing context across sessions
  
  Use --no-memory to disable these features.
  Note: --agent and --no-memory cannot be used together.

[bold]Agent Storage:[/bold]
  Agents are stored in: ~/.deepagents/AGENT_NAME/
  Each agent has an agent.md file containing its prompt

[bold]Interactive Commands:[/bold]
  quit, exit, q    Exit the session
  help             Show usage examples
    """
    console.print(help_text)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    subparsers.add_parser("list", help="List all available agents")
    
    # Help command
    subparsers.add_parser("help", help="Show help information")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset an agent")
    reset_parser.add_argument("--agent", required=True, help="Name of agent to reset")
    reset_parser.add_argument("--target", dest="source_agent", help="Copy prompt from another agent")
    
    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable long-term memory features (no agent.md or /memories/ access).",
    )
    
    return parser.parse_args()


async def main(agent_name: str, no_memory: bool):
    """Main entry point."""
    # Error if both --agent and --no-memory are specified
    if agent_name != "agent" and no_memory:
        console.print("[bold red]Error:[/bold red] Cannot use --agent with --no-memory flag.")
        console.print("Either specify --agent for memory-enabled mode, or use --no-memory for memory-disabled mode.")
        return
    
    # Disable long-term memory if --no-memory flag is set
    long_term_memory = not no_memory
    
    # If memory is disabled, set assistant_id to None
    assistant_id = None if no_memory else agent_name
    
    # Create agent with conditional tools
    tools = [http_request]
    if tavily_client is not None:
        tools.append(web_search)
    
    # Always use default instructions - middleware handles agent.md loading
    agent = create_deep_agent(
        tools=tools,
        system_prompt=get_default_coding_instructions(),
        use_local_filesystem=True,
        long_term_memory=long_term_memory,
    ).with_config(config)
    
    agent.checkpointer = InMemorySaver()
    
    try:
        await simple_cli(agent, assistant_id)
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]👋 Goodbye![/bold cyan]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")


def cli_main():
    """Entry point for console script."""
    args = parse_args()
    
    if args.command == "help":
        show_help()
    elif args.command == "list":
        list_agents()
    elif args.command == "reset":
        reset_agent(args.agent, args.source_agent)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable is not set.")
            console.print("Please set your Anthropic API key:")
            console.print("  export ANTHROPIC_API_KEY=your_api_key_here")
            console.print("\nOr add it to your .env file.")
            return
        asyncio.run(main(args.agent, args.no_memory))


if __name__ == "__main__":
    cli_main()
