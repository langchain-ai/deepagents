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
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends import CompositeBackend
from deepagents.middleware.agent_memory import AgentMemoryMiddleware
from pathlib import Path
import shutil
from rich import box

import dotenv

dotenv.load_dotenv()

COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}

DEEP_AGENTS_ASCII = """
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
"""

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

MAX_ARG_LENGTH = 150


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def execute_task(user_input: str, agent, assistant_id: str | None):
    """Execute any task by passing it directly to the AI agent."""
    console.print()
    
    config = {
        "configurable": {"thread_id": "main"},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {}
    }
    
    has_responded = False
    current_text = ""
    printed_tool_calls_after_text = False
    
    status = console.status(f"[bold {COLORS['thinking']}]Agent is thinking...", spinner="dots")
    status.start()
    spinner_active = True
    
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
    
    for _, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        chunk_data = list(chunk.values())[0]
        if not chunk_data or "messages" not in chunk_data:
            continue
        
        last_message = chunk_data["messages"][-1]
        message_role = getattr(last_message, "type", None)
        message_content = getattr(last_message, "content", None)
        
        # Skip tool results
        if message_role == "tool":
            continue
        
        # Handle AI messages
        if message_role == "ai":
            # First, extract and display text content
            text_content = ""
            
            if isinstance(message_content, str):
                text_content = message_content
            elif isinstance(message_content, list):
                for block in message_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        break
            
            if text_content.strip():
                if spinner_active:
                    status.stop()
                    spinner_active = False
                
                if not has_responded:
                    console.print("... ", style=COLORS["agent"], end="", markup=False)
                    has_responded = True
                    printed_tool_calls_after_text = False
                
                if text_content != current_text:
                    new_text = text_content[len(current_text):]
                    console.print(new_text, style=COLORS["agent"], end="", markup=False)
                    current_text = text_content
                    printed_tool_calls_after_text = False
            
            # Then, handle tool calls from tool_calls attribute
            tool_calls = getattr(last_message, "tool_calls", None)
            if tool_calls:
                # If we've printed text, ensure tool calls go on new line
                if has_responded and current_text and not printed_tool_calls_after_text:
                    console.print()
                    printed_tool_calls_after_text = True
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    
                    icon = tool_icons.get(tool_name, "🔧")
                    args_str = ", ".join(
                        f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items()
                    )
                    
                    if spinner_active:
                        status.stop()
                    console.print(f"  {icon} {tool_name}({args_str})", style=f"dim {COLORS['tool']}")
                    if spinner_active:
                        status.start()
            
            # Handle tool calls from content blocks (alternative format) - only if not already handled
            elif isinstance(message_content, list):
                has_tool_use = False
                for block in message_content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        has_tool_use = True
                        break
                
                if has_tool_use:
                    # If we've printed text, ensure tool calls go on new line
                    if has_responded and current_text and not printed_tool_calls_after_text:
                        console.print()
                        printed_tool_calls_after_text = True
                    
                    for block in message_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            
                            icon = tool_icons.get(tool_name, "🔧")
                            args = ", ".join(
                                f"{k}={truncate_value(str(v), 50)}" for k, v in tool_input.items()
                            )
                            
                            if spinner_active:
                                status.stop()
                            console.print(f"  {icon} {tool_name}({args})", style=f"dim {COLORS['tool']}")
                            if spinner_active:
                                status.start()
    
    if spinner_active:
        status.stop()
    
    if has_responded:
        console.print()
        console.print()


async def simple_cli(agent, assistant_id: str | None):
    """Main CLI loop."""
    console.clear()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()
    
    if tavily_client is None:
        console.print(f"[yellow]⚠ Web search disabled:[/yellow] TAVILY_API_KEY not found.", style=COLORS["dim"])
        console.print(f"  To enable web search, set your Tavily API key:", style=COLORS["dim"])
        console.print(f"    export TAVILY_API_KEY=your_api_key_here", style=COLORS["dim"])
        console.print(f"  Or add it to your .env file. Get your key at: https://tavily.com", style=COLORS["dim"])
        console.print()
    
    console.print("... Ready to code! What would you like to build?", style=COLORS["agent"])
    console.print()
    console.print(f"  Tip: Type 'quit' to exit", style=f"dim {COLORS['dim']}")
    console.print()

    while True:
        try:
            console.print(f"> ", style=COLORS["user"], end="")
            user_input = input().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print()
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            console.print(f"\nGoodbye!", style=COLORS["primary"])
            break

        execute_task(user_input, agent, assistant_id)


def list_agents():
    """List all available agents."""
    agents_dir = Path.home() / ".deepagents"
    
    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(f"[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]", style=COLORS["dim"])
        return
    
    console.print(f"\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])
    
    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"
            
            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
    
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
        source_content = get_default_coding_instructions()
        action_desc = "default"
    
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])
    
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)
    
    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def show_help():
    """Show help information."""
    console.print()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()
    
    console.print("[bold]Usage:[/bold]", style=COLORS["primary"])
    console.print("  deepagents [--agent NAME]                      Start interactive session")
    console.print("  deepagents list                                List all available agents")
    console.print("  deepagents reset --agent AGENT                 Reset agent to default prompt")
    console.print("  deepagents reset --agent AGENT --target SOURCE Reset agent to copy of another agent")
    console.print("  deepagents help                                Show this help message")
    console.print()
    
    console.print("[bold]Examples:[/bold]", style=COLORS["primary"])
    console.print("  deepagents                              # Start with default agent", style=COLORS["dim"])
    console.print("  deepagents --agent mybot                # Start with agent named 'mybot'", style=COLORS["dim"])
    console.print("  deepagents list                         # List all agents", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot          # Reset mybot to default", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot --target other # Reset mybot to copy of 'other' agent", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Long-term Memory:[/bold]", style=COLORS["primary"])
    console.print("  By default, long-term memory is ENABLED using agent name 'agent'.", style=COLORS["dim"])
    console.print("  Memory includes:", style=COLORS["dim"])
    console.print("  - Persistent agent.md file with your instructions", style=COLORS["dim"])
    console.print("  - /memories/ folder for storing context across sessions", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Agent Storage:[/bold]", style=COLORS["primary"])
    console.print("  Agents are stored in: ~/.deepagents/AGENT_NAME/", style=COLORS["dim"])
    console.print("  Each agent has an agent.md file containing its prompt", style=COLORS["dim"])
    console.print()
    
    console.print("[bold]Interactive Commands:[/bold]", style=COLORS["primary"])
    console.print("  quit, exit, q    Exit the session", style=COLORS["dim"])
    console.print()


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
    
    return parser.parse_args()


async def main(assistant_id: str):
    """Main entry point."""
    
    # Create agent with conditional tools
    tools = [http_request]
    if tavily_client is not None:
        tools.append(web_search)

    shell_middleware = ShellToolMiddleware(
        workspace_root=os.getcwd(),
        execution_policy=HostExecutionPolicy()
    )

    # For long-term memory, point to ~/.deepagents/AGENT_NAME/ with /memories/ prefix
    agent_dir = Path.home() / ".deepagents" / assistant_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    if not agent_md.exists():
        source_content = get_default_coding_instructions()
        agent_md.write_text(source_content)

    # Long-term backend - rooted at agent directory
    # This handles both /memories/ files and /agent.md
    long_term_backend = FilesystemBackend(root_dir=agent_dir, virtual_mode=True)

    # Composite backend: current working directory for default, agent directory for /memories/
    backend = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/memories/": long_term_backend}
    )

    # Use the same backend for agent memory middleware
    agent_middleware = [AgentMemoryMiddleware(backend=long_term_backend, memory_path="/memories/"), shell_middleware]
    system_prompt = f"""### Current Working Directory

The filesystem backend is currently operating in: `{Path.cwd()}`"""

    agent = create_deep_agent(
        system_prompt=system_prompt,
        tools=tools,
        backend=backend,
        middleware=agent_middleware,
    ).with_config(config)
    
    agent.checkpointer = InMemorySaver()
    
    try:
        await simple_cli(agent, assistant_id)
    except KeyboardInterrupt:
        console.print(f"\n\nGoodbye!", style=COLORS["primary"])
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
        asyncio.run(main(args.agent))


if __name__ == "__main__":
    cli_main()