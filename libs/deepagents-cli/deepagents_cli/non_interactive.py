"""Non-interactive execution mode for deepagents CLI."""

import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search
from deepagents_cli.config import settings


async def run_non_interactive(
    message: str,
    assistant_id: str = "agent",
    model_name: str | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
) -> int:
    """Run a single task non-interactively and exit.

    Args:
        message: The task/message to execute
        assistant_id: Agent identifier for memory storage
        model_name: Optional model name to use
        sandbox_type: Type of sandbox ("none", "modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse

    Returns:
        Exit code: 0 for success, 1 for error
    """
    console = Console()
    
    # Create model
    model = create_model(model_name)
    
    # Generate thread ID
    thread_id = generate_thread_id()
    
    # Create config
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "assistant_id": assistant_id,
            "agent_name": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    }
    
    console.print(f"[dim]Running task non-interactively...[/dim]")
    console.print(f"[dim]Agent: {assistant_id} | Thread: {thread_id}[/dim]\n")
    
    # Setup sandbox if needed
    sandbox_backend = None
    sandbox_cm = None
    
    if sandbox_type != "none":
        from deepagents_cli.integrations.sandbox_factory import create_sandbox
        try:
            sandbox_cm = create_sandbox(sandbox_type, sandbox_id=sandbox_id)
            sandbox_backend = sandbox_cm.__enter__()
        except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
            console.print(f"[red]❌ Sandbox creation failed: {e}[/red]")
            return 1
    
    try:
        # Use async context manager for checkpointer
        async with get_checkpointer() as checkpointer:
            # Create agent with conditional tools
            tools = [http_request, fetch_url]
            if settings.has_tavily:
                tools.append(web_search)
            
            # Create agent with auto-approve enabled
            agent, composite_backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=sandbox_type if sandbox_type != "none" else None,
                auto_approve=True,  # Always auto-approve in non-interactive mode
                checkpointer=checkpointer,
            )
            
            # Execute the task
            stream_input = {"messages": [{"role": "user", "content": message}]}
            
            file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=composite_backend)
            
            # Track assistant response
            full_response = []
            tool_call_buffers = {}
            
            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                
                namespace, stream_mode, data = chunk
                
                # Only show main agent output (empty namespace)
                is_main_agent = not namespace
                
                if stream_mode == "messages" and is_main_agent:
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    
                    message_obj, metadata = data
                    
                    # Skip summarization chunks
                    if metadata and metadata.get("lc_source") == "summarization":
                        continue
                    
                    if isinstance(message_obj, AIMessage):
                        # Process content blocks for streaming text
                        if hasattr(message_obj, "content_blocks"):
                            for block in message_obj.content_blocks:
                                block_type = block.get("type")
                                
                                if block_type == "text":
                                    text = block.get("text", "")
                                    if text:
                                        print(text, end="", flush=True)
                                        full_response.append(text)
                                
                                elif block_type in ("tool_call_chunk", "tool_call"):
                                    # Buffer tool call information
                                    chunk_name = block.get("name")
                                    chunk_id = block.get("id")
                                    chunk_index = block.get("index")
                                    
                                    buffer_key = chunk_index if chunk_index is not None else (
                                        chunk_id if chunk_id is not None else f"unknown-{len(tool_call_buffers)}"
                                    )
                                    
                                    if buffer_key not in tool_call_buffers:
                                        tool_call_buffers[buffer_key] = {"name": None, "id": None}
                                    
                                    if chunk_name:
                                        tool_call_buffers[buffer_key]["name"] = chunk_name
                                        # Show tool call when we get the name
                                        if full_response:
                                            print()  # Newline after any text
                                        console.print(f"[dim]🔧 Calling tool: {chunk_name}[/dim]")
                    
                    elif isinstance(message_obj, ToolMessage):
                        # Track file operations
                        record = file_op_tracker.complete_with_message(message_obj)
                        
                        # Show file operation diffs
                        if record and record.diff:
                            console.print(f"[dim]📝 {record.display_path}[/dim]")
            
            # Final newline
            if full_response:
                print()
            
            console.print(f"\n[green]✓ Task completed[/green]")
            return 0
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        return 1
    finally:
        # Clean up sandbox if we created one
        if sandbox_cm is not None:
            try:
                sandbox_cm.__exit__(None, None, None)
            except Exception:
                pass
