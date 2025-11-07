"""Factory for creating and managing sandbox backends."""

from typing import Optional, Tuple

from .config import console


def create_sandbox_backend(
    sandbox_type: str,
    sandbox_id: Optional[str] = None,
) -> Tuple[Optional[object], Optional[str]]:
    """Create or connect to a sandbox backend.

    Args:
        sandbox_type: Type of sandbox ("modal", "daytona", "runloop")
        sandbox_id: Optional existing sandbox ID to reuse

    Returns:
        (backend_instance, sandbox_id) or (None, None) on error
    """

    if sandbox_type == "modal":
        try:
            import modal
            from deepagents_cli.integrations.modal import ModalBackend
        except ImportError:
            console.print("[red]Error: Modal SDK not installed[/red]")
            console.print("Install with: [cyan]pip install modal[/cyan]")
            return None, None

        console.print("[yellow]Initializing Modal sandbox...[/yellow]")

        try:
            if sandbox_id:
                # Connect to existing sandbox
                sandbox = modal.Sandbox.from_id(sandbox_id)
                console.print(f"[green]✓ Connected to sandbox: {sandbox_id}[/green]")
            else:
                # Create new sandbox
                sandbox = modal.Sandbox.create()
                console.print(f"[green]✓ Created sandbox: {sandbox.object_id}[/green]")

            return ModalBackend(sandbox), sandbox.object_id

        except Exception as e:
            console.print(f"[red]Failed to initialize Modal sandbox: {e}[/red]")
            return None, None

    elif sandbox_type == "daytona":
        # TODO: Implement Daytona support
        console.print("[yellow]Daytona support coming soon[/yellow]")
        console.print("[dim]Track progress at: https://github.com/langchain-ai/deepagents/pull/320[/dim]")
        return None, None

    elif sandbox_type == "runloop":
        try:
            from runloop_api_client import Runloop
            from deepagents_cli.integrations.runloop import RunloopBackend
        except ImportError:
            console.print("[red]Error: Runloop SDK not installed[/red]")
            console.print("Install with: [cyan]pip install runloop-api-client[/cyan]")
            return None, None

        console.print("[yellow]Initializing Runloop devbox...[/yellow]")

        try:
            import os
            import time

            bearer_token = os.environ.get("RUNLOOP_API_KEY")
            if not bearer_token:
                console.print("[red]Error: RUNLOOP_API_KEY environment variable not set[/red]")
                return None, None

            client = Runloop(bearer_token=bearer_token)

            if sandbox_id:
                # Connect to existing devbox
                devbox = client.devboxes.retrieve(id=sandbox_id)
                if devbox.status != "running":
                    console.print(f"[yellow]Devbox {sandbox_id} status: {devbox.status}[/yellow]")
                    console.print("[yellow]Waiting for devbox to be ready...[/yellow]")
                    while devbox.status != "running":
                        time.sleep(2)
                        devbox = client.devboxes.retrieve(id=sandbox_id)
                console.print(f"[green]✓ Connected to devbox: {sandbox_id}[/green]")
            else:
                # Create new devbox
                devbox = client.devboxes.create()
                console.print(f"[green]✓ Created devbox: {devbox.id}[/green]")
                console.print("[yellow]Waiting for devbox to be ready...[/yellow]")

                waited = 0
                while waited < 180:  # 3 minute timeout
                    status = client.devboxes.retrieve(id=devbox.id)
                    if status.status == "running":
                        console.print("[green]✓ Devbox ready[/green]")
                        break
                    time.sleep(2)
                    waited += 2
                else:
                    console.print("[red]Timeout: Devbox never reached running state[/red]")
                    client.devboxes.shutdown(id=devbox.id)
                    return None, None

            return RunloopBackend(devbox_id=devbox.id, client=client), devbox.id

        except Exception as e:
            console.print(f"[red]Failed to initialize Runloop devbox: {e}[/red]")
            return None, None

    else:
        # No sandbox ("none" or invalid type)
        return None, None


def cleanup_sandbox(
    sandbox_id: str,
    sandbox_type: str,
) -> None:
    """Cleanup sandbox

    Args:
        sandbox_id: ID of sandbox to terminate
        sandbox_type: Type of sandbox ("modal", "daytona", "runloop")
    """

    if not sandbox_id:
        return

    try:
        if sandbox_type == "modal":
            import modal

            console.print(f"[dim]Terminating sandbox {sandbox_id}...[/dim]")
            sandbox = modal.Sandbox.from_id(sandbox_id)
            sandbox.terminate()
            console.print(f"[green]✓ Sandbox terminated[/green]")

        elif sandbox_type == "daytona":
            # TODO: Implement Daytona cleanup
            pass

        elif sandbox_type == "runloop":
            from runloop_api_client import Runloop
            import os

            bearer_token = os.environ.get("RUNLOOP_API_KEY")
            if not bearer_token:
                console.print("[yellow]Warning: RUNLOOP_API_KEY not set, cannot cleanup[/yellow]")
                return

            client = Runloop(bearer_token=bearer_token)
            console.print(f"[dim]Shutting down devbox {sandbox_id}...[/dim]")
            client.devboxes.shutdown(id=sandbox_id)
            console.print(f"[green]✓ Devbox shut down[/green]")

    except Exception as e:
        # Log but don't crash on cleanup failure
        console.print(f"[yellow]Warning: Sandbox cleanup failed: {e}[/yellow]")
        console.print(f"[dim]Sandbox {sandbox_id} may still be running[/dim]")
