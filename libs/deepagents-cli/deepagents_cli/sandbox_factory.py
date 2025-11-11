"""Sandbox lifecycle management with context managers."""

import os
import time
from contextlib import contextmanager

from .config import console


@contextmanager
def create_modal_sandbox(sandbox_id: str | None = None):
    """Create or connect to Modal sandbox.

    Args:
        sandbox_id: Optional existing sandbox ID to reuse

    Yields:
        (ModalBackend, sandbox_id)

    Raises:
        ImportError: Modal SDK not installed
        Exception: Sandbox creation/connection failed
    """
    import modal

    from deepagents_cli.integrations.modal import ModalBackend

    console.print("[yellow]Starting Modal sandbox...[/yellow]")

    # Create ephemeral app (auto-cleans up on exit)
    app = modal.App("deepagents-sandbox")

    with app.run():
        if sandbox_id:
            sandbox = modal.Sandbox.from_id(sandbox_id, app=app)
            should_cleanup = False
        else:
            sandbox = modal.Sandbox.create(app=app)
            sandbox_id = sandbox.object_id
            should_cleanup = True

        console.print(f"[green]✓ Modal sandbox ready: {sandbox_id}[/green]")

        try:
            yield ModalBackend(sandbox), sandbox_id
        finally:
            if should_cleanup:
                try:
                    console.print(f"[dim]Terminating Modal sandbox {sandbox_id}...[/dim]")
                    sandbox.terminate()
                    console.print(f"[dim]✓ Modal sandbox {sandbox_id} terminated[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")
    # Ephemeral app auto-terminates here


@contextmanager
def create_runloop_sandbox(sandbox_id: str | None = None):
    """Create or connect to Runloop devbox.

    Args:
        sandbox_id: Optional existing devbox ID to reuse

    Yields:
        (RunloopBackend, devbox_id)

    Raises:
        ImportError: Runloop SDK not installed
        ValueError: RUNLOOP_API_KEY not set
        RuntimeError: Devbox failed to start within timeout
    """
    from runloop_api_client import Runloop

    from deepagents_cli.integrations.runloop import RunloopBackend

    bearer_token = os.environ.get("RUNLOOP_API_KEY")
    if not bearer_token:
        raise ValueError("RUNLOOP_API_KEY environment variable not set")

    client = Runloop(bearer_token=bearer_token)

    console.print("[yellow]Starting Runloop devbox...[/yellow]")

    if sandbox_id:
        devbox = client.devboxes.retrieve(id=sandbox_id)
        should_cleanup = False
    else:
        devbox = client.devboxes.create()
        sandbox_id = devbox.id
        should_cleanup = True

        # Poll until running (Runloop requires this)
        for _ in range(90):  # 180s timeout (90 * 2s)
            status = client.devboxes.retrieve(id=devbox.id)
            if status.status == "running":
                break
            time.sleep(2)
        else:
            # Timeout - cleanup and fail
            client.devboxes.shutdown(id=devbox.id)
            raise RuntimeError("Devbox failed to start within 180 seconds")

    console.print(f"[green]✓ Runloop devbox ready: {sandbox_id}[/green]")

    try:
        yield RunloopBackend(devbox_id=devbox.id, client=client), devbox.id
    finally:
        if should_cleanup:
            try:
                console.print(f"[dim]Shutting down Runloop devbox {sandbox_id}...[/dim]")
                client.devboxes.shutdown(id=devbox.id)
                console.print(f"[dim]✓ Runloop devbox {sandbox_id} terminated[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


@contextmanager
def create_daytona_sandbox(sandbox_id: str | None = None):
    """Create Daytona sandbox.

    Args:
        sandbox_id: Optional existing sandbox ID to reuse

    Yields:
        (DaytonaBackend, sandbox_id)

    Raises:
        ImportError: Daytona SDK not installed
        ValueError: DAYTONA_API_KEY not set
        NotImplementedError: If sandbox_id provided (not yet supported)

    Note:
        Connecting to existing Daytona sandbox by ID may not be supported yet.
        If sandbox_id is provided, this will raise NotImplementedError.
    """
    from daytona import Daytona, DaytonaConfig

    from deepagents_cli.integrations.daytona import DaytonaBackend

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable not set")

    if sandbox_id:
        raise NotImplementedError(
            "Connecting to existing Daytona sandbox by ID not yet supported. "
            "Create a new sandbox by omitting --sandbox-id."
        )

    console.print("[yellow]Starting Daytona sandbox...[/yellow]")

    daytona = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = daytona.create()

    # Try to get sandbox ID - fallback to placeholder if not available
    try:
        sandbox_id = sandbox.id
    except AttributeError:
        # Daytona SDK may not expose ID - use hash as placeholder
        sandbox_id = f"daytona-{id(sandbox)}"

    console.print(f"[green]✓ Daytona sandbox ready: {sandbox_id}[/green]")

    try:
        yield DaytonaBackend(sandbox), sandbox_id
    finally:
        try:
            console.print(f"[dim]Deleting Daytona sandbox {sandbox_id}...[/dim]")
            sandbox.delete()
            console.print(f"[dim]✓ Daytona sandbox {sandbox_id} terminated[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")
