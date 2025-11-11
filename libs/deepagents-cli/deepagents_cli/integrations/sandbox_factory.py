"""Sandbox lifecycle management with context managers."""

import os
import shlex
import string
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

from deepagents_cli.config import console


def _run_sandbox_setup(backend: SandboxBackendProtocol, setup_script_path: str) -> None:
    """Run users setup script in sandbox with env var expansion.

    Args:
        backend: Sandbox backend instance
        setup_script_path: Path to setup script file

    Raises:
        FileNotFoundError: If script file doesn't exist
        RuntimeError: If setup script fails (non-zero exit code)
    """
    script_path = Path(setup_script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_script_path}")

    console.print(f"[dim]Running setup script: {setup_script_path}...[/dim]")

    # Read script content
    script_content = script_path.read_text()

    # Expand ${VAR} syntax using local environment
    template = string.Template(script_content)
    expanded_script = template.safe_substitute(os.environ)

    # Execute in sandbox with 5-minute timeout
    result = backend.execute(f"bash -c {shlex.quote(expanded_script)}")

    if result.exit_code != 0:
        console.print(f"[red]❌ Setup script failed (exit {result.exit_code}):[/red]")
        console.print(f"[dim]{result.output}[/dim]")
        raise RuntimeError("Setup failed - aborting")

    console.print("[green]✓ Setup complete[/green]")


@contextmanager
def create_modal_sandbox(
    sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[tuple["ModalBackend", str], None, None]:
    """Create or connect to Modal sandbox.

    Args:
        sandbox_id: Optional existing sandbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts

    Yields:
        (ModalBackend, sandbox_id)

    Raises:
        ImportError: Modal SDK not installed
        Exception: Sandbox creation/connection failed
        FileNotFoundError: Setup script not found
        RuntimeError: Setup script failed
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
            sandbox = modal.Sandbox.create(app=app, workdir="/workspace")
            sandbox_id = sandbox.object_id
            should_cleanup = True

            # Poll until running (Modal requires this)
            for _ in range(90):  # 180s timeout (90 * 2s)
                if sandbox.poll() is not None:  # Sandbox terminated unexpectedly
                    raise RuntimeError("Modal sandbox terminated unexpectedly during startup")
                # Check if sandbox is ready by attempting a simple command
                try:
                    process = sandbox.exec("echo", "ready", timeout=5)
                    process.wait()
                    if process.returncode == 0:
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                # Timeout - cleanup and fail
                sandbox.terminate()
                raise RuntimeError("Modal sandbox failed to start within 180 seconds")

        console.print(f"[green]✓ Modal sandbox ready: {sandbox_id}[/green]")

        backend = ModalBackend(sandbox)

        # Run setup script if provided
        if setup_script_path:
            _run_sandbox_setup(backend, setup_script_path)

        try:
            yield backend, sandbox_id
        finally:
            if should_cleanup:
                try:
                    console.print(f"[dim]Terminating Modal sandbox {sandbox_id}...[/dim]")
                    sandbox.terminate()
                    console.print(f"[dim]✓ Modal sandbox {sandbox_id} terminated[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


@contextmanager
def create_runloop_sandbox(
    sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[tuple["RunloopBackend", str], None, None]:
    """Create or connect to Runloop devbox.

    Args:
        sandbox_id: Optional existing devbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts

    Yields:
        (RunloopBackend, devbox_id)

    Raises:
        ImportError: Runloop SDK not installed
        ValueError: RUNLOOP_API_KEY not set
        RuntimeError: Devbox failed to start within timeout
        FileNotFoundError: Setup script not found
        RuntimeError: Setup script failed
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

    backend = RunloopBackend(devbox_id=devbox.id, client=client)

    # Run setup script if provided
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)

    try:
        yield backend, devbox.id
    finally:
        if should_cleanup:
            try:
                console.print(f"[dim]Shutting down Runloop devbox {sandbox_id}...[/dim]")
                client.devboxes.shutdown(id=devbox.id)
                console.print(f"[dim]✓ Runloop devbox {sandbox_id} terminated[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


@contextmanager
def create_daytona_sandbox(
    sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[tuple["DaytonaBackend", str], None, None]:
    """Create Daytona sandbox.

    Args:
        sandbox_id: Optional existing sandbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts

    Yields:
        (DaytonaBackend, sandbox_id)

    Raises:
        ImportError: Daytona SDK not installed
        ValueError: DAYTONA_API_KEY not set
        NotImplementedError: If sandbox_id provided (not yet supported)
        FileNotFoundError: Setup script not found
        RuntimeError: Setup script failed

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
    sandbox_id = sandbox.id

    # Poll until running (Daytona requires this)
    for _ in range(90):  # 180s timeout (90 * 2s)
        # Check if sandbox is ready by attempting a simple command
        try:
            result = sandbox.process.exec("echo ready", timeout=5)
            if result.exit_code == 0:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        try:
            # Clean up if possible
            sandbox.delete()
        finally:
            raise RuntimeError("Daytona sandbox failed to start within 180 seconds")

    console.print(f"[green]✓ Daytona sandbox ready: {sandbox_id}[/green]")

    backend = DaytonaBackend(sandbox)

    # Run setup script if provided
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)
    try:
        yield backend, sandbox_id
    finally:
        console.print(f"[dim]Deleting Daytona sandbox {sandbox_id}...[/dim]")
        try:
            sandbox.delete()
            console.print(f"[dim]✓ Daytona sandbox {sandbox_id} terminated[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


__all__ = [
    "create_daytona_sandbox",
    "create_modal_sandbox",
    "create_runloop_sandbox",
]
