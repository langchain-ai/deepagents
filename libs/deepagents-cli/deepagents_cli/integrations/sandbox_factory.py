"""Sandbox lifecycle management with provider abstraction."""

import shlex
import string
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.sandbox import SandboxProvider
import os

from deepagents_cli.config import console
from deepagents_cli.integrations.daytona import DaytonaProvider
from deepagents_cli.integrations.modal import ModalProvider
from deepagents_cli.integrations.runloop import RunloopProvider


def _run_sandbox_setup(backend: SandboxBackendProtocol, setup_script_path: str) -> None:
    """Run users setup script in sandbox with env var expansion.

    Args:
        backend: Sandbox backend instance
        setup_script_path: Path to setup script file
    """
    script_path = Path(setup_script_path)
    if not script_path.exists():
        msg = f"Setup script not found: {setup_script_path}"
        raise FileNotFoundError(msg)

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
        msg = "Setup failed - aborting"
        raise RuntimeError(msg)

    console.print("[green]✓ Setup complete[/green]")


# ============================================================================
# Unified Sandbox Interface
# ============================================================================

_PROVIDER_TO_WORKING_DIR = {
    "modal": "/workspace",
    "runloop": "/home/user",
    "daytona": "/home/daytona",
}


@contextmanager
def create_sandbox(
    provider: str,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
) -> Generator[SandboxBackendProtocol, None, None]:
    """Create or connect to a sandbox of the specified provider.

    This is the unified interface for sandbox creation using the provider abstraction.

    Args:
        provider: Sandbox provider ("modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts

    Yields:
        SandboxBackendProtocol instance

    Raises:
        ValueError: If provider is unknown
        ImportError: If required SDK is not installed
        RuntimeError: If sandbox creation/startup fails
    """
    with managed_sandbox(
        provider,
        sandbox_id=sandbox_id,
        setup_script_path=setup_script_path,
        cleanup=True,
    ) as backend:
        yield backend


def get_available_sandbox_types() -> list[str]:
    """Get list of available sandbox provider types.

    Returns:
        List of sandbox type names (e.g., ["modal", "runloop", "daytona"])
    """
    return ["modal", "runloop", "daytona"]


def get_default_working_dir(provider: str) -> str:
    """Get the default working directory for a given sandbox provider.

    Args:
        provider: Sandbox provider name ("modal", "runloop", "daytona")

    Returns:
        Default working directory path as string

    Raises:
        ValueError: If provider is unknown
    """
    if provider in _PROVIDER_TO_WORKING_DIR:
        return _PROVIDER_TO_WORKING_DIR[provider]
    msg = f"Unknown sandbox provider: {provider}"
    raise ValueError(msg)


def get_provider(provider_name: str) -> SandboxProvider:
    """Get a SandboxProvider instance for the specified provider.
    
    Args:
        provider_name: Name of the provider ("modal", "runloop", "daytona")
        
    Returns:
        SandboxProvider instance
        
    Raises:
        ValueError: If provider_name is unknown
        ImportError: If required SDK is not installed
    """
    if provider_name == "modal":
        return ModalProvider()
    elif provider_name == "runloop":
        return RunloopProvider()
    elif provider_name == "daytona":
        return DaytonaProvider()
    else:
        msg = (
            f"Unknown sandbox provider: {provider_name}. "
            f"Available providers: {', '.join(get_available_sandbox_types())}"
        )
        raise ValueError(msg)


@contextmanager
def managed_sandbox(
    provider: str | SandboxProvider,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
    cleanup: bool = True,
) -> Generator[SandboxBackendProtocol, None, None]:
    """Create or connect to a sandbox using the provider abstraction with lifecycle management.
    
    This is the new recommended interface that uses the SandboxProvider abstraction.
    
    Args:
        provider: Provider name ("modal", "runloop", "daytona") or SandboxProvider instance
        sandbox_id: Optional existing sandbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts
        cleanup: Whether to delete sandbox on exit (default: True, ignored if sandbox_id provided)

    Yields:
        SandboxBackendProtocol instance
        
    Example:
        ```python
        # Using provider name
        with managed_sandbox("modal", workdir="/workspace") as backend:
            result = backend.execute("echo hello")
            
        # Using provider instance
        provider = ModalProvider()
        with managed_sandbox(provider, sandbox_id="existing_id") as backend:
            result = backend.execute("ls")
        ```
    """
    # Get provider instance
    if isinstance(provider, str):
        provider_obj = get_provider(provider)
    else:
        provider_obj = provider
    
    # Determine if we should cleanup
    should_cleanup = cleanup and sandbox_id is None
    
    # Create or connect to sandbox
    backend = provider_obj.get_or_create(sandbox_id=sandbox_id)
    
    # Run setup script if provided
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)
    
    try:
        yield backend
    finally:
        if should_cleanup:
            try:
                provider_obj.delete(backend.id)
            except Exception as e:
                console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


__all__ = [
    "create_sandbox",
    "get_available_sandbox_types",
    "get_default_working_dir",
    "managed_sandbox",
]
