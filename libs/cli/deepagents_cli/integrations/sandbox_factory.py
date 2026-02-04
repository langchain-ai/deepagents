"""Sandbox lifecycle management with provider abstraction."""

import asyncio
import os
import shlex
import string
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.sandbox import SandboxProvider

from deepagents_cli.config import console
from deepagents_cli.integrations.daytona import DaytonaProvider
from deepagents_cli.integrations.langsmith import LangSmithProvider
from deepagents_cli.integrations.modal import ModalProvider
from deepagents_cli.integrations.runloop import RunloopProvider


def _run_sandbox_setup(backend: SandboxBackendProtocol, setup_script_path: str) -> None:
    """Run users setup script in sandbox with env var expansion.

    Args:
        backend: Sandbox backend instance
        setup_script_path: Path to setup script file

    Raises:
        FileNotFoundError: If the setup script does not exist.
        RuntimeError: If the setup script fails to execute.
    """
    script_path = Path(setup_script_path)
    if not script_path.exists():
        msg = f"Setup script not found: {setup_script_path}"
        raise FileNotFoundError(msg)

    console.print(f"[dim]Running setup script: {setup_script_path}...[/dim]")

    # Read script content
    script_content = script_path.read_text(encoding="utf-8")

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


_PROVIDER_TO_WORKING_DIR = {
    "modal": "/workspace",
    "runloop": "/home/user",
    "daytona": "/home/daytona",
    "langsmith": "/tmp",  # noqa: S108
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
    """
    # Get provider instance
    provider_obj = _get_provider(provider)

    # Determine if we should cleanup (only cleanup if we created it)
    should_cleanup = sandbox_id is None

    # Create or connect to sandbox
    console.print(f"[yellow]Starting {provider} sandbox...[/yellow]")
    backend = provider_obj.get_or_create(sandbox_id=sandbox_id)
    console.print(
        f"[green]✓ {provider.capitalize()} sandbox ready: {backend.id}[/green]"
    )

    # Run setup script if provided
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)

    try:
        yield backend
    finally:
        if should_cleanup:
            try:
                console.print(
                    f"[dim]Terminating {provider} sandbox {backend.id}...[/dim]"
                )
                provider_obj.delete(sandbox_id=backend.id)
                console.print(
                    f"[dim]✓ {provider.capitalize()} sandbox "
                    f"{backend.id} terminated[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


def _get_available_sandbox_types() -> list[str]:
    """Get list of available sandbox provider types (internal).

    Returns:
        List of available sandbox provider type names
    """
    return sorted(_PROVIDER_TO_WORKING_DIR.keys())


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


def _get_provider(provider_name: str) -> SandboxProvider:
    """Get a SandboxProvider instance for the specified provider (internal).

    Args:
        provider_name: Name of the provider ("modal", "runloop", "daytona", "langsmith")

    Returns:
        SandboxProvider instance

    Raises:
        ValueError: If provider_name is unknown
    """
    if provider_name == "modal":
        return ModalProvider()
    if provider_name == "runloop":
        return RunloopProvider()
    if provider_name == "daytona":
        return DaytonaProvider()
    if provider_name == "langsmith":
        return LangSmithProvider()
    msg = (
        f"Unknown sandbox provider: {provider_name}. "
        f"Available providers: {', '.join(_get_available_sandbox_types())}"
    )
    raise ValueError(msg)


@asynccontextmanager
async def create_langsmith_sandbox_async(
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
    cleanup: bool = True,
) -> AsyncGenerator[SandboxBackendProtocol, None]:
    """Create or connect to LangSmith sandbox (async version).

    Args:
        sandbox_id: Optional existing sandbox name to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts
        cleanup: If True, delete sandbox on exit. If False, sandbox persists.

    Yields:
        LangSmithBackend instance

    Raises:
        ImportError: LangSmith SDK not installed
        ValueError: LANGSMITH_API_KEY not set
        RuntimeError: Sandbox failed to start within timeout
        FileNotFoundError: Setup script not found
        RuntimeError: Setup script failed
    """
    from langsmith import sandbox

    from deepagents_cli.integrations.langsmith import (
        LangSmithBackend,
        create_sandbox_instance,
        ensure_template,
        verify_sandbox_ready,
    )

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGSMITH_API_KEY_PROD")
    if not api_key:
        msg = "LANGSMITH_API_KEY or LANGSMITH_API_KEY_PROD environment variable not set"
        raise ValueError(msg)

    langsmith_endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    sandbox_endpoint = f"{langsmith_endpoint.rstrip('/')}/v2/sandboxes"

    console.print("[yellow]Starting LangSmith sandbox...[/yellow]")

    # Run blocking SDK calls in executor
    loop = asyncio.get_event_loop()
    client = await loop.run_in_executor(
        None,
        lambda: sandbox.SandboxClient(
            api_endpoint=sandbox_endpoint,
            api_key=api_key,
        ),
    )

    if sandbox_id:
        # Connect to existing sandbox by name
        try:
            sb = await loop.run_in_executor(None, client.get_sandbox, sandbox_id)
        except Exception as e:
            msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
            raise RuntimeError(msg) from e

        # Verify the existing sandbox is ready
        await loop.run_in_executor(None, verify_sandbox_ready, sb, client)

        should_cleanup = False
        console.print(f"[green]✓ Connected to existing LangSmith sandbox: {sb.name}[/green]")
    else:
        # Ensure template exists
        await loop.run_in_executor(None, ensure_template, client)

        # Create sandbox instance (already includes readiness check)
        sb = await loop.run_in_executor(None, create_sandbox_instance, client)
        should_cleanup = cleanup  # Only cleanup if requested

    backend = LangSmithBackend(sb)

    # Run setup script if provided
    if setup_script_path:
        await loop.run_in_executor(None, _run_sandbox_setup, backend, setup_script_path)

    try:
        yield backend
    finally:
        if should_cleanup:
            console.print(f"[dim]Deleting LangSmith sandbox {sb.name}...[/dim]")
            try:
                await loop.run_in_executor(None, client.delete_sandbox, sb.name)
                console.print(f"[dim]✓ LangSmith sandbox {sb.name} terminated[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ Cleanup failed: {e}[/yellow]")


# Mapping of sandbox types to their async context manager factories
_SANDBOX_PROVIDERS_ASYNC = {
    "langsmith": create_langsmith_sandbox_async,
}


@asynccontextmanager
async def create_sandbox_async(
    provider: str,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
    cleanup: bool = True,
) -> AsyncGenerator[SandboxBackendProtocol, None]:
    """Create or connect to a sandbox of the specified provider (async version).

    This is the unified async interface for sandbox creation that delegates to
    the appropriate provider-specific async context manager.

    Args:
        provider: Sandbox provider (currently only "langsmith" is supported)
        sandbox_id: Optional existing sandbox ID to reuse
        setup_script_path: Optional path to setup script to run after sandbox starts
        cleanup: If True, delete sandbox on exit. If False, sandbox persists.

    Yields:
        SandboxBackend instance

    Raises:
        ValueError: If provider is unknown or doesn't have async support
    """
    if provider not in _SANDBOX_PROVIDERS_ASYNC:
        msg = (
            f"Async sandbox provider not available for: {provider}. "
            f"Available async providers: {', '.join(_SANDBOX_PROVIDERS_ASYNC.keys())}"
        )
        raise ValueError(msg)

    sandbox_provider = _SANDBOX_PROVIDERS_ASYNC[provider]

    async with sandbox_provider(
        sandbox_id=sandbox_id, setup_script_path=setup_script_path, cleanup=cleanup
    ) as backend:
        yield backend


__all__ = [
    "create_sandbox",
    "create_sandbox_async",
    "get_default_working_dir",
]
