"""Sandbox lifecycle management with provider abstraction."""

from __future__ import annotations

import os
import shlex
import string
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_cli.config import console, get_glyphs

if TYPE_CHECKING:
    from collections.abc import Generator

    from deepagents.backends.protocol import SandboxBackendProtocol

    from deepagents_cli.integrations.sandbox_provider import SandboxProvider


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
        console.print(f"[red]Setup script failed (exit {result.exit_code}):[/red]")
        console.print(f"[dim]{result.output}[/dim]")
        msg = "Setup failed - aborting"
        raise RuntimeError(msg)

    console.print(f"[green]{get_glyphs().checkmark} Setup complete[/green]")


_PROVIDER_TO_WORKING_DIR = {
    "daytona": "/home/daytona",
    "langsmith": "/tmp",  # noqa: S108  # LangSmith sandbox working directory
    "modal": "/workspace",
    "runloop": "/home/user",
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
        provider: Sandbox provider ("daytona", "langsmith", "modal", "runloop")
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
    glyphs = get_glyphs()
    console.print(
        f"[green]{glyphs.checkmark} {provider.capitalize()} sandbox ready: "
        f"{backend.id}[/green]"
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
                glyphs = get_glyphs()
                console.print(
                    f"[dim]{glyphs.checkmark} {provider.capitalize()} sandbox "
                    f"{backend.id} terminated[/dim]"
                )
            except Exception as e:  # noqa: BLE001  # Cleanup errors should not mask the original sandbox failure
                warning = get_glyphs().warning
                console.print(
                    f"[yellow]{warning} Cleanup failed for {provider} sandbox "
                    f"{backend.id}: {e}[/yellow]"
                )


def _get_available_sandbox_types() -> list[str]:
    """Get list of available sandbox provider types (internal).

    Returns:
        List of available sandbox provider type names
    """
    return sorted(_PROVIDER_TO_WORKING_DIR.keys())


def get_default_working_dir(provider: str) -> str:
    """Get the default working directory for a given sandbox provider.

    Args:
        provider: Sandbox provider name ("daytona", "langsmith", "modal", "runloop")

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
        provider_name: Name of the provider ("daytona", "langsmith", "modal", "runloop")

    Returns:
        SandboxProvider instance

    Raises:
        ValueError: If provider_name is unknown
    """
    if provider_name == "daytona":
        from deepagents_cli.integrations.daytona import DaytonaProvider

        return DaytonaProvider()
    if provider_name == "langsmith":
        from deepagents_cli.integrations.langsmith import LangSmithProvider

        return LangSmithProvider()
    if provider_name == "modal":
        from deepagents_cli.integrations.modal import ModalProvider

        return ModalProvider()
    if provider_name == "runloop":
        from deepagents_cli.integrations.runloop import RunloopProvider

        return RunloopProvider()
    msg = (
        f"Unknown sandbox provider: {provider_name}. "
        f"Available providers: {', '.join(_get_available_sandbox_types())}"
    )
    raise ValueError(msg)


__all__ = [
    "create_sandbox",
    "get_default_working_dir",
]
