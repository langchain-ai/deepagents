"""Sandbox lifecycle management with provider abstraction."""

from __future__ import annotations

import contextlib
import importlib
import os
import shlex
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape as escape_markup

from deepagents_cli.config import console, get_glyphs
from deepagents_cli.integrations.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from deepagents.backends.protocol import SandboxBackendProtocol


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

    console.print(
        f"[dim]Running setup script: {escape_markup(setup_script_path)}...[/dim]"
    )

    # Read script content
    script_content = script_path.read_text(encoding="utf-8")

    # Expand ${VAR} syntax using local environment
    template = string.Template(script_content)
    expanded_script = template.safe_substitute(os.environ)

    # Execute expanded script in sandbox
    result = backend.execute(f"bash -c {shlex.quote(expanded_script)}")

    if result.exit_code != 0:
        console.print(f"[red]Setup script failed (exit {result.exit_code}):[/red]")
        console.print(f"[dim]{escape_markup(result.output)}[/dim]")
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


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _import_provider_module(
    module_name: str,
    *,
    provider: str,
    package: str,
) -> ModuleType:
    """Import an optional provider module with a provider-specific error message.

    Args:
        module_name: Python module name to import.
        provider: Sandbox provider name (e.g. "daytona").
        package: PyPI package name exposed by the CLI extra.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = (
            f"The '{provider}' sandbox provider requires the '{package}' package. "
            f"Install it with: pip install 'deepagents-cli[{provider}]'"
        )
        raise ImportError(msg) from exc


class _DaytonaProvider(SandboxProvider):
    """Daytona sandbox provider — lifecycle management for Daytona sandboxes."""

    def __init__(self) -> None:
        daytona_module = _import_provider_module(
            "daytona",
            provider="daytona",
            package="langchain-daytona",
        )

        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            msg = "DAYTONA_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = daytona_module.Daytona(
            daytona_module.DaytonaConfig(
                api_key=api_key,
                api_url=os.environ.get("DAYTONA_API_URL"),
            )
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Get or create a Daytona sandbox.

        Args:
            sandbox_id: Not supported yet — must be None.
            timeout: Seconds to wait for startup.
            **kwargs: Unused.

        Returns:
            DaytonaSandbox instance.

        Raises:
            NotImplementedError: If sandbox_id is provided.
            RuntimeError: If the sandbox fails to start.
        """
        daytona_backend = _import_provider_module(
            "langchain_daytona",
            provider="daytona",
            package="langchain-daytona",
        )

        if sandbox_id:
            msg = (
                "Connecting to existing Daytona sandbox by ID not yet supported. "
                "Create a new sandbox by omitting sandbox_id parameter."
            )
            raise NotImplementedError(msg)

        sandbox = self._client.create()
        last_exc: Exception | None = None
        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception as exc:  # noqa: BLE001  # Transient failures expected during readiness polling
                last_exc = exc
            time.sleep(2)
        else:
            with contextlib.suppress(Exception):  # Best-effort cleanup
                sandbox.delete()
            detail = f" Last error: {last_exc}" if last_exc else ""
            msg = f"Daytona sandbox failed to start within {timeout} seconds.{detail}"
            raise RuntimeError(msg)

        return daytona_backend.DaytonaSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Delete a Daytona sandbox by id."""
        sandbox = self._client.get(sandbox_id)
        self._client.delete(sandbox)


class _ModalProvider(SandboxProvider):
    """Modal sandbox provider — lifecycle management for Modal sandboxes."""

    def __init__(self) -> None:
        self._modal = _import_provider_module(
            "modal",
            provider="modal",
            package="langchain-modal",
        )

        self._app = self._modal.App.lookup(
            name="deepagents-sandbox",
            create_if_missing=True,
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Get or create a Modal sandbox.

        Args:
            sandbox_id: Existing sandbox ID, or None to create.
            timeout: Seconds to wait for startup.
            **kwargs: Unused.

        Returns:
            ModalSandbox instance.

        Raises:
            RuntimeError: If the sandbox fails to start.
        """
        modal_backend = _import_provider_module(
            "langchain_modal",
            provider="modal",
            package="langchain-modal",
        )

        if sandbox_id:
            sandbox = self._modal.Sandbox.from_id(
                sandbox_id=sandbox_id,
                app=self._app,
            )
        else:
            sandbox = self._modal.Sandbox.create(app=self._app, workdir="/workspace")
            last_exc: Exception | None = None
            for _ in range(timeout // 2):
                if sandbox.poll() is not None:
                    msg = "Modal sandbox terminated unexpectedly during startup"
                    raise RuntimeError(msg)
                try:
                    process = sandbox.exec("echo", "ready", timeout=5)
                    process.wait()
                    if process.returncode == 0:
                        break
                except Exception as exc:  # noqa: BLE001  # Transient failures expected during readiness polling
                    last_exc = exc
                time.sleep(2)
            else:
                sandbox.terminate()
                detail = f" Last error: {last_exc}" if last_exc else ""
                msg = f"Modal sandbox failed to start within {timeout} seconds.{detail}"
                raise RuntimeError(msg)

        return modal_backend.ModalSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Terminate a Modal sandbox by id."""
        sandbox = self._modal.Sandbox.from_id(sandbox_id=sandbox_id, app=self._app)
        sandbox.terminate()


class _RunloopProvider(SandboxProvider):
    """Runloop sandbox provider — lifecycle management for Runloop devboxes."""

    def __init__(self) -> None:
        runloop_module = _import_provider_module(
            "runloop_api_client",
            provider="runloop",
            package="langchain-runloop",
        )

        api_key = os.environ.get("RUNLOOP_API_KEY")
        if not api_key:
            msg = "RUNLOOP_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client = runloop_module.Runloop(bearer_token=api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Get or create a Runloop devbox.

        Args:
            sandbox_id: Existing devbox ID, or None to create.
            timeout: Seconds to wait for startup.
            **kwargs: Unused.

        Returns:
            RunloopSandbox instance.

        Raises:
            RuntimeError: If the devbox fails to start.
            SandboxNotFoundError: If sandbox_id does not exist.
        """
        runloop_backend = _import_provider_module(
            "langchain_runloop",
            provider="runloop",
            package="langchain-runloop",
        )
        runloop_sdk = _import_provider_module(
            "runloop_api_client.sdk",
            provider="runloop",
            package="langchain-runloop",
        )

        if sandbox_id:
            try:
                self._client.devboxes.retrieve(id=sandbox_id)
            except KeyError as e:
                raise SandboxNotFoundError(sandbox_id) from e
        else:
            view = self._client.devboxes.create()
            sandbox_id = view.id
            for _ in range(timeout // 2):
                status = self._client.devboxes.retrieve(id=sandbox_id)
                if status.status == "running":
                    break
                time.sleep(2)
            else:
                self._client.devboxes.shutdown(id=sandbox_id)
                msg = f"Devbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        devbox = runloop_sdk.Devbox(self._client, sandbox_id)
        return runloop_backend.RunloopSandbox(devbox=devbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Shut down a Runloop devbox by id."""
        self._client.devboxes.shutdown(id=sandbox_id)


def _get_provider(provider_name: str) -> SandboxProvider:
    """Get a SandboxProvider instance for the specified provider (internal).

    Args:
        provider_name: Name of the provider ("daytona", "langsmith", "modal", "runloop")

    Returns:
        SandboxProvider instance

    Raises:
        ValueError: If provider_name is unknown.
    """
    if provider_name == "daytona":
        return _DaytonaProvider()
    if provider_name == "langsmith":
        from deepagents_cli.integrations.langsmith import LangSmithProvider

        return LangSmithProvider()
    if provider_name == "modal":
        return _ModalProvider()
    if provider_name == "runloop":
        return _RunloopProvider()
    msg = (
        f"Unknown sandbox provider: {provider_name}. "
        f"Available providers: {', '.join(_get_available_sandbox_types())}"
    )
    raise ValueError(msg)


__all__ = [
    "create_sandbox",
    "get_default_working_dir",
]
