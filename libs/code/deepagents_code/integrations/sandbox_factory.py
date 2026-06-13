"""Sandbox lifecycle management with provider abstraction."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import shlex
import string
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rich.markup import escape as escape_markup

from deepagents_code.config import console, get_glyphs
from deepagents_code.integrations.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from deepagents.backends.protocol import SandboxBackendProtocol

    from deepagents_code.integrations.sandbox_registry import SandboxRegistry


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


@contextmanager
def create_sandbox(
    provider: str,
    *,
    sandbox_id: str | None = None,
    snapshot_name: str | None = None,
    setup_script_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> Generator[SandboxBackendProtocol, None, None]:
    """Create or connect to a sandbox of the specified provider.

    This is the unified interface for sandbox creation using the
    provider abstraction.

    Args:
        provider: Sandbox provider name. Built-ins (`'agentcore'`, `'daytona'`,
            `'langsmith'`, `'modal'`, `'runloop'`, `'vercel'`), entry-point
            providers, and config-declared providers are all resolved through
            the registry.
        sandbox_id: Optional existing sandbox ID to reuse
        snapshot_name: Optional sandbox snapshot name to use or create.
            Honored by providers whose metadata sets `supports_snapshot_name`
            (built-ins: `'langsmith'` snapshot, `'runloop'` blueprint); must be
            `None` for other providers.
        setup_script_path: Optional path to setup script to run after sandbox starts
        params: Extra keyword arguments forwarded to `provider.get_or_create()`
            (e.g. config-declared `[sandboxes.providers.<name>.params]`).

    Yields:
        `SandboxBackendProtocol` instance

    Raises:
        ValueError: If `snapshot_name` is provided for an unsupported provider,
            or combined with `sandbox_id` (snapshots only apply to fresh sandboxes).
    """
    registry = _get_registry()
    metadata = registry.get_metadata(provider)
    if snapshot_name is not None and (
        metadata is None or not metadata.supports_snapshot_name
    ):
        msg = (
            f"snapshot_name is not supported by provider {provider!r} "
            f"(got snapshot_name={snapshot_name!r})"
        )
        raise ValueError(msg)
    if snapshot_name is not None and sandbox_id is not None:
        msg = (
            "snapshot_name cannot be combined with sandbox_id; "
            "snapshots only apply when creating a fresh sandbox"
        )
        raise ValueError(msg)

    # Get provider instance (reuse the registry already built above so we
    # don't re-read the config file and re-scan entry points).
    provider_obj = _get_provider(provider, registry=registry)

    # Determine if we should cleanup (only cleanup if we created it)
    should_cleanup = sandbox_id is None
    provider_kwargs: dict[str, Any] = dict(registry.get_params(provider))
    if params:
        provider_kwargs.update(params)
    if snapshot_name is not None:
        provider_kwargs["snapshot"] = snapshot_name

    # Create or connect to sandbox
    console.print(f"[yellow]Starting {provider} sandbox...[/yellow]")
    backend = provider_obj.get_or_create(sandbox_id=sandbox_id, **provider_kwargs)
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


def _get_registry() -> SandboxRegistry:
    """Build a `SandboxRegistry` from the current user config.

    Not cached: each call re-reads the config file and re-scans entry points so
    the registry reflects the latest state. Reuse a single instance within one
    operation (see `create_sandbox`) rather than calling this repeatedly.

    Returns:
        A fresh `SandboxRegistry`.
    """
    from deepagents_code.integrations.sandbox_registry import SandboxRegistry

    return SandboxRegistry.load()


def _get_available_sandbox_types() -> list[str]:
    """Get list of available sandbox provider types (internal).

    Returns:
        List of available sandbox provider type names
    """
    return _get_registry().available_providers()


def get_default_working_dir(provider: str) -> str:
    """Get the default working directory for a given sandbox provider.

    Args:
        provider: Sandbox provider name. Resolved through the registry so
            built-in, entry-point, and config providers are all supported.

    Returns:
        Default working directory path as string

    Raises:
        ValueError: If provider is unknown
    """
    metadata = _get_registry().get_metadata(provider)
    if metadata is None:
        msg = f"Unknown sandbox provider: {provider}"
        raise ValueError(msg)
    return metadata.working_dir


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
        provider: Sandbox provider name (e.g. `'daytona'`).
        package: PyPI package name exposed by the package extra.

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
            f"Install it with: /install {provider} (in-app) or "
            f"dcode --install {provider} (CLI)"
        )
        raise ImportError(msg) from exc


_LANGSMITH_DEFAULT_SNAPSHOT = "deepagents-code"
"""Default LangSmith sandbox snapshot name used when none is specified."""

_LANGSMITH_DEFAULT_IMAGE = "python:3"
"""Default Docker image for LangSmith sandbox snapshots when none is provided."""

_LANGSMITH_DEFAULT_FS_CAPACITY_BYTES = 16 * 1024**3
"""Default filesystem capacity (16 GiB) for LangSmith sandbox snapshots."""


class _LangSmithProvider(SandboxProvider):
    """LangSmith sandbox provider implementation.

    Manages LangSmith sandbox lifecycle using the LangSmith SDK, booting
    sandboxes from snapshots built from a Docker image.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize LangSmith provider.

        Args:
            api_key: LangSmith API key (defaults to `LANGSMITH_SANDBOX_API_KEY`,
                then `LANGSMITH_API_KEY` env var).

        Raises:
            ValueError: If no LangSmith API key is found.
        """
        from langsmith.sandbox import SandboxClient

        from deepagents_code.model_config import resolve_env_var

        sandbox_key = resolve_env_var("LANGSMITH_SANDBOX_API_KEY")
        if sandbox_key:
            logger.debug("Using LangSmith API key from LANGSMITH_SANDBOX_API_KEY")
        self._api_key: str | None = (
            api_key
            or sandbox_key
            or resolve_env_var("LANGSMITH_API_KEY")
            or resolve_env_var("LANGCHAIN_API_KEY")
        )
        if not self._api_key:
            msg = (
                "No LangSmith sandbox API key found. Set "
                "LANGSMITH_API_KEY, LANGCHAIN_API_KEY, or LANGSMITH_SANDBOX_API_KEY "
                "(or the DEEPAGENTS_CODE_-prefixed equivalents)."
            )
            raise ValueError(msg)
        self._client: SandboxClient = SandboxClient(api_key=self._api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        snapshot: str | None = None,
        snapshot_image: str | None = None,
        fs_capacity_bytes: int | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing or create new LangSmith sandbox.

        Args:
            sandbox_id: Optional existing sandbox name to reuse.
            timeout: Timeout in seconds for sandbox startup.
            snapshot: Snapshot name to boot from.

                Resolved to a snapshot ID, creating the snapshot from
                `snapshot_image` if missing. Overrides
                `LANGSMITH_SANDBOX_SNAPSHOT_NAME`; overridden by
                `LANGSMITH_SANDBOX_SNAPSHOT_ID` (ID, wins over everything).
            snapshot_image: Docker image used when building the snapshot.
            fs_capacity_bytes: Filesystem capacity when building the snapshot.
            **kwargs: Additional LangSmith-specific parameters.

        Returns:
            `LangSmithSandbox` instance.

        Raises:
            RuntimeError: If sandbox connection or startup fails.
            TypeError: If unsupported keyword arguments are provided.
        """
        from deepagents.backends.langsmith import LangSmithSandbox

        from deepagents_code.model_config import resolve_env_var

        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)
        if sandbox_id:
            # Connect to existing sandbox by name
            try:
                sandbox = self._client.get_sandbox(name=sandbox_id)
            except Exception as e:
                msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e
            return LangSmithSandbox(sandbox)

        # Explicit snapshot ID wins — skip name lookup and auto-build.
        env_snapshot_id = resolve_env_var("LANGSMITH_SANDBOX_SNAPSHOT_ID")
        if env_snapshot_id:
            snapshot_id = env_snapshot_id
            snapshot_name = env_snapshot_id
        else:
            env_snapshot_name = resolve_env_var("LANGSMITH_SANDBOX_SNAPSHOT_NAME")
            snapshot_name = snapshot or env_snapshot_name or _LANGSMITH_DEFAULT_SNAPSHOT
            image = snapshot_image or _LANGSMITH_DEFAULT_IMAGE
            capacity = fs_capacity_bytes or _LANGSMITH_DEFAULT_FS_CAPACITY_BYTES
            snapshot_id = self._ensure_snapshot(snapshot_name, image, capacity)

        try:
            sandbox = self._client.create_sandbox(
                snapshot_id=snapshot_id, timeout=timeout
            )
        except Exception as e:
            msg = f"Failed to create sandbox from snapshot '{snapshot_name}': {e}"
            raise RuntimeError(msg) from e

        # Verify sandbox is ready by polling
        for _ in range(timeout // 2):
            try:
                result = sandbox.run("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: S110, BLE001  # Sandbox not ready yet, continue polling
                pass
            time.sleep(2)
        else:
            # Cleanup on failure
            with contextlib.suppress(Exception):
                self._client.delete_sandbox(sandbox.name)
            msg = f"LangSmith sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return LangSmithSandbox(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """Delete a LangSmith sandbox.

        Args:
            sandbox_id: Sandbox name to delete
            **kwargs: Additional parameters
        """
        self._client.delete_sandbox(sandbox_id)

    def _ensure_snapshot(
        self,
        snapshot_name: str,
        image: str,
        fs_capacity_bytes: int,
    ) -> str:
        """Resolve a snapshot by name, building it from `image` if missing.

        The LangSmith API exposes snapshots by ID, so we list and filter by
        name. Only snapshots with `status == "ready"` are returned; a
        matching-name snapshot in a non-ready state (`"building"`,
        `"failed"`, etc.) raises rather than triggering a duplicate build,
        which would mask the in-flight/failed snapshot.

        When no matching snapshot exists at all, we build one with
        `create_snapshot`, which blocks until the snapshot is ready.

        Returns:
            The snapshot ID ready to be passed to `create_sandbox`.

        Raises:
            RuntimeError: If listing or building the snapshot fails, or if
                a matching-name snapshot exists but is not ready.
        """
        try:
            snapshots = self._client.list_snapshots()
        except Exception as e:
            msg = f"Failed to list snapshots: {e}"
            raise RuntimeError(msg) from e

        non_ready_status: str | None = None
        for snap in snapshots:
            if snap.name != snapshot_name:
                continue
            if snap.status == "ready":
                return snap.id
            non_ready_status = snap.status

        if non_ready_status is not None:
            msg = (
                f"Snapshot '{snapshot_name}' exists but is in state "
                f"'{non_ready_status}'. Wait for it to finish building, or "
                f"delete it to rebuild."
            )
            raise RuntimeError(msg)

        try:
            snapshot = self._client.create_snapshot(
                name=snapshot_name,
                docker_image=image,
                fs_capacity_bytes=fs_capacity_bytes,
            )
        except Exception as create_err:
            msg = f"Failed to build snapshot '{snapshot_name}': {create_err}"
            raise RuntimeError(msg) from create_err
        return snapshot.id


class _DaytonaProvider(SandboxProvider):
    """Daytona sandbox provider — lifecycle management for Daytona sandboxes."""

    def __init__(self) -> None:
        daytona_module = _import_provider_module(
            "daytona",
            provider="daytona",
            package="langchain-daytona",
        )

        from deepagents_code.model_config import resolve_env_var

        api_key = resolve_env_var("DAYTONA_API_KEY")
        if not api_key:
            msg = (
                "No Daytona API key found. Set DAYTONA_API_KEY "
                "or DEEPAGENTS_CODE_DAYTONA_API_KEY."
            )
            raise ValueError(msg)
        self._client = daytona_module.Daytona(
            daytona_module.DaytonaConfig(
                api_key=api_key,
                api_url=resolve_env_var("DAYTONA_API_URL"),
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
            `DaytonaSandbox` instance.

        Raises:
            NotImplementedError: If `sandbox_id` is provided.
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

        from deepagents_code.model_config import resolve_env_var

        token_id = resolve_env_var("MODAL_TOKEN_ID")
        token_secret = resolve_env_var("MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            try:
                self._client = self._modal.Client.from_credentials(
                    token_id, token_secret
                )
            except Exception as exc:
                msg = (
                    "Failed to authenticate with Modal using "
                    "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET "
                    "(or the DEEPAGENTS_CODE_-prefixed equivalents). "
                    "Verify your credentials are valid."
                )
                raise ValueError(msg) from exc
        elif token_id or token_secret:
            logger.warning(
                "Only one of MODAL_TOKEN_ID / MODAL_TOKEN_SECRET is set; "
                "both are required for explicit credential auth. "
                "Falling back to default Modal authentication.",
            )
            self._client = None
        else:
            self._client = None

        lookup_kwargs: dict[str, Any] = {
            "name": "deepagents-sandbox",
            "create_if_missing": True,
        }
        if self._client is not None:
            lookup_kwargs["client"] = self._client
        self._app = self._modal.App.lookup(**lookup_kwargs)

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
            `ModalSandbox` instance.

        Raises:
            RuntimeError: If the sandbox fails to start.
        """
        modal_backend = _import_provider_module(
            "langchain_modal",
            provider="modal",
            package="langchain-modal",
        )

        client_kwargs: dict[str, Any] = {}
        if self._client is not None:
            client_kwargs["client"] = self._client

        if sandbox_id:
            sandbox = self._modal.Sandbox.from_id(
                sandbox_id=sandbox_id,
                app=self._app,
                **client_kwargs,
            )
        else:
            sandbox = self._modal.Sandbox.create(
                app=self._app, workdir="/workspace", **client_kwargs
            )
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
        del_kwargs: dict[str, Any] = {"sandbox_id": sandbox_id, "app": self._app}
        if self._client is not None:
            del_kwargs["client"] = self._client
        sandbox = self._modal.Sandbox.from_id(**del_kwargs)
        sandbox.terminate()


class _RunloopProvider(SandboxProvider):
    """Runloop sandbox provider — delegates to `langchain_runloop.RunloopProvider`."""

    def __init__(self) -> None:
        runloop_module = _import_provider_module(
            "langchain_runloop",
            provider="runloop",
            package="langchain-runloop",
        )

        from deepagents_code.model_config import resolve_env_var

        api_key = resolve_env_var("RUNLOOP_API_KEY")
        if not api_key:
            msg = (
                "No Runloop API key found. Set RUNLOOP_API_KEY "
                "or DEEPAGENTS_CODE_RUNLOOP_API_KEY."
            )
            raise ValueError(msg)
        self._provider = runloop_module.RunloopProvider(
            api_key=api_key,
            resolve_env_var=resolve_env_var,
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get or create a Runloop devbox.

        Args:
            sandbox_id: Existing devbox ID, or None to create.
            timeout: Accepted for parity with other providers; currently
                forwarded but unused by the Runloop backend (the SDK manages
                its own startup wait).
            **kwargs: Runloop-specific options (`snapshot` blueprint name,
                `blueprint_dockerfile`).

        Returns:
            `RunloopSandbox` instance.

        Raises:
            SandboxNotFoundError: If `sandbox_id` does not exist. `RunloopProvider`
                translates the SDK's not-found error into a `KeyError`, which is
                mapped here.
            KeyError: If a `KeyError` is raised while no `sandbox_id` was supplied
                (re-raised unchanged rather than mislabeled as not-found).
        """
        try:
            return self._provider.get_or_create(
                sandbox_id=sandbox_id,
                timeout=timeout,
                **kwargs,
            )
        except KeyError as e:
            if sandbox_id is None:
                raise
            raise SandboxNotFoundError(sandbox_id) from e

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Shut down a Runloop devbox by id."""
        self._provider.delete(sandbox_id=sandbox_id)


class _AgentCoreProvider(SandboxProvider):
    """AgentCore Code Interpreter sandbox provider.

    Manages AgentCore session lifecycle. Sessions cannot be reconnected after
    the app exits — the `sandbox_id` parameter is not supported.
    """

    def __init__(self, region: str | None = None) -> None:
        """Initialize AgentCore provider.

        Args:
            region: AWS region (defaults to `AWS_REGION` /
                `AWS_DEFAULT_REGION` / `us-west-2`).

        Raises:
            ValueError: If boto3 is installed and AWS credentials cannot
                be resolved.
        """
        self._region = region or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )

        # Validate AWS credentials early for a clear error message.
        try:
            import boto3  # ty: ignore[unresolved-import]

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                msg = (
                    "AWS credentials not found. Configure via "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN, "
                    "~/.aws/credentials, or an IAM role."
                )
                raise ValueError(msg)  # noqa: TRY301  # intentional raise for early credential validation
        except ImportError:
            logger.debug("boto3 not installed; skipping credential pre-check")
        except ValueError:
            raise
        except Exception:
            logger.warning(
                "AWS credential pre-validation failed — the session may "
                "fail to start. Check your AWS configuration.",
                exc_info=True,
            )

        self._active_interpreters: dict[str, Any] = {}

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,  # noqa: ARG002  # required by SandboxProvider interface
    ) -> SandboxBackendProtocol:
        """Create a new AgentCore Code Interpreter session.

        Args:
            sandbox_id: Not supported — raises `NotImplementedError`
                if provided.
            **kwargs: Additional parameters (unused).

        Returns:
            `AgentCoreSandbox` instance wrapping the started interpreter.

        Raises:
            NotImplementedError: If `sandbox_id` is provided.
        """
        if sandbox_id:
            msg = (
                "AgentCore does not support reconnecting to existing sessions. "
                "Remove the --sandbox-id option."
            )
            raise NotImplementedError(msg)

        agentcore_module = _import_provider_module(
            "bedrock_agentcore.tools.code_interpreter_client",
            provider="agentcore",
            package="langchain-agentcore-codeinterpreter",
        )
        agentcore_backend = _import_provider_module(
            "langchain_agentcore_codeinterpreter",
            provider="agentcore",
            package="langchain-agentcore-codeinterpreter",
        )

        interpreter = agentcore_module.CodeInterpreter(
            region=self._region,
            integration_source="deepagents-code",
        )
        try:
            interpreter.start()
        except Exception:
            with contextlib.suppress(Exception):
                interpreter.stop()
            raise

        backend = agentcore_backend.AgentCoreSandbox(interpreter=interpreter)
        self._active_interpreters[backend.id] = interpreter
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # required by SandboxProvider interface
        """Stop an AgentCore session.

        Args:
            sandbox_id: Session ID to stop.
            **kwargs: Additional parameters (unused).
        """
        interpreter = self._active_interpreters.pop(sandbox_id, None)
        if interpreter:
            try:
                interpreter.stop()
                logger.info("AgentCore session %s stopped", sandbox_id)
            except Exception:
                logger.warning(
                    "Failed to stop AgentCore session %s — the session may "
                    "still be running and incurring costs. Check the AWS "
                    "console to verify.",
                    sandbox_id,
                    exc_info=True,
                )
        else:
            logger.info(
                "AgentCore session %s not tracked (may have already expired)",
                sandbox_id,
            )


_VercelStatus = Literal[
    "pending",
    "running",
    "stopping",
    "stopped",
    "failed",
    "aborted",
    "snapshotting",
]
"""Vercel sandbox lifecycle statuses (mirrors the SDK's `SandboxStatus` enum)."""

_VERCEL_TERMINAL_STATUSES: frozenset[_VercelStatus] = frozenset(
    {"aborted", "failed", "stopped"}
)
"""Vercel sandbox statuses that cannot become ready without a new sandbox.

Typing the members as `_VercelStatus` turns a typo into a type error rather than
a silently-never-matching string.
"""

_VERCEL_DEFAULT_RUNTIME = "python3.13"
"""Runtime used when creating a fresh Vercel sandbox."""

_VERCEL_SANDBOX_TIMEOUT = timedelta(minutes=30)
"""Lifetime used for freshly created Vercel sandboxes."""


class _VercelSandboxHandle(Protocol):
    """Minimal Vercel SDK sandbox surface used by the built-in provider."""

    sandbox_id: str
    status: str

    def wait_for_status(self, status: str, *, timeout: int) -> None:
        """Wait for the sandbox to reach the requested status."""

    def stop(self) -> None:
        """Stop the sandbox."""


class _VercelProvider(SandboxProvider):
    """Vercel Sandbox provider implementation."""

    _CREDENTIAL_ENV_NAMES = (
        "VERCEL_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
    )

    def __init__(self) -> None:
        self._sdk_kwargs = self._resolve_sdk_kwargs()

    @classmethod
    def _resolve_sdk_kwargs(cls) -> dict[str, str]:
        """Resolve explicit Vercel credentials when a prefixed override is set.

        Credentials stay SDK-managed (canonical `VERCEL_*` variables or OIDC)
        unless at least one of the prefixed overrides
        `DEEPAGENTS_CODE_VERCEL_TOKEN`, `DEEPAGENTS_CODE_VERCEL_PROJECT_ID`, or
        `DEEPAGENTS_CODE_VERCEL_TEAM_ID` is present. When triggered, each value
        still resolves prefixed-first then canonical via `resolve_env_var`.

        Returns:
            Explicit SDK credential arguments, or an empty mapping to delegate
            credential resolution to the Vercel SDK.
        """
        prefix = "DEEPAGENTS_CODE_"
        has_override = any(
            f"{prefix}{name}" in os.environ for name in cls._CREDENTIAL_ENV_NAMES
        )
        if not has_override:
            return {}

        from deepagents_code.model_config import resolve_env_var

        values = {
            "token": resolve_env_var("VERCEL_TOKEN"),
            "project_id": resolve_env_var("VERCEL_PROJECT_ID"),
            "team_id": resolve_env_var("VERCEL_TEAM_ID"),
        }
        missing = sorted(key for key, value in values.items() if not value)
        if missing:
            logger.warning(
                "Incomplete explicit Vercel credentials; VERCEL_TOKEN, "
                "VERCEL_PROJECT_ID, and VERCEL_TEAM_ID are all required. "
                "Falling back to default Vercel authentication. Missing: %s",
                ", ".join(missing),
            )
            return {}
        return {key: value for key, value in values.items() if value}

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get or create a Vercel sandbox.

        Args:
            sandbox_id: Existing sandbox ID, or None to create.
            timeout: Seconds to wait for startup.
            **kwargs: Rejected; passing any keyword argument raises `TypeError`.

        Returns:
            `VercelSandbox` instance.

        Raises:
            RuntimeError: If the sandbox fails to start.
            TypeError: If unsupported keyword arguments are provided.
        """
        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        vercel_backend = _import_provider_module(
            "langchain_vercel_sandbox",
            provider="vercel",
            package="langchain-vercel-sandbox",
        )
        vercel_sandbox = _import_provider_module(
            "vercel.sandbox",
            provider="vercel",
            package="vercel",
        )

        # The SDK is imported dynamically, so `Sandbox.get`/`create` are typed
        # `Any`. Annotating pins the result to the Protocol so the type checker
        # verifies the `.status`/`.wait_for_status`/`.stop` calls below.
        sandbox: _VercelSandboxHandle
        try:
            if sandbox_id:
                sandbox = vercel_sandbox.Sandbox.get(
                    sandbox_id=sandbox_id,
                    **self._sdk_kwargs,
                )
            else:
                sandbox = vercel_sandbox.Sandbox.create(
                    runtime=_VERCEL_DEFAULT_RUNTIME,
                    timeout=_VERCEL_SANDBOX_TIMEOUT,
                    **self._sdk_kwargs,
                )
        except Exception as exc:  # Vercel SDK exception types vary by version
            # Keep the message generic so SDK errors cannot leak credentials,
            # but chain the original exception so tracebacks retain root cause.
            action = "connect to existing" if sandbox_id else "create"
            msg = f"Failed to {action} Vercel sandbox."
            raise RuntimeError(msg) from exc

        try:
            self._wait_until_running(sandbox, timeout=timeout)
        except Exception:
            if sandbox_id is None:
                with contextlib.suppress(Exception):
                    sandbox.stop()
            raise

        return vercel_backend.VercelSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # **kwargs required by SandboxProvider interface
        """Stop a Vercel sandbox by id.

        Raises:
            RuntimeError: If the Vercel SDK cannot find or stop the sandbox.
        """
        vercel_sandbox = _import_provider_module(
            "vercel.sandbox",
            provider="vercel",
            package="vercel",
        )
        try:
            sandbox = vercel_sandbox.Sandbox.get(
                sandbox_id=sandbox_id,
                **self._sdk_kwargs,
            )
            sandbox.stop()
        except Exception as exc:  # Vercel SDK exception types vary by version
            # Generic message avoids leaking credentials; chain preserves cause.
            msg = "Failed to stop Vercel sandbox."
            raise RuntimeError(msg) from exc

    @staticmethod
    def _wait_until_running(
        sandbox: _VercelSandboxHandle,
        *,
        timeout: int,
    ) -> None:
        """Wait for a Vercel sandbox to become ready.

        Raises:
            RuntimeError: If the sandbox reaches a terminal state or does not
                become ready before the timeout.
        """
        status = str(sandbox.status)
        if status == "running":
            return
        if status in _VERCEL_TERMINAL_STATUSES:
            msg = f"Vercel sandbox {sandbox.sandbox_id} is in terminal state {status!r}"
            raise RuntimeError(msg)

        try:
            sandbox.wait_for_status("running", timeout=timeout)
        except TimeoutError as exc:
            status = str(sandbox.status)
            msg = (
                f"Vercel sandbox {sandbox.sandbox_id} failed to start within "
                f"{timeout} seconds; current status is {status!r}"
            )
            raise RuntimeError(msg) from exc
        except Exception as exc:  # Vercel SDK exception types vary by version
            # Generic message avoids leaking credentials; chain preserves cause.
            msg = "Failed while waiting for Vercel sandbox startup."
            raise RuntimeError(msg) from exc


def _get_provider(
    provider_name: str,
    registry: SandboxRegistry | None = None,
) -> SandboxProvider:
    """Get a `SandboxProvider` instance for the specified provider (internal).

    Args:
        provider_name: Name of the provider. Resolved through the registry so
            built-in, entry-point, and config providers are all supported.
        registry: An already-built registry to reuse. A fresh one is loaded
            when omitted.

    Returns:
        `SandboxProvider` instance. Propagates `ValueError` from the registry
            if `provider_name` is unknown.
    """
    reg = registry if registry is not None else _get_registry()
    return reg.create_provider(provider_name)


def verify_sandbox_deps(provider: str) -> None:
    """Check that the required packages for a sandbox provider are installed.

    Uses `importlib.util.find_spec` for a lightweight check with no actual
    imports. Call this in the app's process *before* spawning the server
    subprocess so users get a clear, actionable error instead of an opaque
    server crash. The backend module to probe and the install hint both come
    from provider metadata.

    Args:
        provider: Sandbox provider name (e.g. `'daytona'`).

    Raises:
        ImportError: If the provider's backend package is not installed.
    """
    if not provider or provider == "none":
        return

    metadata = _get_registry().get_metadata(provider)
    if metadata is None or metadata.backend_module is None:
        logger.debug(
            "No backend module to probe for provider %r; skipping pre-flight check",
            provider,
        )
        return

    try:
        found = importlib.util.find_spec(metadata.backend_module) is not None
    except (ImportError, ValueError):
        found = False

    if not found:
        if metadata.install is not None:
            install_hint = (
                f"Install with: {metadata.install.command(in_app=True)} (in-app) "
                f"or {metadata.install.command(in_app=False)} (CLI)"
            )
        else:
            install_hint = "Install the provider's package."
        msg = f"Missing dependencies for '{provider}' sandbox. {install_hint}"
        raise ImportError(msg)


__all__ = [
    "create_sandbox",
    "get_default_working_dir",
    "verify_sandbox_deps",
]
