"""LangSmith sandbox provider for the CLI."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from deepagents.backends.langsmith import (
    LangSmithSandbox as LangSmithBackend,
)
from langsmith.sandbox import ResourceNotFoundError

from deepagents_cli.integrations.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol
    from langsmith.sandbox import SandboxClient, SandboxTemplate


# Default template configuration
DEFAULT_TEMPLATE_NAME = "deepagents-cli"
DEFAULT_TEMPLATE_IMAGE = "python:3"


class LangSmithProvider(SandboxProvider):
    """LangSmith sandbox provider implementation.

    Manages LangSmith sandbox lifecycle using the LangSmith SDK.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize LangSmith provider.

        Args:
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)

        Raises:
            ValueError: If LANGSMITH_API_KEY environment variable not set
        """
        from langsmith import sandbox

        self._api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        if not self._api_key:
            msg = "LANGSMITH_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client: SandboxClient = sandbox.SandboxClient(api_key=self._api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        template: str | None = None,
        template_image: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing or create new LangSmith sandbox.

        Args:
            sandbox_id: Optional existing sandbox name to reuse
            timeout: Timeout in seconds for sandbox startup (default: 180)
            template: Template name for the sandbox
            template_image: Docker image for the template
            **kwargs: Additional LangSmith-specific parameters

        Returns:
            LangSmithBackend instance

        Raises:
            RuntimeError: If sandbox connection or startup fails
            TypeError: If unsupported keyword arguments are provided
        """
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
            return LangSmithBackend(sandbox)

        resolved_template_name, resolved_image_name = self._resolve_template(
            template, template_image
        )

        # Create new sandbox - ensure template exists first
        self._ensure_template(resolved_template_name, resolved_image_name)

        try:
            sandbox = self._client.create_sandbox(
                template_name=resolved_template_name, timeout=timeout
            )
        except Exception as e:
            msg = (
                f"Failed to create sandbox from template "
                f"'{resolved_template_name}': {e}"
            )
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

        return LangSmithBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """Delete a LangSmith sandbox.

        Args:
            sandbox_id: Sandbox name to delete
            **kwargs: Additional parameters
        """
        self._client.delete_sandbox(sandbox_id)

    @staticmethod
    def _resolve_template(
        template: SandboxTemplate | str | None,
        template_image: str | None = None,
    ) -> tuple[str, str]:
        """Resolve template name and image from kwargs.

        Returns:
            Tuple of (template_name, template_image). Always returns values,
            using defaults if not provided.
        """
        resolved_image = template_image or DEFAULT_TEMPLATE_IMAGE
        if template is None:
            return DEFAULT_TEMPLATE_NAME, resolved_image
        if isinstance(template, str):
            return template, resolved_image
        # SandboxTemplate object - extract image if not provided
        if template_image is None and template.image:
            resolved_image = template.image
        return template.name, resolved_image

    def _ensure_template(
        self,
        template_name: str,
        template_image: str,
    ) -> None:
        """Ensure template exists, creating it if needed.

        Raises:
            RuntimeError: If template check or creation fails
        """
        try:
            self._client.get_template(template_name)
        except ResourceNotFoundError as e:
            if e.resource_type != "template":
                msg = f"Unexpected resource not found: {e}"
                raise RuntimeError(msg) from e
            # Template doesn't exist, create it
            try:
                self._client.create_template(name=template_name, image=template_image)
            except Exception as create_err:
                msg = f"Failed to create template '{template_name}': {create_err}"
                raise RuntimeError(msg) from create_err
        except Exception as e:
            msg = f"Failed to check template '{template_name}': {e}"
            raise RuntimeError(msg) from e
