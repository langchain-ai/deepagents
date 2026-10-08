# Copyright (c) 2026 Mainbrella
"""Generation-safe Mainbrella container lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mainbrella import Mainbrella

from langchain_mainbrella.sandbox import MainbrellaSandbox

if TYPE_CHECKING:
    from mainbrella import Sandbox


class MainbrellaProvider:
    """Create, attach to, and stop Mainbrella container generations."""

    def __init__(
        self, *, api_key: str, base_url: str = "https://api.mainbrella.com"
    ) -> None:
        """Initialize the official client.

        Args:
            api_key: Mainbrella API key; keep it outside the guest environment.
            base_url: API origin. HTTPS is required except on loopback.
        """
        self._client = Mainbrella(api_key, base_url=base_url)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        image_id: str | None = None,
        size: str = "lite",
        idempotency_key: str | None = None,
    ) -> MainbrellaSandbox:
        """Connect to an exact generation or create a Python container.

        Args:
            sandbox_id: `slot@createdAt` from `MainbrellaSandbox.id` to attach.
            timeout: Seconds to wait for creation readiness.
            image_id: Custom image ID, overriding the catalog selection.
            size: Mainbrella machine size for a new container.
            idempotency_key: Creation key to retain for reconciliation.

        Returns:
            A connected sandbox backend.

        Raises:
            ValueError: If the supplied sandbox ID omits its generation.
            KeyError: If the requested generation is no longer running.
            MainbrellaError: If creation fails; ambiguous errors retain their
                `idempotency_key` for reconciliation.
        """
        sandbox = (
            self._connect(sandbox_id)
            if sandbox_id is not None
            else self._client.create(
                catalog_id=None if image_id else "python",
                image_id=image_id,
                size=size,
                wait_timeout=timeout,
                idempotency_key=idempotency_key,
            )
        )
        return MainbrellaSandbox(sandbox=sandbox)

    def _connect(self, sandbox_id: str) -> Sandbox:
        container_id, separator, created_at = sandbox_id.partition("@")
        if not separator or not container_id or not created_at:
            msg = (
                "Mainbrella sandbox ID must be slot@createdAt, including its generation"
            )
            raise ValueError(msg)
        sandbox = self._client.connect(container_id, created_at)
        containers = self._client.list()["containers"]
        if not any(
            c["id"] == container_id
            and c["createdAt"] == created_at
            and c["status"] == "running"
            for c in containers
        ):
            raise KeyError(sandbox_id)
        return sandbox

    def delete(self, *, sandbox_id: str) -> None:
        """Stop only the selected generation, without affecting replacements.

        Args:
            sandbox_id: Generation-qualified ID from `MainbrellaSandbox.id`.

        Raises:
            KeyError: If the requested generation is no longer running.
            MainbrellaError: If stopping cannot be confirmed.
        """
        self._connect(sandbox_id).kill()
