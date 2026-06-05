"""Runloop devbox lifecycle (create, attach, delete)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from runloop_api_client import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)
from runloop_api_client.sdk import RunloopSDK

if TYPE_CHECKING:
    from collections.abc import Callable

    from runloop_api_client import Runloop
    from runloop_api_client.sdk import Devbox

from langchain_runloop.sandbox import RunloopSandbox

logger = logging.getLogger(__name__)

_DEFAULT_BLUEPRINT_DOCKERFILE = "FROM python:3\n"
"""Default Dockerfile when auto-building a missing blueprint."""


def _default_resolve_env(name: str) -> str | None:
    """Resolve an env var with optional `DEEPAGENTS_CODE_` prefix override.

    Mirrors `deepagents_code.model_config.resolve_env_var` so that standalone
    use of this package (without the CLI injecting its resolver) matches CLI
    behavior, including treating a present-but-empty value as unset.
    """
    if not name.startswith("DEEPAGENTS_CODE_"):
        prefixed = f"DEEPAGENTS_CODE_{name}"
        if prefixed in os.environ:
            value = os.environ[prefixed]
            if not value:
                logger.debug("%s is set but empty; treating as unset", prefixed)
            return value or None
    value = os.environ.get(name)
    if name in os.environ and not value:
        logger.debug("%s is set but empty; treating as unset", name)
    return value or None


def _not_ready_message(blueprint_name: str, status: str) -> str:
    """Build an actionable message for a same-name blueprint that isn't ready.

    A `failed` status is terminal, so waiting is futile; any other non-complete
    status (`queued`, `provisioning`, `building`) is still in progress.
    """
    if status == "failed":
        return (
            f"Blueprint '{blueprint_name}' exists but its last build failed. "
            f"Delete it and retry, or fix the Dockerfile."
        )
    return (
        f"Blueprint '{blueprint_name}' exists but is still building "
        f"(state '{status}'). Wait for it to finish, or delete it to rebuild."
    )


def _ensure_blueprint(
    client: Runloop,
    blueprint_name: str,
    *,
    dockerfile: str,
) -> None:
    """Guarantee a blueprint named `blueprint_name` has finished building.

    Args:
        client: Runloop API client.
        blueprint_name: Blueprint name to resolve or create.
        dockerfile: Dockerfile used when creating a new blueprint.

    Raises:
        RuntimeError: If listing or building fails, or a same-name blueprint
            exists but is not ready.
    """
    non_ready_status: str | None = None
    starting_after: str | None = None

    while True:
        list_kwargs: dict[str, Any] = {"name": blueprint_name, "limit": 100}
        if starting_after is not None:
            list_kwargs["starting_after"] = starting_after
        try:
            page = client.blueprints.list(**list_kwargs)
        except Exception as e:
            msg = f"Failed to list blueprints: {e}"
            raise RuntimeError(msg) from e

        for blueprint in page.blueprints:
            if blueprint.name != blueprint_name:
                continue
            if blueprint.status == "build_complete":
                return
            non_ready_status = blueprint.status

        if not page.has_more or not page.blueprints:
            break
        starting_after = page.blueprints[-1].id

    if non_ready_status is not None:
        raise RuntimeError(_not_ready_message(blueprint_name, non_ready_status))

    try:
        client.blueprints.create_and_await_build_complete(
            name=blueprint_name,
            dockerfile=dockerfile,
        )
    except Exception as create_err:
        msg = f"Failed to build blueprint '{blueprint_name}': {create_err}"
        raise RuntimeError(msg) from create_err


class RunloopProvider:
    """Create or attach Runloop devboxes, optionally from named blueprints."""

    def __init__(
        self,
        *,
        api_key: str,
        resolve_env_var: Callable[[str], str | None] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: Runloop API bearer token.
            resolve_env_var: Optional env lookup (e.g. for `DEEPAGENTS_CODE_`
                overrides). Defaults to prefix-aware `os.environ` reads.
        """
        self._sdk = RunloopSDK(bearer_token=api_key)
        self._client = self._sdk.api
        self._resolve_env = resolve_env_var or _default_resolve_env

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,  # noqa: ARG002  # Reserved for API parity with other providers
        snapshot: str | None = None,
        blueprint_dockerfile: str | None = None,
        **kwargs: Any,
    ) -> RunloopSandbox:
        """Return a sandbox backend for an existing or new devbox.

        Blueprint boot runs only when `snapshot`, `RUNLOOP_SANDBOX_BLUEPRINT_ID`,
        or `RUNLOOP_SANDBOX_BLUEPRINT_NAME` is set. Otherwise a fresh empty devbox
        is created (same as before this feature). Blueprint resolution order:
        `RUNLOOP_SANDBOX_BLUEPRINT_ID` (boots by ID, skips listing/building) →
        `snapshot` kwarg → `RUNLOOP_SANDBOX_BLUEPRINT_NAME` (both create-if-missing
        by name) → empty devbox.

        Args:
            sandbox_id: Existing devbox ID to attach to, or `None` to create.
            timeout: Reserved for parity with other sandbox providers.
            snapshot: Blueprint name to boot from (create-if-missing).
            blueprint_dockerfile: Dockerfile when auto-building a blueprint.
            **kwargs: Unsupported.

        Returns:
            Connected `RunloopSandbox` instance.

        Raises:
            TypeError: If unsupported keyword arguments are passed.
            KeyError: If `sandbox_id` does not refer to an existing devbox
                (the SDK's `NotFoundError` is translated to `KeyError` so
                callers can map a missing sandbox without importing the SDK).
            RuntimeError: If devbox or blueprint creation fails.
        """
        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        dockerfile = blueprint_dockerfile or _DEFAULT_BLUEPRINT_DOCKERFILE

        if sandbox_id is not None:
            try:
                devbox = self._sdk.devbox.from_id(sandbox_id)
            except NotFoundError as e:
                # Translate the SDK's not-found error into a stable KeyError so
                # callers (e.g. the deepagents-code factory) can detect a
                # missing sandbox without importing runloop_api_client.
                raise KeyError(sandbox_id) from e
            return RunloopSandbox(devbox=devbox)

        env_blueprint_id = self._resolve_env("RUNLOOP_SANDBOX_BLUEPRINT_ID")
        env_blueprint_name = self._resolve_env("RUNLOOP_SANDBOX_BLUEPRINT_NAME")
        blueprint_name = snapshot or env_blueprint_name
        use_blueprint = env_blueprint_id is not None or blueprint_name is not None

        try:
            if use_blueprint:
                devbox = self._create_from_blueprint(
                    blueprint_id=env_blueprint_id,
                    blueprint_name=blueprint_name,
                    dockerfile=dockerfile,
                )
            else:
                devbox = self._sdk.devbox.create()
        except (AuthenticationError, PermissionDeniedError) as e:
            msg = (
                "Runloop rejected the credentials; check RUNLOOP_API_KEY "
                f"(or DEEPAGENTS_CODE_RUNLOOP_API_KEY): {e}"
            )
            raise RuntimeError(msg) from e
        except (APIConnectionError, APITimeoutError) as e:
            msg = f"Runloop API unreachable (transient — safe to retry): {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            target = blueprint_name or env_blueprint_id or "devbox"
            msg = f"Failed to create Runloop devbox from '{target}': {e}"
            raise RuntimeError(msg) from e

        return RunloopSandbox(devbox=devbox)

    def _create_from_blueprint(
        self,
        *,
        blueprint_id: str | None,
        blueprint_name: str | None,
        dockerfile: str,
    ) -> Devbox:
        if blueprint_id is not None:
            return self._sdk.devbox.create_from_blueprint_id(blueprint_id)
        if blueprint_name is None:
            msg = "Blueprint name is required when no blueprint ID is set"
            raise RuntimeError(msg)
        _ensure_blueprint(self._client, blueprint_name, dockerfile=dockerfile)
        return self._sdk.devbox.create_from_blueprint_name(blueprint_name)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Shut down a devbox by ID.

        Raises:
            runloop_api_client.NotFoundError: If `sandbox_id` does not refer to
                an existing devbox.
        """
        self._client.devboxes.shutdown(id=sandbox_id)
