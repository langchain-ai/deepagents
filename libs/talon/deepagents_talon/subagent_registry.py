"""Immutable remote subagent revisions retained outside conversation checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from deepagents.middleware.async_subagents import (
    ASYNC_TASK_TOOL_DESCRIPTION,
    AsyncSubAgent,
    AsyncSubAgentMiddleware,
)
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.types import Command

_MAX_REVISIONS = 256
_MAX_REGISTRY_BYTES = 4 * 1024 * 1024


class _Definition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    graph_id: str = Field(min_length=1)
    url: str | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict, repr=False)

    def spec(self, name: str) -> AsyncSubAgent:
        result: AsyncSubAgent = {
            "name": name,
            "description": self.description,
            "graph_id": self.graph_id,
        }
        if self.url is not None:
            result["url"] = self.url
        if self.headers:
            result["headers"] = {
                key: value.get_secret_value() for key, value in self.headers.items()
            }
        return result


class SubagentRegistry:
    """Retain original connection configuration for tasks across reloads and restarts.

    Args:
        path: Private registry file, or `None` for an in-memory runtime.
    """

    def __init__(self, path: Path | None) -> None:
        """Load private configuration revisions without constructing clients."""
        self.path = path
        self._revisions: dict[str, _Definition] = {}
        if path is not None and path.exists():
            if path.is_symlink() or path.stat().st_size > _MAX_REGISTRY_BYTES:
                msg = "Invalid subagent registry file"
                raise ValueError(msg)
            self._revisions = _read_revisions(path)
            path.chmod(0o600)

    def prepare(
        self, specs: Sequence[AsyncSubAgent]
    ) -> tuple[dict[str, _Definition], dict[str, str]]:
        """Prepare a revision snapshot without modifying the active registry.

        Args:
            specs: Current configured remote agents.

        Returns:
            Candidate revisions and the current name-to-revision mapping.
        """
        revisions = dict(self._revisions)
        current: dict[str, str] = {}
        for spec in specs:
            try:
                definition = _Definition.model_validate(spec)
            except ValidationError:
                msg = "Invalid remote subagent configuration"
                raise ValueError(msg) from None
            # Unversioned names keep checkpoints created before this registry usable.
            revisions.setdefault(definition.name, definition)
            revision = next((key for key, value in revisions.items() if value == definition), None)
            if revision is None:
                revision = f"talon-{uuid4().hex}"
                revisions[revision] = definition
            current[definition.name] = revision
        if len(revisions) > _MAX_REVISIONS:
            msg = "Subagent registry revision limit reached; previous configuration retained"
            raise ValueError(msg)
        return revisions, current

    def commit(self, revisions: dict[str, _Definition]) -> None:
        """Persist a successfully compiled snapshot before activating it.

        Args:
            revisions: Validated registry returned by `prepare`.
        """
        if revisions == self._revisions:
            return
        if self.path is not None:
            payload = json.dumps(
                {
                    "version": 1,
                    "revisions": {key: value.spec(value.name) for key, value in revisions.items()},
                }
            )
            if len(payload.encode()) > _MAX_REGISTRY_BYTES:
                msg = "Subagent registry size limit reached"
                raise ValueError(msg)
            _write_private(self.path, payload)
        self._revisions = revisions


def _read_revisions(path: Path) -> dict[str, _Definition]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] == 1 and len(data["revisions"]) <= _MAX_REVISIONS:
            return {
                key: _Definition.model_validate(value) for key, value in data["revisions"].items()
            }
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    msg = "Invalid subagent registry; original configuration must be recovered"
    raise ValueError(msg) from None


def _write_private(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        msg = "Subagent registry must not be a symlink"
        raise ValueError(msg)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".subagents-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class VersionedAsyncSubagents(AsyncSubAgentMiddleware):
    """Expose current agents for launches while retaining prior targets for management.

    Args:
        revisions: Immutable configuration snapshot for this graph.
        current: Public agent names mapped to their current revision.
    """

    def __init__(self, revisions: Mapping[str, _Definition], current: Mapping[str, str]) -> None:
        """Construct management tools for all revisions and launches for current names."""
        self._current = dict(current)
        super().__init__(async_subagents=[value.spec(key) for key, value in revisions.items()])
        available = (
            "\n".join(
                f"- {name}: {revisions[revision].description}" for name, revision in current.items()
            )
            or "No agents available for new tasks. Existing tasks can still be managed."
        )
        self.tools[0].description = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=available)

    def _launch_request(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        if request.tool_call["name"] != "start_async_task":
            return request
        name = request.tool_call["args"].get("subagent_type")
        revision = self._current.get(name) if isinstance(name, str) else None
        if revision is None:
            return ToolMessage(
                "Subagent is unavailable for new tasks. Reload definitions or use a current name.",
                tool_call_id=request.tool_call["id"],
            )
        return request.override(
            tool_call={
                **request.tool_call,
                "args": {**request.tool_call["args"], "subagent_type": revision},
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Resolve launches using the current snapshot; management uses persisted task IDs."""
        resolved = self._launch_request(request)
        return resolved if isinstance(resolved, ToolMessage) else handler(resolved)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Resolve asynchronous launches without changing already tracked tasks."""
        resolved = self._launch_request(request)
        return resolved if isinstance(resolved, ToolMessage) else await handler(resolved)
