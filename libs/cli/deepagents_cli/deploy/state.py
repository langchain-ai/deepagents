"""User-local deploy state persisted outside the project checkout.

Tracks the managed agent ID returned by the last successful deploy so that
subsequent runs of `deepagents deploy` issue `PATCH` rather than `POST`. Also
caches the `{mcp_server_url → mcp_server_id}` map to skip the list-call on
every deploy.

State is keyed by project root and API endpoint under `~/.deepagents/`.
Repository-local state is intentionally ignored because cloned projects can be
attacker controlled and must not silently steer authenticated deployments.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    _STATE_ROOT = Path.home() / ".deepagents" / "deployments"
except RuntimeError:
    _STATE_ROOT = Path("/nonexistent/.deepagents/deployments")
_SCHEMA_VERSION = 1


def _state_path(project_root: Path, endpoint: str) -> Path:
    root = str(project_root.resolve())
    normalized_endpoint = endpoint.rstrip("/")
    material = json.dumps(
        {"endpoint": normalized_endpoint, "project_root": root},
        sort_keys=True,
        separators=(",", ":"),
    )
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return _STATE_ROOT / f"{key}.json"


@dataclass
class State:
    """In-memory view of user-local deploy state.

    Use `State.load(project_root, endpoint=...)` to read; mutate fields freely;
    call `state.save(...)` to persist.
    """

    project_root: Path
    state_path: Path
    agent_id: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    last_deployed_at: str | None = None
    mcp_servers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path, *, endpoint: str, reset: bool = False) -> State:
        """Load state from the user-local deploy cache.

        Returns an empty state if the file does not exist. With `reset=True`,
        deletes the file (if present) before returning the empty state.
        """
        project_root = project_root.resolve()
        endpoint = endpoint.rstrip("/")
        path = _state_path(project_root, endpoint)
        if reset and path.exists():
            path.unlink()
        if not path.is_file():
            return cls(project_root=project_root, state_path=path, endpoint=endpoint)
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != _SCHEMA_VERSION:
            msg = (
                f"Unknown schema_version {version!r} in {path}. "
                f"Expected {_SCHEMA_VERSION}. Delete the file to start fresh."
            )
            raise ValueError(msg)
        return cls(
            project_root=project_root,
            state_path=path,
            agent_id=data.get("agent_id"),
            revision=data.get("revision"),
            endpoint=endpoint,
            last_deployed_at=data.get("last_deployed_at"),
            mcp_servers=dict(data.get("mcp_servers") or {}),
        )

    def save(self, *, agent_id: str | None = None, revision: str | None = None) -> None:
        """Persist state, optionally updating agent_id / revision in the same call."""
        if agent_id is not None:
            self.agent_id = agent_id
        if revision is not None:
            self.revision = revision
        self.last_deployed_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "project_root": str(self.project_root),
            "endpoint": self.endpoint,
            "agent_id": self.agent_id,
            "revision": self.revision,
            "last_deployed_at": self.last_deployed_at,
            "mcp_servers": self.mcp_servers,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear_agent(self) -> None:
        """Remove agent_id / revision from state and persist."""
        self.agent_id = None
        self.revision = None
        self.save()
