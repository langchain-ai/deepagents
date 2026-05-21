"""Local deploy state persisted under `.deepagents/state.json`.

Tracks the managed agent ID returned by the last successful deploy so that
subsequent runs of `deepagents deploy` issue `PATCH` rather than `POST`. Also
caches the `{mcp_server_url → mcp_server_id}` map to skip the list-call on
every deploy.

The file is gitignored by the scaffold written by `deepagents init`; humans
should never need to edit it by hand.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

_STATE_DIR = ".deepagents"
_STATE_FILE = "state.json"
_SCHEMA_VERSION = 1


@dataclass
class State:
    """In-memory view of `.deepagents/state.json`.

    Use `State.load(project_root)` to read; mutate fields freely; call
    `state.save(...)` to persist.
    """

    project_root: Path
    agent_id: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    last_deployed_at: str | None = None
    mcp_servers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path, *, reset: bool = False) -> State:
        """Load state from `<project_root>/.deepagents/state.json`.

        Returns an empty state if the file does not exist. With `reset=True`,
        deletes the file (if present) before returning the empty state.
        """
        path = project_root / _STATE_DIR / _STATE_FILE
        if reset and path.exists():
            path.unlink()
        if not path.is_file():
            return cls(project_root=project_root)
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
            agent_id=data.get("agent_id"),
            revision=data.get("revision"),
            endpoint=data.get("endpoint"),
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
        directory = self.project_root / _STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "endpoint": self.endpoint,
            "agent_id": self.agent_id,
            "revision": self.revision,
            "last_deployed_at": self.last_deployed_at,
            "mcp_servers": self.mcp_servers,
        }
        (directory / _STATE_FILE).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def clear_agent(self) -> None:
        """Remove agent_id / revision from state and persist."""
        self.agent_id = None
        self.revision = None
        self.save()
