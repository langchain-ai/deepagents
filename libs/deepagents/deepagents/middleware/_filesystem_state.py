"""State schemas for backends that store files in agent state."""

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import AgentState
from langgraph.channels.delta import DeltaChannel

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendProtocol, FileData
from deepagents.backends.state import StateBackend


def _file_data_delta_reducer(
    left: dict[str, FileData] | None,
    values: list[dict[str, FileData | None]],
) -> dict[str, FileData]:
    """Merge batches of file updates with support for deletions."""
    result: dict[str, FileData] = dict(left) if left else {}
    for writes in values:
        for key, value in writes.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
    return result


class FilesystemState(AgentState):
    """State for backends that store files in agent state."""

    files: Annotated[NotRequired[dict[str, FileData]], DeltaChannel(_file_data_delta_reducer, snapshot_frequency=50)]  # ty: ignore[invalid-argument-type]
    """Files in the filesystem. Uses DeltaChannel with snapshots every ~50 pregel steps to bound read depth."""


def _uses_state_backend(backend: BackendProtocol) -> bool:
    """Return whether a backend stores any files in agent state."""
    if isinstance(backend, StateBackend):
        return True
    if not isinstance(backend, CompositeBackend):
        return False
    return _uses_state_backend(backend.default) or any(_uses_state_backend(route) for route in backend.routes.values())


def _state_schema_for_backend(
    backend: BackendProtocol,
    base_schema: type[AgentState],
    state_backend_schema: type[AgentState],
) -> type[AgentState]:
    """Select the schema required by a backend's storage strategy."""
    return state_backend_schema if _uses_state_backend(backend) else base_schema
