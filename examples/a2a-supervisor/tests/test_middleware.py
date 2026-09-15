"""Tests for the A2A supervisor middleware."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.types import (
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)

from middleware import A2ASupervisorMiddleware, AgentCardReference, _json_result, _validate_url

HOST = "8.8.8.8"


def card(name: str = "Researcher") -> AgentCard:
    """Create a minimal test Agent Card."""
    return AgentCard(
        name=name,
        description="Researches a topic",
        version="1.0",
        supported_interfaces=[
            AgentInterface(
                url=f"https://{HOST}/a2a",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        skills=[AgentSkill(id="research", name="Research", description="Find evidence")],
    )


class FakeClient:
    """Record A2A requests and return deterministic responses."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.gotten: list[Any] = []
        self.cancelled: list[Any] = []
        self.closed = False

    async def send_message(self, request: Any) -> AsyncIterator[StreamResponse]:
        """Yield a completed task."""
        self.sent.append(request)
        task = Task(
            id="task-1",
            context_id=request.message.context_id or "context-1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                Artifact(
                    artifact_id="artifact-1",
                    name="answer",
                    parts=[Part(text="done")],
                )
            ],
        )
        yield StreamResponse(task=task)

    async def get_task(self, request: Any) -> Task:
        """Return the requested task."""
        self.gotten.append(request)
        return Task(
            id=request.id,
            context_id="context-1",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )

    async def cancel_task(self, request: Any) -> Task:
        """Return a canceled task."""
        self.cancelled.append(request)
        return Task(
            id=request.id,
            context_id="context-1",
            status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
        )

    async def close(self) -> None:
        """Record closure."""
        self.closed = True


class FakeFactory:
    """Create inspectable fake clients."""

    def __init__(self) -> None:
        self.clients: list[FakeClient] = []

    def create(
        self,
        card: AgentCard,
        interceptors: list[Any] | None = None,
    ) -> FakeClient:
        """Create a fake client."""
        assert card.name
        assert interceptors is not None
        client = FakeClient()
        self.clients.append(client)
        return client


class FakeRegistry:
    """Return one referenced Agent Card."""

    async def discover(self, query: str | None = None, *, limit: int = 20) -> list[AgentCardReference]:
        """Return deterministic discovery data."""
        assert query == "finance"
        assert limit == 5
        return [AgentCardReference(id="finance", card=card("Financial analyst"))]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:password@example.com",
        "https://127.0.0.1",
        "https://169.254.169.254/latest/meta-data",
        "https://metadata.google.internal",
    ],
)
async def test_validate_url_rejects_unsafe_targets(url: str) -> None:
    """Reject unsafe schemes, credentials, and internal targets."""
    with pytest.raises(ValueError):
        await _validate_url(url, allowed_hosts=None, allow_local_http=False)


async def test_validate_url_enforces_allowlist() -> None:
    """Reject a public host outside the explicit allowlist."""
    with pytest.raises(ValueError, match="not allowlisted"):
        await _validate_url(f"https://{HOST}", allowed_hosts=frozenset({"example.com"}), allow_local_http=False)


async def test_discovery_returns_stable_registry_id() -> None:
    """Expose compact registry results under their stable IDs."""
    middleware = A2ASupervisorMiddleware(
        registry=FakeRegistry(),
        allowed_hosts=[HOST],
        client_factory=FakeFactory(),
    )
    result = await middleware.find_a2a_agents("finance", limit=5)
    payload = json.loads(result.splitlines()[1])
    assert payload["agents"][0]["id"] == "registry:finance"
    assert payload["agents"][0]["skills"][0]["id"] == "research"


async def test_delegate_preserves_continuation_ids_and_tracks_task() -> None:
    """Forward task/context identifiers and use tracked ownership for polling."""
    factory = FakeFactory()
    middleware = A2ASupervisorMiddleware(
        agents=[card()],
        allowed_hosts=[HOST],
        client_factory=factory,
    )
    result = await middleware.delegate_to_a2a_agent(
        "direct:1",
        "continue",
        task_id="prior-task",
        context_id="context-1",
    )
    sent = factory.clients[0].sent[0].message
    assert sent.task_id == "prior-task"
    assert sent.context_id == "context-1"
    assert factory.clients[0].sent[0].configuration.return_immediately
    assert factory.clients[0].closed
    assert '"taskId":"task-1"' in result

    status = await middleware.get_a2a_task("task-1")
    assert '"status":"working"' in status
    assert factory.clients[1].gotten[0].id == "task-1"
    assert factory.clients[1].closed


async def test_cancel_and_bound_untrusted_output() -> None:
    """Cancel tracked work and bound remote output size."""
    factory = FakeFactory()
    middleware = A2ASupervisorMiddleware(
        agents=[card()],
        allowed_hosts=[HOST],
        client_factory=factory,
        max_output_bytes=1_000,
    )
    await middleware.delegate_to_a2a_agent("direct:1", "start")
    result = await middleware.cancel_a2a_task("task-1")
    assert '"status":"canceled"' in result
    assert factory.clients[-1].cancelled[0].id == "task-1"
    assert len(result.encode()) <= 1_000


def test_json_result_strictly_bounds_escaped_data() -> None:
    """Keep the byte ceiling even when JSON escaping expands content."""
    result = _json_result({"value": '"\\' * 2_000}, 1_000)
    assert len(result.encode()) <= 1_000


def test_reference_requires_one_source() -> None:
    """Require exactly one source for a registry reference."""
    with pytest.raises(ValueError):
        AgentCardReference(id="bad")
    with pytest.raises(ValueError):
        AgentCardReference(id="bad", card=card(), url="https://example.com")
    with pytest.raises(ValueError):
        AgentCardReference(id="unsafe:id", card=card())
