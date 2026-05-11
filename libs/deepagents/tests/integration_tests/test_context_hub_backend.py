"""Integration tests for ContextHubBackend against a real LangSmith Hub.

Skipped unless ``LANGSMITH_API_KEY`` is set. Each test fixture creates a
uniquely-named throwaway agent repo and deletes it on teardown, so these
tests are safe to run against a real tenant.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langsmith import Client

from deepagents.backends import ContextHubBackend
from tests.integration_tests._backend_standard_suite import BackendIntegrationTests

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="LANGSMITH_API_KEY not set; skipping Context Hub integration tests.",
)

logger = logging.getLogger(__name__)


@pytest.fixture
def identifier() -> str:
    """Unique throwaway agent-repo handle under the current tenant."""
    return f"-/deepagents-ctx-hub-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def backend(identifier: str) -> Iterator[ContextHubBackend]:
    """Build a ContextHubBackend and delete the underlying repo on teardown."""
    client = Client()
    yield ContextHubBackend(identifier, client=client)

    try:
        client.delete_agent(identifier)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to delete test repo %r", identifier, exc_info=True)


class TestContextHubBackendStandard(BackendIntegrationTests):
    """Run the standard backend integration contract against Context Hub."""


def test_persists_across_backend_instances(backend, identifier) -> None:
    """A fresh ContextHubBackend on the same identifier sees prior writes."""
    assert backend.write("/persist.md", "original").error is None

    second = ContextHubBackend(identifier, client=Client())
    result = second.read("/persist.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "original"


def test_parent_commit_conflict_surfaces_error(backend, identifier) -> None:
    """Concurrent writes against a stale parent_commit should be rejected."""
    assert backend.write("/shared.md", "v0").error is None

    stale = ContextHubBackend(identifier, client=Client())
    stale.read("/shared.md")  # prime stale's commit_hash with current state

    # `backend` advances the repo.
    assert backend.write("/shared.md", "v1").error is None

    # `stale` now has an outdated parent_commit; server rejects.
    result = stale.write("/other.md", "should-fail")
    assert result.error is not None
    assert "Hub unavailable" in result.error


def test_upload_files_produces_single_commit(identifier) -> None:
    """Batch upload of N files should make exactly one ``push_agent`` call.

    Unit tests assert this with a fully-mocked client; this test wraps a
    real ``langsmith.Client`` so we get the same guarantee against the
    actual Hub API surface, and additionally confirms the resulting commit
    persists every file in one shot.
    """
    real_client = Client()
    push_calls: list[dict[str, Any]] = []
    original_push = type(real_client).push_agent

    def spy_push(self, identifier: str, **kwargs: Any) -> str:
        push_calls.append({"identifier": identifier, **kwargs})
        return original_push(self, identifier, **kwargs)

    backend = ContextHubBackend(identifier, client=real_client)
    try:
        with patch.object(type(real_client), "push_agent", spy_push):
            responses = backend.upload_files(
                [
                    ("/batch/a.md", b"alpha"),
                    ("/batch/b.md", b"beta"),
                    ("/batch/c.md", b"gamma"),
                    ("/batch/d.md", b"delta"),
                ]
            )
        assert all(r.error is None for r in responses), responses
        assert len(push_calls) == 1, f"expected one push_agent call, got {len(push_calls)}"
        assert set(push_calls[0]["files"].keys()) == {
            "batch/a.md",
            "batch/b.md",
            "batch/c.md",
            "batch/d.md",
        }

        # Confirm the commit actually landed and contains all four files.
        pulled = Client().pull_agent(identifier)
        pulled_paths = set(pulled.files.keys())
        assert {"batch/a.md", "batch/b.md", "batch/c.md", "batch/d.md"} <= pulled_paths
    finally:
        try:
            Client().delete_agent(identifier)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete test repo %r", identifier, exc_info=True)
