"""Integration tests for the hub-backed `deepagents deploy` bundle.

These tests scaffold a tiny project, bundle it with
``[memories].backend = "hub"``, load the vendored ContextHubBackend from
the bundle directory, and exercise the seed flow against a real LangSmith
Hub. Each test provisions a unique throwaway agent repo and deletes it on
teardown so the suite is safe to run against a real tenant.

Skipped unless ``LANGSMITH_API_KEY`` is set.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
import uuid
from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.bundler import bundle
from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    SKILLS_DIRNAME,
    AgentConfig,
    DeployConfig,
    MemoriesConfig,
)

if TYPE_CHECKING:
    import types
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="LANGSMITH_API_KEY not set; skipping hub deploy integration tests.",
)

logger = logging.getLogger(__name__)


@pytest.fixture
def hub_identifier() -> Iterator[str]:
    """Unique throwaway agent-repo handle; deleted on teardown."""
    ident = f"-/deepagents-deploy-test-{uuid.uuid4().hex[:12]}"
    yield ident

    try:
        from langsmith import Client

        Client().delete_agent(ident)
    except Exception:
        logger.warning("Failed to delete test repo %r", ident, exc_info=True)


def _scaffold_project(project: Path) -> None:
    """Drop a minimal AGENTS.md + one skill into *project*."""
    project.mkdir(parents=True, exist_ok=True)
    (project / AGENTS_MD_FILENAME).write_text(
        "# Deploy hub integration test\n\nThis agent exists to validate "
        "the hub-backed bundle.\n",
        encoding="utf-8",
    )
    skill_dir = project / SKILLS_DIRNAME / "echo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: Echo the user input.\n---\n"
        "# Echo\n\nReturn the user input verbatim.\n",
        encoding="utf-8",
    )


def _load_vendored_hub_backend(build_dir: Path) -> types.ModuleType:
    """Import the bundle's vendored ContextHubBackend module."""
    sys.path.insert(0, str(build_dir))
    try:
        # Fresh import so we pick up the copy in this specific build dir.
        if "_context_hub" in sys.modules:
            del sys.modules["_context_hub"]
        return importlib.import_module("_context_hub")
    finally:
        sys.path.remove(str(build_dir))


def test_hub_bundle_seeds_through_composite(
    tmp_path: Path, hub_identifier: str
) -> None:
    """End-to-end: bundle a hub-backed project and seed it into a real repo."""
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from langsmith import Client

    project = tmp_path / "project"
    _scaffold_project(project)

    build = tmp_path / "build"
    config = DeployConfig(
        agent=AgentConfig(name="deploy-hub-test"),
        memories=MemoriesConfig(backend="hub", identifier=hub_identifier),
    )
    bundle(config, project, build)

    # Bundle-level assertions: the vendored module and the hub wiring land.
    assert (build / "_context_hub.py").exists()
    graph_src = (build / "deploy_graph.py").read_text(encoding="utf-8")
    assert f"MEMORIES_HUB_IDENTIFIER = '{hub_identifier}'" in graph_src
    assert "from _context_hub import ContextHubBackend" in graph_src

    # Build a composite matching the one the generated graph builds, and
    # exercise the seed path end-to-end against a real hub.
    ctx_hub_mod = _load_vendored_hub_backend(build)
    hub_backend = ctx_hub_mod.ContextHubBackend(identifier=hub_identifier)
    composite = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": hub_backend},
    )
    seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))

    async def _run() -> None:
        for path, content in seed.get("memories", {}).items():
            result = await composite.awrite(f"/memories/{path.lstrip('/')}", content)
            assert result.error is None, result.error
        for path, content in seed.get("skills", {}).items():
            result = await composite.awrite(
                f"/memories/skills/{path.lstrip('/')}", content
            )
            assert result.error is None, result.error

    asyncio.run(_run())

    # Fresh client pull verifies writes reached the hub (not just cache).
    pulled = Client().pull_agent(hub_identifier)
    pulled_paths = set(pulled.files.keys())
    assert "AGENTS.md" in pulled_paths
    assert "skills/echo/SKILL.md" in pulled_paths

    # Reads through a brand-new backend should round-trip the seeded content.
    fresh_backend = ctx_hub_mod.ContextHubBackend(identifier=hub_identifier)
    read = fresh_backend.read("/AGENTS.md")
    assert read.error is None
    assert read.file_data is not None
    assert "Deploy hub integration test" in read.file_data["content"]


def test_seed_hub_repo_creates_repo_before_invocation(
    tmp_path: Path, hub_identifier: str
) -> None:
    """`_seed_hub_repo` must create the hub repo at deploy time, not first run.

    Bundles a minimal hub-backed project, runs the CLI seed helper directly,
    and asserts the agent repo exists in LangSmith Hub — proving the agent
    is created before any graph invocation.
    """
    from langsmith import Client

    from deepagents_cli.deploy.commands import _seed_hub_repo

    project = tmp_path / "project"
    _scaffold_project(project)
    build = tmp_path / "build"
    config = DeployConfig(
        agent=AgentConfig(name="deploy-hub-test"),
        memories=MemoriesConfig(backend="hub", identifier=hub_identifier),
    )
    bundle(config, project, build)

    _seed_hub_repo(config, build)

    pulled = Client().pull_agent(hub_identifier)
    assert "AGENTS.md" in pulled.files


def test_store_bundle_omits_vendored_hub(tmp_path: Path) -> None:
    """Regression: default store mode must not ship the hub module."""
    project = tmp_path / "project"
    _scaffold_project(project)
    build = tmp_path / "build"
    bundle(
        DeployConfig(agent=AgentConfig(name="store-only")),
        project,
        build,
    )
    # The vendored module is only shipped in hub mode.
    assert not (build / "_context_hub.py").exists()
    graph_src = (build / "deploy_graph.py").read_text(encoding="utf-8")
    assert "MEMORIES_BACKEND = 'store'" in graph_src
