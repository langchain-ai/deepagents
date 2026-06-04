"""Tests for Fleet-style agent manifest import."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from deepagents_code.agent_manifest import (
    AgentManifestError,
    fetch_fleet_agent_manifest,
    load_agent_manifest,
    load_manifest_backend,
    load_manifest_model,
    materialize_agent_manifest,
    parse_fleet_agent_payload,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_fleet_payload_drops_hosted_only_fields() -> None:
    """Unsupported Fleet fields are removed with visible dropped-field notes."""
    payload = {
        "name": "Researcher",
        "runtime": {"model": {"model_id": "openai:gpt-5.5"}},
        "backend": {"type": "local", "PolicyIDs": ["policy-1"]},
        "permissions": {"users": ["user-1"]},
        "sharedUsers": ["user-2"],
        "files": {
            "AGENTS.md": "You research carefully.",
            "tools.json": json.dumps(
                {
                    "mcpServers": {
                        "filesystem": {"command": "uvx", "args": ["mcp-fs"]},
                        "hosted": {
                            "url": "https://example.com/mcp",
                            "oauth": {"client_id": "hosted"},
                        },
                    }
                }
            ),
            "subagents/writer/AGENTS.md": (
                "---\n"
                "name: writer\n"
                "description: Write summaries\n"
                "---\n"
                "Write concise summaries.\n"
            ),
        },
    }

    manifest = parse_fleet_agent_payload(payload)

    assert manifest.name == "Researcher"
    assert manifest.runtime_model == "openai:gpt-5.5"
    assert manifest.system_prompt == "You research carefully."
    assert manifest.tools == {
        "mcpServers": {
            "filesystem": {"command": "uvx", "args": ["mcp-fs"]},
        }
    }
    assert set(manifest.metadata["_dropped_fields"]) == {
        "backend.PolicyIDs",
        "permissions",
        "sharedUsers",
        "tools.mcpServers.hosted.oauth",
    }


def test_materialized_manifest_round_trips(tmp_path: Path) -> None:
    """A local file-tree manifest can be imported and loaded back."""
    payload = {
        "name": "Writer",
        "runtime": {"model": {"model_id": "anthropic:claude-sonnet-4-6"}},
        "files": {
            "AGENTS.md": "Main prompt",
            "tools.json": '{"mcpServers": {}}',
            "skills/editor/SKILL.md": (
                "---\nname: editor\ndescription: Edit prose\n---\nEdit carefully.\n"
            ),
        },
    }
    source = parse_fleet_agent_payload(payload)
    target = tmp_path / "writer"

    result = materialize_agent_manifest(
        source,
        target,
        agent_name="writer",
        backend_type="modal",
    )
    loaded = load_agent_manifest(target)

    assert result.path == target
    assert result.backend_type == "modal"
    assert loaded.runtime_model == "anthropic:claude-sonnet-4-6"
    assert loaded.system_prompt == "Main prompt"
    assert (target / "skills" / "editor" / "SKILL.md").is_file()
    assert load_manifest_model(target) == "anthropic:claude-sonnet-4-6"
    assert load_manifest_backend(target) == "modal"


def test_materialize_refuses_existing_directory_without_force(tmp_path: Path) -> None:
    """Imports do not overwrite local agents unless explicitly forced."""
    target = tmp_path / "agent"
    target.mkdir()
    manifest = parse_fleet_agent_payload({"files": {"AGENTS.md": "Prompt"}})

    with pytest.raises(AgentManifestError, match="already exists"):
        materialize_agent_manifest(manifest, target, agent_name="agent")


def test_manifest_rejects_path_traversal() -> None:
    """Fleet file paths must stay inside the agent tree."""
    with pytest.raises(AgentManifestError, match="inside the manifest tree"):
        parse_fleet_agent_payload({"files": {"../AGENTS.md": "escape"}})


def test_fetch_fleet_agent_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fleet id import requests the manifest with files and parses the payload."""
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "runtime": {"model": {"model_id": "openai:gpt-5.5"}},
                "files": {"AGENTS.md": "Fetched prompt"},
            }

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("LANGSMITH_API_KEY", "key")
    monkeypatch.setattr("httpx.get", fake_get)

    manifest = fetch_fleet_agent_manifest(
        "agent/with space",
        api_base="https://smith.example",
        timeout=12.0,
    )

    assert manifest.runtime_model == "openai:gpt-5.5"
    assert manifest.system_prompt == "Fetched prompt"
    assert captured["url"] == (
        "https://smith.example/api/v1/fleet/agents/"
        "agent%2Fwith%20space?include_files=true"
    )
    assert captured["headers"] == {"x-api-key": "key"}
    assert captured["timeout"] == 12.0
