"""No-network integration coverage for Talon ticket 07 dependent flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from deepagents_code.agent import create_cli_agent, import_agent_manifest
from deepagents_code.agent_manifest import (
    load_agent_manifest,
    load_manifest_backend,
    load_manifest_model,
)
from deepagents_code.integrations.sandbox_factory import create_sandbox

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


@dataclass(frozen=True)
class ExecResult:
    """Minimal execution result returned by the fake sandbox."""

    output: str
    exit_code: int = 0


class FakeSandbox:
    """Small sandbox backend for exercising factory lifecycle without network."""

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.commands: list[str] = []

    def execute(self, command: str) -> ExecResult:
        """Record a command and return a deterministic result."""
        self.commands.append(command)
        return ExecResult(output=f"ran:{command}")


def _fake_model() -> GenericFakeChatModel:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}
    return model


def test_local_execution_backend_runs_against_selected_working_directory(
    tmp_path: Path,
) -> None:
    """Local-default agent construction should expose one host execution backend."""
    _agent, backend = create_cli_agent(
        _fake_model(),
        "ticket07-local",
        auto_approve=True,
        cwd=tmp_path,
        enable_ask_user=False,
        enable_memory=False,
        enable_skills=False,
    )

    result = backend.execute("printf local > result.txt")
    read = backend.read("result.txt")

    assert result.exit_code == 0
    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "local"


def test_sandbox_factory_reuses_persisted_backend_and_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sandboxed execution should reconnect by persisted id for reusable providers."""
    from deepagents_code import model_config

    monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path)
    first = FakeSandbox("sandbox-1")
    second = FakeSandbox("sandbox-1")
    provider = MagicMock()
    provider.get_or_create.side_effect = [first, second]

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        create_sandbox("modal") as sandbox,
    ):
        assert sandbox.execute("echo first").output == "ran:echo first"

    with (
        patch(
            "deepagents_code.integrations.sandbox_factory._get_provider",
            return_value=provider,
        ),
        create_sandbox("modal") as sandbox,
    ):
        assert sandbox.execute("echo second").output == "ran:echo second"

    assert provider.get_or_create.call_args_list[0].kwargs == {"sandbox_id": None}
    assert provider.get_or_create.call_args_list[1].kwargs == {
        "sandbox_id": "sandbox-1"
    }
    provider.delete.assert_not_called()


def test_fleet_manifest_import_materializes_runnable_local_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fleet files should import locally with hosted-only fields dropped."""
    from deepagents_code import agent as agent_module

    source = tmp_path / "fleet-agent.json"
    source.write_text(
        json.dumps(
            {
                "name": "Imported",
                "runtime": {"model": {"model_id": "provider:test-model"}},
                "backend": {"type": "langsmith", "PolicyIDs": ["hosted-policy"]},
                "permissions": {"users": ["hosted-user"]},
                "files": {
                    "AGENTS.md": "Imported system prompt",
                    "tools.json": json.dumps(
                        {
                            "mcpServers": {
                                "local": {"command": "uvx", "args": ["mcp-server"]},
                                "hosted": {
                                    "url": "https://example.com/mcp",
                                    "auth": "oauth",
                                },
                            }
                        }
                    ),
                    "subagents/reviewer/AGENTS.md": (
                        "---\n"
                        "name: reviewer\n"
                        "description: Review work\n"
                        "---\n"
                        "Review carefully.\n"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "agents" / "imported"
    monkeypatch.setattr(agent_module.settings, "get_agent_dir", lambda _name: target)

    import_agent_manifest(
        str(source),
        "imported",
        backend_type="modal",
        output_format="json",
    )
    output = json.loads(capsys.readouterr().out)

    manifest = load_agent_manifest(target)
    assert manifest.system_prompt == "Imported system prompt"
    assert load_manifest_model(target) == "provider:test-model"
    assert load_manifest_backend(target) == "modal"
    assert (target / "subagents" / "reviewer" / "AGENTS.md").is_file()
    assert manifest.tools == {
        "mcpServers": {"local": {"command": "uvx", "args": ["mcp-server"]}}
    }
    assert set(output["data"]["dropped_fields"]) == {
        "backend.PolicyIDs",
        "permissions",
        "tools.mcpServers.hosted.oauth",
    }
