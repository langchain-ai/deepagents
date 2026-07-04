from __future__ import annotations

import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import pytest

import deepagents_talon.__main__ as talon_main
from deepagents_talon.mcp import MCPTools
from tests.conftest import RecordingChannel

if TYPE_CHECKING:
    from pathlib import Path

Channel = Literal["telegram", "whatsapp"]

CHANNEL_TOKEN = "channel-token-should-not-appear"  # noqa: S105  # fake leak sentinel
OAUTH_TOKEN = "oauth-token-should-not-appear"  # noqa: S105  # fake leak sentinel
BEARER_HEADER = "Bearer bearer-header-should-not-appear"
QUERY_TOKEN = "query-token-should-not-appear"  # noqa: S105  # fake leak sentinel


@dataclass(frozen=True, slots=True)
class ImportContext:
    channel: Channel
    fleet: Path
    assistant_id: str
    mcp_config_path: Path


@pytest.mark.parametrize("channel", ["telegram", "whatsapp"])
def test_fleet_direct_run_import_and_once_startup_path(
    channel: Channel,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet_dir = _fleet_export(tmp_path)
    fleet = _fleet_zip(tmp_path, fleet_dir)
    home = tmp_path / "home"
    assistant_id = f"fleet-{channel}"
    mcp_config_path = home / assistant_id / "agent" / ".mcp.json"
    context = ImportContext(
        channel=channel,
        fleet=fleet,
        assistant_id=assistant_id,
        mcp_config_path=mcp_config_path,
    )

    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(home))
    monkeypatch.setenv("LANGSMITH_API_KEY", OAUTH_TOKEN)
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "tenant-id")
    monkeypatch.setenv("LANGSMITH_ORGANIZATION_ID", "organization-id")
    _set_channel_env(channel, monkeypatch)

    summary, manifest_text, payload = _import_fleet_through_cli(context, capsys, monkeypatch)
    _assert_import_result(
        context,
        summary=summary,
        manifest_text=manifest_text,
        payload=payload,
    )
    _assert_secret_free(summary, manifest_text, caplog.text)
    _run_once_startup_path(
        context,
        caplog=caplog,
        monkeypatch=monkeypatch,
    )


def _import_fleet_through_cli(
    context: ImportContext,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str, dict[str, object]]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deepagents-talon",
            "import-fleet",
            str(context.fleet),
            "--assistant-id",
            context.assistant_id,
            "--channel",
            context.channel,
            "--non-interactive",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        talon_main.main()

    assert exc.value.code == 0
    summary = capsys.readouterr().out
    manifest_text = context.mcp_config_path.read_text(encoding="utf-8")
    manifest = cast("dict[str, object]", json.loads(manifest_text))
    payload = cast("dict[str, object]", manifest["_fleet_import"])
    return summary, manifest_text, payload


def _assert_import_result(
    context: ImportContext,
    *,
    summary: str,
    manifest_text: str,
    payload: dict[str, object],
) -> None:
    del manifest_text
    assert summary.startswith("Imported Fleet export for Talon.")
    assert f"assistant_id: {context.assistant_id}" in summary
    assert "tools_summarized: 4" in summary
    assert "mcp_servers: 3" in summary
    assert "interrupt_tools: 1" in summary
    assert "root_mcp_config:" in summary
    assert payload["source"] == "fleet"
    assert payload["assistant_id"] == context.assistant_id
    assert payload["fleet_export"] == str(context.fleet.resolve())
    assert context.mcp_config_path.parent.joinpath("AGENTS.md").read_text(encoding="utf-8") == (
        "system prompt"
    )
    assert (context.mcp_config_path.parent / "subagents" / "researcher" / "AGENTS.md").read_text(
        encoding="utf-8"
    ) == "research prompt"
    assert payload["servers"] == [
        {
            "auth_path": "builtin",
            "endpoint": "https://builtin.example/mcp",
            "interrupt_tools": [],
            "scope": "root",
            "server_name": "builtin",
            "tool_names": ["builtin_search"],
        },
        {
            "auth_path": "headers",
            "endpoint": "https://missing.example/mcp",
            "interrupt_tools": ["search"],
            "scope": "root",
            "server_name": "missing",
            "tool_names": ["lookup", "search"],
        },
        {
            "auth_path": "oauth",
            "endpoint": "https://calendar.example/mcp",
            "interrupt_tools": [],
            "scope": "subagent:researcher",
            "server_name": "calendar",
            "tool_names": ["calendar"],
        },
    ]


def _run_once_startup_path(
    context: ImportContext,
    *,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels: list[RecordingChannel] = []
    runtimes: list[RecordingRuntime] = []
    mcp_configs: list[object] = []

    async def fake_load_mcp(config: object, *, allow_empty: bool = False) -> MCPTools:
        assert allow_empty is True
        mcp_configs.append(config)
        return MCPTools(tools=(), servers=())

    class RecordingRuntime:
        def __init__(
            self,
            *,
            model: str,
            env: dict[str, str],
            **kwargs: object,
        ) -> None:
            self.model = model
            self.env = env
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            runtimes.append(self)

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    class RecordingTelegramChannel(RecordingChannel):
        def __init__(self, config: object) -> None:
            self.config = config
            super().__init__(provider="telegram")
            channels.append(self)

    class RecordingWhatsAppChannel(RecordingChannel):
        def __init__(self, config: object) -> None:
            self.config = config
            super().__init__(provider="whatsapp")
            channels.append(self)

    monkeypatch.setattr(talon_main, "load_mcp_tools", fake_load_mcp)
    monkeypatch.setattr(talon_main, "DeepAgentRuntime", RecordingRuntime)
    monkeypatch.setattr(talon_main, "TelegramChannel", RecordingTelegramChannel)
    monkeypatch.setattr(talon_main, "WhatsAppChannel", RecordingWhatsAppChannel)
    monkeypatch.setenv("AGENT_ASSISTANT_ID", context.assistant_id)
    monkeypatch.setenv("AGENT_MODEL", "runtime:model")
    monkeypatch.delenv("DEEPAGENTS_TALON_FLEET_DIR", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deepagents-talon",
            "--once",
        ],
    )

    with caplog.at_level(logging.INFO):
        talon_main.main()

    assert [adapter.provider for adapter in channels] == [context.channel]
    assert len(runtimes) == 1
    assert runtimes[0].model == "runtime:model"
    assert runtimes[0].kwargs["assistant_dir"] == context.mcp_config_path.parent
    assert runtimes[0].started is True
    assert runtimes[0].stopped is True
    assert runtimes[0].env["LANGSMITH_API_KEY"] == OAUTH_TOKEN
    assert len(mcp_configs) == 1
    _assert_secret_free(caplog.text)


def _set_channel_env(channel: Channel, monkeypatch: pytest.MonkeyPatch) -> None:
    if channel == "telegram":
        monkeypatch.setenv("DEEPAGENTS_TALON_TELEGRAM_ENABLED", "true")
        monkeypatch.setenv("DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN", CHANNEL_TOKEN)
        monkeypatch.setenv("DEEPAGENTS_TALON_TELEGRAM_EXPOSURE", "allowlist")
        monkeypatch.setenv("DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS", "123456")
        return

    monkeypatch.setenv("DEEPAGENTS_TALON_WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("DEEPAGENTS_TALON_WHATSAPP_BRIDGE_TOKEN", CHANNEL_TOKEN)
    monkeypatch.setenv("DEEPAGENTS_TALON_WHATSAPP_EXPOSURE", "self")


def _fleet_export(tmp_path: Path) -> Path:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "AGENTS.md").write_text("system prompt", encoding="utf-8")
    (fleet / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "search",
                        "mcp_server_url": f"https://missing.example/mcp?token={QUERY_TOKEN}",
                        "mcp_server_name": "missing",
                        "auth_type": "headers",
                        "headers": {"Authorization": BEARER_HEADER},
                    },
                    {
                        "name": "lookup",
                        "mcp_server_url": f"https://missing.example/mcp?api_key={QUERY_TOKEN}",
                        "mcp_server_name": "missing",
                        "auth_type": "headers",
                    },
                    {
                        "name": "builtin_search",
                        "mcp_server_url": f"https://builtin.example/mcp?token={QUERY_TOKEN}",
                        "mcp_server_name": "builtin",
                        "auth_type": "builtin",
                    },
                ],
                "interrupt_config": {
                    "https://missing.example/mcp::search::missing": True,
                    "https://missing.example/mcp::lookup::missing": False,
                    "https://builtin.example/mcp::builtin_search::builtin": False,
                },
            }
        ),
        encoding="utf-8",
    )
    subagent = fleet / "subagents" / "researcher"
    subagent.mkdir(parents=True)
    (subagent / "AGENTS.md").write_text("research prompt", encoding="utf-8")
    (subagent / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "calendar",
                        "mcp_server_url": f"https://calendar.example/mcp?oauth={OAUTH_TOKEN}",
                        "mcp_server_name": "calendar",
                        "auth_type": "oauth",
                        "oauth_access_token": OAUTH_TOKEN,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return fleet


def _fleet_zip(tmp_path: Path, fleet: Path) -> Path:
    export = tmp_path / f"{fleet.name}.zip"
    with zipfile.ZipFile(export, "w") as archive:
        for path in sorted(fleet.rglob("*")):
            archive.write(path, path.relative_to(fleet))
    return export


def _assert_secret_free(*values: str) -> None:
    for value in values:
        assert CHANNEL_TOKEN not in value
        assert OAUTH_TOKEN not in value
        assert BEARER_HEADER not in value
        assert QUERY_TOKEN not in value
