from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from deepagents_talon.__main__ import _run_mcp_add, _server_config_from_args
from deepagents_talon.config import TalonConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_server_config_from_args_builds_http_server() -> None:
    args = argparse.Namespace(
        transport="http",
        url="https://mcp.example/mcp",
        server_command=None,
        arg=[],
        header=["Authorization=Bearer ${TOKEN}"],
        env=[],
        allow=["search*"],
        disable=[],
        oauth=True,
    )

    server = _server_config_from_args(args)

    assert server == {
        "type": "http",
        "url": "https://mcp.example/mcp",
        "auth": "oauth",
        "headers": {"Authorization": "Bearer ${TOKEN}"},
        "allowedTools": ["search*"],
    }


def test_run_mcp_add_writes_manifest_config(tmp_path: Path) -> None:
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    args = argparse.Namespace(
        config_path=None,
        name="local",
        transport="stdio",
        url=None,
        server_command="server",
        arg=["--flag"],
        header=[],
        env=["TOKEN=${TOKEN}"],
        allow=[],
        disable=["dangerous"],
        oauth=False,
        overwrite=False,
    )

    exit_code = _run_mcp_add(args, config)

    assert exit_code == 0
    data = json.loads((config.manifest_dir / "tools.json").read_text())
    assert data["mcpServers"]["local"] == {
        "type": "stdio",
        "command": "server",
        "args": ["--flag"],
        "env": {"TOKEN": "${TOKEN}"},
        "disabledTools": ["dangerous"],
    }
