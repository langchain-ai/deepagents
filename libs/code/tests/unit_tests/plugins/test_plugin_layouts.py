from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_code.mcp_config import resolve_mcp_server_env
from deepagents_code.plugins import (
    add_local_marketplace,
    discover_plugins,
    install_plugin,
)
from deepagents_code.plugins.adapters.hooks import discover_plugin_hook_sources
from deepagents_code.plugins.adapters.mcp import (
    plugin_mcp_configs,
    scoped_mcp_server_name,
)
from deepagents_code.plugins.layouts import (
    AGENT_PLUGIN_V1_MCP_SCHEMA,
    AGENT_PLUGIN_V1_SCHEMA,
)
from deepagents_code.plugins.manifest import (
    PluginManifestError,
    discover_components,
    load_manifest,
    select_manifest_path,
)
from deepagents_code.plugins.models import PluginInstance

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import JsonObject, PluginManifest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_skill(path: Path, name: str = "review") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Review code.\n---\n\nReview code.",
        encoding="utf-8",
    )


def _agent_manifest(name: str = "quality") -> dict[str, object]:
    return {
        "$schema": AGENT_PLUGIN_V1_SCHEMA,
        "name": name,
        "version": "1.0.0",
    }


def _plugin_instance(
    root: Path,
    manifest: PluginManifest,
    *,
    plugin_id: str = "quality@tools",
) -> PluginInstance:
    return PluginInstance(
        plugin_id=plugin_id,
        name=plugin_id.rsplit("@", 1)[0],
        marketplace=plugin_id.rsplit("@", 1)[1],
        version=manifest.version,
        root=root,
        data_dir=root.parent / "data" / plugin_id,
        manifest=manifest,
        inventory=discover_components(root, manifest),
    )


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (AGENT_PLUGIN_V1_SCHEMA, "agent-plugin-v1"),
        (None, "claude"),
        ("https://example.com/plugin.schema.json", "claude"),
    ],
)
def test_root_manifest_schema_selects_dialect(
    tmp_path: Path,
    schema: str | None,
    expected: str,
) -> None:
    document: dict[str, object] = {"name": "quality", "futureField": True}
    if schema is not None:
        document["$schema"] = schema
    _write_json(tmp_path / "plugin.json", document)

    manifest, path, warnings = load_manifest(tmp_path)

    assert manifest is not None
    assert manifest.dialect == expected
    assert manifest.schema == schema
    assert path == tmp_path / "plugin.json"
    assert warnings == ()


def test_future_agent_plugin_schema_is_not_misread_as_claude(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "plugin.json",
        {
            "$schema": "https://agent-plugins.org/schemas/2.0.0/plugin.schema.json",
            "name": "quality",
        },
    )

    with pytest.raises(PluginManifestError, match="Unsupported Agent Plugins schema"):
        load_manifest(tmp_path)


def test_root_manifest_is_authoritative(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", _agent_manifest())
    nested = tmp_path / ".claude-plugin" / "plugin.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{invalid", encoding="utf-8")

    manifest, path, _warnings = load_manifest(tmp_path)

    assert manifest is not None
    assert manifest.dialect == "agent-plugin-v1"
    assert path == tmp_path / "plugin.json"
    assert select_manifest_path(tmp_path) == tmp_path / "plugin.json"


def test_invalid_root_manifest_does_not_fall_through(tmp_path: Path) -> None:
    (tmp_path / "plugin.json").write_text("{invalid", encoding="utf-8")
    _write_json(
        tmp_path / ".claude-plugin" / "plugin.json",
        {"name": "quality"},
    )

    with pytest.raises(PluginManifestError, match="Invalid JSON syntax"):
        load_manifest(tmp_path)


def test_manifest_symlink_cannot_escape_plugin_root(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    outside = tmp_path / "outside.json"
    _write_json(outside, _agent_manifest())
    try:
        (root / "plugin.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(PluginManifestError, match="escapes plugin root"):
        load_manifest(root)


def test_agent_layout_composes_standard_paths_hooks_and_declarations(
    tmp_path: Path,
) -> None:
    manifest_document = {
        **_agent_manifest(),
        "skills": "./extra-skills",
        "mcpServers": "./extra-mcp.json",
        "hooks": "./extra-hooks.json",
    }
    _write_json(tmp_path / "plugin.json", manifest_document)
    _write_skill(tmp_path / "skills" / "review" / "SKILL.md")
    _write_skill(tmp_path / "extra-skills" / "audit" / "SKILL.md", "audit")
    _write_skill(tmp_path / "SKILL.md", "root")
    _write_json(tmp_path / "mcp.json", {"mcpServers": {}})
    _write_json(tmp_path / ".mcp.json", {"mcpServers": {"ignored": {}}})
    _write_json(tmp_path / "extra-mcp.json", {"mcpServers": {}})
    _write_json(tmp_path / "hooks" / "hooks.json", {"hooks": {}})
    _write_json(tmp_path / "extra-hooks.json", {"hooks": {}})

    manifest, _path, warnings = load_manifest(tmp_path)
    assert manifest is not None
    inventory = discover_components(tmp_path, manifest, warnings)

    assert inventory.skills == (
        (tmp_path / "skills").resolve(),
        (tmp_path / "extra-skills").resolve(),
    )
    assert inventory.mcp_files == (
        (tmp_path / "mcp.json").resolve(),
        (tmp_path / "extra-mcp.json").resolve(),
    )
    assert inventory.hook_files == (
        (tmp_path / "hooks" / "hooks.json").resolve(),
        (tmp_path / "extra-hooks.json").resolve(),
    )


def test_claude_layout_preserves_existing_conventions(tmp_path: Path) -> None:
    _write_json(
        tmp_path / ".claude-plugin" / "plugin.json",
        {"name": "quality"},
    )
    _write_skill(tmp_path / "SKILL.md")
    _write_json(tmp_path / ".mcp.json", {"mcpServers": {}})
    _write_json(tmp_path / "mcp.json", {"mcpServers": {"ignored": {}}})
    _write_json(tmp_path / "hooks" / "hooks.json", {"hooks": {}})

    manifest, _path, warnings = load_manifest(tmp_path)
    assert manifest is not None
    inventory = discover_components(tmp_path, manifest, warnings)

    assert manifest.dialect == "claude"
    assert inventory.skills == ((tmp_path / "SKILL.md").resolve(),)
    assert inventory.mcp_files == ((tmp_path / ".mcp.json").resolve(),)
    assert inventory.hook_files == ((tmp_path / "hooks" / "hooks.json").resolve(),)


@pytest.mark.parametrize("dialect", ["agent-plugin-v1", "claude"])
def test_plugin_stdio_defaults_are_shared_across_dialects(
    tmp_path: Path,
    dialect: str,
) -> None:
    root = tmp_path / dialect
    if dialect == "agent-plugin-v1":
        _write_json(root / "plugin.json", _agent_manifest())
        mcp_path = root / "mcp.json"
    else:
        _write_json(
            root / ".claude-plugin" / "plugin.json",
            {"name": "quality"},
        )
        mcp_path = root / ".mcp.json"
    _write_json(
        mcp_path,
        {"mcpServers": {"stdio": {"command": "./bin/server"}}},
    )
    manifest, _path, _warnings = load_manifest(root)
    assert manifest is not None
    plugin = _plugin_instance(root, manifest)

    configs = plugin_mcp_configs((plugin,))
    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    server = servers[scoped_mcp_server_name(plugin.plugin_id, "stdio")]

    assert server["command"] == str((root / "bin" / "server").resolve())
    assert server["cwd"] == str(root.resolve())


def test_agent_and_claude_auto_update_semantics_are_independent(
    tmp_path: Path,
) -> None:
    setting = {"com.langchain.deepagents.code": {"autoUpdate": True}}
    agent_root = tmp_path / "agent"
    claude_root = tmp_path / "claude"
    _write_json(
        agent_root / "plugin.json",
        {**_agent_manifest(), "extensions": setting},
    )
    _write_json(
        claude_root / ".claude-plugin" / "plugin.json",
        {"name": "quality", "extensions": setting},
    )

    agent, _path, _warnings = load_manifest(agent_root)
    claude, _path, _warnings = load_manifest(claude_root)

    assert agent is not None
    assert claude is not None
    assert agent.auto_update is False
    assert claude.auto_update is True


def test_unsupported_agent_mcp_schema_disables_only_mcp(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    _write_json(root / "plugin.json", _agent_manifest())
    _write_skill(root / "skills" / "review" / "SKILL.md")
    _write_json(
        root / "mcp.json",
        {
            "$schema": "https://agent-plugins.org/schemas/2.0.0/mcp.schema.json",
            "mcpServers": {"server": {"command": "python"}},
        },
    )
    manifest, _path, _warnings = load_manifest(root)
    assert manifest is not None
    plugin = _plugin_instance(root, manifest)

    assert plugin.inventory.skills == ((root / "skills").resolve(),)
    assert plugin_mcp_configs((plugin,)) == []


def test_agent_mcp_path_escapes_are_isolated(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    _write_json(root / "plugin.json", _agent_manifest())
    outside = tmp_path / "outside"
    outside.write_text("server", encoding="utf-8")
    (root / "bin").mkdir(parents=True)
    try:
        (root / "bin" / "linked").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _write_json(
        root / "mcp.json",
        {
            "mcpServers": {
                "valid": {"command": "python"},
                "command-traversal": {"command": "./../outside"},
                "cwd-traversal": {"command": "python", "cwd": "../outside"},
                "symlink-escape": {"command": "./bin/linked"},
            }
        },
    )
    manifest, _path, _warnings = load_manifest(root)
    assert manifest is not None
    plugin = _plugin_instance(root, manifest)

    configs = plugin_mcp_configs((plugin,))
    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])

    assert set(servers) == {scoped_mcp_server_name(plugin.plugin_id, "valid")}


def test_agent_mcp_uses_shared_substitution_and_environment_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "plugin"
    _write_json(root / "plugin.json", _agent_manifest())
    _write_json(
        root / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_V1_MCP_SCHEMA,
            "mcpServers": {
                "stdio": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "args": ["${PLUGIN_ROOT}", "${PLUGIN_TOKEN}"],
                    "cwd": "work",
                    "env": {
                        "TOKEN": "${PLUGIN_TOKEN}",
                        "PLUGIN_ROOT": "/configured/root",
                        "PLUGIN_DATA": "/configured/data",
                    },
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.com/${PLUGIN_PATH}",
                    "headers": {"Authorization": "Bearer ${PLUGIN_TOKEN}"},
                },
                "ambient-cwd": {
                    "command": "python",
                    "cwd": "${PLUGIN_WORK_DIR}",
                },
                "invalid": {"type": []},
            },
        },
    )
    manifest, _path, _warnings = load_manifest(root)
    assert manifest is not None
    plugin = _plugin_instance(root, manifest)

    configs = plugin_mcp_configs((plugin,), project_dir=tmp_path / "project")
    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    stdio_name = scoped_mcp_server_name(plugin.plugin_id, "stdio")
    remote_name = scoped_mcp_server_name(plugin.plugin_id, "remote")
    ambient_cwd_name = scoped_mcp_server_name(plugin.plugin_id, "ambient-cwd")
    invalid_name = scoped_mcp_server_name(plugin.plugin_id, "invalid")
    assert invalid_name in servers
    stdio = servers[stdio_name]
    remote = servers[remote_name]
    ambient_cwd = servers[ambient_cwd_name]
    stdio_env = cast("dict[str, str]", stdio["env"])
    remote_env = cast("dict[str, str]", remote["env"])

    assert stdio["command"] == str((root / "bin" / "server").resolve())
    assert stdio["cwd"] == str((root / "work").resolve())
    assert stdio_env["PLUGIN_ROOT"] == str(root.resolve())
    assert stdio_env["PLUGIN_DATA"] == str(plugin.data_dir.resolve())
    assert stdio_env["TOKEN"] == "${PLUGIN_TOKEN}"
    assert remote_env["PLUGIN_ROOT"] == str(root.resolve())
    assert ambient_cwd["cwd"] == "${PLUGIN_WORK_DIR}"

    monkeypatch.setenv("PLUGIN_TOKEN", "token-value")
    monkeypatch.setenv("PLUGIN_PATH", "mcp")
    monkeypatch.setenv("PLUGIN_WORK_DIR", "/tmp/plugin-work")
    resolved_stdio = resolve_mcp_server_env(stdio_name, stdio)
    resolved_remote = resolve_mcp_server_env(remote_name, remote)
    resolved_ambient_cwd = resolve_mcp_server_env(
        ambient_cwd_name,
        ambient_cwd,
    )

    assert resolved_stdio["args"] == [str(root.resolve()), "token-value"]
    assert resolved_stdio["env"]["TOKEN"] == "token-value"
    assert resolved_remote["url"] == "https://example.com/mcp"
    assert resolved_remote["headers"]["Authorization"] == "Bearer token-value"
    assert resolved_ambient_cwd["cwd"] == "/tmp/plugin-work"


def test_agent_hooks_use_the_shared_hook_adapter(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    _write_json(
        root / "plugin.json",
        {
            **_agent_manifest(),
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "inline"}]}]},
        },
    )
    _write_json(
        root / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command", "command": "file"}]}]
            }
        },
    )
    manifest, _path, _warnings = load_manifest(root)
    assert manifest is not None
    plugin = _plugin_instance(root, manifest)

    documents, diagnostics = discover_plugin_hook_sources(
        project_dir=tmp_path / "project",
        plugins=(plugin,),
    )

    assert diagnostics == ()
    assert len(documents) == 2
    assert documents[0][0].location == str((root / "hooks" / "hooks.json").resolve())
    assert documents[0][0].env["PLUGIN_ROOT"] == str(root.resolve())
    assert documents[1][0].location == str(root / "plugin.json")


def test_marketplace_installs_and_discovers_agent_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        tmp_path / "state",
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR",
        tmp_path / "config",
    )
    marketplace = tmp_path / "marketplace"
    _write_json(
        marketplace / ".claude-plugin" / "marketplace.json",
        {
            "name": "tools",
            "plugins": [{"name": "quality", "source": "./plugins/quality"}],
        },
    )
    source = marketplace / "plugins" / "quality"
    _write_json(source / "plugin.json", _agent_manifest())
    _write_skill(source / "skills" / "review" / "SKILL.md")
    _write_json(source / "mcp.json", {"mcpServers": {}})
    _write_json(source / "hooks" / "hooks.json", {"hooks": {}})

    add_local_marketplace(marketplace)
    installed = install_plugin("quality@tools")
    discovered = discover_plugins().plugins

    assert installed.manifest is not None
    assert installed.manifest.dialect == "agent-plugin-v1"
    assert installed.root != source
    assert installed.inventory.skills == ((installed.root / "skills").resolve(),)
    assert installed.inventory.mcp_files == ((installed.root / "mcp.json").resolve(),)
    assert installed.inventory.hook_files == (
        (installed.root / "hooks" / "hooks.json").resolve(),
    )
    assert discovered == (installed,)
