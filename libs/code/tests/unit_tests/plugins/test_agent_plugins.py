from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from deepagents_code.mcp_config import (
    MCP_ENV_RESOLUTION_DISABLED,
    MCP_REDIRECTS_DISABLED,
    resolve_mcp_server_env,
)
from deepagents_code.plugins import (
    add_local_marketplace,
    discover_plugins,
    install_plugin,
)
from deepagents_code.plugins.adapters.mcp import (
    plugin_mcp_configs,
    plugin_mcp_server_entries,
    scoped_mcp_server_name,
)
from deepagents_code.plugins.agent_plugins import (
    AGENT_PLUGIN_MANIFEST_SCHEMA,
    AGENT_PLUGIN_MCP_SCHEMA,
    _expand_plugin_value,
)
from deepagents_code.plugins.manifest import (
    PluginManifestError,
    build_inventory,
    load_manifest,
)
from deepagents_code.plugins.models import JsonObject, PluginInstance


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_skill(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n---\n\nRun {name}.",
        encoding="utf-8",
    )


def _write_manifest(root: Path, **overrides: object) -> None:
    manifest: dict[str, object] = {
        "$schema": AGENT_PLUGIN_MANIFEST_SCHEMA,
        "name": "portable-plugin",
        "version": "1.2.3",
    }
    manifest.update(overrides)
    _write_json(root / "plugin.json", manifest)


def _instance(root: Path, data_dir: Path) -> PluginInstance:
    manifest, _path, warnings = load_manifest(root)
    assert manifest is not None
    return PluginInstance(
        plugin_id="portable-plugin@tools",
        name="portable-plugin",
        marketplace="tools",
        version=manifest.version,
        root=root.resolve(),
        data_dir=data_dir.resolve(),
        manifest=manifest,
        inventory=build_inventory(root, manifest, warnings),
    )


def test_plugin_placeholder_expansion_is_single_pass(tmp_path: Path) -> None:
    root = tmp_path / "${PLUGIN_DATA}" / "plugin"
    data = tmp_path / "data"

    result = _expand_plugin_value(
        "${PLUGIN_ROOT}/config",
        plugin_root=root,
        plugin_data=data,
    )

    assert result == f"{root}/config"


def test_agent_plugin_manifest_precedes_legacy_and_reports_nonfatal_fields(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        unexpected={"value": True},
        extensions=["invalid"],
    )
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text("{", encoding="utf-8")

    manifest, path, warnings = load_manifest(tmp_path)

    assert manifest is not None
    assert manifest.name == "portable-plugin"
    assert manifest.version == "1.2.3"
    assert manifest.plugin_format == "agent-plugins-v1"
    assert path == tmp_path / "plugin.json"
    assert warnings == (
        "ignoring unknown Agent Plugins manifest field 'unexpected'",
        "ignoring non-object Agent Plugins extensions field",
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"$schema": "https://example.com/schema.json"}, r"\$schema"),
        ({"name": "Invalid Name"}, "name"),
        ({"version": 1}, "version"),
        ({"author": {"name": "Team", "extra": True}}, "author"),
        ({"keywords": ["valid", 1]}, "keywords"),
        ({"extensions": {"com.example": "invalid"}}, "extensions"),
    ],
)
def test_agent_plugin_manifest_rejects_schema_violations(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    _write_manifest(tmp_path, **override)

    with pytest.raises(PluginManifestError, match=message):
        load_manifest(tmp_path)


def test_agent_plugin_manifest_rejects_nonstandard_json(tmp_path: Path) -> None:
    (tmp_path / "plugin.json").write_text(
        '{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",'
        '"name":"portable-plugin","version":NaN}',
        encoding="utf-8",
    )

    with pytest.raises(PluginManifestError, match="invalid JSON constant"):
        load_manifest(tmp_path)


def test_invalid_root_manifest_does_not_fall_back_to_legacy(tmp_path: Path) -> None:
    _write_json(tmp_path / "plugin.json", {"name": "portable-plugin"})
    _write_json(
        tmp_path / ".claude-plugin" / "plugin.json",
        {"name": "portable-plugin"},
    )

    with pytest.raises(PluginManifestError, match=r"\$schema"):
        load_manifest(tmp_path)


def test_agent_plugin_manifest_symlink_cannot_escape_root(tmp_path: Path) -> None:
    external = tmp_path / "external.json"
    _write_json(
        external,
        {"$schema": AGENT_PLUGIN_MANIFEST_SCHEMA, "name": "portable-plugin"},
    )
    root = tmp_path / "plugin"
    root.mkdir()
    (root / "plugin.json").symlink_to(external)

    with pytest.raises(PluginManifestError, match="escapes plugin root"):
        load_manifest(root)


def test_marketplace_installs_and_discovers_agent_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / "state"
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    marketplace = tmp_path / "marketplace"
    _write_json(
        marketplace / ".claude-plugin" / "marketplace.json",
        {
            "name": "tools",
            "plugins": [
                {"name": "portable-plugin", "source": "./plugins/portable-plugin"}
            ],
        },
    )
    source = marketplace / "plugins" / "portable-plugin"
    _write_manifest(source)
    _write_skill(source / "skills" / "review" / "SKILL.md", "review")

    add_local_marketplace(marketplace)
    installed = install_plugin("portable-plugin@tools")
    discovered = discover_plugins()

    assert installed.manifest is not None
    assert installed.manifest.plugin_format == "agent-plugins-v1"
    assert installed.root != source
    assert installed.inventory.skills == (
        (installed.root / "skills" / "review" / "SKILL.md").resolve(),
    )
    assert discovered.plugins == (installed,)


def test_agent_plugin_inventory_uses_only_fixed_immediate_components(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    _write_skill(tmp_path / "skills" / "review" / "SKILL.md", "review")
    _write_skill(
        tmp_path / "skills" / "nested" / "audit" / "SKILL.md",
        "audit",
    )
    _write_skill(tmp_path / "SKILL.md", "root")
    _write_json(tmp_path / ".mcp.json", {"mcpServers": {}})
    _write_json(
        tmp_path / "mcp.json",
        {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": {}},
    )
    _write_json(tmp_path / "hooks" / "hooks.json", {"hooks": {}})

    manifest, _path, warnings = load_manifest(tmp_path)
    assert manifest is not None
    inventory = build_inventory(tmp_path, manifest, warnings)

    assert inventory.skills == (
        (tmp_path / "skills" / "review" / "SKILL.md").resolve(),
    )
    assert inventory.mcp_files == ((tmp_path / "mcp.json").resolve(),)
    assert inventory.hook_files == ()
    assert inventory.unsupported == ()


def test_agent_plugin_inventory_isolates_component_symlink_escapes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "plugin"
    _write_manifest(root)
    external_skill = tmp_path / "external-skill"
    _write_skill(external_skill / "SKILL.md", "external")
    (root / "skills").mkdir()
    (root / "skills" / "external").symlink_to(external_skill, target_is_directory=True)
    external_mcp = tmp_path / "external-mcp.json"
    _write_json(
        external_mcp,
        {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": {}},
    )
    (root / "mcp.json").symlink_to(external_mcp)

    manifest, _path, warnings = load_manifest(root)
    assert manifest is not None
    inventory = build_inventory(root, manifest, warnings)

    assert inventory.skills == ()
    assert inventory.mcp_files == ()
    assert any("skill outside root" in warning for warning in inventory.warnings)
    assert any("invalid Agent Plugins MCP" in warning for warning in inventory.warnings)


def test_agent_plugin_mcp_adapts_portable_servers(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(root)
    _write_json(
        root / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "local": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "args": [
                        "--cache",
                        "${PLUGIN_DATA}/cache",
                        "${UNKNOWN_LITERAL}",
                    ],
                    "env": {"CONFIG": "${PLUGIN_ROOT}/config.json"},
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Literal": "${TOKEN}"},
                },
                "loopback": {
                    "type": "sse",
                    "url": "http://127.0.0.1:8000/sse",
                },
            },
        },
    )
    plugin = _instance(root, data_dir)

    configs = plugin_mcp_configs((plugin,), project_dir=tmp_path / "project")

    assert len(configs) == 1
    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    local = servers[scoped_mcp_server_name(plugin.plugin_id, "local")]
    assert local["command"] == str((root / "bin" / "server").resolve())
    assert local["cwd"] == str(root.resolve())
    assert local["args"] == [
        "--cache",
        f"{data_dir.resolve()}/cache",
        "${UNKNOWN_LITERAL}",
    ]
    env = cast("dict[str, str]", local["env"])
    assert env["CONFIG"] == f"{root.resolve()}/config.json"
    assert env["PLUGIN_ROOT"] == str(root.resolve())
    assert env["PLUGIN_DATA"] == str(data_dir.resolve())

    remote = servers[scoped_mcp_server_name(plugin.plugin_id, "remote")]
    assert remote["type"] == "streamable-http"
    assert remote["headers"] == {"X-Literal": "${TOKEN}"}
    assert remote[MCP_REDIRECTS_DISABLED] is True
    assert "env" not in remote

    resolved_local = resolve_mcp_server_env("local", local)
    resolved_remote = resolve_mcp_server_env("remote", remote)
    assert resolved_local["args"] == local["args"]
    assert resolved_remote["headers"] == remote["headers"]
    assert MCP_ENV_RESOLUTION_DISABLED not in resolved_local
    assert MCP_ENV_RESOLUTION_DISABLED not in resolved_remote
    assert {entry[0] for entry in plugin_mcp_server_entries(plugin)} == {
        "local",
        "loopback",
        "remote",
    }


def test_agent_plugin_mcp_resolves_relative_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("plugin")
    data_dir = Path("cache/data")
    _write_manifest(root)
    _write_json(
        root / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "local": {
                    "type": "stdio",
                    "command": "./bin/server",
                    "args": ["${PLUGIN_DATA}/state"],
                }
            },
        },
    )
    manifest, _path, warnings = load_manifest(root)
    assert manifest is not None
    plugin = PluginInstance(
        plugin_id="portable-plugin@tools",
        name="portable-plugin",
        marketplace="tools",
        version=manifest.version,
        root=root,
        data_dir=data_dir,
        manifest=manifest,
        inventory=build_inventory(root, manifest, warnings),
    )

    configs = plugin_mcp_configs((plugin,))

    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    local = servers[scoped_mcp_server_name(plugin.plugin_id, "local")]
    expected_root = root.resolve()
    expected_data = data_dir.resolve()
    assert local["command"] == str(expected_root / "bin/server")
    assert local["cwd"] == str(expected_root)
    assert local["args"] == [str(expected_data / "state")]
    env = cast("dict[str, str]", local["env"])
    assert env["PLUGIN_ROOT"] == str(expected_root)
    assert env["PLUGIN_DATA"] == str(expected_data)


def test_unwritable_plugin_data_skips_only_stdio_servers(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    data_dir = tmp_path / "data"
    data_dir.write_text("not a directory", encoding="utf-8")
    _write_manifest(root)
    _write_json(
        root / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "local": {"type": "stdio", "command": "python"},
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                },
            },
        },
    )
    plugin = _instance(root, data_dir)

    configs = plugin_mcp_configs((plugin,))

    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    assert set(servers) == {scoped_mcp_server_name(plugin.plugin_id, "remote")}


def test_agent_plugin_mcp_skips_invalid_entries_independently(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(root)
    _write_json(
        root / "mcp.json",
        {
            "$schema": AGENT_PLUGIN_MCP_SCHEMA,
            "mcpServers": {
                "valid": {"type": "stdio", "command": "python"},
                "shell-command": {"type": "stdio", "command": "sh -c true"},
                "escaping-command": {
                    "type": "stdio",
                    "command": "./../outside",
                },
                "invalid-command-path": {
                    "type": "stdio",
                    "command": "./bin/\x00server",
                },
                "escaping-cwd": {
                    "type": "stdio",
                    "command": "python",
                    "cwd": "${PLUGIN_DATA}/../outside",
                },
                "reserved-env": {
                    "type": "stdio",
                    "command": "python",
                    "env": {"PLUGIN_ROOT": "/tmp"},
                },
                "insecure-remote": {
                    "type": "streamable-http",
                    "url": "http://example.com/mcp",
                },
                "userinfo": {
                    "type": "streamable-http",
                    "url": "https://user@example.com/mcp",
                },
                "invalid-url": {
                    "type": "streamable-http",
                    "url": "https://exa mple.com/mcp",
                },
                "url-placeholder": {
                    "type": "streamable-http",
                    "url": "https://${HOST}/mcp",
                },
                "duplicate-headers": {
                    "type": "sse",
                    "url": "https://example.com/sse",
                    "headers": {"X-Test": "a", "x-test": "b"},
                },
                "unicode-header": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"X-Label": "café"},
                },
                "unknown-field": {
                    "type": "stdio",
                    "command": "python",
                    "extra": True,
                },
            },
        },
    )
    plugin = _instance(root, data_dir)

    configs = plugin_mcp_configs((plugin,))

    servers = cast("dict[str, JsonObject]", configs[0]["mcpServers"])
    assert set(servers) == {scoped_mcp_server_name(plugin.plugin_id, "valid")}


def test_nonstandard_json_disables_agent_plugin_mcp(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(root)
    (root / "mcp.json").write_text(
        '{"$schema":"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",'
        '"mcpServers":{"invalid":NaN}}',
        encoding="utf-8",
    )
    plugin = _instance(root, data_dir)

    assert plugin_mcp_configs((plugin,)) == []


@pytest.mark.parametrize(
    "document",
    [
        None,
        {},
        {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": {}, "extra": True},
        {"$schema": "https://example.com/mcp.schema.json", "mcpServers": {}},
        {"$schema": AGENT_PLUGIN_MCP_SCHEMA, "mcpServers": []},
    ],
)
def test_invalid_agent_plugin_mcp_document_disables_only_mcp(
    tmp_path: Path, document: object
) -> None:
    root = tmp_path / "plugin"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(root)
    _write_skill(root / "skills" / "review" / "SKILL.md", "review")
    _write_json(root / "mcp.json", document)
    plugin = _instance(root, data_dir)

    assert plugin.inventory.skills
    assert plugin_mcp_configs((plugin,)) == []


def test_legacy_plugin_discovery_remains_unchanged(tmp_path: Path) -> None:
    _write_json(
        tmp_path / ".claude-plugin" / "plugin.json",
        {"name": "legacy-plugin", "version": "1.0.0"},
    )
    _write_skill(tmp_path / "skills" / "nested" / "SKILL.md", "nested")
    _write_json(tmp_path / ".mcp.json", {"mcpServers": {}})

    manifest, _path, warnings = load_manifest(tmp_path)
    assert manifest is not None
    inventory = build_inventory(tmp_path, manifest, warnings)

    assert manifest.plugin_format == "legacy"
    assert inventory.skills == ((tmp_path / "skills").resolve(),)
    assert inventory.mcp_files == ((tmp_path / ".mcp.json").resolve(),)
