"""Tests for the public extension factory API."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.extensions import ExtensionAPI
from deepagents_code.extensions.discovery import discover_extension_files
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
)
from deepagents_code.plugins.manifest import load_manifest
from deepagents_code.plugins.models import ComponentInventory, PluginInstance


class _Middleware(AgentMiddleware):
    """Minimal extension middleware."""


def _write_manifest(
    root: Path, *, path: str = "./extension.py", version: str | None = None
) -> None:
    manifest: dict[str, Any] = {
        "name": "example",
        "extensions": {"com.langchain.deepagents.code": {"pythonExtensions": path}},
    }
    if version is not None:
        manifest["version"] = version
    (root / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def api() -> ExtensionAPI:
    """Return a factory-scoped registrar."""
    return ExtensionAPI(
        ExtensionRegistry(),
        SourceInfo(Path("/extensions/example.py")),
        cwd=Path("/workspace"),
        mode="interactive",
    )


def test_registers_supported_units_and_context(api: ExtensionAPI) -> None:
    """Factories can register every supported unit and read session context."""
    backend = FilesystemBackend(virtual_mode=False)

    def status() -> str:
        """Report status."""
        return "ready"

    api.register_middleware(_Middleware)
    api.register_tool(status)
    api.register_backend_route("/memories/", backend)
    api.on_shutdown(lambda: None)

    registry = api._registry
    assert [item.name for item in registry.middleware] == ["_Middleware"]
    assert [item.name for item in registry.tools] == ["status"]
    assert registry.backend_routes[0].unit is backend
    assert len(registry.shutdown_hooks) == 1
    assert (api.cwd, api.mode, api.path) == (
        Path("/workspace"),
        "interactive",
        Path("/extensions/example.py"),
    )


@pytest.mark.parametrize(
    "prefix",
    ["memories/", "/memories", "/../memories/", "/memory//notes/", "/UPPER/"],
)
def test_rejects_unsafe_routes(api: ExtensionAPI, prefix: str) -> None:
    """Backend prefixes must be canonical virtual paths."""
    with pytest.raises(ExtensionError, match="Invalid backend route prefix"):
        api.register_backend_route(prefix, FilesystemBackend(virtual_mode=False))


def test_rejects_invalid_and_allows_runtime_registrations(api: ExtensionAPI) -> None:
    """Invalid types fail while valid runtime registrations remain supported."""
    with pytest.raises(ExtensionError, match="AgentMiddleware"):
        api.register_middleware(cast("Any", object()))
    with pytest.raises(ExtensionError, match="BackendProtocol"):
        api.register_backend_route("/memories/", cast("Any", object()))
    api.register_middleware(_Middleware())
    assert api._registry.middleware[0].name == "_Middleware"


def test_registrar_closes_after_session(api: ExtensionAPI) -> None:
    """A retained registrar cannot mutate a registry after teardown."""
    api._registry.retain_api(api)
    api._registry.deactivate_apis()

    with pytest.raises(ExtensionError, match="closed for this session"):
        api.register_middleware(_Middleware())


def test_plugin_extension_discovery_requires_experimental_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed plugin entries remain invisible until explicitly enabled."""
    path = tmp_path / "example.py"
    path.touch()
    _write_manifest(tmp_path, path="./example.py", version="1.2.3")
    manifest, _, warnings = load_manifest(tmp_path)
    assert manifest is not None
    assert not warnings
    plugin = PluginInstance(
        plugin_id="example@test",
        name="example",
        marketplace="test",
        version=manifest.version,
        root=tmp_path,
        data_dir=tmp_path / "data",
        manifest=manifest,
        inventory=ComponentInventory(),
    )
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    assert not discover_extension_files(plugins=(plugin,))
    monkeypatch.setenv(EXPERIMENTAL, "1")
    sources = discover_extension_files(plugins=(plugin,))
    assert [source.path for source in sources] == [path.resolve()]
    assert sources[0].plugin_id == "example@test"


def test_plugin_python_extension_requires_version(tmp_path: Path) -> None:
    """User-wide executable extensions must have a plugin version."""
    path = tmp_path / "extension.py"
    path.touch()
    _write_manifest(tmp_path)

    manifest, _, warnings = load_manifest(tmp_path)

    assert manifest is not None
    assert not manifest.python_extensions
    assert any("require a non-empty plugin version" in warning for warning in warnings)
