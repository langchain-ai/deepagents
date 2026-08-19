"""Tests for the public extension factory API."""

from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.extensions import ExtensionAPI
from deepagents_code.extensions.discovery import discover_extension_files
from deepagents_code.extensions.models import ExtensionError, SourceInfo
from deepagents_code.extensions.registry import ExtensionRegistry


class _Middleware(AgentMiddleware):
    """Minimal extension middleware."""


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


def test_rejects_invalid_or_late_registrations(api: ExtensionAPI) -> None:
    """Invalid types and post-factory mutation fail at the API boundary."""
    with pytest.raises(ExtensionError, match="AgentMiddleware"):
        api.register_middleware(cast("Any", object()))
    with pytest.raises(ExtensionError, match="BackendProtocol"):
        api.register_backend_route("/memories/", cast("Any", object()))
    api._close()
    with pytest.raises(ExtensionError, match="factory is running"):
        api.register_middleware(_Middleware())


def test_discovery_requires_experimental_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extension files remain invisible until explicitly enabled."""
    path = tmp_path / "example.py"
    path.touch()
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    assert not discover_extension_files(user_dir=tmp_path)
    monkeypatch.setenv(EXPERIMENTAL, "1")
    assert [source.path for source in discover_extension_files(user_dir=tmp_path)] == [
        path
    ]
