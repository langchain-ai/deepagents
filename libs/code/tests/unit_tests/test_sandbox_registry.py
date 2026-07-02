"""Tests for sandbox provider discovery and the registry."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_provider import (
    SandboxProvider,
    SandboxProviderMetadata,
)
from deepagents_code.integrations.sandbox_registry import SandboxRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.backends.protocol import SandboxBackendProtocol


class FakeProvider(SandboxProvider):
    """Minimal provider used by config/entry-point discovery tests."""

    metadata = SandboxProviderMetadata(
        name="acme",
        working_dir="/acme",
        supports_snapshot_name=True,
    )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        return MagicMock()

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class ModalEntryPointProvider(SandboxProvider):
    """Entry-point provider that intentionally collides with a built-in name."""

    metadata = SandboxProviderMetadata(
        name="modal",
        working_dir="/third-party-modal",
        supports_snapshot_name=True,
    )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        return MagicMock()

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class RaisingProvider(SandboxProvider):
    """Entry-point provider whose constructor fails (e.g. missing creds)."""

    def __init__(self) -> None:
        msg = "no credentials configured"
        raise RuntimeError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        return MagicMock()

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        return None


_FAKE_CLASS_PATH = f"{__name__}:FakeProvider"
_MODAL_ENTRY_POINT_CLASS_PATH = f"{__name__}:ModalEntryPointProvider"
_RAISING_CLASS_PATH = f"{__name__}:RaisingProvider"


def _metadata(registry: SandboxRegistry, name: str) -> SandboxProviderMetadata:
    """Return non-`None` metadata for `name`, asserting it exists."""
    meta = registry.get_metadata(name)
    assert meta is not None
    return meta


def _empty_registry() -> SandboxRegistry:
    return SandboxRegistry(config=SandboxConfig(), include_entry_points=False)


def test_builtins_are_available() -> None:
    registry = _empty_registry()
    assert registry.available_providers() == [
        "agentcore",
        "daytona",
        "langsmith",
        "modal",
        "runloop",
        "vercel",
    ]
    assert registry.is_available("daytona")
    assert not registry.is_available("acme")


def test_builtin_metadata_working_dir() -> None:
    registry = _empty_registry()
    assert _metadata(registry, "modal").working_dir == "/workspace"
    assert _metadata(registry, "langsmith").supports_snapshot_name is True
    assert _metadata(registry, "daytona").supports_snapshot_name is False
    vercel = _metadata(registry, "vercel")
    assert vercel.working_dir == "/vercel/sandbox"
    assert vercel.backend_module == "langchain_vercel_sandbox"
    assert vercel.supports_sandbox_id is True
    assert vercel.supports_snapshot_name is False
    assert vercel.install is not None
    assert vercel.install.command(in_app=False) == "dcode --install vercel"


def test_unknown_provider_metadata_is_none() -> None:
    assert _empty_registry().get_metadata("acme") is None


def test_config_provider_is_discovered() -> None:
    config = SandboxConfig(
        default="acme",
        providers={
            "acme": {
                "class_path": _FAKE_CLASS_PATH,
                "working_dir": "/workspace",
                "package": "acme-dcode-sandbox",
                "params": {"region": "us-east-1"},
            }
        },
    )
    registry = SandboxRegistry(config=config, include_entry_points=False)
    assert registry.is_available("acme")
    assert registry.default == "acme"
    metadata = _metadata(registry, "acme")
    assert metadata.working_dir == "/workspace"
    assert metadata.install is not None
    assert metadata.install.kind == "package"
    assert metadata.install.name == "acme-dcode-sandbox"
    assert registry.get_params("acme") == {"region": "us-east-1"}
    provider = registry.create_provider("acme")
    assert isinstance(provider, FakeProvider)


def test_config_provider_without_class_path_raises() -> None:
    config = SandboxConfig(providers={"acme": {"working_dir": "/x"}})
    registry = SandboxRegistry(config=config, include_entry_points=False)
    with pytest.raises(ValueError, match="missing 'class_path'"):
        registry.create_provider("acme")


def test_entry_point_provider_is_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = importlib.metadata.EntryPoint(
        name="acme",
        value=_FAKE_CLASS_PATH.replace(":", ":"),
        group="deepagents_code.sandbox_providers",
    )

    def fake_entry_points(*, group: str) -> list[importlib.metadata.EntryPoint]:
        assert group == "deepagents_code.sandbox_providers"
        return [entry]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=True)
    assert registry.is_available("acme")
    provider = registry.create_provider("acme")
    assert isinstance(provider, FakeProvider)
    assert registry.provider_metadata("acme").working_dir == "/acme"
    metadata = _metadata(registry, "acme")
    assert metadata.working_dir == "/acme"
    assert metadata.supports_snapshot_name is True


def test_entry_point_metadata_overrides_builtin_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = importlib.metadata.EntryPoint(
        name="modal",
        value=_MODAL_ENTRY_POINT_CLASS_PATH,
        group="deepagents_code.sandbox_providers",
    )

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [entry],  # noqa: ARG005
    )
    registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=True)
    metadata = _metadata(registry, "modal")
    assert metadata.working_dir == "/third-party-modal"
    assert metadata.supports_snapshot_name is True
    assert isinstance(registry.create_provider("modal"), ModalEntryPointProvider)


def test_config_overrides_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = importlib.metadata.EntryPoint(
        name="acme",
        value="some.other:Provider",
        group="deepagents_code.sandbox_providers",
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [entry],  # noqa: ARG005
    )
    config = SandboxConfig(
        providers={"acme": {"class_path": _FAKE_CLASS_PATH, "working_dir": "/cfg"}}
    )
    registry = SandboxRegistry(config=config, include_entry_points=True)
    provider = registry.create_provider("acme")
    assert isinstance(provider, FakeProvider)
    assert _metadata(registry, "acme").working_dir == "/cfg"


def test_create_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox provider: acme"):
        _empty_registry().create_provider("acme")


def test_builtin_supports_sandbox_id_flags() -> None:
    """`supports_sandbox_id` reflects each built-in's capability."""
    registry = _empty_registry()
    assert _metadata(registry, "agentcore").supports_sandbox_id is False
    assert _metadata(registry, "daytona").supports_sandbox_id is True


def test_config_override_of_builtin_carries_backend_module() -> None:
    """Overriding a built-in keeps its probe module so deps stay verifiable."""
    config = SandboxConfig(
        providers={
            "daytona": {
                "class_path": _FAKE_CLASS_PATH,
                "package": "my-daytona",
            }
        }
    )
    registry = SandboxRegistry(config=config, include_entry_points=False)
    metadata = _metadata(registry, "daytona")
    # Probe module inherited from the built-in base...
    assert metadata.backend_module == "langchain_daytona"
    # ...while the install hint reflects the user's package override.
    assert metadata.install is not None
    assert metadata.install.kind == "package"
    assert metadata.install.name == "my-daytona"
    # Working dir falls back to the built-in default when not overridden.
    assert metadata.working_dir == "/home/daytona"


def test_pure_config_provider_has_no_backend_module() -> None:
    """A config provider with no built-in base skips the dependency probe."""
    config = SandboxConfig(providers={"acme": {"class_path": _FAKE_CLASS_PATH}})
    registry = SandboxRegistry(config=config, include_entry_points=False)
    assert _metadata(registry, "acme").backend_module is None


def test_discover_entry_points_failure_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure enumerating entry points degrades to built-ins only."""

    def boom(*, group: str) -> list[importlib.metadata.EntryPoint]:  # noqa: ARG001
        msg = "broken dist-info"
        raise RuntimeError(msg)

    monkeypatch.setattr(importlib.metadata, "entry_points", boom)
    registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=True)
    # Built-ins still resolve; discovery failure didn't crash construction.
    assert registry.is_available("daytona")
    assert "daytona" in registry.available_providers()


def test_provider_metadata_falls_back_when_instantiation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry-point provider that can't construct yields a safe placeholder."""
    entry = importlib.metadata.EntryPoint(
        name="acme",
        value=_RAISING_CLASS_PATH,
        group="deepagents_code.sandbox_providers",
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [entry],  # noqa: ARG005
    )
    registry = SandboxRegistry(config=SandboxConfig(), include_entry_points=True)
    metadata = registry.provider_metadata("acme")
    assert metadata.name == "acme"
    assert metadata.working_dir == "/workspace"


def test_config_error_surfaces_through_registry(tmp_path: Path) -> None:
    """A malformed config file is reported via `config_error`."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is not = valid = toml", encoding="utf-8")
    registry = SandboxRegistry.load(config_path)
    assert registry.config_error is not None
    assert "invalid TOML" in registry.config_error
