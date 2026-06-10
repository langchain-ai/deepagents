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


_FAKE_CLASS_PATH = f"{__name__}:FakeProvider"


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
    ]
    assert registry.is_available("daytona")
    assert not registry.is_available("acme")


def test_builtin_metadata_working_dir() -> None:
    registry = _empty_registry()
    assert _metadata(registry, "modal").working_dir == "/workspace"
    assert _metadata(registry, "langsmith").supports_snapshot_name is True
    assert _metadata(registry, "daytona").supports_snapshot_name is False


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
