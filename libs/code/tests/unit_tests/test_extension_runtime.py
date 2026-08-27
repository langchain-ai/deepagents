"""Tests for extension loading, trust gating, and teardown."""

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from deepagents_code.extensions import load_extensions
from deepagents_code.extensions.registry import ExtensionError
from deepagents_code.extensions.runtime import (
    ExtensionLoadResult,
    bind_server_extensions,
    shutdown_extensions,
    shutdown_server_extensions,
)
from deepagents_code.extensions.settings import ExtensionSettings, TrustPolicy
from deepagents_code.plugins.models import (
    ComponentInventory,
    PluginDiscoveryResult,
    PluginInstance,
    PluginManifest,
)


@pytest.fixture(autouse=True)
def _isolate_plugins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent tests from loading executable code from real user state."""
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "1")
    monkeypatch.setattr(
        "deepagents_code.plugins.discover_plugins",
        lambda: PluginDiscoveryResult(plugins=()),
    )
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: tmp_path / "user-extensions",
    )
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.importlib.metadata.entry_points",
        lambda **_: (),
    )


def _plugin(root: Path, entries: list[str]) -> PluginInstance:
    """Create one installed plugin declaring Python entry files."""
    return PluginInstance(
        plugin_id="test-extension@test",
        name="test-extension",
        marketplace="test",
        version="1.0.0",
        root=root,
        data_dir=root / "data",
        manifest=PluginManifest(
            name="test-extension",
            version="1.0.0",
            component_paths={},
            inline_mcp={},
            python_extensions=tuple(root / entry for entry in entries),
        ),
        inventory=ComponentInventory(),
    )


async def test_experimental_mode_gates_runtime_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled experiment performs no config, trust, or plugin lookup."""
    from deepagents_code.extensions import runtime

    monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL")

    def fail(*_: object, **__: object) -> None:
        msg = "extension runtime crossed the experimental gate"
        raise AssertionError(msg)

    monkeypatch.setattr(runtime, "load_extension_settings", fail)
    monkeypatch.setattr(runtime, "is_project_extensions_trusted", fail)
    monkeypatch.setattr("deepagents_code.plugins.discover_plugins", fail)

    result = await load_extensions(
        project_root=tmp_path,
        project_trust_granted=True,
        cli_paths=(tmp_path / "extension.py",),
    )

    assert not result.registry.registrations()
    assert not result.errors
    assert not result.active


async def test_failures_roll_back_and_teardown_stays_on_factory_loop(
    tmp_path: Path,
) -> None:
    """One bad factory cannot leak units or block same-loop teardown."""
    directory = tmp_path / "plugin"
    directory.mkdir()
    (directory / "a_broken.py").write_text(
        """
async def extension(d):
    def partial():
        \"\"\"Never become visible.\"\"\"
    d.register_tool(partial)
    raise RuntimeError(\"boom\")
""",
        encoding="utf-8",
    )
    marker = tmp_path / "closed"
    (directory / "b_valid.py").write_text(
        f"""
import asyncio

async def extension(d):
    loop = asyncio.get_running_loop()
    def ready():
        \"\"\"Report readiness.\"\"\"
        return \"ready\"
    async def shutdown():
        assert asyncio.get_running_loop() is loop
        open({str(marker)!r}, \"w\").close()
    d.register_tool(ready)
    d.on_shutdown(shutdown)
""",
        encoding="utf-8",
    )
    plugin = _plugin(directory, ["a_broken.py", "b_valid.py"])

    with patch(
        "deepagents_code.plugins.discover_plugins",
        return_value=PluginDiscoveryResult(plugins=(plugin,)),
    ):
        result = await load_extensions(cwd=tmp_path)

    assert [item.name for item in result.registry.tools] == ["ready"]
    assert len(result.errors) == 1
    bind_server_extensions(result)
    await shutdown_server_extensions()
    assert marker.exists()


@pytest.mark.parametrize(
    ("policy", "grant", "expected"),
    [
        (TrustPolicy.ASK, False, None),
        (TrustPolicy.ASK, True, "extensions"),
        (TrustPolicy.NEVER, True, None),
    ],
)
async def test_project_directory_is_hidden_until_trusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: TrustPolicy,
    grant: bool,
    expected: str | None,
) -> None:
    """Discovery never receives an unauthorized project directory."""
    from deepagents_code.extensions import runtime

    observed: list[Path | None] = []

    def discover(**kwargs: object) -> object:
        from deepagents_code.extensions.discovery import DiscoveryResult

        observed.append(cast("Path | None", kwargs["project_dir"]))
        return DiscoveryResult()

    monkeypatch.setattr(runtime, "discover_extensions", discover)
    monkeypatch.setattr(
        runtime,
        "load_extension_settings",
        lambda: ExtensionSettings(trust=policy),
    )
    monkeypatch.setattr(runtime, "is_project_extensions_trusted", lambda _: False)

    await load_extensions(project_root=tmp_path, project_trust_granted=grant)

    assert (observed[0].name if observed[0] else None) == expected
