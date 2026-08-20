"""Tests for extension loading, trust gating, and teardown."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from deepagents_code.extensions import load_extensions
from deepagents_code.extensions.runtime import (
    bind_server_extensions,
    shutdown_server_extensions,
)
from deepagents_code.extensions.settings import ExtensionSettings, TrustPolicy
from deepagents_code.plugins.manifest import load_manifest
from deepagents_code.plugins.models import (
    ComponentInventory,
    PluginDiscoveryResult,
    PluginInstance,
)


@pytest.fixture(autouse=True)
def _isolate_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from loading installed plugins from the real user state."""
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "1")
    monkeypatch.setattr(
        "deepagents_code.plugins.discover_plugins",
        lambda: PluginDiscoveryResult(plugins=()),
    )


def _plugin(root: Path, entries: list[str]) -> PluginInstance:
    """Create one installed plugin declaring Python entry files."""
    (root / "plugin.json").write_text(
        json.dumps(
            {
                "name": "test-extension",
                "version": "1.0.0",
                "extensions": {
                    "com.langchain.deepagents.code": {
                        "pythonExtensions": [f"./{entry}" for entry in entries]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest, _, warnings = load_manifest(root)
    assert manifest is not None
    assert not warnings
    return PluginInstance(
        plugin_id="test-extension@test",
        name="test-extension",
        marketplace="test",
        version=manifest.version,
        root=root,
        data_dir=root / "data",
        manifest=manifest,
        inventory=ComponentInventory(),
    )


async def test_failures_roll_back_and_teardown_stays_on_factory_loop(
    tmp_path: Path,
) -> None:
    """One bad factory cannot leak units or block same-loop teardown."""
    directory = tmp_path / "plugin"
    directory.mkdir()
    (directory / "a_broken.py").write_text(
        """
def extension(d):
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
    bind_server_extensions(result.registry)
    await shutdown_server_extensions()
    assert marker.exists()


async def test_package_extensions_support_relative_imports(
    tmp_path: Path,
) -> None:
    """A discovered package can import sibling modules."""
    root = tmp_path / "plugin"
    package = root / "sample"
    package.mkdir(parents=True)
    (package / "helper.py").write_text("VALUE = 'ready'\n", encoding="utf-8")
    (package / "__init__.py").write_text(
        """
from .helper import VALUE
def extension(d):
    def package_status():
        \"\"\"Report package status.\"\"\"
        return VALUE
    d.register_tool(package_status)
""",
        encoding="utf-8",
    )
    plugin = _plugin(root, ["sample/__init__.py"])

    with patch(
        "deepagents_code.plugins.discover_plugins",
        return_value=PluginDiscoveryResult(plugins=(plugin,)),
    ):
        result = await load_extensions()

    assert result.registry.tools[0].unit.invoke({}) == "ready"


async def test_plugin_setup_runs_eagerly(tmp_path: Path) -> None:
    """Every enabled plugin setup function finishes before loading returns."""
    root = tmp_path / "plugin"
    root.mkdir()
    marker = tmp_path / "initialized"
    (root / "extension.py").write_text(
        f"def extension(d):\n    open({str(marker)!r}, 'w').close()\n",
        encoding="utf-8",
    )
    plugin = _plugin(root, ["extension.py"])

    with patch(
        "deepagents_code.plugins.discover_plugins",
        return_value=PluginDiscoveryResult(plugins=(plugin,)),
    ):
        await load_extensions()

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

    def discover(**kwargs: object) -> list[object]:
        observed.append(cast("Path | None", kwargs["project_dir"]))
        return []

    monkeypatch.setattr(runtime, "discover_extension_files", discover)
    monkeypatch.setattr(
        runtime,
        "load_extension_settings",
        lambda: ExtensionSettings(trust=policy),
    )
    monkeypatch.setattr(runtime, "is_project_extensions_trusted", lambda _: False)

    await load_extensions(project_root=tmp_path, project_trust_granted=grant)

    assert (observed[0].name if observed[0] else None) == expected
