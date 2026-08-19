"""Tests for extension loading, trust gating, and teardown."""

from pathlib import Path
from typing import cast

import pytest

from deepagents_code.extensions import load_extensions
from deepagents_code.extensions.runtime import (
    bind_server_extensions,
    shutdown_server_extensions,
)
from deepagents_code.extensions.settings import ExtensionSettings, TrustPolicy


@pytest.fixture(autouse=True)
def _isolate_user_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from loading real user extensions."""
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: tmp_path / "no-user-extensions",
    )


async def test_failures_roll_back_and_teardown_stays_on_factory_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad factory cannot leak units or block same-loop teardown."""
    directory = tmp_path / "extensions"
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
    monkeypatch.setattr(
        "deepagents_code.extensions.runtime.load_extension_settings",
        lambda: ExtensionSettings(paths=(directory,)),
    )

    result = await load_extensions(cwd=tmp_path)

    assert [item.name for item in result.registry.tools] == ["ready"]
    assert len(result.errors) == 1
    bind_server_extensions(result.registry)
    await shutdown_server_extensions()
    assert marker.exists()


async def test_package_extensions_support_relative_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovered package can import sibling modules."""
    package = tmp_path / "extensions" / "sample"
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
    monkeypatch.setattr(
        "deepagents_code.extensions.runtime.load_extension_settings",
        lambda: ExtensionSettings(paths=(package.parent,)),
    )

    result = await load_extensions()

    assert result.registry.tools[0].unit.invoke({}) == "ready"


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
