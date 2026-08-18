"""Tests for extension loading orchestration and lifecycle."""

from pathlib import Path
from typing import cast

import pytest

from deepagents_code.extensions import (
    ExtensionSettings,
    TrustPolicy,
    load_extensions,
    project_extensions_trusted,
    shutdown_extensions,
)
from deepagents_code.extensions.runtime import (
    bind_server_extensions,
    get_server_extensions,
    reset_server_extensions,
    shutdown_server_extensions,
)


@pytest.fixture(autouse=True)
def _isolate_user_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real user extensions out of runtime tests."""
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: tmp_path / "isolated-user-extensions",
    )


def _write(directory: Path, name: str, body: str) -> Path:
    """Write one test extension source file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


async def test_loads_user_and_explicitly_trusted_project_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project code should join user sources only after explicit trust."""
    user_dir = tmp_path / "user"
    project_root = tmp_path / "project"
    project_dir = project_root / ".deepagents" / "extensions"
    body = """
def extension(d):
    def status():
        \"\"\"Report extension status.\"\"\"
        return d.path.stem
    d.register_tool(status)
"""
    _write(user_dir, "user.py", body)
    _write(project_dir, "project.py", body.replace("status", "project_status"))
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir", lambda: user_dir
    )

    result = await load_extensions(
        cwd=tmp_path,
        project_root=project_root,
        project_trust_granted=True,
        settings=ExtensionSettings(),
    )

    assert [unit.name for unit in result.registry.tools] == [
        "status",
        "project_status",
    ]
    assert not result.errors


async def test_untrusted_project_directory_is_not_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project path must remain absent from discovery until authorized."""
    from deepagents_code.extensions import runtime

    observed: list[Path | None] = []

    def record_discovery(**kwargs: object) -> list[object]:
        observed.append(cast("Path | None", kwargs.get("project_dir")))
        return []

    monkeypatch.setattr(runtime, "discover_extension_files", record_discovery)

    await load_extensions(
        project_root=tmp_path,
        settings=ExtensionSettings(trust=TrustPolicy.ASK),
    )

    assert observed == [None]


async def test_bad_extension_does_not_block_later_files(tmp_path: Path) -> None:
    """Factory failures should be reported without aborting the load pass."""
    directory = tmp_path / "extensions"
    _write(directory, "a_broken.py", "raise RuntimeError('boom')\n")
    _write(
        directory,
        "b_valid.py",
        """
def extension(d):
    def status():
        \"\"\"Report status.\"\"\"
        return \"ready\"
    d.register_tool(status)
""",
    )

    result = await load_extensions(
        settings=ExtensionSettings(paths=(directory,)),
    )

    assert [unit.name for unit in result.registry.tools] == ["status"]
    assert len(result.errors) == 1
    assert "boom" in result.errors[0]


def test_project_trust_policy_is_fail_closed(tmp_path: Path) -> None:
    """Never must override grants and ask must require a stored decision."""
    store = tmp_path / "missing.json"

    assert not project_extensions_trusted(
        tmp_path, policy=TrustPolicy.NEVER, granted=True, store_path=store
    )
    assert project_extensions_trusted(
        tmp_path, policy=TrustPolicy.ALWAYS, store_path=store
    )
    assert not project_extensions_trusted(
        tmp_path, policy=TrustPolicy.ASK, store_path=store
    )


async def test_shutdown_uses_factory_loop_and_isolates_failures(tmp_path: Path) -> None:
    """Async resources should initialize and close on the same event loop."""
    directory = tmp_path / "extensions"
    _write(
        directory,
        "lifecycle.py",
        """
import asyncio


async def extension(d):
    loop = asyncio.get_running_loop()
    d.on_shutdown(lambda: (_ for _ in ()).throw(RuntimeError("ignored")))

    async def shutdown():
        assert asyncio.get_running_loop() is loop
    d.on_shutdown(shutdown)
""",
    )
    result = await load_extensions(
        settings=ExtensionSettings(paths=(directory,)),
    )

    await shutdown_extensions(result.registry)


async def test_server_binding_exposes_and_releases_registry(tmp_path: Path) -> None:
    """Thread context binding should retain hooks for lifespan teardown."""
    directory = tmp_path / "extensions"
    marker = tmp_path / "closed"
    _write(
        directory,
        "lifecycle.py",
        f"""
def extension(d):
    d.on_shutdown(lambda: open({str(marker)!r}, "w").close())
""",
    )
    result = await load_extensions(
        settings=ExtensionSettings(paths=(directory,)),
    )

    token = bind_server_extensions(result)
    try:
        assert get_server_extensions() is result
    finally:
        reset_server_extensions(token)
    await shutdown_server_extensions()

    assert marker.exists()
