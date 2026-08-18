"""Tests for transactional extension import and initialization."""

from pathlib import Path

import pytest

from deepagents_code.extensions import (
    ExtensionError,
    ExtensionFile,
    ExtensionRegistry,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)
from deepagents_code.extensions.loader import load_extension

_VALID_EXTENSION = """
from deepagents.backends import FilesystemBackend


def extension(d):
    def status():
        \"\"\"Report status.\"\"\"
        return \"ready\"

    d.register_tool(status)
    d.register_backend_route(
        \"/memories/\", FilesystemBackend(virtual_mode=False)
    )
    d.on_shutdown(lambda: None)
"""

_BROKEN_EXTENSION = """
def extension(d):
    def partial():
        \"\"\"Never become visible.\"\"\"
        return \"partial\"

    d.register_tool(partial)
    d.on_shutdown(lambda: None)
    raise RuntimeError(\"boom\")
"""


def _file(path: Path, *, origin: UnitOrigin = UnitOrigin.TOP_LEVEL) -> ExtensionFile:
    """Build a discovered extension record for a test file."""
    return ExtensionFile(
        path=path,
        source=SourceInfo(path=path, scope=UnitScope.USER, origin=origin),
    )


async def test_loads_registered_units(tmp_path: Path) -> None:
    """A valid factory should return the units it registered."""
    path = tmp_path / "valid.py"
    path.write_text(_VALID_EXTENSION, encoding="utf-8")
    registry = ExtensionRegistry()

    loaded = await load_extension(
        _file(path), registry, cwd=tmp_path, mode="interactive"
    )

    assert [tool.name for tool in loaded.tools] == ["status"]
    assert loaded.backend_routes == ("/memories/",)
    assert len(registry.shutdown_hooks) == 1


async def test_package_extension_supports_relative_imports(tmp_path: Path) -> None:
    """Package entry points should resolve sibling helper modules."""
    package = tmp_path / "package"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 'package-ready'\n", encoding="utf-8")
    entry = package / "__init__.py"
    entry.write_text(
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

    loaded = await load_extension(
        _file(entry, origin=UnitOrigin.PACKAGE),
        ExtensionRegistry(),
        cwd=tmp_path,
        mode="headless",
    )

    assert loaded.tools[0].invoke({}) == "package-ready"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("VALUE = 1\n", "does not define a callable"),
        ("raise RuntimeError('import boom')\n", "Failed to import"),
    ],
)
async def test_import_failures_are_extension_errors(
    tmp_path: Path, body: str, message: str
) -> None:
    """Invalid modules should fail without leaving registrations."""
    path = tmp_path / "invalid.py"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ExtensionError, match=message):
        await load_extension(
            _file(path), ExtensionRegistry(), cwd=tmp_path, mode="interactive"
        )


async def test_failed_factory_rolls_back_partial_state(tmp_path: Path) -> None:
    """A factory exception should remove its tools and teardown hooks."""
    path = tmp_path / "broken.py"
    path.write_text(_BROKEN_EXTENSION, encoding="utf-8")
    registry = ExtensionRegistry()

    with pytest.raises(ExtensionError, match="boom"):
        await load_extension(_file(path), registry, cwd=tmp_path, mode="interactive")

    assert registry.is_empty()
    assert not registry.shutdown_hooks
