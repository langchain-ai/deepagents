"""Tests for sandbox provider metadata value types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest

from deepagents_code.integrations.sandbox_provider import (
    SandboxInstallHint,
    SandboxProvider,
)

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol


@pytest.mark.parametrize(
    ("kind", "in_app", "expected"),
    [
        ("extra", True, "/install daytona"),
        ("extra", False, "dcode --install daytona"),
        ("package", True, "/install daytona --package"),
        ("package", False, "dcode --install daytona --package"),
    ],
)
def test_install_hint_command(
    kind: Literal["extra", "package"], in_app: bool, expected: str
) -> None:
    """`command()` renders the right prefix and `--package` suffix per kind."""
    hint = SandboxInstallHint(kind=kind, name="daytona")
    assert hint.command(in_app=in_app) == expected


def test_provider_metadata_defaults_to_none() -> None:
    """The ABC's `metadata` property returns `None` unless a provider overrides."""

    class BareProvider(SandboxProvider):
        def get_or_create(
            self,
            *,
            sandbox_id: str | None = None,
            **kwargs: Any,
        ) -> SandboxBackendProtocol:
            raise NotImplementedError

        def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
            raise NotImplementedError

    assert BareProvider().metadata is None
