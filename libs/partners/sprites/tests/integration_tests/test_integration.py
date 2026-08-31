from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from sprites import SpritesClient

from langchain_sprites import SpritesSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestSpritesSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        token = os.environ.get("SPRITES_TOKEN")
        if not token:
            msg = "Missing secret for Sprites integration test: set SPRITES_TOKEN"
            raise RuntimeError(msg)

        client = SpritesClient(token)
        name = f"ci-deepagents-py-{secrets.token_hex(3)}"
        sprite = client.create_sprite(name)
        backend = SpritesSandbox(sprite=sprite)
        try:
            yield backend
        finally:
            sprite.destroy()
