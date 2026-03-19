from __future__ import annotations

import os
from typing import TYPE_CHECKING

import modal
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

from langchain_modal import ModalSandbox


class TestModalSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        if not token_id or not token_secret:
            msg = (
                "Missing secrets for Modal integration test: set MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET"
            )
            raise RuntimeError(msg)

        sandbox = _create_modal_sandbox()
        backend = ModalSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            sandbox.terminate()


def _create_modal_sandbox() -> modal.Sandbox:
    sandbox = modal.Sandbox.create(
        "python:3.11-slim",
        secrets=[modal.Secret.from_name("modal-token")],
    )
    sandbox.wait()
    return sandbox
