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


# Ignore the pytest warning about class-scoped fixtures defined as instance
# methods. The `SandboxIntegrationTests` base class in `langchain-tests`
# defines its fixtures this way, so the warning can't be fixed here.
@pytest.mark.filterwarnings("ignore::pytest.PytestRemovedIn10Warning")
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

        sandbox = _create_modal_sandbox(token_id, token_secret)
        backend = ModalSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            sandbox.terminate()


def _create_modal_sandbox(token_id: str, token_secret: str) -> modal.Sandbox:
    app = modal.App.lookup("langchain-modal-tests", create_if_missing=True)
    return modal.Sandbox.create(
        image=modal.Image.from_registry("python:3.11-slim"),
        app=app,
        timeout=600,
        secrets=[
            modal.Secret.from_dict(
                {"MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}
            )
        ],
    )
