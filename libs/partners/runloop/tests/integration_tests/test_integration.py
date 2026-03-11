from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

from langchain_runloop import RunloopSandbox

if TYPE_CHECKING:
    from runloop_api_client.sdk import Devbox

from runloop_api_client import Runloop


class TestRunloopSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("RUNLOOP_API_KEY")
        if not api_key:
            msg = "Missing secrets for Runloop integration test: set RUNLOOP_API_KEY"
            raise RuntimeError(msg)

        client, devbox = _create_runloop_devbox(api_key=api_key)
        backend = RunloopSandbox(devbox=devbox)
        try:
            yield backend
        finally:
            client.devboxes.delete(devbox_id=devbox.id)


def _create_runloop_devbox(*, api_key: str) -> tuple[Runloop, Devbox]:

    client = Runloop(bearer_token=api_key)
    devbox = client.devboxes.create()
    return client, devbox
