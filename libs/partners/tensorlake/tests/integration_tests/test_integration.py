from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from tensorlake.sandbox import SandboxClient

from langchain_tensorlake import TensorlakeSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestTensorlakeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("TENSORLAKE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing secrets for Tensorlake integration test: set TENSORLAKE_API_KEY")

        try:
            client = SandboxClient.for_cloud(api_key=api_key)
            sandbox_response = client.create()

            # Create the actual sandbox object from the response
            from tensorlake.sandbox import Sandbox
            sandbox = Sandbox(sandbox_id=sandbox_response.sandbox_id, api_key=api_key)

            backend = TensorlakeSandbox(sandbox=sandbox)
            yield backend
        except Exception as e:
            # Print detailed error for debugging
            import traceback
            print(f"Tensorlake integration test failed: {e}")
            print("Traceback:")
            traceback.print_exc()
            raise
        finally:
            # Only try to delete if we successfully created the sandbox
            if 'client' in locals() and 'sandbox_response' in locals():
                try:
                    client.delete(sandbox_response.sandbox_id)
                except Exception as delete_error:
                    print(f"Failed to delete sandbox: {delete_error}")
