"""Network-free checks of shared workspace identity and lifetime."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from graph import parent, sandbox_name, worker
from langsmith.sandbox import ResourceNotFoundError


@pytest.mark.parametrize("existing", [True, False])
async def test_parent_and_worker_share_sandbox(existing: bool) -> None:
    """Separate checkpoint threads resolve the same sandbox and never delete it."""
    parent_id, worker_id = str(uuid4()), str(uuid4())
    with (
        patch("graph.SandboxClient") as clients,
        patch("graph.create_deep_agent", return_value=MagicMock()),
        patch.dict("os.environ", {"SANDBOX_SNAPSHOT": "test-snapshot"}),
    ):
        client = clients.return_value.__enter__.return_value
        sandbox = MagicMock()
        sandbox.name = sandbox_name(parent_id)
        client.create_sandbox.return_value = sandbox
        client.get_sandbox.side_effect = [
            sandbox
            if existing
            else ResourceNotFoundError("missing", resource_type="sandbox"),
            sandbox,
        ]
        async with parent({"configurable": {"thread_id": parent_id}}):
            async with worker(
                {
                    "configurable": {
                        "thread_id": worker_id,
                        "parent_thread_id": parent_id,
                    }
                }
            ):
                pass
        assert [call.args[0] for call in client.get_sandbox.call_args_list] == [
            sandbox.name,
            sandbox.name,
        ]
        if not existing:
            client.create_sandbox.assert_called_once_with(
                snapshot_name="test-snapshot", name=sandbox.name
            )
        else:
            client.create_sandbox.assert_not_called()
        client.delete_sandbox.assert_not_called()
