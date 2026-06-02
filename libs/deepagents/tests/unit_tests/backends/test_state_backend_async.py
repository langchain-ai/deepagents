"""Async unit tests for StateBackend.

StateBackend requires a LangGraph graph execution context (get_config()).
Functional tests (write/read/edit/ls/grep/glob) are covered by
TestStateBackendConfigKeys in test_end_to_end.py using create_deep_agent
with a fake model.  This file only contains async-specific error tests.
"""

import pytest

from deepagents.backends.state import StateBackend


async def test_state_backend_raises_outside_graph_context_async():
    """Async StateBackend operations outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        await be.aread("/anything.txt")
