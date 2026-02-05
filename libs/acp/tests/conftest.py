"""Pytest configuration and shared fixtures for langchain-acp tests."""

import pytest
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def memory_checkpointer():
    """Fixture providing a MemorySaver checkpointer."""
    return MemorySaver()
