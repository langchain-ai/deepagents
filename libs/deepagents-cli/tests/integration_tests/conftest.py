"""Pytest configuration for benchmark tests."""

import os
from collections.abc import Generator

import pytest
from langsmith import Client, get_tracing_context


@pytest.fixture(scope="session", autouse=True)
def langsmith_client() -> Generator[Client | None, None, None]:
    """Create a LangSmith client if LANGSMITH_API_KEY is set.

    This fixture is session-scoped and automatically used by all tests.
    It creates a single client instance and ensures it's flushed after each test.
    """
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")

    if langsmith_api_key:
        client = get_tracing_context()["client"] or Client()
        yield client

        # Final flush at end of session
        client.flush()
    else:
        yield None


@pytest.fixture(autouse=True)
def flush_langsmith_after_test(langsmith_client: Client) -> Generator[None, None, None]:
    """Automatically flush LangSmith client after each test."""
    yield

    # This runs after each test
    if langsmith_client is not None:
        langsmith_client.flush()
