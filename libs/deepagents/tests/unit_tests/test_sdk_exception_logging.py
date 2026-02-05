import logging
from unittest.mock import MagicMock, patch

import pytest
from _pytest.logging import LogCaptureFixture

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.summarization import SummarizationMiddleware

# --- StoreBackend Tests ---


@pytest.fixture
def store_backend() -> StoreBackend:
    runtime = MagicMock()
    runtime.store = MagicMock()
    # Mock config to avoid "Could not get namespace" log noise, or let it be.
    # We'll patch get_config in tests if needed.
    return StoreBackend(runtime=runtime)


def test_store_backend_ls_info_logs_value_error(store_backend: StoreBackend, caplog: LogCaptureFixture) -> None:
    """Verify that ls_info logs debug message when item conversion fails."""
    # Create a mock item that will cause ValueError logic
    mock_item = MagicMock()
    mock_item.key = "/bad_file"

    store_backend.runtime.store.search.return_value = [mock_item]

    # Mock _convert_store_item_to_file_data to raise ValueError
    with patch.object(store_backend, "_convert_store_item_to_file_data", side_effect=ValueError("Invalid content")), caplog.at_level(logging.DEBUG):
        # We don't care about namespace failure log, just check for our log
        store_backend.ls_info("/")

    assert "Skipping invalid store item /bad_file" in caplog.text
    assert "Invalid content" in caplog.text


def test_store_backend_grep_raw_logs_value_error(store_backend: StoreBackend, caplog: LogCaptureFixture) -> None:
    """Verify that grep_raw logs debug message when item conversion fails."""
    mock_item = MagicMock()
    mock_item.key = "bad_file"
    store_backend.runtime.store.search.return_value = [mock_item]

    with patch.object(store_backend, "_convert_store_item_to_file_data", side_effect=ValueError("Invalid content")), caplog.at_level(logging.DEBUG):
        store_backend.grep_raw("some_pattern", "/")

    assert "Skipping invalid store item bad_file during grep" in caplog.text


def test_store_backend_glob_info_logs_value_error(store_backend: StoreBackend, caplog: LogCaptureFixture) -> None:
    """Verify that glob_info logs debug message when item conversion fails."""
    mock_item = MagicMock()
    mock_item.key = "bad_file"
    store_backend.runtime.store.search.return_value = [mock_item]

    with patch.object(store_backend, "_convert_store_item_to_file_data", side_effect=ValueError("Invalid content")), caplog.at_level(logging.DEBUG):
        store_backend.glob_info("*.py", "/")

    assert "Skipping invalid store item bad_file during glob" in caplog.text


# --- LocalShellBackend Tests ---


def test_local_shell_execute_logs_exception(caplog: LogCaptureFixture) -> None:
    """Verify that execute logs debug message with exception info on failure."""
    # LocalShellBackend does not take runtime argument
    backend = LocalShellBackend()

    # subprocess.run is mocked to raise a generic Exception
    with patch("subprocess.run", side_effect=Exception("Something went wrong")), caplog.at_level(logging.DEBUG):
        response = backend.execute("ls")

    assert response.exit_code == 1
    assert "Error executing command: Something went wrong" in response.output

    # Check logs
    assert "Local shell execution failed: Something went wrong" in caplog.text


# --- SummarizationMiddleware Tests ---


def test_summarization_get_thread_id_logs_runtime_error(caplog: LogCaptureFixture) -> None:
    """Verify that _get_thread_id logs debug message when get_config fails."""
    # SummarizationMiddleware requires model and backend
    middleware = SummarizationMiddleware(model=MagicMock(), backend=MagicMock())

    with patch("deepagents.middleware.summarization.get_config", side_effect=RuntimeError("No context")), caplog.at_level(logging.DEBUG):
        thread_id = middleware._get_thread_id()

    assert thread_id.startswith("session_")
    assert "Could not get thread_id from config (RuntimeError)" in caplog.text
