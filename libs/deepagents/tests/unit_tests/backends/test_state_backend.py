"""Unit tests for StateBackend.

StateBackend requires a LangGraph graph execution context (get_config()).
Functional tests (write/read/edit/ls/grep/glob) are covered by
TestStateBackendConfigKeys in test_end_to_end.py using create_deep_agent
with a fake model.  This file only contains tests that don't need graph
context: deprecation warnings and error messages.
"""

import warnings

import pytest

from deepagents._api.deprecation import LangChainDeprecationWarning
from deepagents.backends.state import StateBackend


def test_state_backend_runtime_deprecation_warning():
    """Passing runtime= to StateBackend should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateBackend(runtime="ignored_value")
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert deprecation_warnings[0].category is LangChainDeprecationWarning
        assert "0.7.0" in str(deprecation_warnings[0].message)
        assert "runtime" in str(deprecation_warnings[0].message)
        # The warning should attribute to this test file, not deepagents internals.
        assert deprecation_warnings[0].filename == __file__


def test_state_backend_no_deprecation_without_runtime():
    """StateBackend() without runtime should NOT emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateBackend()
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


def test_state_backend_raises_outside_graph_context():
    """StateBackend operations outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        be.read("/anything.txt")


def test_upload_files_raises_outside_graph_context():
    """upload_files outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        be.upload_files([("/hello.txt", b"hello")])
