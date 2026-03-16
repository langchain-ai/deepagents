"""Tests for EMmsMemoryMiddleware.

These tests run without a live EMMS server by mocking the emms module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_emms import EMmsMemoryMiddleware, EMmsMemoryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_emms(
    *,
    long_term: dict[str, Any] | None = None,
    semantic: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    rag_context: str = "## Memory\n- test memory",
) -> MagicMock:
    emms = MagicMock()
    emms.memory.working = []
    emms.memory.short_term = []
    emms.memory.long_term = long_term or {}
    emms.memory.semantic = semantic or {}
    emms.stats = stats or {"identity": {"narrative_coherence": 0.8, "ego_boundary_strength": 0.9}}
    emms.build_rag_context.return_value = rag_context
    return emms


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestEMmsMemoryMiddlewareInit:
    def test_default_params(self) -> None:
        mw = EMmsMemoryMiddleware()
        assert mw.state_path == "~/.emms/emms_state.json"
        assert mw.token_budget == 4000
        assert mw.auto_store is True
        assert mw.context_query is None

    def test_custom_params(self) -> None:
        mw = EMmsMemoryMiddleware(
            state_path="/tmp/test.json",
            token_budget=2000,
            auto_store=False,
            context_query="test query",
        )
        assert mw.state_path == "/tmp/test.json"
        assert mw.token_budget == 2000
        assert mw.auto_store is False
        assert mw.context_query == "test query"


class TestExtractSelfNarrative:
    def test_empty_memory_returns_freshness_message(self) -> None:
        mw = EMmsMemoryMiddleware()
        emms = _make_mock_emms()
        narrative = mw._extract_self_narrative(emms)
        assert "newly initialized" in narrative

    def test_goals_appear_in_narrative(self) -> None:
        mw = EMmsMemoryMiddleware()

        goal_item = MagicMock()
        goal_item.experience.domain = "identity"
        goal_item.experience.content = "committed to: ship the v2 release; committed to: grow user base"

        emms = _make_mock_emms(semantic={"g1": goal_item})
        narrative = mw._extract_self_narrative(emms)
        assert "ship the v2 release" in narrative


class TestFormatPrompt:
    def test_renders_without_error(self) -> None:
        mw = EMmsMemoryMiddleware()
        prompt = mw._format_prompt(
            context="- memory 1\n- memory 2",
            stats={
                "identity": {"narrative_coherence": 0.75, "ego_boundary_strength": 0.85},
                "_total_across_tiers": 42,
                "_self_narrative": "I know who I am.",
            },
        )
        assert "<identity>" in prompt
        assert "<my_memories>" in prompt
        assert "42" in prompt

    def test_empty_context_shows_fallback(self) -> None:
        mw = EMmsMemoryMiddleware()
        prompt = mw._format_prompt(context="", stats={})
        assert "No memories retrieved" in prompt


class TestBeforeAgent:
    @patch("langchain_emms.middleware._get_emms")
    def test_returns_state_update_on_first_call(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_mock_emms()
        mw = EMmsMemoryMiddleware()
        state: EMmsMemoryState = {"messages": []}  # type: ignore[typeddict-item]
        result = mw.before_agent(state, MagicMock(), MagicMock())
        assert result is not None
        assert "emms_context" in result
        assert "emms_stats" in result

    @patch("langchain_emms.middleware._get_emms")
    def test_skips_on_subsequent_calls(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_mock_emms()
        mw = EMmsMemoryMiddleware()
        state: EMmsMemoryState = {"emms_context": "already loaded", "messages": []}  # type: ignore[typeddict-item]
        result = mw.before_agent(state, MagicMock(), MagicMock())
        assert result is None

    @patch("langchain_emms.middleware._get_emms")
    def test_uses_context_query_when_set(self, mock_get: MagicMock) -> None:
        mock_emms = _make_mock_emms()
        mock_get.return_value = mock_emms
        mw = EMmsMemoryMiddleware(context_query="fixed query")
        state: EMmsMemoryState = {"messages": []}  # type: ignore[typeddict-item]
        mw.before_agent(state, MagicMock(), MagicMock())
        mock_emms.build_rag_context.assert_called_once_with(
            query="fixed query",
            token_budget=4000,
            fmt="markdown",
            include_metadata=True,
        )


@pytest.mark.compile
def test_import() -> None:
    """Smoke test — package imports cleanly."""
    from langchain_emms import EMmsMemoryMiddleware  # noqa: F401
