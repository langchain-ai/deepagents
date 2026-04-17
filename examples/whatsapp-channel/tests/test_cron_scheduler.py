"""Tests for cron.scheduler — tick loop and run/deliver helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cron.scheduler import (
    SILENT_MARKER,
    _build_prompt,
    _extract_final_text,
)


class TestBuildPrompt:
    def test_prepends_cron_hint(self) -> None:
        out = _build_prompt("do the thing")
        assert "SYSTEM:" in out
        assert SILENT_MARKER in out
        assert out.endswith("do the thing")


class TestExtractFinalText:
    def test_extracts_string_content(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        out = {"messages": [
            HumanMessage(content="q"),
            AIMessage(content="the answer"),
        ]}
        assert _extract_final_text(out) == "the answer"

    def test_extracts_list_content_text_block(self) -> None:
        from langchain_core.messages import AIMessage

        out = {"messages": [AIMessage(content=[
            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
            {"type": "text", "text": "the answer"},
        ])]}
        assert _extract_final_text(out) == "the answer"

    def test_empty_when_no_ai_message(self) -> None:
        from langchain_core.messages import HumanMessage

        assert _extract_final_text({"messages": [HumanMessage(content="q")]}) == ""

    def test_empty_on_missing_messages(self) -> None:
        assert _extract_final_text({}) == ""
        assert _extract_final_text(None) == ""
