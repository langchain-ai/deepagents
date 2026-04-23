"""Tests that `create_deep_agent` installs / skips VideoFrameExtractionMiddleware.

We can't reliably introspect the compiled LangGraph for its middleware stack, so
instead we mock `deepagents.graph.create_agent` to capture the middleware list
that `create_deep_agent` builds and passes downstream.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic

from deepagents.graph import create_deep_agent
from deepagents.middleware.video_frames import VideoFrameExtractionMiddleware


def _capture_middleware(**kwargs: Any) -> list[Any]:
    """Build an agent under a mocked create_agent and return the middleware list."""
    captured: dict[str, list[Any]] = {}

    def fake_create_agent(*_a: Any, **kw: Any) -> MagicMock:
        captured["middleware"] = list(kw.get("middleware", []))
        return MagicMock()

    with patch("deepagents.graph.create_agent", side_effect=fake_create_agent):
        create_deep_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), **kwargs)
    return captured["middleware"]


class TestCreateDeepAgentVideoMiddleware:
    def test_installed_by_default(self) -> None:
        middleware = _capture_middleware()
        types = [type(m) for m in middleware]
        assert VideoFrameExtractionMiddleware in types

    def test_disabled_when_false(self) -> None:
        middleware = _capture_middleware(video_frame_extraction=False)
        types = [type(m) for m in middleware]
        assert VideoFrameExtractionMiddleware not in types

    def test_uses_provided_instance(self) -> None:
        custom = VideoFrameExtractionMiddleware(max_frames=42)
        middleware = _capture_middleware(video_frame_extraction=custom)
        installed = [m for m in middleware if isinstance(m, VideoFrameExtractionMiddleware)]
        assert len(installed) == 1
        assert installed[0].max_frames == 42
        assert installed[0] is custom

    def test_order_before_anthropic_prompt_caching(self) -> None:
        """Video expansion must finalise message shape before prompt caching."""
        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

        middleware = _capture_middleware()
        video_idx = next(
            i for i, m in enumerate(middleware)
            if isinstance(m, VideoFrameExtractionMiddleware)
        )
        cache_idx = next(
            i for i, m in enumerate(middleware)
            if isinstance(m, AnthropicPromptCachingMiddleware)
        )
        assert video_idx < cache_idx
