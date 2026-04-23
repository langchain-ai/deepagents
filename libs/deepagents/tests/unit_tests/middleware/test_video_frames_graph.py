"""Tests that `create_deep_agent` installs / skips VideoFrameExtractionMiddleware.

We can't reliably introspect the compiled LangGraph for its middleware stack, so
instead we mock `deepagents.graph.create_agent` to capture the middleware list
that `create_deep_agent` builds and passes downstream.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.video_frames import VideoFrameExtractionMiddleware


def _capture_middleware(**kwargs: Any) -> list[Any]:
    """Build an agent under a mocked create_agent and return the middleware list."""
    captured: dict[str, list[Any]] = {}

    def fake_create_agent(*_a: Any, **kw: Any) -> MagicMock:
        captured["middleware"] = list(kw.get("middleware", []))
        return MagicMock()

    with patch("deepagents.graph.create_agent", side_effect=fake_create_agent):
        create_deep_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), **kwargs)  # type: ignore[call-arg]
    return captured["middleware"]


def _capture_all(**kwargs: Any) -> dict[str, Any]:
    """Capture the full kwarg dict passed to create_agent."""
    captured: dict[str, Any] = {}

    def fake_create_agent(*_a: Any, **kw: Any) -> MagicMock:
        captured.update(kw)
        return MagicMock()

    with patch("deepagents.graph.create_agent", side_effect=fake_create_agent):
        create_deep_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), **kwargs)  # type: ignore[call-arg]
    return captured


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
        middleware = _capture_middleware()
        video_idx = next(i for i, m in enumerate(middleware) if isinstance(m, VideoFrameExtractionMiddleware))
        cache_idx = next(i for i, m in enumerate(middleware) if isinstance(m, AnthropicPromptCachingMiddleware))
        assert video_idx < cache_idx


class TestSubagentVideoMiddleware:
    def _get_subagent_middleware_stacks(self, **kwargs: Any) -> list[list[Any]]:
        """Return the middleware list for every SubAgent spec inside SubAgentMiddleware."""
        captured = _capture_all(**kwargs)
        main_middleware: list[Any] = captured.get("middleware", [])
        sub_mw_obj = next(
            (m for m in main_middleware if isinstance(m, SubAgentMiddleware)), None
        )
        assert sub_mw_obj is not None, "SubAgentMiddleware not found in main middleware"
        stacks = []
        for spec in sub_mw_obj._subagents:
            if "middleware" in spec:
                stacks.append(list(spec["middleware"]))
        return stacks

    def test_gp_subagent_has_video_middleware(self) -> None:
        """General-purpose subagent must include VideoFrameExtractionMiddleware."""
        stacks = self._get_subagent_middleware_stacks()
        # There is always at least one stack (the gp subagent).
        assert len(stacks) >= 1
        gp_stack = stacks[0]
        assert any(isinstance(m, VideoFrameExtractionMiddleware) for m in gp_stack), (
            "VideoFrameExtractionMiddleware not found in general-purpose subagent middleware"
        )

    def test_gp_subagent_video_middleware_before_prompt_caching(self) -> None:
        """Video middleware must come before AnthropicPromptCachingMiddleware in gp subagent."""
        stacks = self._get_subagent_middleware_stacks()
        gp_stack = stacks[0]
        video_idx = next(i for i, m in enumerate(gp_stack) if isinstance(m, VideoFrameExtractionMiddleware))
        cache_idx = next(i for i, m in enumerate(gp_stack) if isinstance(m, AnthropicPromptCachingMiddleware))
        assert video_idx < cache_idx

    def test_inline_subagent_has_video_middleware(self) -> None:
        """Inline SubAgent specs must also include VideoFrameExtractionMiddleware."""
        stacks = self._get_subagent_middleware_stacks(
            subagents=[{"name": "custom", "description": "x", "system_prompt": "y"}]
        )
        # stacks[0] is gp, stacks[1] is the custom inline subagent
        assert len(stacks) >= 2
        inline_stack = stacks[1]
        assert any(isinstance(m, VideoFrameExtractionMiddleware) for m in inline_stack), (
            "VideoFrameExtractionMiddleware not found in inline subagent middleware"
        )

    def test_inline_subagent_video_disabled_main_still_installs_subagent(self) -> None:
        """When video_frame_extraction=False on main agent, subagents still get it."""
        stacks = self._get_subagent_middleware_stacks(
            video_frame_extraction=False,
            subagents=[{"name": "custom", "description": "x", "system_prompt": "y"}],
        )
        # All subagent stacks should still have it.
        for stack in stacks:
            assert any(isinstance(m, VideoFrameExtractionMiddleware) for m in stack), (
                "VideoFrameExtractionMiddleware missing from subagent stack even though it is unconditional"
            )
