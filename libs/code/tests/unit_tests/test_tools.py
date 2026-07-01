"""Unit tests for dcode built-in tool selection."""

from __future__ import annotations

from deepagents_code import tools as tools_module
from deepagents_code.tools import build_builtin_tools


def test_build_builtin_tools_uses_provider_web_search_for_openai() -> None:
    """OpenAI-backed models use the provider-hosted web search declaration."""
    tools = build_builtin_tools(provider="openai", has_tavily=False)

    assert tools == [
        tools_module.fetch_url,
        tools_module.get_current_thread_id,
        {"type": "web_search"},
    ]


def test_build_builtin_tools_prefers_provider_web_search_over_tavily() -> None:
    """OpenAI-backed models should not expose both hosted search and Tavily."""
    tools = build_builtin_tools(provider="openai_codex", has_tavily=True)

    assert tools == [
        tools_module.fetch_url,
        tools_module.get_current_thread_id,
        {"type": "web_search"},
    ]


def test_build_builtin_tools_uses_tavily_for_other_providers() -> None:
    """Non-OpenAI providers keep the existing Tavily tool behavior."""
    tools = build_builtin_tools(provider="anthropic", has_tavily=True)

    assert tools == [
        tools_module.fetch_url,
        tools_module.get_current_thread_id,
        tools_module.web_search,
    ]
