"""The built-in and workspace-bound web search tools must stay interchangeable."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents_code.tools import (
    create_web_search_tool,
    is_web_search_tool,
    web_search,
)


class TestSchemaParity:
    """`is_web_search_tool` treats both variants as one, so they must match."""

    def test_workspace_variant_presents_the_same_arguments(self) -> None:
        """A `**kwargs` closure must not erode the advertised schema."""
        builtin = convert_to_openai_tool(tool("web_search")(web_search))
        workspace = convert_to_openai_tool(create_web_search_tool("key"))

        assert workspace == builtin


class TestToolIdentity:
    """Both variants are recognized; unrelated tools are not."""

    def test_recognizes_both_variants(self) -> None:
        assert is_web_search_tool(web_search)
        assert is_web_search_tool(create_web_search_tool("key"))

    def test_rejects_an_unrelated_tool(self) -> None:
        from deepagents_code.tools import fetch_url

        assert not is_web_search_tool(fetch_url)
        assert not is_web_search_tool(object())


class TestWorkspaceErrorTranslation:
    """Failures return a payload the model can act on, never an exception."""

    def test_missing_package_is_translated(self) -> None:
        """An uninstalled `tavily` must not raise out of the tool."""
        search = create_web_search_tool("key")

        with patch.dict(sys.modules, {"tavily": None}):
            result = search.invoke({"query": "anything"})

        assert result["error"].startswith("Required package not installed")

    def test_empty_key_reports_configuration_not_a_tavily_error(self) -> None:
        """`_build_tools` can pass `""`; that must read as unconfigured."""
        search = create_web_search_tool("")

        result = search.invoke({"query": "anything"})

        assert "Tavily API key not configured" in result["error"]
        assert result["query"] == "anything"

    def test_search_errors_are_translated_like_the_builtin(self) -> None:
        """A request failure returns the shared translated payload."""
        import requests

        search = create_web_search_tool("key")

        class _Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def search(self, *_args: Any, **_kwargs: Any) -> object:
                raise requests.exceptions.ConnectionError

        with patch("tavily.TavilyClient", _Client):
            result = search.invoke({"query": "anything"})

        assert "error" in result
