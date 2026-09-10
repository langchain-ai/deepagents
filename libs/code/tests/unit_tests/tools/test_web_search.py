"""The built-in and workspace-bound web search tools must stay interchangeable."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents_code.tools import (
    _search_with_tavily,
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


class TestTavilyQueryHandling:
    def test_rewritten_query_is_reported(self) -> None:
        class _Client:
            def search(self, *_args: Any, **_kwargs: Any) -> object:
                return {"query": "LangChain security", "results": []}

        result = _search_with_tavily(
            _Client(),
            query='site:docs.langchain.com "LangChain security"',
            max_results=5,
            topic="general",
            include_raw_content=False,
            include_domains=None,
            exclude_domains=None,
        )

        assert result["query_rewritten"] == {
            "requested": 'site:docs.langchain.com "LangChain security"',
            "executed": "LangChain security",
        }
        assert result["warning"]

    def test_domain_filters_are_forwarded(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.kwargs: dict[str, Any] = {}

            def search(self, *_args: Any, **kwargs: Any) -> object:
                self.kwargs = kwargs
                return {"query": "anything", "results": []}

        client = _Client()
        _search_with_tavily(
            client,
            query="anything",
            max_results=5,
            topic="general",
            include_raw_content=False,
            include_domains=["example.com"],
            exclude_domains=["blocked.example"],
        )

        assert client.kwargs["include_domains"] == ["example.com"]
        assert client.kwargs["exclude_domains"] == ["blocked.example"]

    def test_unchanged_query_has_no_rewrite_marker(self) -> None:
        class _Client:
            def search(self, *_args: Any, **_kwargs: Any) -> object:
                return {"query": "  anything   useful ", "results": []}

        result = _search_with_tavily(
            _Client(),
            query="anything useful",
            max_results=5,
            topic="general",
            include_raw_content=False,
            include_domains=None,
            exclude_domains=None,
        )

        assert "query_rewritten" not in result
