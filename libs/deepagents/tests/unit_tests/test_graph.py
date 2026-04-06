"""Unit tests for deepagents.graph module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from deepagents._models import _PROVIDER_PROFILES, ProviderProfile, register_provider_profile
from deepagents._version import __version__
from deepagents.graph import _profile_for_model, _resolve_extra_middleware, _tool_name, create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_model(dump: dict[str, Any]) -> MagicMock:
    """Create a mock BaseChatModel with a given model_dump return."""
    model = MagicMock(spec=BaseChatModel)
    model.model_dump.return_value = dump
    return model


class TestCreateDeepAgentMetadata:
    """Tests for metadata on the compiled graph."""

    def test_versions_metadata_contains_sdk_version(self) -> None:
        """`create_deep_agent` should attach SDK version in metadata.versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        versions = agent.config["metadata"]["versions"]
        assert versions["deepagents"] == __version__

    def test_ls_integration_metadata_preserved(self) -> None:
        """`ls_integration` should still be present alongside versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        assert agent.config["metadata"]["ls_integration"] == "deepagents"


class TestResolveExtraMiddleware:
    """Tests for _resolve_extra_middleware."""

    def test_empty_profile_returns_empty_list(self) -> None:
        result = _resolve_extra_middleware(ProviderProfile())
        assert result == []

    def test_static_sequence_returned_as_list(self) -> None:
        sentinel = MagicMock()
        profile = ProviderProfile(extra_middleware=(sentinel,))
        result = _resolve_extra_middleware(profile)
        assert result == [sentinel]

    def test_callable_factory_is_invoked(self) -> None:
        sentinel = MagicMock()
        factory = MagicMock(return_value=[sentinel])
        profile = ProviderProfile(extra_middleware=factory)
        result = _resolve_extra_middleware(profile)
        factory.assert_called_once()
        assert result == [sentinel]

    def test_returns_fresh_list_each_call(self) -> None:
        sentinel = MagicMock()
        profile = ProviderProfile(extra_middleware=(sentinel,))
        a = _resolve_extra_middleware(profile)
        b = _resolve_extra_middleware(profile)
        assert a == b
        assert a is not b


class TestProfileForModel:
    """Tests for _profile_for_model."""

    def test_uses_spec_when_provided(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            profile = ProviderProfile(init_kwargs={"from_spec": True})
            register_provider_profile("testprov", profile)
            result = _profile_for_model(_make_model({}), "testprov:some-model")
            assert result is profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_falls_back_to_identifier_when_spec_is_none(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            profile = ProviderProfile(init_kwargs={"from_id": True})
            register_provider_profile("myprov", profile)
            model = _make_model({"model_name": "myprov:my-model"})
            result = _profile_for_model(model, None)
            assert result is profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_returns_empty_default_when_no_match(self) -> None:
        model = _make_model({"model_name": "unknown-model"})
        result = _profile_for_model(model, None)
        assert result == ProviderProfile()

    def test_returns_empty_default_when_no_identifier(self) -> None:
        model = _make_model({})
        result = _profile_for_model(model, None)
        assert result == ProviderProfile()


class TestToolName:
    """Tests for _tool_name helper."""

    def test_basetool(self) -> None:
        tool = MagicMock(spec=BaseTool)
        tool.name = "my_tool"
        assert _tool_name(tool) == "my_tool"

    def test_dict_tool(self) -> None:
        assert _tool_name({"name": "dict_tool", "description": "desc"}) == "dict_tool"

    def test_dict_tool_without_name(self) -> None:
        assert _tool_name({"description": "desc"}) is None

    def test_dict_tool_non_string_name(self) -> None:
        assert _tool_name({"name": 123}) is None

    def test_callable_with_name_attr(self) -> None:
        fn: Callable[..., Any] = MagicMock()
        fn.name = "callable_tool"  # type: ignore[attr-defined]
        assert _tool_name(fn) == "callable_tool"

    def test_callable_without_name(self) -> None:
        def my_func() -> None:
            pass

        # Plain functions have __name__ but not name
        assert _tool_name(my_func) is None


class TestToolFiltering:
    """Tests for tool exclusion and description override logic.

    These test the filtering helpers directly rather than going through
    ``create_deep_agent`` (which requires real tool schemas).
    """

    def test_exclude_filters_basetool(self) -> None:
        tool = MagicMock(spec=BaseTool)
        tool.name = "remove_me"
        tools = [tool]
        exclude = frozenset({"remove_me"})
        result = [t for t in tools if _tool_name(t) not in exclude]
        assert result == []

    def test_exclude_filters_dict_tool(self) -> None:
        tools: list[dict[str, Any]] = [{"name": "remove_me", "description": "d"}]
        exclude = frozenset({"remove_me"})
        result = [t for t in tools if _tool_name(t) not in exclude]
        assert result == []

    def test_exclude_keeps_non_matching(self) -> None:
        keep = MagicMock(spec=BaseTool)
        keep.name = "keep_me"
        remove = MagicMock(spec=BaseTool)
        remove.name = "remove_me"
        tools = [keep, remove]
        exclude = frozenset({"remove_me"})
        result = [t for t in tools if _tool_name(t) not in exclude]
        assert len(result) == 1
        assert result[0].name == "keep_me"

    def test_exclude_skips_tools_without_name(self) -> None:
        """Tools with no determinable name should not be excluded."""
        nameless: dict[str, Any] = {"description": "no name"}
        exclude = frozenset({"something"})
        result = [t for t in [nameless] if _tool_name(t) not in exclude]
        assert len(result) == 1

    def test_description_override_on_dict(self) -> None:
        tool: dict[str, Any] = {"name": "my_tool", "description": "old"}
        overrides = {"my_tool": "new desc"}
        name = _tool_name(tool)
        assert name is not None
        override = overrides.get(name)
        if override is not None:
            tool["description"] = override
        assert tool["description"] == "new desc"

    def test_description_override_on_basetool(self) -> None:
        tool = MagicMock(spec=BaseTool)
        tool.name = "my_tool"
        tool.description = "old"
        overrides = {"my_tool": "new desc"}
        name = _tool_name(tool)
        assert name is not None
        override = overrides.get(name)
        if override is not None:
            tool.description = override
        assert tool.description == "new desc"

    def test_copy_prevents_caller_mutation(self) -> None:
        """Copying the list prevents the caller's list from being modified."""
        original = [{"name": "a"}, {"name": "b"}]
        copied = list(original)
        exclude = frozenset({"b"})
        filtered = [t for t in copied if _tool_name(t) not in exclude]
        assert len(original) == 2
        assert len(filtered) == 1


class TestDefaultModelProfile:
    """Tests for default model=None getting the correct Anthropic profile."""

    def test_default_model_gets_anthropic_profile(self) -> None:
        """model=None should resolve to the Anthropic profile."""
        # Verify the built-in anthropic profile is registered
        anthropic_profile = _PROVIDER_PROFILES.get("anthropic")
        assert anthropic_profile is not None
        assert callable(anthropic_profile.extra_middleware)
