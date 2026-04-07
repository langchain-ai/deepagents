"""Unit tests for deepagents.graph module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents._profiles import _PROVIDER_PROFILES, ProviderProfile, register_provider_profile
from deepagents._version import __version__
from deepagents.graph import (
    _apply_tool_description_overrides,
    _profile_for_model,
    _resolve_extra_middleware,
    _tool_name,
    create_deep_agent,
)
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

    def test_falls_back_to_provider_for_bare_identifier(self) -> None:
        """Pre-built models with bare identifiers (no colon) resolve via provider."""
        original = dict(_PROVIDER_PROFILES)
        try:
            profile = ProviderProfile(init_kwargs={"from_provider": True})
            register_provider_profile("fakeprov", profile)
            model = _make_model({"model": "some-model-name"})
            # Simulate _get_ls_params returning the provider
            model._get_ls_params = MagicMock(return_value={"ls_provider": "fakeprov"})
            result = _profile_for_model(model, None)
            assert result is profile
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)

    def test_returns_empty_default_when_no_match(self) -> None:
        model = _make_model({"model_name": "unknown-model"})
        model._get_ls_params = MagicMock(return_value={})
        result = _profile_for_model(model, None)
        assert result == ProviderProfile()

    def test_returns_empty_default_when_no_identifier(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={})
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


class TestToolDescriptionOverrides:
    """Tests for copying and rewriting supported user-supplied tools.

    These test the helper directly rather than going through `create_deep_agent`
    (which requires full agent assembly).
    """

    def test_description_override_on_dict_copies_without_mutation(self) -> None:
        tool: dict[str, Any] = {"name": "my_tool", "description": "old"}
        result = _apply_tool_description_overrides([tool], {"my_tool": "new desc"})
        assert result is not None
        assert result[0]["description"] == "new desc"
        assert result[0] is not tool
        assert tool["description"] == "old"

    def test_description_override_on_basetool_copies_without_mutation(self) -> None:
        def sample_tool(text: str) -> str:
            return text

        tool = StructuredTool.from_function(
            func=sample_tool,
            name="my_tool",
            description="old",
        )
        result = _apply_tool_description_overrides([tool], {"my_tool": "new desc"})
        assert result is not None
        rewritten = result[0]
        assert isinstance(rewritten, BaseTool)
        assert rewritten.description == "new desc"
        assert rewritten is not tool
        assert tool.description == "old"

    def test_plain_callable_is_left_unchanged(self) -> None:
        def my_func() -> None:
            pass

        my_func.name = "my_tool"  # type: ignore[attr-defined]
        result = _apply_tool_description_overrides([my_func], {"my_tool": "new desc"})
        assert result == [my_func]


class TestDefaultModelProfile:
    """Tests for default model=None getting the correct Anthropic profile."""

    def test_default_model_gets_anthropic_profile(self) -> None:
        """model=None should resolve to the Anthropic profile."""
        # Verify the built-in anthropic profile is registered
        anthropic_profile = _PROVIDER_PROFILES.get("anthropic")
        assert anthropic_profile is not None
        assert callable(anthropic_profile.extra_middleware)


class TestToolDescriptionOverrideWiring:
    """Tests that supported built-in tool overrides are wired into middleware."""

    def test_create_deep_agent_passes_overrides_to_filesystem_and_task(self) -> None:
        original = dict(_PROVIDER_PROFILES)
        try:
            register_provider_profile(
                "testprov",
                ProviderProfile(
                    tool_description_overrides={
                        "ls": "custom ls",
                        "task": "custom task",
                    }
                ),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.FilesystemMiddleware", side_effect=[MagicMock(), MagicMock()]) as mock_fs,
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.TodoListMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.PatchToolCallsMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                result = create_deep_agent(model="testprov:some-model")

            assert result == "compiled-agent"
            assert mock_fs.call_count == 2
            for call in mock_fs.call_args_list:
                assert call.kwargs["custom_tool_descriptions"] == {
                    "ls": "custom ls",
                    "task": "custom task",
                }
            assert mock_subagents.call_args is not None
            assert mock_subagents.call_args.kwargs["task_description"] == "custom task"
        finally:
            _PROVIDER_PROFILES.clear()
            _PROVIDER_PROFILES.update(original)
