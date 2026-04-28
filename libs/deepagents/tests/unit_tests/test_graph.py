"""Unit tests for deepagents.graph module."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents._api.deprecation import LangChainDeprecationWarning
from deepagents._version import __version__
from deepagents.graph import (
    BASE_AGENT_PROMPT,
    _apply_tool_description_overrides,
    _harness_profile_for_model,
    _resolve_extra_middleware,
    _tool_name,
    create_deep_agent,
)
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.profiles import _HARNESS_PROFILES, _get_harness_profile, _HarnessProfile, _register_harness_profile
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ModelRequest


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
        result = _resolve_extra_middleware(_HarnessProfile())
        assert result == []

    def test_static_sequence_returned_as_list(self) -> None:
        sentinel = MagicMock()
        profile = _HarnessProfile(extra_middleware=(sentinel,))
        result = _resolve_extra_middleware(profile)
        assert result == [sentinel]

    def test_callable_factory_is_invoked(self) -> None:
        sentinel = MagicMock()
        factory = MagicMock(return_value=[sentinel])
        profile = _HarnessProfile(extra_middleware=factory)
        result = _resolve_extra_middleware(profile)
        factory.assert_called_once()
        assert result == [sentinel]

    def test_returns_fresh_list_each_call(self) -> None:
        sentinel = MagicMock()
        profile = _HarnessProfile(extra_middleware=(sentinel,))
        a = _resolve_extra_middleware(profile)
        b = _resolve_extra_middleware(profile)
        assert a == b
        assert a is not b


class TestProfileForModel:
    """Tests for _harness_profile_for_model."""

    def test_uses_spec_when_provided(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            profile = _HarnessProfile(init_kwargs={"from_spec": True})
            _register_harness_profile("testprov", profile)
            result = _harness_profile_for_model(_make_model({}), "testprov:some-model")
            assert result is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_falls_back_to_identifier_when_spec_is_none(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            profile = _HarnessProfile(init_kwargs={"from_id": True})
            _register_harness_profile("myprov", profile)
            model = _make_model({"model_name": "myprov:my-model"})
            result = _harness_profile_for_model(model, None)
            assert result is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_falls_back_to_provider_for_bare_identifier(self) -> None:
        """Pre-built models with bare identifiers (no colon) resolve via provider."""
        original = dict(_HARNESS_PROFILES)
        try:
            profile = _HarnessProfile(init_kwargs={"from_provider": True})
            _register_harness_profile("fakeprov", profile)
            model = _make_model({"model": "some-model-name"})
            # Simulate _get_ls_params returning the provider
            model._get_ls_params = MagicMock(return_value={"ls_provider": "fakeprov"})
            result = _harness_profile_for_model(model, None)
            assert result is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_returns_empty_default_when_no_match(self) -> None:
        model = _make_model({"model_name": "unknown-model"})
        model._get_ls_params = MagicMock(return_value={})
        result = _harness_profile_for_model(model, None)
        assert result == _HarnessProfile()

    def test_returns_empty_default_when_no_identifier(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={})
        result = _harness_profile_for_model(model, None)
        assert result == _HarnessProfile()


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
    """Tests for default model=None getting the default profile."""

    def test_default_model_gets_default_profile(self) -> None:
        """model=None resolves to default profile (no Anthropic-specific registration)."""
        profile = _get_harness_profile("anthropic:claude-sonnet-4-6")
        assert profile == _HarnessProfile()


class TestToolDescriptionOverrideWiring:
    """Tests that supported built-in tool overrides are wired into middleware."""

    def test_create_deep_agent_passes_overrides_to_filesystem_and_task(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "testprov",
                _HarnessProfile(
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
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestSystemPromptAssembly:
    """Tests for system prompt assembly: profile base_system_prompt, suffix, and user prompt interaction."""

    def _build_and_capture_system_prompt(self, profile_key: str, profile: _HarnessProfile, **kwargs: Any) -> str | SystemMessage:
        """Register a profile, call create_deep_agent, return the system_prompt passed to create_agent."""
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(profile_key, profile)
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.FilesystemMiddleware", side_effect=[MagicMock(), MagicMock()]),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.TodoListMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.PatchToolCallsMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model=f"{profile_key}:some-model", **kwargs)

            return mock_create.call_args.kwargs["system_prompt"]
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_default_uses_base_agent_prompt(self) -> None:
        prompt = self._build_and_capture_system_prompt("defprov", _HarnessProfile())
        assert prompt == BASE_AGENT_PROMPT

    def test_profile_base_system_prompt_replaces_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(base_system_prompt="You are a custom agent."),
        )
        assert prompt == "You are a custom agent."
        assert BASE_AGENT_PROMPT not in prompt

    def test_profile_base_system_prompt_with_suffix(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(
                base_system_prompt="You are a custom agent.",
                system_prompt_suffix="Be concise.",
            ),
        )
        assert prompt == "You are a custom agent.\n\nBe concise."
        assert BASE_AGENT_PROMPT not in prompt

    def test_suffix_without_base_system_prompt_appends_to_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "suffprov",
            _HarnessProfile(system_prompt_suffix="Think step by step."),
        )
        assert prompt == BASE_AGENT_PROMPT + "\n\nThink step by step."

    def test_user_system_prompt_prepended_before_profile_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(base_system_prompt="Custom base."),
            system_prompt="User instructions.",
        )
        assert prompt == "User instructions.\n\nCustom base."
        assert BASE_AGENT_PROMPT not in prompt

    def test_user_system_prompt_prepended_before_default_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "defprov",
            _HarnessProfile(),
            system_prompt="User instructions.",
        )
        assert prompt == f"User instructions.\n\n{BASE_AGENT_PROMPT}"

    def test_triple_combo_all_three_inputs(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(
                base_system_prompt="Custom base.",
                system_prompt_suffix="Extra.",
            ),
            system_prompt="User instructions.",
        )
        assert prompt == "User instructions.\n\nCustom base.\n\nExtra."
        assert BASE_AGENT_PROMPT not in prompt

    def test_system_message_with_profile_base(self) -> None:
        msg = SystemMessage(content="User content.")
        result = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(base_system_prompt="Custom base."),
            system_prompt=msg,
        )
        assert isinstance(result, SystemMessage)
        # Last content block should contain the custom base, not BASE_AGENT_PROMPT
        last_block = result.content_blocks[-1]
        assert "Custom base." in last_block["text"]
        assert BASE_AGENT_PROMPT not in last_block["text"]

    def test_empty_string_base_system_prompt_replaces_with_empty(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(base_system_prompt=""),
        )
        assert prompt == ""
        assert BASE_AGENT_PROMPT not in prompt

    def test_empty_string_suffix_still_appended(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            _HarnessProfile(
                base_system_prompt="Custom base.",
                system_prompt_suffix="",
            ),
        )
        assert prompt == "Custom base.\n\n"


class TestToolExclusionMiddleware:
    """Tests for _ToolExclusionMiddleware."""

    def test_filters_tools_from_request(self) -> None:
        tool_a = MagicMock()
        tool_a.name = "keep"
        tool_b = MagicMock()
        tool_b.name = "drop"
        request = MagicMock()
        request.tools = [tool_a, tool_b]

        # override should be called with filtered tools
        overridden_request = MagicMock()
        request.override.return_value = overridden_request

        handler = MagicMock(return_value="response")

        mw = _ToolExclusionMiddleware(excluded=frozenset({"drop"}))
        result = mw.wrap_model_call(request, handler)

        request.override.assert_called_once()
        filtered = request.override.call_args.kwargs["tools"]
        assert len(filtered) == 1
        assert filtered[0].name == "keep"
        handler.assert_called_once_with(overridden_request)
        assert result == "response"

    def test_empty_excluded_passes_through(self) -> None:
        request = MagicMock()
        handler = MagicMock(return_value="response")

        mw = _ToolExclusionMiddleware(excluded=frozenset())
        result = mw.wrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    async def test_async_filters_tools(self) -> None:
        tool_a = MagicMock()
        tool_a.name = "keep"
        tool_b = MagicMock()
        tool_b.name = "drop"
        request = MagicMock()
        request.tools = [tool_a, tool_b]

        overridden_request = MagicMock()
        request.override.return_value = overridden_request

        async def async_handler(_req: ModelRequest) -> str:  # type: ignore[type-arg]
            return "async_response"

        mw = _ToolExclusionMiddleware(excluded=frozenset({"drop"}))
        result = await mw.awrap_model_call(request, async_handler)  # type: ignore[arg-type]

        filtered = request.override.call_args.kwargs["tools"]
        assert len(filtered) == 1
        assert filtered[0].name == "keep"
        assert result == "async_response"


class TestToolExclusionWiring:
    """Tests that excluded_tools on a profile wires _ToolExclusionMiddleware into create_deep_agent."""

    def test_exclusion_middleware_added_when_profile_has_excluded_tools(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "exclprov",
                _HarnessProfile(excluded_tools=frozenset({"execute", "write_file"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.FilesystemMiddleware", side_effect=[MagicMock(), MagicMock()]),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.TodoListMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.PatchToolCallsMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                result = create_deep_agent(model="exclprov:some-model")

            assert result == "compiled-agent"
            # The middleware stack should contain a _ToolExclusionMiddleware
            mw_stack = mock_create.call_args.kwargs["middleware"]
            exclusion_mws = [m for m in mw_stack if isinstance(m, _ToolExclusionMiddleware)]
            assert len(exclusion_mws) == 1
            assert exclusion_mws[0]._excluded == frozenset({"execute", "write_file"})
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_no_exclusion_middleware_when_no_excluded_tools(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "noxprov",
                _HarnessProfile(init_kwargs={"x": 1}),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.FilesystemMiddleware", side_effect=[MagicMock(), MagicMock()]),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.TodoListMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.PatchToolCallsMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model="noxprov:some-model")

            mw_stack = mock_create.call_args.kwargs["middleware"]
            exclusion_mws = [m for m in mw_stack if isinstance(m, _ToolExclusionMiddleware)]
            assert len(exclusion_mws) == 0
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_user_tools_pass_through_to_middleware_for_exclusion(self) -> None:
        """User tools are not pre-filtered; the middleware handles exclusion."""
        original = dict(_HARNESS_PROFILES)
        try:
            _register_harness_profile(
                "exclprov",
                _HarnessProfile(excluded_tools=frozenset({"my_tool"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            user_tool_keep = {"name": "keeper", "description": "keep me"}
            user_tool_drop = {"name": "my_tool", "description": "drop me"}

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.FilesystemMiddleware", side_effect=[MagicMock(), MagicMock()]),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.TodoListMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.PatchToolCallsMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="exclprov:some-model",
                    tools=[user_tool_keep, user_tool_drop],
                )

            # User tools are passed through unfiltered; middleware strips them
            passed_tools = mock_create.call_args.kwargs["tools"]
            names = [t["name"] for t in passed_tools]
            assert "keeper" in names
            assert "my_tool" in names

            # But the middleware is in the stack to handle filtering at call time
            mw_stack = mock_create.call_args.kwargs["middleware"]
            exclusion_mws = [m for m in mw_stack if isinstance(m, _ToolExclusionMiddleware)]
            assert len(exclusion_mws) == 1
            assert "my_tool" in exclusion_mws[0]._excluded
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestModelNoneDeprecationWarning:
    """Tests for the deprecation warning when model=None."""

    def test_model_none_emits_deprecation_warning(self) -> None:
        """Passing model=None should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_deep_agent(model=None)

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "model=None" in str(w.message)]
        assert len(deprecations) == 1
        msg = str(deprecations[0].message)
        assert "deprecated" in msg
        assert "BaseChatModel | str" in msg
        assert "https://docs.langchain.com/oss/python/deepagents/models" in msg
        # The warning must be a `LangChainDeprecationWarning`, not stdlib —
        # this is the strongest signal that we routed through `warn_deprecated`.
        assert deprecations[0].category is LangChainDeprecationWarning
        # And it should not be attributed to a frame inside deepagents itself.
        assert "/deepagents/graph.py" not in deprecations[0].filename

    def test_model_none_default_emits_deprecation_warning(self) -> None:
        """Calling create_deep_agent() with no model arg should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_deep_agent()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "model=None" in str(w.message)]
        assert len(deprecations) == 1

    def test_explicit_model_no_deprecation_warning(self) -> None:
        """Passing an explicit model should not emit a DeprecationWarning."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            create_deep_agent(model=model)

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "model=None" in str(w.message)]
        assert len(deprecations) == 0
