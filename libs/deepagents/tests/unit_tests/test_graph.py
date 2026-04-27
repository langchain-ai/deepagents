"""Unit tests for deepagents.graph module."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents._tools import _apply_tool_description_overrides, _tool_name
from deepagents._version import __version__
from deepagents.graph import (
    _REQUIRED_MIDDLEWARE_CLASSES,
    _REQUIRED_MIDDLEWARE_NAMES,
    BASE_AGENT_PROMPT,
    _resolve_extra_middleware,
    create_deep_agent,
)
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.permissions import _PermissionMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
    _harness_profile_for_model,
)
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
        result = _resolve_extra_middleware(HarnessProfile())
        assert result == []

    def test_static_sequence_returned_as_list(self) -> None:
        sentinel = MagicMock()
        profile = HarnessProfile(extra_middleware=(sentinel,))
        result = _resolve_extra_middleware(profile)
        assert result == [sentinel]

    def test_callable_factory_is_invoked(self) -> None:
        sentinel = MagicMock()
        factory = MagicMock(return_value=[sentinel])
        profile = HarnessProfile(extra_middleware=factory)
        result = _resolve_extra_middleware(profile)
        factory.assert_called_once()
        assert result == [sentinel]

    def test_returns_fresh_list_each_call(self) -> None:
        sentinel = MagicMock()
        profile = HarnessProfile(extra_middleware=(sentinel,))
        a = _resolve_extra_middleware(profile)
        b = _resolve_extra_middleware(profile)
        assert a == b
        assert a is not b


class TestProfileForModel:
    """Tests for _harness_profile_for_model."""

    def test_uses_spec_when_provided(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            profile = HarnessProfile(system_prompt_suffix="from spec")
            register_harness_profile("testprov", profile)
            result = _harness_profile_for_model(_make_model({}), "testprov:some-model")
            assert result is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_falls_back_to_identifier_when_spec_is_none(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            profile = HarnessProfile(system_prompt_suffix="from identifier")
            register_harness_profile("myprov", profile)
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
            profile = HarnessProfile(system_prompt_suffix="from provider")
            register_harness_profile("fakeprov", profile)
            model = _make_model({"model": "some-model-name"})
            # Simulate _get_ls_params returning the provider
            model._get_ls_params = MagicMock(return_value={"ls_provider": "fakeprov"})
            result = _harness_profile_for_model(model, None)
            assert result is profile
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_matches_combined_provider_model_key_for_prebuilt(self) -> None:
        """Model-level keys (`provider:model`) resolve for pre-built models.

        Prior to combining the identifier and provider into a single registry
        key, a profile registered under `"fakeprov:my-model"` was unreachable
        when the caller handed in a pre-built model without a spec string: the
        bare identifier `"my-model"` never matched the full key, and lookup
        silently fell through to any provider-level profile (or the empty
        default). This asserts the combined key is consulted first.
        """
        original = dict(_HARNESS_PROFILES)
        try:
            provider_profile = HarnessProfile(system_prompt_suffix="provider level")
            model_profile = HarnessProfile(system_prompt_suffix="model level")
            register_harness_profile("fakeprov", provider_profile)
            register_harness_profile("fakeprov:my-model", model_profile)
            model = _make_model({"model_name": "my-model"})
            model._get_ls_params = MagicMock(return_value={"ls_provider": "fakeprov"})
            result = _harness_profile_for_model(model, None)
            # Model-level wins on merge; suffix reflects model-level registration.
            assert result.system_prompt_suffix == "model level"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_returns_empty_default_when_no_match(self) -> None:
        model = _make_model({"model_name": "unknown-model"})
        model._get_ls_params = MagicMock(return_value={})
        result = _harness_profile_for_model(model, None)
        assert result == HarnessProfile()

    def test_returns_empty_default_when_no_identifier(self) -> None:
        model = _make_model({})
        model._get_ls_params = MagicMock(return_value={})
        result = _harness_profile_for_model(model, None)
        assert result == HarnessProfile()


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
    """Tests for harness-profile lookup on Anthropic model keys."""

    def test_unregistered_anthropic_model_returns_none(self) -> None:
        """An Anthropic model without a built-in registration falls through."""
        assert _get_harness_profile("anthropic:claude-sonnet-4-5") is None


class TestToolDescriptionOverrideWiring:
    """Tests that supported built-in tool overrides are wired into middleware."""

    def test_create_deep_agent_passes_overrides_to_filesystem_and_task(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "testprov",
                HarnessProfile(
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


class TestGeneralPurposeSubagentProfileWiring:
    """Tests for harness-level general-purpose subagent controls."""

    def test_create_deep_agent_applies_general_purpose_subagent_edits(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "testprov",
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(
                        description="Custom general-purpose description",
                        system_prompt="Custom general-purpose prompt.",
                    )
                ),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            # Patch only what we need to inspect (`SubAgentMiddleware` call
            # args) plus the boundary (`resolve_model`) and `create_agent`,
            # which would otherwise reject the mocked middleware. All other
            # middleware constructors run for real.
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(model="testprov:some-model")

            subagents = mock_subagents.call_args.kwargs["subagents"]
            general_purpose = next(spec for spec in subagents if spec["name"] == "general-purpose")
            assert general_purpose["description"] == "Custom general-purpose description"
            assert general_purpose["system_prompt"] == "Custom general-purpose prompt."
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_disabling_default_general_purpose_removes_task_tool(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "testprov",
                HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with patch("deepagents.graph.resolve_model", return_value=fake_model):
                agent = create_deep_agent(model="testprov:some-model")
            assert "task" not in agent.nodes["tools"].bound._tools_by_name
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_explicit_sync_subagent_still_keeps_task_tool_when_default_disabled(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "testprov",
                HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with patch("deepagents.graph.resolve_model", return_value=fake_model):
                agent = create_deep_agent(
                    model="testprov:some-model",
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker subagent.",
                            "system_prompt": "Do work.",
                        }
                    ],
                )
            assert "task" in agent.nodes["tools"].bound._tools_by_name
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestSystemPromptAssembly:
    """Tests for system prompt assembly: profile base_system_prompt, suffix, and user prompt interaction."""

    def _build_and_capture_system_prompt(self, profile_key: str, profile: HarnessProfile, **kwargs: Any) -> str | SystemMessage:
        """Register a profile, call create_deep_agent, return the system_prompt passed to create_agent."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(profile_key, profile)
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
        prompt = self._build_and_capture_system_prompt("defprov", HarnessProfile())
        assert prompt == BASE_AGENT_PROMPT

    def test_profile_base_system_prompt_replaces_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            HarnessProfile(base_system_prompt="You are a custom agent."),
        )
        assert prompt == "You are a custom agent."
        assert BASE_AGENT_PROMPT not in prompt

    def test_profile_base_system_prompt_with_suffix(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            HarnessProfile(
                base_system_prompt="You are a custom agent.",
                system_prompt_suffix="Be concise.",
            ),
        )
        assert prompt == "You are a custom agent.\n\nBe concise."
        assert BASE_AGENT_PROMPT not in prompt

    def test_suffix_without_base_system_prompt_appends_to_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "suffprov",
            HarnessProfile(system_prompt_suffix="Think step by step."),
        )
        assert prompt == BASE_AGENT_PROMPT + "\n\nThink step by step."

    def test_user_system_prompt_prepended_before_profile_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            HarnessProfile(base_system_prompt="Custom base."),
            system_prompt="User instructions.",
        )
        assert prompt == "User instructions.\n\nCustom base."
        assert BASE_AGENT_PROMPT not in prompt

    def test_user_system_prompt_prepended_before_default_base(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "defprov",
            HarnessProfile(),
            system_prompt="User instructions.",
        )
        assert prompt == f"User instructions.\n\n{BASE_AGENT_PROMPT}"

    def test_triple_combo_all_three_inputs(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            HarnessProfile(
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
            HarnessProfile(base_system_prompt="Custom base."),
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
            HarnessProfile(base_system_prompt=""),
        )
        assert prompt == ""
        assert BASE_AGENT_PROMPT not in prompt

    def test_empty_string_suffix_still_appended(self) -> None:
        prompt = self._build_and_capture_system_prompt(
            "custprov",
            HarnessProfile(
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
            register_harness_profile(
                "exclprov",
                HarnessProfile(excluded_tools=frozenset({"execute", "write_file"})),
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
            register_harness_profile(
                "noxprov",
                HarnessProfile(system_prompt_suffix="present"),
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
            register_harness_profile(
                "exclprov",
                HarnessProfile(excluded_tools=frozenset({"my_tool"})),
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


class _StubMW(AgentMiddleware[Any, Any, Any]):
    """Minimal real `AgentMiddleware` subclass so langchain's agent factory accepts it."""


class TestExtraMiddlewareWiring:
    """End-to-end tests that `extra_middleware` from a harness profile reaches the compiled agent."""

    def test_extra_middleware_is_added_to_main_stack(self) -> None:
        mw_instance = _StubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "emprov",
                HarnessProfile(extra_middleware=[mw_instance]),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model="emprov:some-model")

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert any(m is mw_instance for m in mw_stack), "extra_middleware instance missing from main middleware stack"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_extra_middleware_factory_is_invoked_per_stack(self) -> None:
        """A callable factory is called once for each middleware stack (main + general-purpose)."""
        calls: list[None] = []

        def factory() -> list[AgentMiddleware]:
            calls.append(None)
            return [_StubMW()]

        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "emfactprov",
                HarnessProfile(extra_middleware=factory),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(model="emfactprov:some-model")

            # Invoked at least twice: once for the general-purpose subagent, once for the main stack.
            assert len(calls) >= 2
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class _OtherStubMW(AgentMiddleware[Any, Any, Any]):
    """Second stub class so exclusion tests can assert coexistence with `_StubMW`."""


class _StubSubMW(_StubMW):
    """Subclass of `_StubMW` used to verify exact-type (not isinstance) exclusion."""


class TestMiddlewareExclusionWiring:
    """End-to-end tests that `excluded_middleware` filters the assembled stack."""

    def test_excluded_middleware_strips_user_middleware_from_main_stack(self) -> None:
        """User-supplied middleware whose class is excluded is filtered out."""
        dropped = _StubMW()
        kept = _OtherStubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_StubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="excmwprov:some-model",
                    middleware=[dropped, kept],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is dropped for m in mw_stack), "excluded user middleware leaked into stack"
            assert any(m is kept for m in mw_stack), "non-excluded user middleware was dropped"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_strips_profile_extra_middleware(self) -> None:
        """A profile can exclude a class it also provides via `extra_middleware`.

        This covers the merge case where a provider-level profile adds a
        middleware and a model-level profile removes it.
        """
        provided = _StubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("excmwprov", HarnessProfile(extra_middleware=[provided]))
            register_harness_profile(
                "excmwprov:some-model",
                HarnessProfile(excluded_middleware=frozenset({_StubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model="excmwprov:some-model")

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is provided for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_preserves_subclass(self) -> None:
        """Exclusion matches on exact type, so subclasses of an excluded class are kept.

        Passes both the base class and a subclass instance so the exclusion
        entry has something to match — required by the coverage guard.
        """
        base_instance = _StubMW()
        subclass_instance = _StubSubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_StubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="excmwprov:some-model",
                    middleware=[base_instance, subclass_instance],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is base_instance for m in mw_stack), "exact base class should be filtered"
            assert any(m is subclass_instance for m in mw_stack), "subclass instance should be preserved"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_strips_from_general_purpose_subagent_stack(self) -> None:
        """The auto-added general-purpose subagent has its stack filtered too."""
        provided = _StubMW()

        def factory() -> list[AgentMiddleware]:
            return [provided]

        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(
                    extra_middleware=factory,
                    excluded_middleware=frozenset({_StubMW}),
                ),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(model="excmwprov:some-model")

            gp_spec = next(s for s in mock_subagents.call_args.kwargs["subagents"] if s["name"] == "general-purpose")
            gp_stack = gp_spec["middleware"]
            assert not any(type(m) is _StubMW for m in gp_stack), "general-purpose stack not filtered"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_strips_from_declarative_subagent_stack(self) -> None:
        """Declarative `SubAgent` specs built by `create_deep_agent` are filtered too."""
        provided = _StubMW()

        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(
                    extra_middleware=[provided],
                    excluded_middleware=frozenset({_StubMW}),
                ),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(
                    model="excmwprov:main-model",
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker.",
                            "system_prompt": "Do work.",
                            "model": "excmwprov:worker-model",
                        }
                    ],
                )

            worker_spec = next(s for s in mock_subagents.call_args.kwargs["subagents"] if s.get("name") == "worker")
            worker_stack = worker_spec["middleware"]
            assert not any(type(m) is _StubMW for m in worker_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_register_additive_unions_excluded_middleware(self) -> None:
        """Re-registering under the same key unions `excluded_middleware` with the prior set."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_StubMW})),
            )
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_OtherStubMW})),
            )
            merged = _get_harness_profile("excmwprov")
            assert merged is not None
            assert merged.excluded_middleware == frozenset({_StubMW, _OtherStubMW})
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_strips_async_subagent_middleware(self) -> None:
        """Async subagents add `AsyncSubAgentMiddleware` to the parent stack — it can be excluded."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({AsyncSubAgentMiddleware})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="excmwprov:some-model",
                    subagents=[
                        {
                            "name": "remote",
                            "description": "Remote worker.",
                            "graph_id": "my-graph",
                        }
                    ],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(type(m) is AsyncSubAgentMiddleware for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_handles_multiple_classes_in_one_set(self) -> None:
        """A single exclusion set with two classes removes instances of both."""
        stub = _StubMW()
        other = _OtherStubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_StubMW, _OtherStubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model="excmwprov:some-model", middleware=[stub, other])

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is stub for m in mw_stack)
            assert not any(m is other for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_preserves_order_of_kept_entries(self) -> None:
        """Filtering an excluded class keeps surrounding middleware in original relative order."""
        before = _OtherStubMW()
        dropped = _StubMW()
        after = _OtherStubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "excmwprov",
                HarnessProfile(excluded_middleware=frozenset({_StubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="excmwprov:some-model",
                    middleware=[before, dropped, after],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            before_idx = mw_stack.index(before)
            after_idx = mw_stack.index(after)
            assert before_idx < after_idx, "relative order of non-excluded middleware changed after filter"
            assert not any(m is dropped for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_excluded_middleware_rejects_required_scaffolding(self) -> None:
        """Excluding a required class raises `ValueError` at assembly time.

        Splitting the assertion per class makes the failure message point at
        the specific class that slipped past the deny-list, rather than a
        single composite test that hides which class regressed.
        """
        forbidden_classes: tuple[type[AgentMiddleware[Any, Any, Any]], ...] = (
            FilesystemMiddleware,
            SubAgentMiddleware,
            _PermissionMiddleware,
        )
        original = dict(_HARNESS_PROFILES)
        try:
            for forbidden_cls in forbidden_classes:
                _HARNESS_PROFILES.clear()
                _HARNESS_PROFILES.update(original)
                # cast silences ty's narrow inference when HarnessProfile's
                # field type is `frozenset[type[AgentMiddleware[Any, Any, Any]]]`
                # and the loop variable is a specific scaffolding class — the
                # runtime behavior is identical, but ty sees the exact class
                # instead of the parameterized base.
                register_harness_profile(
                    "denyprov",
                    HarnessProfile(
                        excluded_middleware=cast(
                            "frozenset[type[AgentMiddleware[Any, Any, Any]]]",
                            frozenset({forbidden_cls}),
                        )
                    ),
                )
                fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
                with (
                    patch("deepagents.graph.resolve_model", return_value=fake_model),
                    pytest.raises(ValueError, match=forbidden_cls.__name__),
                ):
                    create_deep_agent(model="denyprov:some-model")
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestScaffoldingViolationAggregation:
    """Multiple scaffolding exclusions surface together in a single `ValueError`."""

    def test_class_and_name_violations_report_together(self) -> None:
        """Mixing class-form and name-form scaffolding entries reports both in one message."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "multiviol",
                HarnessProfile(excluded_middleware=frozenset({FilesystemMiddleware, "SubAgentMiddleware"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                pytest.raises(ValueError, match="scaffolding") as excinfo,
            ):
                create_deep_agent(model="multiviol:some-model")
            message = str(excinfo.value)
            assert "FilesystemMiddleware" in message
            assert "SubAgentMiddleware" in message
            assert "scaffolding" in message
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestRequiredMiddlewareNamesCoverage:
    """Drift guard: `_REQUIRED_MIDDLEWARE_NAMES` must cover every required class's `.name`.

    `_REQUIRED_MIDDLEWARE_NAMES` is a hand-maintained frozenset of the string
    forms that should be rejected by the scaffolding guard. If a future
    refactor renames a required class or overrides its `.name` without
    updating the names constant, string-form exclusion would silently bypass
    the guard. This test catches that drift.
    """

    def test_every_required_class_name_is_listed(self) -> None:
        """Every `_REQUIRED_MIDDLEWARE_CLASSES` entry's `.name` must appear in `_REQUIRED_MIDDLEWARE_NAMES`.

        `.name` is a property on `AgentMiddleware` that doesn't require
        `__init__` to have run, so `__new__` gives us the reportable name
        without invoking constructor arguments we don't have here.
        """
        for cls in _REQUIRED_MIDDLEWARE_CLASSES:
            instance = cls.__new__(cls)
            assert instance.name in _REQUIRED_MIDDLEWARE_NAMES, (
                f"{cls.__name__}.name={instance.name!r} is not in _REQUIRED_MIDDLEWARE_NAMES "
                f"({sorted(_REQUIRED_MIDDLEWARE_NAMES)!r}) — string-form exclusion would "
                f"silently bypass the scaffolding guard. Add it to _REQUIRED_MIDDLEWARE_NAMES."
            )


class PublicStubMW(AgentMiddleware[Any, Any, Any]):
    """Middleware with a public (non-underscore-prefixed) class name.

    String-form `excluded_middleware` matches against `AgentMiddleware.name`,
    which defaults to the class's `__name__`. The underscore-prefix guard
    rejects private-looking names, so string-form exclusion tests need a stub
    class whose name does not start with `_`.
    """


class OtherPublicStubMW(AgentMiddleware[Any, Any, Any]):
    """Second public-named stub used to assert coexistence with `PublicStubMW`."""


class TestStringFormExcludedMiddleware:
    """End-to-end tests that string-form entries in `excluded_middleware` filter the stack.

    String entries match `AgentMiddleware.name` exactly (no normalization), so
    a caller can exclude middleware without importing its class — useful for
    profiles loaded from config files or for excluding middleware injected
    internally by `create_deep_agent`.
    """

    def test_string_entry_excludes_user_middleware_by_name(self) -> None:
        """An exact `.name` match drops the corresponding user middleware."""
        dropped = PublicStubMW()
        kept = OtherPublicStubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"PublicStubMW"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="strexcprov:some-model",
                    middleware=[dropped, kept],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is dropped for m in mw_stack)
            assert any(m is kept for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_string_entry_matches_overridden_name_on_summarization(self) -> None:
        """`"SummarizationMiddleware"` drops `_DeepAgentsSummarizationMiddleware` via the `.name` override."""
        dropped = _DeepAgentsSummarizationMiddleware.__new__(_DeepAgentsSummarizationMiddleware)
        kept = PublicStubMW()
        assert dropped.name == "SummarizationMiddleware", "summarization impl must report its public alias via `.name`"
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"SummarizationMiddleware"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="strexcprov:some-model",
                    middleware=[dropped, kept],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(type(m) is _DeepAgentsSummarizationMiddleware for m in mw_stack)
            assert any(m is kept for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_mixed_class_and_string_entries_both_apply(self) -> None:
        """A single `excluded_middleware` set may hold both classes and strings."""
        dropped_by_class = PublicStubMW()
        dropped_by_string = OtherPublicStubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({PublicStubMW, "OtherPublicStubMW"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(
                    model="strexcprov:some-model",
                    middleware=[dropped_by_class, dropped_by_string],
                )

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(m is dropped_by_class for m in mw_stack)
            assert not any(m is dropped_by_string for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    @pytest.mark.parametrize(
        "forbidden",
        [
            "FilesystemMiddleware",
            "SubAgentMiddleware",
            "PermissionMiddleware",
        ],
    )
    def test_string_entry_rejects_required_scaffolding(self, forbidden: str) -> None:
        """Public-name strings of required scaffolding raise `ValueError` at assembly time.

        The private `"_PermissionMiddleware"` spelling is rejected earlier at
        `HarnessProfile` construction by the grammar guard (see
        `test_string_entry_rejects_private_underscore_prefixed_names`), so
        only public names reach the scaffolding check.
        """
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "denyprov",
                HarnessProfile(excluded_middleware=frozenset({forbidden})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                pytest.raises(ValueError, match="scaffolding"),
            ):
                create_deep_agent(model="denyprov:some-model")
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    @pytest.mark.parametrize("entry", ["_ToolExclusionMiddleware", "_PermissionMiddleware"])
    def test_string_entry_rejects_private_underscore_prefixed_names(self, entry: str) -> None:
        """Underscore-prefixed string entries raise `ValueError` at construction.

        The grammar guard on `HarnessProfile.__post_init__` rejects private
        plain names up front. This catches typos eagerly; callers who
        genuinely need to exclude a private middleware can pass the class
        directly via the runtime `HarnessProfile`.
        """
        with pytest.raises(ValueError, match="cannot start with '_'"):
            HarnessProfile(excluded_middleware=frozenset({entry}))

    def test_string_entry_unknown_name_raises_coverage_error(self) -> None:
        """A string entry that matches nothing across any stack raises `ValueError`.

        Typos and stale profiles fail loudly rather than silently no-opping —
        a typo'd exclusion that silently has no effect is a common source of
        "my profile isn't working" confusion.
        """
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"DoesNotExistMiddleware"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                pytest.raises(ValueError, match="matched no middleware"),
            ):
                create_deep_agent(model="strexcprov:some-model", middleware=[PublicStubMW()])
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_class_entry_unknown_class_raises_coverage_error(self) -> None:
        """A class entry that matches nothing across any stack raises `ValueError`.

        The coverage guard is symmetric across class and string forms — class
        typos (or stale profiles referencing removed middleware) are caught
        at assembly time alongside string typos.
        """
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({_OtherStubMW})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                pytest.raises(ValueError, match="matched no middleware"),
            ):
                create_deep_agent(model="strexcprov:some-model", middleware=[_StubMW()])
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_entry_matching_only_gp_subagent_stack_is_accepted(self) -> None:
        """An entry matching only the GP subagent stack (not the main stack) is accepted.

        Coverage is aggregated across all stacks the profile applies to, so a
        profile-level exclusion only has to match somewhere — not in every
        stack. `TodoListMiddleware` is added unconditionally to the GP
        subagent stack; excluding it should work even though the main agent
        also has one (both count as matches).
        """
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"TodoListMiddleware"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create,
            ):
                create_deep_agent(model="strexcprov:some-model")

            mw_stack = mock_create.call_args.kwargs["middleware"]
            assert not any(type(m).__name__ == "TodoListMiddleware" for m in mw_stack)
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_string_entry_matching_multiple_classes_raises(self) -> None:
        """A string entry matching multiple distinct classes in one stack raises `ValueError`.

        Protects against accidental collisions where a user-supplied
        middleware's `.name` happens to match a built-in alias. Dropping
        every instance under the shared name would silently widen the blast
        radius beyond what the caller asked for.
        """

        class ShadowingStubMW(AgentMiddleware[Any, Any, Any]):
            """Second class whose `.name` collides with `PublicStubMW` by override."""

            name = "PublicStubMW"

        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"PublicStubMW"})),
            )
            fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            with (
                patch("deepagents.graph.resolve_model", return_value=fake_model),
                pytest.raises(ValueError, match="matched multiple distinct"),
            ):
                create_deep_agent(
                    model="strexcprov:some-model",
                    middleware=[PublicStubMW(), ShadowingStubMW()],
                )
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_string_entry_unions_with_class_entry_on_merge(self) -> None:
        """Re-registering with a string-form entry unions with an existing class-form set."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({PublicStubMW})),
            )
            register_harness_profile(
                "strexcprov",
                HarnessProfile(excluded_middleware=frozenset({"OtherPublicStubMW"})),
            )
            merged = _get_harness_profile("strexcprov")
            assert merged is not None
            assert merged.excluded_middleware == frozenset({PublicStubMW, "OtherPublicStubMW"})
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestSubagentLevelProfileResolution:
    """Tests that sync subagents with their own `model` get their own harness profile applied."""

    def test_subagent_with_different_model_resolves_its_own_profile(self) -> None:
        parent_mw = _StubMW()
        subagent_mw = _StubMW()

        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "parentprov",
                HarnessProfile(extra_middleware=[parent_mw]),
            )
            register_harness_profile(
                "subprov",
                HarnessProfile(extra_middleware=[subagent_mw]),
            )

            parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

            def fake_resolve(spec: str | BaseChatModel) -> BaseChatModel:
                if isinstance(spec, BaseChatModel):
                    return spec
                if spec.startswith("subprov"):
                    return subagent_model
                return parent_model

            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", side_effect=fake_resolve),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(
                    model="parentprov:main-model",
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker.",
                            "system_prompt": "Do work.",
                            "model": "subprov:worker-model",
                        }
                    ],
                )

            sub_specs = mock_subagents.call_args.kwargs["subagents"]
            worker = next(s for s in sub_specs if s.get("name") == "worker")
            worker_middleware = worker["middleware"]
            assert any(m is subagent_mw for m in worker_middleware), "Subagent profile's middleware not applied"
            assert not any(m is parent_mw for m in worker_middleware), "Parent profile's middleware leaked into subagent"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestProfileMissLogLevel:
    """Tests that pre-built-model profile-miss logs escalate to warning when profiles are registered."""

    def test_no_registered_profiles_logs_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        model = MagicMock(spec=BaseChatModel)
        model.model_dump.return_value = {}
        model._get_ls_params = MagicMock(return_value={})
        with caplog.at_level(logging.DEBUG, logger="deepagents.profiles.harness.harness_profiles"):
            result = _harness_profile_for_model(model, None)
        assert result == HarnessProfile()
        records = [r for r in caplog.records if "No harness profile matched" in r.getMessage()]
        assert records, "Expected a profile-miss log record"
        assert all(r.levelno == logging.DEBUG for r in records)

    def test_registered_profiles_but_no_match_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile("someprov", HarnessProfile(system_prompt_suffix="x"))
            model = MagicMock(spec=BaseChatModel)
            model.model_dump.return_value = {}
            model._get_ls_params = MagicMock(return_value={})
            with caplog.at_level(logging.DEBUG, logger="deepagents.profiles.harness.harness_profiles"):
                result = _harness_profile_for_model(model, None)
            assert result == HarnessProfile()
            records = [r for r in caplog.records if "No harness profile matched" in r.getMessage()]
            assert records, "Expected a profile-miss log record"
            assert all(r.levelno == logging.WARNING for r in records)
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
        # stacklevel=2 should point at the caller, not inside graph.py
        assert deprecations[0].filename == __file__

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


class TestPrebuiltModelIdentifierDoesNotMatchProviderKey:
    """Regression tests: a bare model identifier must not match a registered provider key.

    Previously, `_harness_profile_for_model` would call `_get_harness_profile(identifier)`
    for a pre-built model whose identifier happened to coincide with a registered
    provider (e.g. an in-house proxy whose `model_name` is `"openai"`), silently
    picking up that provider's profile. The fix skips bare-identifier lookups.
    """

    def test_bare_identifier_coinciding_with_provider_key_does_not_match(self) -> None:
        original = dict(_HARNESS_PROFILES)
        try:
            # A harness profile is registered under bare "openai".
            register_harness_profile(
                "openai",
                HarnessProfile(system_prompt_suffix="openai-specific"),
            )
            # Build a pre-built model whose identifier happens to equal "openai"
            # but whose provider (from _get_ls_params) is a different vendor.
            model = _make_model({"model_name": "openai"})
            model._get_ls_params = MagicMock(return_value={"ls_provider": "custom_proxy"})

            result = _harness_profile_for_model(model, None)
            assert result == HarnessProfile(), "Bare-identifier lookup should not match the openai provider profile"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_colon_qualified_identifier_still_matches(self) -> None:
        """If the identifier itself is in `provider:model` shape, lookup still resolves."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "colprov",
                HarnessProfile(system_prompt_suffix="from provider"),
            )
            model = _make_model({"model_name": "colprov:some-model"})
            # No ls_provider in _get_ls_params; the identifier alone carries the provider.
            model._get_ls_params = MagicMock(return_value={})
            result = _harness_profile_for_model(model, None)
            assert result.system_prompt_suffix == "from provider"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestSubagentLevelToolExclusionAndOverrides:
    """Tests that sync subagents with their own profile get their own `excluded_tools` and `tool_description_overrides`."""

    def test_subagent_excluded_tools_not_leaked_from_parent(self) -> None:
        """A subagent's profile `excluded_tools` must apply to the subagent only, not inherit parent's."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "parentexcl",
                HarnessProfile(excluded_tools=frozenset({"parent_only"})),
            )
            register_harness_profile(
                "subexcl",
                HarnessProfile(excluded_tools=frozenset({"child_only"})),
            )

            parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

            def fake_resolve(spec: str | BaseChatModel) -> BaseChatModel:
                if isinstance(spec, BaseChatModel):
                    return spec
                if spec.startswith("subexcl"):
                    return subagent_model
                return parent_model

            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", side_effect=fake_resolve),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(
                    model="parentexcl:main-model",
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker.",
                            "system_prompt": "Do work.",
                            "model": "subexcl:worker-model",
                        }
                    ],
                )

            sub_specs = mock_subagents.call_args.kwargs["subagents"]
            worker = next(s for s in sub_specs if s.get("name") == "worker")
            exclusions = [m for m in worker["middleware"] if isinstance(m, _ToolExclusionMiddleware)]
            assert len(exclusions) == 1
            assert exclusions[0]._excluded == frozenset({"child_only"})
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)

    def test_subagent_tool_description_overrides_not_leaked_from_parent(self) -> None:
        """A subagent's profile `tool_description_overrides` must reach its FilesystemMiddleware, not the parent's."""
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "parentdesc",
                HarnessProfile(tool_description_overrides={"ls": "parent ls"}),
            )
            register_harness_profile(
                "subdesc",
                HarnessProfile(tool_description_overrides={"ls": "child ls"}),
            )

            parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            subagent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

            def fake_resolve(spec: str | BaseChatModel) -> BaseChatModel:
                if isinstance(spec, BaseChatModel):
                    return spec
                if spec.startswith("subdesc"):
                    return subagent_model
                return parent_model

            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", side_effect=fake_resolve),
                patch("deepagents.graph.FilesystemMiddleware", return_value=MagicMock()) as mock_fs,
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(
                    model="parentdesc:main-model",
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker.",
                            "system_prompt": "Do work.",
                            "model": "subdesc:worker-model",
                        }
                    ],
                )

            # The subagent's FilesystemMiddleware should receive the child overrides.
            subagent_descriptions = [
                call.kwargs["custom_tool_descriptions"]
                for call in mock_fs.call_args_list
                if dict(call.kwargs["custom_tool_descriptions"]) == {"ls": "child ls"}
            ]
            assert subagent_descriptions, "Subagent FilesystemMiddleware did not receive its own tool_description_overrides"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestPrebuiltSubagentModelResolvesProfile:
    """A pre-built `BaseChatModel` passed as a subagent's `model` must still get a harness profile via identifier/provider extraction."""

    def test_prebuilt_subagent_model_uses_provider_profile_by_ls_params(self) -> None:
        subagent_mw = _StubMW()
        original = dict(_HARNESS_PROFILES)
        try:
            register_harness_profile(
                "prebuiltprov",
                HarnessProfile(extra_middleware=[subagent_mw]),
            )

            parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
            # Subagent pre-built model: expose model_dump + _get_ls_params so
            # `_harness_profile_for_model` can extract identifier and provider.
            subagent_model = MagicMock(spec=BaseChatModel)
            subagent_model.model_dump.return_value = {"model_name": "sub-model"}
            subagent_model._get_ls_params = MagicMock(return_value={"ls_provider": "prebuiltprov"})

            def fake_resolve(spec: str | BaseChatModel) -> BaseChatModel:
                if isinstance(spec, BaseChatModel):
                    return spec
                return parent_model

            fake_agent = MagicMock()
            fake_agent.with_config.return_value = "compiled-agent"

            with (
                patch("deepagents.graph.resolve_model", side_effect=fake_resolve),
                patch("deepagents.graph.SubAgentMiddleware", return_value=MagicMock()) as mock_subagents,
                patch("deepagents.graph.create_summarization_middleware", return_value=MagicMock()),
                patch("deepagents.graph.create_agent", return_value=fake_agent),
            ):
                create_deep_agent(
                    model=parent_model,
                    subagents=[
                        {
                            "name": "worker",
                            "description": "Worker.",
                            "system_prompt": "Do work.",
                            "model": subagent_model,
                        }
                    ],
                )

            sub_specs = mock_subagents.call_args.kwargs["subagents"]
            worker = next(s for s in sub_specs if s.get("name") == "worker")
            assert any(m is subagent_mw for m in worker["middleware"]), "Pre-built subagent model did not pick up registered profile"
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)


class TestHasAnyHarnessProfile:
    """Regression test for `_has_any_harness_profile` helper."""

    def test_reports_true_when_registered(self) -> None:
        from deepagents.profiles.harness.harness_profiles import _has_any_harness_profile  # noqa: PLC0415

        original = dict(_HARNESS_PROFILES)
        try:
            assert _has_any_harness_profile() is False
            register_harness_profile("helperprov", HarnessProfile(system_prompt_suffix="x"))
            assert _has_any_harness_profile() is True
        finally:
            _HARNESS_PROFILES.clear()
            _HARNESS_PROFILES.update(original)
