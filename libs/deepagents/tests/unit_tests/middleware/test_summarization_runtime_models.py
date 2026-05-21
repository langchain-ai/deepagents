"""Unit tests for runtime-resolved summarization model selection.

Exercises the precedence rules documented on
`_DeepAgentsSummarizationMiddleware._resolve_models`:

- Threshold model: `runtime.context["model"]` then construction-time fallback.
- Summarizer model: `runtime.context["summarization_model"]` then threshold model.
"""

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.chat_models import BaseChatModel
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage

from deepagents.middleware import SummarizationContext
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
)
from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    call_awrap_model_call,
    call_wrap_model_call,
    make_conversation_messages,
    make_mock_model,
    make_mock_runtime,
    make_model_request,
    mock_get_config,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


def _as_chat_model(mock_model: MagicMock) -> MagicMock:
    """Make a `MagicMock` pass `isinstance(..., BaseChatModel)`.

    The middleware's runtime-context type check rejects non-`BaseChatModel`
    specs; `make_mock_model` returns a bare `MagicMock`, so we tag `__class__`
    here to opt that mock into the type contract.
    """
    mock_model.__class__ = BaseChatModel  # ty: ignore[invalid-assignment]
    return mock_model


def _runtime_with_context(**kwargs: Any) -> MagicMock:
    """Build a mock runtime with the given context dict."""
    runtime = make_mock_runtime()
    runtime.context = dict(kwargs)
    return runtime


class TestSummarizerModelResolution:
    """Compaction picks the right summarizer model based on runtime context."""

    def test_explicit_summarization_model_overrides_main(self) -> None:
        """`runtime.context["summarization_model"]` wins over `model`."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="main"))
        summ_override = _as_chat_model(make_mock_model(summary_response="explicit"))

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(
            model=main_override,
            summarization_model=summ_override,
        )

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        summ_override.invoke.assert_called_once()
        main_override.invoke.assert_not_called()
        construction_model.invoke.assert_not_called()

    def test_summarizer_follows_main_model_when_only_model_set(self) -> None:
        """Without `summarization_model`, the summarizer follows the `model` override."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="main"))

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(model=main_override)

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        main_override.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_construction_time_model_when_no_context(self) -> None:
        """With no runtime context, summarization uses the construction-time model."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        construction_model.invoke.assert_called_once()

    async def test_async_explicit_summarization_model(self) -> None:
        """Async variant honors `summarization_model` override."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        summ_override = _as_chat_model(make_mock_model(summary_response="explicit"))

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(summarization_model=summ_override)

        with mock_get_config():
            result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        summ_override.ainvoke.assert_called_once()
        construction_model.ainvoke.assert_not_called()


class TestThresholdModelResolution:
    """Threshold computation follows `model`, not `summarization_model`."""

    def test_count_tokens_supports_tool_aware_and_legacy_counters(self) -> None:
        """`_count_tokens` supports counters with and without `tools`."""
        messages = [MagicMock()]
        tools = [{"name": "example"}]

        tool_aware_counter = MagicMock(return_value=12)
        tool_aware_helper = MagicMock(token_counter=tool_aware_counter)
        assert SummarizationMiddleware._count_tokens(tool_aware_helper, messages, tools) == 12
        tool_aware_counter.assert_called_once_with(messages, tools=tools)

        def legacy_counter(msgs: list[Any]) -> int:
            return len(msgs)

        legacy = MagicMock(side_effect=legacy_counter)
        legacy_helper = MagicMock(token_counter=legacy)
        assert SummarizationMiddleware._count_tokens(legacy_helper, messages, tools) == 1
        assert legacy.call_count == 2

    def test_threshold_uses_main_model_profile(self) -> None:
        """A bigger-context `model` override raises the fraction-based trigger."""
        backend = MockBackend()
        construction_model = make_mock_model()
        construction_model.profile = {"max_input_tokens": 1000}

        # Bigger context window — fraction-based trigger threshold goes up
        # so the same message bucket falls below the new trigger.
        main_override = _as_chat_model(make_mock_model(summary_response="main"))
        main_override.profile = {"max_input_tokens": 10_000_000}

        # Use the lower-level constructor since `create_summarization_middleware`
        # enforces a real `BaseChatModel` instance.
        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("fraction", 0.001),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})

        # Sanity: with no override, the construction profile fires the trigger.
        runtime_base = make_mock_runtime()
        with mock_get_config():
            base_result, _ = call_wrap_model_call(middleware, state, runtime_base)
        assert isinstance(base_result, ExtendedModelResponse)

        # Reset mock invocation history before the override-driven call so we
        # only see the post-override behavior in the next assertion.
        construction_model.invoke.reset_mock()

        runtime = _runtime_with_context(model=main_override)
        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        # Bigger context → no summarization triggered.
        assert not isinstance(result, ExtendedModelResponse)
        assert main_override.invoke.call_count == 0
        assert construction_model.invoke.call_count == 0

    def test_threshold_uses_runtime_model_token_counter(self) -> None:
        """Token thresholds are counted with the runtime model helper."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="main"))
        construction_counter = MagicMock(return_value=1)
        runtime_counter = MagicMock(return_value=1_000)

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("tokens", 100),
            keep=("messages", 2),
            token_counter=construction_counter,
        )
        middleware._get_helper_for(main_override).token_counter = runtime_counter

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(model=main_override)

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        runtime_counter.assert_called()
        construction_counter.assert_not_called()
        main_override.invoke.assert_called_once()

    async def test_async_threshold_uses_runtime_model_token_counter(self) -> None:
        """Async token thresholds are counted with the runtime model helper."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="main"))
        construction_counter = MagicMock(return_value=1)
        runtime_counter = MagicMock(return_value=1_000)

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("tokens", 100),
            keep=("messages", 2),
            token_counter=construction_counter,
        )
        middleware._get_helper_for(main_override).token_counter = runtime_counter

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(model=main_override)

        with mock_get_config():
            result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        runtime_counter.assert_called()
        construction_counter.assert_not_called()
        main_override.ainvoke.assert_called_once()

    def test_overflow_clipping_uses_runtime_model_token_counter(self) -> None:
        """Overflow fallback clips preserved messages with runtime helper counter."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="main"))
        construction_counter = MagicMock(return_value=1)
        runtime_counter = MagicMock(return_value=1)

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 100),
            keep=("messages", 2),
            token_counter=construction_counter,
        )
        middleware._get_helper_for(main_override).token_counter = runtime_counter

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(model=main_override)
        request = make_model_request(state, runtime)
        call_count = 0

        def handler(_req: Any) -> AIMessage:  # noqa: ANN401
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextOverflowError
            return AIMessage(content="ok")

        def clip_side_effect(preserved: list[Any], *_args: Any, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
            return preserved, []

        with (
            mock_get_config(),
            mock.patch(
                "deepagents.middleware.summarization._clip_overflow_tail",
                side_effect=clip_side_effect,
            ) as clip,
        ):
            result = middleware.wrap_model_call(request, handler)

        assert isinstance(result, ExtendedModelResponse)
        assert clip.call_args.kwargs["token_counter"] is runtime_counter
        construction_counter.assert_not_called()

    def test_summarization_model_does_not_change_thresholds(self) -> None:
        """`summarization_model` alone must not move the trigger boundary."""
        backend = MockBackend()
        construction_model = make_mock_model()
        construction_model.profile = {"max_input_tokens": 1000}

        # Configure a fraction-based trigger that the messages WILL exceed
        # with the construction-time profile (1000 tokens, fraction 0.001).
        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("fraction", 0.001),
            keep=("messages", 2),
        )

        # Summarization_model has a huge profile — but it should NOT affect
        # thresholds, only the summarizer invocation.
        summ_override = _as_chat_model(make_mock_model(summary_response="summ"))
        summ_override.profile = {"max_input_tokens": 10_000_000}

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(summarization_model=summ_override)

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        # Compaction still fires because the threshold is determined by the
        # construction-time profile, not by `summarization_model`.
        assert isinstance(result, ExtendedModelResponse)
        summ_override.invoke.assert_called_once()


class TestModelResolverCallback:
    """String model specs in runtime context are resolved via `model_resolver`."""

    def test_string_spec_resolved_through_callback(self) -> None:
        """A string in context is passed to `model_resolver` and the result is used."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        resolved = make_mock_model(summary_response="resolved")

        seen: list[str] = []

        def resolver(spec: str) -> Any:  # noqa: ANN401
            seen.append(spec)
            return resolved

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            model_resolver=resolver,
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(summarization_model="openai:gpt-5.4-mini")

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert seen == ["openai:gpt-5.4-mini"]
        resolved.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_string_spec_without_resolver_falls_back_to_resolve_model(self) -> None:
        """A string spec with no `model_resolver` falls back to `resolve_model`."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        resolved = make_mock_model(summary_response="resolved")

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(summarization_model="openai:gpt-5.4-mini")

        seen: list[str] = []

        def fake_resolve_model(spec: str) -> Any:  # noqa: ANN401
            seen.append(spec)
            return resolved

        with (
            mock_get_config(),
            mock.patch(
                "deepagents._models.resolve_model",
                side_effect=fake_resolve_model,
            ),
        ):
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert seen == ["openai:gpt-5.4-mini"]
        resolved.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_non_string_non_model_spec_raises_type_error(self) -> None:
        """A spec that is neither `str` nor `BaseChatModel` is rejected loudly."""
        backend = MockBackend()
        construction_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        # A dict is not a valid model spec.
        runtime = _runtime_with_context(summarization_model={"oops": True})

        with mock_get_config(), pytest.raises(TypeError, match="summarization_model"):
            call_wrap_model_call(middleware, state, runtime)

    def test_resolver_exception_propagates(self) -> None:
        """Errors raised by a host-supplied resolver surface unchanged."""
        backend = MockBackend()
        construction_model = make_mock_model()

        class ResolverError(RuntimeError):
            pass

        def resolver(_spec: str) -> Any:  # noqa: ANN401
            msg = "boom"
            raise ResolverError(msg)

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            model_resolver=resolver,
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(summarization_model="openai:gpt-5.4-mini")

        with mock_get_config(), pytest.raises(ResolverError, match="boom"):
            call_wrap_model_call(middleware, state, runtime)

    def test_string_spec_resolved_via_runtime_context(self) -> None:
        """A resolver placed in `runtime.context["model_resolver"]` is honored.

        This is how hosts (e.g. `deepagents-code`) inject the resolver without
        a constructor-level configuration step. The SDK never imports CLI code
        — the callable simply rides along in `runtime.context`.
        """
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        resolved = make_mock_model(summary_response="resolved")

        seen: list[str] = []

        def resolver(spec: str) -> Any:  # noqa: ANN401
            seen.append(spec)
            return resolved

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(
            summarization_model="openai:gpt-5.4-mini",
            model_resolver=resolver,
        )

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert seen == ["openai:gpt-5.4-mini"]
        resolved.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_constructor_resolver_takes_precedence_over_runtime(self) -> None:
        """A constructor-level resolver beats a `runtime.context` resolver."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        ctor_resolved = make_mock_model(summary_response="ctor")
        ctx_resolved = make_mock_model(summary_response="ctx")

        def ctor_resolver(spec: str) -> Any:  # noqa: ANN401, ARG001
            return ctor_resolved

        def ctx_resolver(spec: str) -> Any:  # noqa: ANN401, ARG001
            return ctx_resolved

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            model_resolver=ctor_resolver,
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = _runtime_with_context(
            summarization_model="openai:gpt-5.4-mini",
            model_resolver=ctx_resolver,
        )

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        ctor_resolved.invoke.assert_called_once()
        ctx_resolved.invoke.assert_not_called()


class TestDataclassContext:
    """Dataclass / attribute-style contexts are honored alongside dicts.

    LangGraph documents both `typing.TypedDict` and `dataclasses.dataclass` as
    valid `context_schema` shapes — these tests lock in that the summarization
    middleware reads overrides from either form.
    """

    def test_dataclass_context_model_override(self) -> None:
        """`runtime.context.model` on a dataclass swaps the threshold model."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="override"))

        @dataclass
        class Ctx:
            model: Any = None

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = Ctx(model=main_override)

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        main_override.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_dataclass_context_resolver_injection(self) -> None:
        """Dataclass contexts can inject `model_resolver` for string specs."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        resolved = make_mock_model(summary_response="resolved")
        seen: list[str] = []

        def resolver(spec: str) -> Any:  # noqa: ANN401
            seen.append(spec)
            return resolved

        @dataclass
        class Ctx:
            summarization_model: Any = None
            model_resolver: Any = None

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = Ctx(
            summarization_model="openai:gpt-5.4-mini",
            model_resolver=resolver,
        )

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert seen == ["openai:gpt-5.4-mini"]
        resolved.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_dataclass_context_without_override_keys(self) -> None:
        """A dataclass missing the override keys falls back to construction model."""
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")

        @dataclass
        class Ctx:
            user_id: str = "u1"

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = Ctx()

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        construction_model.invoke.assert_called_once()

    def test_simplenamespace_context_model_override(self) -> None:
        """`SimpleNamespace` (attribute-style, non-dataclass) is honored.

        The ctx-extraction refactor dropped the prior `isinstance(raw, dict)`
        gate, so any attribute-bearing object — Pydantic models,
        `SimpleNamespace`, plain instances — should now route through
        `getattr`. Locks that in.
        """
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="override"))

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = SimpleNamespace(model=main_override)

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        main_override.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_mapping_proxy_context_model_override(self) -> None:
        """`MappingProxyType` (Mapping but not `dict`) is honored.

        `_ctx_get` keys off `isinstance(ctx, Mapping)`, not `dict` — this test
        pins the broader contract a host could rely on (read-only mapping
        views, custom Mapping subclasses).
        """
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        main_override = _as_chat_model(make_mock_model(summary_response="override"))

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = MappingProxyType({"model": main_override})

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        main_override.invoke.assert_called_once()
        construction_model.invoke.assert_not_called()

    def test_ctor_resolver_beats_dataclass_ctx_resolver(self) -> None:
        """Constructor `model_resolver` wins over `ctx.model_resolver` on dataclasses.

        Mirrors the dict-ctx case in `TestModelResolverCallback` to lock
        precedence symmetry across the two context shapes.
        """
        backend = MockBackend()
        construction_model = make_mock_model(summary_response="construction")
        ctor_resolved = make_mock_model(summary_response="ctor")
        ctx_resolved = make_mock_model(summary_response="ctx")

        def ctor_resolver(spec: str) -> Any:  # noqa: ANN401
            assert spec == "openai:gpt-5.4-mini"
            return ctor_resolved

        def ctx_resolver(spec: str) -> Any:  # noqa: ANN401, ARG001  # signature must match `Callable[[str], BaseChatModel]`; spec unused since this branch should never fire.
            return ctx_resolved

        @dataclass
        class Ctx:
            summarization_model: Any = None
            model_resolver: Any = None

        middleware = SummarizationMiddleware(
            model=construction_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            model_resolver=ctor_resolver,
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()
        runtime.context = Ctx(
            summarization_model="openai:gpt-5.4-mini",
            model_resolver=ctx_resolver,
        )

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        ctor_resolved.invoke.assert_called_once()
        ctx_resolved.invoke.assert_not_called()


class TestSummarizationContextExport:
    """`SummarizationContext` is re-exported from the public middleware package."""

    def test_exported_from_middleware_package(self) -> None:
        """Hosts can import `SummarizationContext` from `deepagents.middleware`."""
        # Every documented key is optional, so all three are listed in
        # `__optional_keys__` and none in `__required_keys__`.
        assert SummarizationContext.__optional_keys__ == frozenset({"model", "summarization_model", "model_resolver"})
        assert SummarizationContext.__required_keys__ == frozenset()
