"""Unit tests for runtime-resolved summarization model selection.

Exercises the precedence rules documented on
`_DeepAgentsSummarizationMiddleware._resolve_models`:

- Threshold model: `runtime.context["model"]` then construction-time fallback.
- Summarizer model: `runtime.context["summarization_model"]` then threshold model.
"""

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse

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
    mock_get_config,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


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
        main_override = make_mock_model(summary_response="main")
        summ_override = make_mock_model(summary_response="explicit")

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
        main_override = make_mock_model(summary_response="main")

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
        summ_override = make_mock_model(summary_response="explicit")

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

    def test_threshold_uses_main_model_profile(self) -> None:
        """A bigger-context `model` override raises the fraction-based trigger."""
        backend = MockBackend()
        construction_model = make_mock_model()
        construction_model.profile = {"max_input_tokens": 1000}

        # Bigger context window — fraction-based trigger threshold goes up
        # so the same message bucket falls below the new trigger.
        main_override = make_mock_model(summary_response="main")
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
        summ_override = make_mock_model(summary_response="summ")
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

    def test_string_spec_without_resolver_raises(self) -> None:
        """Missing `model_resolver` with a string spec is a clear error."""
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
        runtime = _runtime_with_context(summarization_model="openai:gpt-5.4-mini")

        with mock_get_config(), pytest.raises(RuntimeError, match="model_resolver"):
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
