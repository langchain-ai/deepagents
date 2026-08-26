"""Tests for mutable runtime model state."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from deepagents_code.config import ModelResult, Settings
from deepagents_code.runtime_state import RuntimeState, get_runtime_state


def test_runtime_state_defaults() -> None:
    state = RuntimeState()

    assert state.model_name is None
    assert state.model_provider is None
    assert state.model_context_limit is None
    assert state.model_unsupported_modalities == frozenset()


def test_runtime_state_is_singleton_across_threads() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(lambda _: get_runtime_state(), range(8)))

    assert all(state is states[0] for state in states)


def test_settings_model_properties_delegate_to_runtime_state() -> None:
    state = get_runtime_state()
    settings = Settings(
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
        nvidia_api_key=None,
        tavily_api_key=None,
        google_cloud_project=None,
        google_cloud_location=None,
        deepagents_langchain_project=None,
        user_langchain_project=None,
    )
    previous = state.model_name

    try:
        settings.model_name = "model"
        assert state.model_name == "model"
        assert settings.model_name == "model"
    finally:
        state.model_name = previous


def test_model_result_updates_runtime_state() -> None:
    state = get_runtime_state()
    previous = RuntimeState(
        model_name=state.model_name,
        model_provider=state.model_provider,
        model_context_limit=state.model_context_limit,
        model_unsupported_modalities=state.model_unsupported_modalities,
    )
    result = ModelResult(
        model=Mock(),
        model_name="model",
        provider="provider",
        context_limit=128_000,
        unsupported_modalities=frozenset({"audio"}),
    )

    try:
        result.apply_to_settings()
        assert state.model_name == "model"
        assert state.model_provider == "provider"
        assert state.model_context_limit == 128_000
        assert state.model_unsupported_modalities == frozenset({"audio"})
    finally:
        state.model_name = previous.model_name
        state.model_provider = previous.model_provider
        state.model_context_limit = previous.model_context_limit
        state.model_unsupported_modalities = previous.model_unsupported_modalities
