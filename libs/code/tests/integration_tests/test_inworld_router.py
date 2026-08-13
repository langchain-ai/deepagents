"""Live tests for the Inworld LLM router.

Skipped unless `INWORLD_API_KEY` is set. These check what the unit tests can
only assume: that catalog discovery, the model ids it reports, and the
completions endpoint agree with each other.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from openai import RateLimitError

from deepagents_code import inworld_catalog
from deepagents_code.config import create_model
from deepagents_code.model_config import clear_caches, get_available_models
from deepagents_code.reasoning_effort import (
    supported_efforts_for_model,
    with_effort_model_params,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

pytestmark = pytest.mark.skipif(
    not os.environ.get("INWORLD_API_KEY"),
    reason="INWORLD_API_KEY not configured",
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Discard cached discovery so each test exercises a real fetch."""
    clear_caches()


def _api_key() -> str:
    """Return the configured credential."""
    return os.environ["INWORLD_API_KEY"]


def _invoke(model: BaseChatModel, prompt: str) -> BaseMessage | None:
    """Invoke `model`, reporting upstream capacity exhaustion as `None`.

    A saturated upstream returns `429 engine_overloaded`, which says nothing
    about this integration. Every other error still propagates.
    """
    try:
        return model.invoke(prompt)
    except RateLimitError:
        return None


def test_catalog_reports_tool_calling_models() -> None:
    """The live catalog parses into tool-calling model profiles."""
    catalog = inworld_catalog.model_profiles(None, _api_key())

    assert len(catalog) > 1, "discovery fell back to the router-only catalog"
    assert any(profile.get("tool_calling") for profile in catalog.values())
    assert all(
        "/" in model for model in catalog if model != inworld_catalog.ROUTER_MODEL
    )


def test_discovered_models_reach_the_switcher() -> None:
    """Discovered models appear in the available-model list."""
    models = get_available_models()["inworld"]

    assert inworld_catalog.ROUTER_MODEL in models
    assert len(models) > 1


def test_router_model_completes_a_tool_call() -> None:
    """The router model completes a tool call."""

    def get_weather(city: str) -> str:  # noqa: ARG001  # bound as a tool schema
        """Get the weather for a city."""
        return "sunny"

    result = create_model(f"inworld:{inworld_catalog.ROUTER_MODEL}")
    response = result.model.bind_tools([get_weather]).invoke(
        "What is the weather in Paris? Use the tool."
    )

    assert [call["name"] for call in response.tool_calls] == ["get_weather"]


def test_discovered_model_completes_with_profile_applied() -> None:
    """A discovered model reaches the router and carries its context limit.

    Covers one model per upstream provider in the catalog.
    """
    catalog = inworld_catalog.model_profiles(None, _api_key())
    by_upstream: dict[str, str] = {}
    for model, profile in catalog.items():
        if model == inworld_catalog.ROUTER_MODEL or not profile.get("tool_calling"):
            continue
        by_upstream.setdefault(model.split("/", 1)[0], model)

    assert by_upstream, "catalog reported no tool-calling models"

    completed: list[str] = []
    for model in by_upstream.values():
        result = create_model(f"inworld:{model}")

        assert result.context_limit == catalog[model].get("max_input_tokens")

        response = _invoke(result.model, "Reply with the single word OK.")
        if response is None:
            continue
        assert response.content, f"{model} returned an empty completion"
        completed.append(model)

    assert completed, "every upstream provider was capacity-limited"


def test_reasoning_models_offer_selectable_effort_levels() -> None:
    """A live reasoning model exposes effort levels to the picker."""
    catalog = inworld_catalog.model_profiles(None, _api_key())
    reasoning_models = [
        model
        for model, profile in catalog.items()
        if profile.get("reasoning_output") and profile.get("tool_calling")
    ]

    assert reasoning_models, "catalog reported no reasoning models"

    spec = f"inworld:{reasoning_models[0]}"
    levels = supported_efforts_for_model(spec)

    assert levels, f"{spec} advertises reasoning but offers no effort levels"
    assert all(level.islower() for level in levels)


def test_selected_effort_reaches_the_router() -> None:
    """A selected effort reaches the router without error."""
    spec = "inworld:openai/gpt-5.2"
    params = with_effort_model_params(spec, None, "low")
    result = create_model(spec, extra_kwargs=dict(params))

    response = _invoke(result.model, "Reply with the single word OK.")
    if response is None:
        pytest.skip(f"{spec} is capacity-limited upstream")

    assert response.content


def test_streaming_yields_incremental_chunks() -> None:
    """Streaming yields more than one chunk."""
    result = create_model(f"inworld:{inworld_catalog.ROUTER_MODEL}")

    chunks = list(result.model.stream("Count from 1 to 5."))

    assert len(chunks) > 1
