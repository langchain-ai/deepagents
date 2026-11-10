"""Load model-routing configuration for the LangSmith LLM gateway."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.language_models import BaseChatModel

    from deepagents_evals.routing.middleware import Granularity

GATEWAY_ANTHROPIC_URL = "https://gateway.smith.langchain.com/anthropic"
GATEWAY_OPENAI_URL = "https://gateway.smith.langchain.com/openai/v1"

DEFAULT_CONFIG = Path(__file__).with_name("default_routing.json")
CONFIG_ENV = "ROUTING_CONFIG"
"""Env var holding a routing config as JSON, to run another ladder without editing the packaged default."""

GATEWAY_URLS = {"anthropic": GATEWAY_ANTHROPIC_URL, "openai": GATEWAY_OPENAI_URL}


@dataclass(frozen=True)
class RoutingConfig:
    """The models one routing run uses.

    Args:
        anchor: The model passed to `create_deep_agent`; serves unrouted calls (summaries, subagents).
        models: The ladder, cheapest first.
        decision_model: OpenAI model the Decisions API estimates difficulty with.
        granularity: Route once per thread or at the start of every turn.
    """

    anchor: BaseChatModel
    models: list[BaseChatModel]
    decision_model: str
    granularity: Granularity = "per_turn"


def _gateway_model(spec: Mapping[str, Any]) -> BaseChatModel:
    """A config entry's `model` (`provider:model`) reached through the gateway, built with its `model_kwargs`.

    Settings such as effort stay on the instance to preserve each rung's cached prefix.

    Raises:
        ValueError: If the provider has no gateway route.
    """
    model = spec["model"]
    provider = model.partition(":")[0]
    if provider not in GATEWAY_URLS:
        msg = f"no gateway route for provider {provider!r} in {model!r}; expected one of {sorted(GATEWAY_URLS)}"
        raise ValueError(msg)
    return init_chat_model(
        model,
        base_url=GATEWAY_URLS[provider],
        api_key=os.environ["LC_GATEWAY_KEY"],
        **spec.get("model_kwargs", {}),
    )


def load_config() -> RoutingConfig:
    """Read `ROUTING_CONFIG`, falling back to the packaged default.

    Raises:
        ValueError: If a rung's context window is smaller than the anchor's, which summarization
            budgets against.
    """
    raw = json.loads(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG.read_text())
    anchor = _gateway_model(raw["anchor"])
    models = [_gateway_model(spec) for spec in raw["models"]]
    window = (anchor.profile or {}).get("max_input_tokens") or 0
    small = [
        spec["model"]
        for spec, model in zip(raw["models"], models, strict=True)
        if (model.profile or {}).get("max_input_tokens", window) < window
    ]
    if small:
        msg = f"rungs {small} have a smaller context window than the anchor; summarization would overflow them"
        raise ValueError(msg)
    return RoutingConfig(
        anchor=anchor,
        models=models,
        decision_model=raw["decision_model"],
        granularity=raw.get("granularity", "per_turn"),
    )
