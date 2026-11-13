"""Load model-routing configuration for the LangSmith LLM gateway."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model

from deepagents_evals.routing.models import ModelDetails, ModelPricing

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


def load_gateway_key() -> str:
    """Remove stray characters from `LC_GATEWAY_KEY` and check its format.

    Returns:
        A personal or service key containing only ASCII letters, digits, underscores, and hyphens.

    Raises:
        KeyError: If `LC_GATEWAY_KEY` is not set.
        ValueError: If the cleaned value does not match a LangSmith personal or service key.
    """
    key = re.sub(r"[^A-Za-z0-9_-]", "", os.environ["LC_GATEWAY_KEY"])
    if not re.fullmatch(r"lsv2_(?:pt|sk)_[A-Za-z0-9_-]+", key):
        msg = "LC_GATEWAY_KEY must match a LangSmith personal or service key after cleanup (lsv2_pt_... or lsv2_sk_...)."
        raise ValueError(msg)
    return key


@dataclass(frozen=True)
class RoutingConfig:
    """The models one routing run uses.

    Args:
        anchor: The model passed to `create_deep_agent`; serves unrouted calls (summaries, subagents).
        models: Candidate models in stable index order, independent of price or suitability.
        decision_model: Jev or OpenAI model used to estimate difficulty.
        granularity: Route once per thread or at the start of every turn.
        difficulty_limits: Optional per-rung ceilings on the 0-4 task difficulty scale.
        model_details: Descriptions and USD token prices corresponding to `models`.
    """

    anchor: BaseChatModel
    models: list[BaseChatModel]
    decision_model: str
    granularity: Granularity = "per_turn"
    difficulty_limits: list[float] | None = field(default=None, kw_only=True)
    model_details: list[ModelDetails] | None = field(default=None, kw_only=True)


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
        api_key=load_gateway_key(),
        **spec.get("model_kwargs", {}),
        **({"profile": spec["capabilities"]} if "capabilities" in spec else {}),
    )


def load_config() -> RoutingConfig:
    """Read `ROUTING_CONFIG`, falling back to the packaged default.

    Raises:
        ValueError: If a rung's context window is smaller than the anchor's, which summarization
            budgets against.
    """
    raw = json.loads(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG.read_text())
    specified = ["max_difficulty" in spec for spec in raw["models"]]
    if any(specified) and not all(specified):
        msg = "Set max_difficulty for every model or omit it for all models."
        raise ValueError(msg)
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
        difficulty_limits=[spec["max_difficulty"] for spec in raw["models"]]
        if any("max_difficulty" in spec for spec in raw["models"])
        else None,
        model_details=[
            ModelDetails(
                description=spec.get("description", ""),
                pricing=ModelPricing(**spec["pricing"]) if "pricing" in spec else None,
            )
            for spec in raw["models"]
        ],
    )
