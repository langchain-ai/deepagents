"""Tests for gateway credentials used by model routing."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from deepagents_evals.routing import middleware as routing_config
from deepagents_evals.routing.middleware import (
    CONFIG_ENV,
    ModelPricing,
    load_config,
    load_gateway_key,
)


@pytest.mark.parametrize("kind", ["pt", "sk"])
def test_gateway_key_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    key = f"lsv2_{kind}_example-key_0123"
    monkeypatch.setenv("LC_GATEWAY_KEY", key)

    assert load_gateway_key() == key


@pytest.mark.parametrize(
    "character",
    [
        " ",
        "\n",
        "\r",
        "\t",
        "\x01",
        "\x7f",
        "é",
        "\u200b",
        "\ufeff",
        '"',
        "'",
        ":",
        "/",
        "\\",
        "=",
        ".",
    ],
)
def test_gateway_key_strips_invalid_characters(
    monkeypatch: pytest.MonkeyPatch, character: str
) -> None:
    key = "lsv2_pt_example-key_0123"
    corrupted = character + key[:6] + character + key[6:] + character
    monkeypatch.setenv("LC_GATEWAY_KEY", corrupted)

    assert load_gateway_key() == key


def test_gateway_key_strips_trailing_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "lsv2_pt_example-key_0123"
    monkeypatch.setenv("LC_GATEWAY_KEY", key + "\n\n")

    assert load_gateway_key() == key


@pytest.mark.parametrize(
    "key",
    ["", "\n\t\u200b", "example-key", "sk-example-key", "lsv2_pt_", "lsv2_sk_", "lsv2_tr_example"],
)
def test_invalid_gateway_key_is_rejected_without_disclosing_it(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", key)

    with pytest.raises(ValueError, match="LC_GATEWAY_KEY must match") as exc:
        load_gateway_key()

    assert str(exc.value) == (
        "LC_GATEWAY_KEY must match a LangSmith personal or service key after cleanup "
        "(lsv2_pt_... or lsv2_sk_...)."
    )


def test_missing_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LC_GATEWAY_KEY", raising=False)

    with pytest.raises(KeyError, match="LC_GATEWAY_KEY"):
        load_gateway_key()


def test_routing_config_rejects_bad_key_before_creating_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", "invalid-key\n\n")
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps(
            {"anchor": {"model": "openai:example"}, "models": [], "decision_model": "example"}
        ),
    )

    with pytest.raises(ValueError, match="LC_GATEWAY_KEY must match"):
        load_config()


@pytest.mark.parametrize("limits", [None, [2, 4]])
def test_routing_config_loads_difficulty_limits(
    monkeypatch: pytest.MonkeyPatch, limits: list[float] | None
) -> None:
    models: list[dict[str, str | float]] = [{"model": "openai:example"} for _ in range(2)]
    if limits is not None:
        for model, limit in zip(models, limits, strict=True):
            model["max_difficulty"] = limit
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps({"anchor": models[-1], "models": models, "decision_model": "jev-1.14.0"}),
    )
    monkeypatch.setattr(
        routing_config, "_gateway_model", lambda _spec: FakeMessagesListChatModel(responses=[])
    )

    assert load_config().difficulty_limits == limits


def test_routing_config_rejects_partial_difficulty_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps(
            {
                "anchor": {"model": "openai:example"},
                "models": [
                    {"model": "openai:example", "max_difficulty": 2},
                    {"model": "openai:example"},
                ],
                "decision_model": "jev-1.14.0",
            }
        ),
    )

    with pytest.raises(ValueError, match="Set max_difficulty for every model"):
        load_config()


def test_config_loads_descriptions_prices_and_unordered_suitability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        {
            "model": "openai:example",
            "description": "More capable",
            "max_difficulty": 4,
            "pricing": {"input": 1, "output": 5, "cache_read": 0.1},
        },
        {
            "model": "anthropic:example",
            "description": "Less capable",
            "max_difficulty": 2,
            "pricing": {"input": 3, "output": 15, "cache_write": 3.75},
        },
    ]
    monkeypatch.setenv(
        CONFIG_ENV, json.dumps({"anchor": specs[0], "models": specs, "decision_model": "example"})
    )
    monkeypatch.setattr(
        routing_config, "_gateway_model", lambda _spec: FakeMessagesListChatModel(responses=[])
    )

    config = load_config()

    assert config.difficulty_limits == [4, 2]
    assert config.model_details is not None
    assert [details.description for details in config.model_details] == [
        "More capable",
        "Less capable",
    ]
    pricing = config.model_details[0].pricing
    assert pricing is not None
    assert pricing.estimate(1000000, 0, 1000000, warm=True) == pytest.approx(5.1)


def test_config_supplies_capabilities_to_the_gateway_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", "lsv2_pt_example")
    monkeypatch.setattr(
        routing_config,
        "init_chat_model",
        lambda _model, **kwargs: FakeMessagesListChatModel(responses=[], profile=kwargs["profile"]),
    )

    model = routing_config._gateway_model(
        {
            "model": "openai:example",
            "capabilities": {"image_inputs": False, "max_input_tokens": 4000},
        }
    )

    assert model.profile == {"image_inputs": False, "max_input_tokens": 4000}


def test_benchmark_defaults_have_usable_routing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.setenv("LC_GATEWAY_KEY", "lsv2_pt_example")
    config = load_config()

    assert config.model_details is not None
    assert len(config.models) == len(config.model_details)
    assert all(
        details.description and isinstance(details.pricing, ModelPricing)
        for details in config.model_details
    )
    assert all((model.profile or {}).get("tool_calling") is True for model in config.models)
