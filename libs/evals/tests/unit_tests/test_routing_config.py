"""Tests for gateway credentials used by model routing."""

import json

import pytest

from deepagents_evals.routing.config import CONFIG_ENV, load_config, load_gateway_key


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
