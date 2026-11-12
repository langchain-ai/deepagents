"""Tests for gateway credentials used by model routing."""

import json

import pytest

from deepagents_evals.routing.config import CONFIG_ENV, load_config, load_gateway_key


def test_gateway_key_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "example-key_0123.AB+/="
    monkeypatch.setenv("LC_GATEWAY_KEY", key)

    assert load_gateway_key() == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        "example-key\n\n",
        "example\rkey",
        "example\nkey",
        "example\tkey",
        " example-key",
        "example-key ",
        "example key",
        "example\x01key",
        "example\x7fkey",
        "example-kéy",
        "example\u200bkey",
    ],
)
def test_invalid_gateway_key_is_rejected_without_disclosing_it(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", key)

    with pytest.raises(ValueError, match="LC_GATEWAY_KEY must be") as exc:
        load_gateway_key()

    assert str(exc.value) == (
        "LC_GATEWAY_KEY must be nonempty printable ASCII without whitespace. "
        "Check the configured secret."
    )


def test_missing_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LC_GATEWAY_KEY", raising=False)

    with pytest.raises(KeyError, match="LC_GATEWAY_KEY"):
        load_gateway_key()


def test_routing_config_rejects_bad_key_before_creating_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", "example-key\n\n")
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps(
            {"anchor": {"model": "openai:example"}, "models": [], "decision_model": "example"}
        ),
    )

    with pytest.raises(ValueError, match="LC_GATEWAY_KEY must be"):
        load_config()
