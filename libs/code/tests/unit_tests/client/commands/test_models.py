"""Tests for the model catalog CLI command."""

from __future__ import annotations

import argparse
import json

from deepagents_code.client.commands.models import (
    build_model_catalog,
    run_models_command,
)
from deepagents_code.model_config import (
    ModelConfig,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
)


def test_build_model_catalog_includes_models_defaults_and_auth(monkeypatch) -> None:
    """The catalog exposes model metadata without credential values."""
    monkeypatch.setattr(
        "deepagents_code.model_config.get_available_models",
        lambda: {"openai": ["gpt-one", "gpt-two"], "anthropic": ["claude-one"]},
    )
    monkeypatch.setattr(
        ModelConfig,
        "load",
        classmethod(
            lambda _cls: ModelConfig(
                default_model="openai:gpt-two",
                recent_model="anthropic:claude-one",
            )
        ),
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.get_provider_auth_status",
        lambda provider: ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider=provider,
            source=ProviderAuthSource.ENV,
            env_var=f"{provider.upper()}_API_KEY",
            detail="configured",
        ),
    )

    assert build_model_catalog() == {
        "default_model": "openai:gpt-two",
        "recent_model": "anthropic:claude-one",
        "providers": [
            {
                "id": "openai",
                "models": ["gpt-one", "gpt-two"],
                "auth": {
                    "state": "configured",
                    "source": "env",
                    "env_var": "OPENAI_API_KEY",
                    "detail": "configured",
                },
            },
            {
                "id": "anthropic",
                "models": ["claude-one"],
                "auth": {
                    "state": "configured",
                    "source": "env",
                    "env_var": "ANTHROPIC_API_KEY",
                    "detail": "configured",
                },
            },
        ],
    }


def test_run_models_command_writes_json_envelope(monkeypatch, capsys) -> None:
    """JSON mode uses the stable CLI envelope."""
    monkeypatch.setattr(
        "deepagents_code.client.commands.models.build_model_catalog",
        lambda: {"default_model": None, "recent_model": None, "providers": []},
    )

    assert run_models_command(argparse.Namespace(output_format="json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "models"
    assert payload["data"]["providers"] == []
