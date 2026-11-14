"""Route metadata stays consistent across the catalog and model construction."""

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from deepagents_code import config, config_manifest, model_catalog, model_config
from deepagents_code._model_profiles import merge_model_profile
from deepagents_code.config import UNICODE_GLYPHS
from deepagents_code.model_config import ModelConfig, ModelProfileEntry
from deepagents_code.reasoning_effort import (
    default_effort_for_model,
    supported_efforts_for_model,
)
from deepagents_code.tui.widgets.model_selector import ModelSelectorScreen

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", path)
    model_config.clear_caches()
    yield path
    model_config.clear_caches()


def test_profileless_installed_provider_keeps_recommendations_and_policy(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = "ollama:mistral-large-4:cloud"
    monkeypatch.setattr(model_catalog, "get_available_models", dict)
    monkeypatch.setattr(
        config_manifest, "is_provider_package_installed", lambda _: True
    )
    catalog = model_catalog.load_model_catalog(recommended_models=[spec])
    assert catalog.models == [spec]
    profile = catalog.profiles[spec].profile
    assert profile["structured_output"] is False
    assert profile["max_output_tokens"] is None
    assert "Cloud" in str(profile["profile_note"])
    assert catalog.providers["ollama"].install_extra is None

    isolated_config.write_text(
        '[models]\nallowed = ["openai:custom"]\n', encoding="utf-8"
    )
    model_config.clear_caches()
    assert model_catalog.load_model_catalog(recommended_models=[spec]).models == []
    isolated_config.write_text(
        "[models.providers.ollama]\nenabled = false\n", encoding="utf-8"
    )
    model_config.clear_caches()
    catalog = model_catalog.load_model_catalog(recommended_models=[spec])
    assert catalog.models == []
    assert "ollama" not in catalog.providers


@pytest.mark.parametrize(
    ("spec", "image", "schema"),
    [
        ("baseten:zai-org/GLM-5.3", True, True),
        ("fireworks:accounts/fireworks/models/glm-5p3", False, True),
        ("openrouter:z-ai/glm-5.3", False, None),
        ("ollama:glm-5.3:cloud", False, False),
        ("baseten:meta-models/muse-spark-open", True, False),
    ],
)
def test_host_capabilities_do_not_inherit_family_claims(
    spec: str, image: bool, schema: bool | None
) -> None:
    profile = merge_model_profile(
        spec, {"image_inputs": True, "structured_output": True, "max_output_tokens": 42}
    )
    assert profile["image_inputs"] is image
    assert profile["structured_output"] is schema
    assert profile["max_output_tokens"] is None


@pytest.mark.parametrize(
    ("spec", "total", "budget", "output"),
    [
        ("openai:gpt-6.1-terra", 1_050_000, 922_000, 128_000),
        ("google_genai:gemini-4-carbon", 1_048_576, 983_040, 65_536),
    ],
)
def test_shared_context_reserves_output(
    spec: str, total: int, budget: int, output: int
) -> None:
    profile = merge_model_profile(spec, {"max_input_tokens": total})
    assert profile["context_window"] == total
    assert profile["max_input_tokens"] == budget
    assert profile["max_output_tokens"] == output
    assert budget + output == total


def test_codex_catalog_does_not_copy_api_capabilities() -> None:
    profiles = model_config.get_model_profiles()
    api = profiles["openai:gpt-6.1-terra"]["profile"]
    oauth = profiles["openai_codex:gpt-6.1-terra"]["profile"]
    assert api["max_input_tokens"] == 922_000
    assert oauth["max_input_tokens"] is None
    assert oauth["max_output_tokens"] is None
    assert oauth["structured_output"] is None
    assert oauth["image_inputs"] is None
    assert supported_efforts_for_model("openai_codex:gpt-6.1-terra") == ()
    assert (
        profiles["openai_codex:gpt-6.1-sol"]["profile"]["max_input_tokens"] == 500_000
    )
    assert "openai_codex:gpt-6-sol" in profiles


def test_unknown_effort_drops_inherited_claims_without_warnings(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = "openrouter:google/gemini-4-hydrogen"
    monkeypatch.setattr(
        model_config,
        "_load_provider_profiles",
        lambda _: {
            "google/gemini-4-hydrogen": {
                "reasoning_output": True,
                "reasoning_effort_levels": ["xhigh"],
                "reasoning_effort_default": "xhigh",
            }
        },
    )
    assert supported_efforts_for_model(spec) == ()
    assert default_effort_for_model(spec) is None
    assert not caplog.records
    override = {
        "reasoning_effort_levels": ["high"],
        "reasoning_effort_default": "high",
    }
    assert supported_efforts_for_model(spec, cli_override=override) == ("high",)
    assert default_effort_for_model(spec, cli_override=override) == "high"


def test_catalog_runtime_and_user_profile_precedence(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_config.write_text(
        '[models.providers.openai.profile."gpt-6.1-terra"]\n'
        "max_input_tokens = 600000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    monkeypatch.setattr(
        config_manifest, "is_provider_package_installed", lambda _: True
    )
    monkeypatch.setattr(model_catalog, "get_available_models", dict)
    spec = "openai:gpt-6.1-terra"
    constructor = {"api_key": "test-key", "profile": {"max_input_tokens": 250_000}}
    result = config.create_model(spec, extra_kwargs=constructor)
    catalog = model_catalog.load_model_catalog(recommended_models=[spec])
    assert (
        result.context_limit
        == catalog.profiles[spec].profile["max_input_tokens"]
        == 600_000
    )
    assert catalog.profiles[spec].overridden_keys == ["max_input_tokens"]
    override: dict[str, object] = {"max_input_tokens": 500_000}
    result = config.create_model(
        spec, extra_kwargs=constructor, profile_overrides=override
    )
    catalog = model_catalog.load_model_catalog(
        recommended_models=[spec], profile_overrides=override
    )
    assert (
        result.context_limit
        == catalog.profiles[spec].profile["max_input_tokens"]
        == 500_000
    )
    assert catalog.profiles[spec].profile["context_window"] == 1_050_000


def test_codex_runtime_discards_api_metadata_but_keeps_explicit_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.integrations import openai_codex

    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    model = Mock()
    model.profile = {"max_input_tokens": 1_050_000, "api_only_claim": True}
    monkeypatch.setattr(
        openai_codex, "build_chat_model", lambda *_args, **_kwargs: model
    )
    result = config.create_model("openai_codex:gpt-6.1-terra")
    assert result.context_limit is None
    assert "api_only_claim" not in model.profile
    assert model.profile["structured_output"] is None
    result = config.create_model(
        "openai_codex:gpt-6.1-terra",
        extra_kwargs={"profile": {"max_input_tokens": 200_000}},
    )
    assert result.context_limit == 200_000


def test_custom_class_profile_is_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    custom = ModelConfig(
        providers={
            "openai": {"class_path": "custom:ChatModel", "models": ["gpt-6.1-terra"]}
        }
    )
    monkeypatch.setattr(ModelConfig, "load", lambda: custom)
    model = Mock()
    model.profile = {"max_input_tokens": 1234}
    monkeypatch.setattr(config, "_create_model_from_class", lambda *_args: model)
    monkeypatch.setattr(model_config, "has_provider_credentials", lambda _: True)
    assert config.create_model("openai:gpt-6.1-terra").context_limit == 1234


def test_catalog_serializes_only_safe_scalar_presentation_fields() -> None:
    entry = ModelProfileEntry(
        profile={
            "context_window": 1_050_000,
            "profile_note": "Provider caveat",
            "source": "https://example.com/model",
            "reasoning_effort_levels": ["high"],
            "api_key": "not-for-presentation",
            "base_url": "https://private.example.com",
        },
        overridden_keys=frozenset({"api_key", "context_window"}),
    )
    public = model_catalog._profile_entry(entry)
    assert set(public.profile) == {"context_window", "profile_note", "source"}
    assert public.overridden_keys == ["context_window"]


def test_footer_distinguishes_total_budget_unknowns_and_overrides() -> None:
    entry = ModelProfileEntry(
        profile={
            "context_window": 1_050_000,
            "max_input_tokens": 500_000,
            "structured_output": None,
            "tool_calling": True,
            "profile_note": "Backend limits vary",
        },
        overridden_keys=frozenset({"max_input_tokens"}),
    )
    footer = str(ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS))
    assert "1.1M total" in footer
    assert "*500.0K in" in footer
    assert "override" in footer
    assert "Backend limits vary" in footer
    assert "tool calling" in footer
    assert "structured output" not in footer


def test_compatibility_label_survives_upstream_display_name() -> None:
    screen = ModelSelectorScreen(default_scope=None)
    spec = "google_genai:gemini-3.8-flash"
    screen._profiles = {
        spec: ModelProfileEntry(
            profile={"name": "Upstream name"}, overridden_keys=frozenset()
        )
    }
    assert "previous generation" in screen._get_model_display_name(spec)
    assert screen._get_model_status(spec) is None
