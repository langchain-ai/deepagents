"""Tests for model_config module."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.model_config import (
    PROVIDER_API_KEY_ENV,
    ModelConfig,
    ModelConfigError,
    ModelSpec,
    _get_builtin_providers,
    _get_provider_profile_modules,
    _load_provider_profiles,
    clear_caches,
    clear_default_model,
    get_available_models,
    has_provider_credentials,
    is_warning_suppressed,
    save_recent_model,
    suppress_warning,
)


@pytest.fixture(autouse=True)
def _clear_model_caches() -> Iterator[None]:
    """Clear module-level caches before and after each test."""
    clear_caches()
    yield
    clear_caches()


class TestModelSpec:
    """Tests for ModelSpec value type."""

    def test_parse_valid_spec(self) -> None:
        """parse() correctly splits provider:model format."""
        spec = ModelSpec.parse("anthropic:claude-sonnet-4-5")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-sonnet-4-5"

    def test_parse_with_colons_in_model_name(self) -> None:
        """parse() handles model names that contain colons."""
        spec = ModelSpec.parse("custom:model:with:colons")
        assert spec.provider == "custom"
        assert spec.model == "model:with:colons"

    def test_parse_raises_on_invalid_format(self) -> None:
        """parse() raises ValueError when spec lacks colon."""
        with pytest.raises(ValueError, match="must be in provider:model format"):
            ModelSpec.parse("invalid-spec")

    def test_parse_raises_on_empty_string(self) -> None:
        """parse() raises ValueError on empty string."""
        with pytest.raises(ValueError, match="must be in provider:model format"):
            ModelSpec.parse("")

    def test_try_parse_returns_spec_on_success(self) -> None:
        """try_parse() returns ModelSpec for valid input."""
        spec = ModelSpec.try_parse("openai:gpt-4o")
        assert spec is not None
        assert spec.provider == "openai"
        assert spec.model == "gpt-4o"

    def test_try_parse_returns_none_on_failure(self) -> None:
        """try_parse() returns None for invalid input."""
        spec = ModelSpec.try_parse("invalid")
        assert spec is None

    def test_str_returns_provider_model_format(self) -> None:
        """str() returns the spec in provider:model format."""
        spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-5")
        assert str(spec) == "anthropic:claude-sonnet-4-5"

    def test_equality(self) -> None:
        """ModelSpec instances with same values are equal."""
        spec1 = ModelSpec(provider="openai", model="gpt-4o")
        spec2 = ModelSpec.parse("openai:gpt-4o")
        assert spec1 == spec2

    def test_immutable(self) -> None:
        """ModelSpec is immutable (frozen dataclass)."""
        spec = ModelSpec(provider="openai", model="gpt-4o")
        with pytest.raises(AttributeError):
            spec.provider = "anthropic"  # type: ignore[misc]

    def test_validates_empty_provider(self) -> None:
        """ModelSpec raises on empty provider."""
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            ModelSpec(provider="", model="gpt-4o")

    def test_validates_empty_model(self) -> None:
        """ModelSpec raises on empty model."""
        with pytest.raises(ValueError, match="Model cannot be empty"):
            ModelSpec(provider="openai", model="")


class TestHasProviderCredentials:
    """Tests for has_provider_credentials() function."""

    def test_returns_false_for_unknown_provider(self):
        """Returns False for unknown provider."""
        assert has_provider_credentials("unknown") is False

    def test_returns_true_when_env_var_set(self):
        """Returns True when provider env var is set."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            assert has_provider_credentials("anthropic") is True

    def test_returns_false_when_env_var_not_set(self):
        """Returns False when provider env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert has_provider_credentials("anthropic") is False


class TestProviderApiKeyEnv:
    """Tests for PROVIDER_API_KEY_ENV constant."""

    def test_contains_major_providers(self):
        """Contains environment variables for major providers."""
        assert PROVIDER_API_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"
        assert PROVIDER_API_KEY_ENV["azure_openai"] == "AZURE_OPENAI_API_KEY"
        assert PROVIDER_API_KEY_ENV["cohere"] == "COHERE_API_KEY"
        assert PROVIDER_API_KEY_ENV["deepseek"] == "DEEPSEEK_API_KEY"
        assert PROVIDER_API_KEY_ENV["fireworks"] == "FIREWORKS_API_KEY"
        assert PROVIDER_API_KEY_ENV["google_genai"] == "GOOGLE_API_KEY"
        assert PROVIDER_API_KEY_ENV["google_vertexai"] == "GOOGLE_CLOUD_PROJECT"
        assert PROVIDER_API_KEY_ENV["groq"] == "GROQ_API_KEY"
        assert PROVIDER_API_KEY_ENV["huggingface"] == "HUGGINGFACEHUB_API_TOKEN"
        assert PROVIDER_API_KEY_ENV["ibm"] == "WATSONX_APIKEY"
        assert PROVIDER_API_KEY_ENV["mistralai"] == "MISTRAL_API_KEY"
        assert PROVIDER_API_KEY_ENV["nvidia"] == "NVIDIA_API_KEY"
        assert PROVIDER_API_KEY_ENV["openai"] == "OPENAI_API_KEY"
        assert PROVIDER_API_KEY_ENV["openrouter"] == "OPENROUTER_API_KEY"
        assert PROVIDER_API_KEY_ENV["perplexity"] == "PPLX_API_KEY"
        assert PROVIDER_API_KEY_ENV["together"] == "TOGETHER_API_KEY"
        assert PROVIDER_API_KEY_ENV["xai"] == "XAI_API_KEY"


class TestModelConfigLoad:
    """Tests for ModelConfig.load() method."""

    def test_returns_empty_config_when_file_not_exists(self, tmp_path):
        """Returns empty config when file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"
        config = ModelConfig.load(config_path)

        assert config.default_model is None
        assert config.providers == {}

    def test_loads_default_model(self, tmp_path):
        """Loads default model from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "claude-sonnet-4-5"
""")
        config = ModelConfig.load(config_path)

        assert config.default_model == "claude-sonnet-4-5"

    def test_loads_providers(self, tmp_path):
        """Loads provider configurations."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5", "claude-haiku-4-5"]
api_key_env = "ANTHROPIC_API_KEY"

[models.providers.openai]
models = ["gpt-4o"]
api_key_env = "OPENAI_API_KEY"
""")
        config = ModelConfig.load(config_path)

        assert "anthropic" in config.providers
        assert "openai" in config.providers
        assert config.providers["anthropic"]["models"] == [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ]
        assert config.providers["anthropic"]["api_key_env"] == "ANTHROPIC_API_KEY"

    def test_loads_custom_base_url(self, tmp_path):
        """Loads custom base_url for providers."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.local-ollama]
base_url = "http://localhost:11434/v1"
models = ["llama3"]
""")
        config = ModelConfig.load(config_path)

        assert (
            config.providers["local-ollama"]["base_url"] == "http://localhost:11434/v1"
        )

    def test_corrupt_toml_returns_empty_config(self, tmp_path, caplog):
        """Corrupt TOML file returns empty config and logs a warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[invalid toml content")

        with caplog.at_level(logging.WARNING):
            config = ModelConfig.load(config_path)

        assert config.default_model is None
        assert config.providers == {}
        assert any("invalid TOML syntax" in r.message for r in caplog.records)

    def test_unreadable_file_returns_empty_config(self, tmp_path, caplog):
        """Unreadable config file returns empty config and logs a warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[models]\ndefault = 'test'")
        config_path.chmod(0o000)

        try:
            with caplog.at_level(logging.WARNING):
                config = ModelConfig.load(config_path)

            assert config.default_model is None
            assert config.providers == {}
            assert any(
                "Could not read config file" in r.message for r in caplog.records
            )
        finally:
            config_path.chmod(0o644)


class TestModelConfigGetAllModels:
    """Tests for ModelConfig.get_all_models() method."""

    def test_returns_empty_list_when_no_providers(self):
        """Returns empty list when no providers configured."""
        config = ModelConfig()
        assert config.get_all_models() == []

    def test_returns_model_provider_tuples(self, tmp_path):
        """Returns list of (model, provider) tuples."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5", "claude-haiku-4-5"]

[models.providers.openai]
models = ["gpt-4o"]
""")
        config = ModelConfig.load(config_path)
        models = config.get_all_models()

        assert ("claude-sonnet-4-5", "anthropic") in models
        assert ("claude-haiku-4-5", "anthropic") in models
        assert ("gpt-4o", "openai") in models


class TestModelConfigGetProviderForModel:
    """Tests for ModelConfig.get_provider_for_model() method."""

    def test_returns_none_for_unknown_model(self):
        """Returns None for model not in any provider."""
        config = ModelConfig()
        assert config.get_provider_for_model("unknown-model") is None

    def test_returns_provider_name(self, tmp_path):
        """Returns provider name for known model."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)

        assert config.get_provider_for_model("claude-sonnet-4-5") == "anthropic"


class TestModelConfigHasCredentials:
    """Tests for ModelConfig.has_credentials() method."""

    def test_returns_false_for_unknown_provider(self):
        """Returns False for unknown provider."""
        config = ModelConfig()
        assert config.has_credentials("unknown") is False

    def test_returns_none_when_no_key_configured(self, tmp_path):
        """Returns None when api_key_env not specified (unknown status)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.local]
models = ["llama3"]
""")
        config = ModelConfig.load(config_path)

        assert config.has_credentials("local") is None

    def test_returns_true_when_env_var_set(self, tmp_path):
        """Returns True when api_key_env is set in environment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            assert config.has_credentials("anthropic") is True

    def test_returns_false_when_env_var_not_set(self, tmp_path):
        """Returns False when api_key_env not set in environment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)

        with patch.dict("os.environ", {}, clear=True):
            assert config.has_credentials("anthropic") is False


class TestModelConfigGetBaseUrl:
    """Tests for ModelConfig.get_base_url() method."""

    def test_returns_none_for_unknown_provider(self):
        """Returns None for unknown provider."""
        config = ModelConfig()
        assert config.get_base_url("unknown") is None

    def test_returns_none_when_not_configured(self, tmp_path):
        """Returns None when base_url not in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)

        assert config.get_base_url("anthropic") is None

    def test_returns_base_url(self, tmp_path):
        """Returns configured base_url."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.local]
base_url = "http://localhost:11434/v1"
models = ["llama3"]
""")
        config = ModelConfig.load(config_path)

        assert config.get_base_url("local") == "http://localhost:11434/v1"


class TestModelConfigGetApiKeyEnv:
    """Tests for ModelConfig.get_api_key_env() method."""

    def test_returns_none_for_unknown_provider(self):
        """Returns None for unknown provider."""
        config = ModelConfig()
        assert config.get_api_key_env("unknown") is None

    def test_returns_env_var_name(self, tmp_path):
        """Returns configured api_key_env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)

        assert config.get_api_key_env("anthropic") == "ANTHROPIC_API_KEY"


class TestSaveDefaultModel:
    """Tests for save_default_model() function."""

    def test_creates_new_file(self, tmp_path):
        """Creates config file when it doesn't exist."""
        config_path = tmp_path / "config.toml"
        model_config.save_default_model("claude-sonnet-4-5", config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert 'default = "claude-sonnet-4-5"' in content

    def test_updates_existing_default(self, tmp_path):
        """Updates existing default model."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "old-model"

[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        model_config.save_default_model("new-model", config_path)

        content = config_path.read_text()
        assert 'default = "new-model"' in content
        assert "old-model" not in content
        # Should preserve other config
        assert "[models.providers.anthropic]" in content

    def test_adds_default_to_models_section(self, tmp_path):
        """Adds default key to [models] section if missing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        model_config.save_default_model("claude-sonnet-4-5", config_path)

        content = config_path.read_text()
        assert 'default = "claude-sonnet-4-5"' in content

    def test_creates_parent_directory(self, tmp_path):
        """Creates parent directory if needed."""
        config_path = tmp_path / "subdir" / "config.toml"
        model_config.save_default_model("claude-sonnet-4-5", config_path)

        assert config_path.exists()

    def test_saves_provider_model_format(self, tmp_path):
        """Saves model in provider:model format."""
        config_path = tmp_path / "config.toml"
        model_config.save_default_model("anthropic:claude-sonnet-4-5", config_path)

        content = config_path.read_text()
        assert 'default = "anthropic:claude-sonnet-4-5"' in content

    def test_updates_to_provider_model_format(self, tmp_path):
        """Updates from bare model name to provider:model format."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "claude-sonnet-4-5"
""")
        model_config.save_default_model("anthropic:claude-opus-4-5", config_path)

        content = config_path.read_text()
        assert 'default = "anthropic:claude-opus-4-5"' in content
        assert "claude-sonnet-4-5" not in content

    def test_preserves_existing_recent(self, tmp_path):
        """Does not overwrite [models].recent when saving default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
recent = "anthropic:claude-sonnet-4-5"
""")
        model_config.save_default_model("ollama:qwen3:4b", config_path)

        content = config_path.read_text()
        assert 'recent = "anthropic:claude-sonnet-4-5"' in content
        assert 'default = "ollama:qwen3:4b"' in content


class TestClearDefaultModel:
    """Tests for clear_default_model() function."""

    def test_removes_default_key(self, tmp_path):
        """Removes [models].default from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "anthropic:claude-sonnet-4-5"
""")
        result = clear_default_model(config_path)

        assert result is True
        content = config_path.read_text()
        assert "default" not in content

    def test_preserves_recent(self, tmp_path):
        """Does not remove [models].recent when clearing default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "anthropic:claude-sonnet-4-5"
recent = "openai:gpt-5.2"
""")
        clear_default_model(config_path)

        content = config_path.read_text()
        assert "default" not in content
        assert 'recent = "openai:gpt-5.2"' in content

    def test_preserves_providers(self, tmp_path):
        """Does not affect provider configuration."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "anthropic:claude-sonnet-4-5"

[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        clear_default_model(config_path)

        content = config_path.read_text()
        assert "default" not in content
        assert "[models.providers.anthropic]" in content

    def test_noop_when_no_default(self, tmp_path):
        """Returns True when no default is set (nothing to clear)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
recent = "openai:gpt-5.2"
""")
        result = clear_default_model(config_path)

        assert result is True
        content = config_path.read_text()
        assert 'recent = "openai:gpt-5.2"' in content

    def test_noop_when_file_missing(self, tmp_path):
        """Returns True when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"
        result = clear_default_model(config_path)

        assert result is True


class TestModelPersistenceBetweenSessions:
    """Tests for model selection persistence across app sessions.

    These tests verify that when a user switches models using /model command,
    the selection persists when the CLI is restarted (new session).
    """

    def test_saved_model_is_used_when_no_model_specified(self, tmp_path):
        """Recently switched model should be used when CLI starts without --model.

        Steps:
        1. Save a model to config via save_recent_model (simulating /model switch)
        2. Call _get_default_model_spec() without specifying a model
        3. Verify the saved recent model is used
        """
        from deepagents_cli.config import _get_default_model_spec

        # Use a temporary config path
        config_path = tmp_path / ".deepagents" / "config.toml"

        # Step 1: Save model to config (simulating /model anthropic:claude-opus-4-5)
        save_recent_model("anthropic:claude-opus-4-5", config_path)

        # Verify the model was saved
        assert config_path.exists()
        content = config_path.read_text()
        assert 'recent = "anthropic:claude-opus-4-5"' in content

        # Step 2: Patch DEFAULT_CONFIG_PATH and call _get_default_model_spec
        # This simulates starting a new CLI session
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key"},
                clear=False,
            ),
        ):
            # Step 3: Get default model spec - should use saved recent model
            result = _get_default_model_spec()

            assert result == "anthropic:claude-opus-4-5", (
                f"Expected saved model 'anthropic:claude-opus-4-5' but got '{result}'. "
                "The saved model selection is not being loaded from config."
            )

    def test_config_file_default_takes_priority_over_env_detection(self, tmp_path):
        """Config file default model should take priority over env var detection.

        When both a config file default AND API keys are present,
        the config file's default model should be used.
        """
        from deepagents_cli.config import _get_default_model_spec
        from deepagents_cli.model_config import save_default_model

        config_path = tmp_path / ".deepagents" / "config.toml"

        # Save an OpenAI model as default
        save_default_model("openai:gpt-5.2", config_path)

        # Even with Anthropic key set, should use saved OpenAI default
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key", "OPENAI_API_KEY": "test-key"},
                clear=False,
            ),
        ):
            result = _get_default_model_spec()

            # Should use the saved config, not auto-detect from env vars
            assert result == "openai:gpt-5.2", (
                f"Expected config default 'openai:gpt-5.2' but got '{result}'. "
                "Config file default should take priority over env var detection."
            )


class TestGetAvailableModels:
    """Tests for get_available_models() function."""

    def test_returns_discovered_models_when_package_installed(self):
        """Returns discovered models when a provider package is installed."""
        fake_profiles = {
            "claude-sonnet-4-5": {"tool_calling": True},
            "claude-haiku-4-5": {"tool_calling": True},
            "claude-instant": {"tool_calling": False},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with patch(
            "deepagents_cli.model_config._load_provider_profiles",
            side_effect=mock_load,
        ):
            models = get_available_models()

        assert "anthropic" in models
        # Should only include models with tool_calling=True
        assert "claude-sonnet-4-5" in models["anthropic"]
        assert "claude-haiku-4-5" in models["anthropic"]
        assert "claude-instant" not in models["anthropic"]

    def test_logs_debug_on_import_error(self, caplog):
        """Logs debug message when provider package is not installed."""
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            caplog.at_level(logging.DEBUG, logger="deepagents_cli.model_config"),
        ):
            get_available_models()

        assert any(
            "Could not import profiles" in record.message for record in caplog.records
        )


class TestGetAvailableModelsMergesConfig:
    """Tests for get_available_models() merging config-file providers."""

    def test_merges_new_provider_from_config(self, tmp_path):
        """Config-file provider not in profiles gets appended."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["accounts/fireworks/models/llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "fireworks" in models
        assert "accounts/fireworks/models/llama-v3p1-70b" in models["fireworks"]

    def test_merges_new_models_into_existing_provider(self, tmp_path):
        """Config-file models for an existing provider get appended."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-custom-finetune"]
""")
        fake_profiles = {
            "claude-sonnet-4-5": {"tool_calling": True},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "claude-sonnet-4-5" in models["anthropic"]
        assert "claude-custom-finetune" in models["anthropic"]

    def test_does_not_duplicate_existing_models(self, tmp_path):
        """Config-file models already in profiles are not duplicated."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        fake_profiles = {
            "claude-sonnet-4-5": {"tool_calling": True},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert models["anthropic"].count("claude-sonnet-4-5") == 1

    def test_skips_config_provider_with_no_models(self, tmp_path):
        """Config provider with empty models list is not added."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.empty]
api_key_env = "SOME_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "empty" not in models


class TestHasProviderCredentialsFallback:
    """Tests for has_provider_credentials() falling back to ModelConfig."""

    def test_falls_back_to_config_no_key_required(self, tmp_path):
        """Returns None for config provider with no api_key_env (unknown)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert has_provider_credentials("ollama") is None

    def test_falls_back_to_config_with_key_set(self, tmp_path):
        """Returns True for config provider with api_key_env set in env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}),
        ):
            assert has_provider_credentials("fireworks") is True

    def test_falls_back_to_config_with_key_missing(self, tmp_path):
        """Returns False for config provider with api_key_env not in env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.fireworks]
models = ["llama-v3p1-70b"]
api_key_env = "FIREWORKS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert has_provider_credentials("fireworks") is False

    def test_returns_false_for_totally_unknown_provider(self):
        """Returns False for provider not in hardcoded map, config, or langchain."""
        assert has_provider_credentials("nonexistent_provider_xyz") is False

    def test_returns_none_for_langchain_known_provider(self):
        """Returns None for a provider known to langchain but not in config."""
        fake_registry = {
            "ollama": ("langchain_ollama", "ChatOllama", None),
        }
        with patch(
            "deepagents_cli.model_config._get_builtin_providers",
            return_value=fake_registry,
        ):
            assert has_provider_credentials("ollama") is None


class TestModelConfigGetClassPath:
    """Tests for ModelConfig.get_class_path() method."""

    def test_returns_none_for_unknown_provider(self):
        """Returns None for unknown provider."""
        config = ModelConfig()
        assert config.get_class_path("unknown") is None

    def test_returns_none_when_not_configured(self, tmp_path):
        """Returns None when class_path not in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)
        assert config.get_class_path("anthropic") is None

    def test_returns_class_path(self, tmp_path):
        """Returns configured class_path."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
class_path = "my_package.models:MyChatModel"
models = ["my-model"]
""")
        config = ModelConfig.load(config_path)
        assert config.get_class_path("custom") == "my_package.models:MyChatModel"


class TestModelConfigGetKwargs:
    """Tests for ModelConfig.get_kwargs() method."""

    def test_returns_empty_for_unknown_provider(self):
        """Returns empty dict for unknown provider."""
        config = ModelConfig()
        assert config.get_kwargs("unknown") == {}

    def test_returns_empty_when_no_params(self, tmp_path):
        """Returns empty dict when params not in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
""")
        config = ModelConfig.load(config_path)
        assert config.get_kwargs("custom") == {}

    def test_returns_params(self, tmp_path):
        """Returns configured params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]

[models.providers.custom.params]
temperature = 0
max_tokens = 4096
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("custom")
        assert kwargs == {"temperature": 0, "max_tokens": 4096}

    def test_returns_copy(self, tmp_path):
        """Returns a copy, not the original dict."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]

[models.providers.custom.params]
temperature = 0
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("custom")
        kwargs["extra"] = "mutated"
        # Original should not be affected
        assert "extra" not in config.get_kwargs("custom")


class TestModelConfigGetKwargsPerModel:
    """Tests for ModelConfig.get_kwargs() with per-model overrides."""

    def test_model_override_replaces_provider_value(self, tmp_path):
        """Per-model sub-table overrides same key from provider params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b", "llama3"]

[models.providers.ollama.params]
temperature = 0
num_ctx = 8192

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
num_ctx = 4000
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="qwen3:4b")
        assert kwargs == {"temperature": 0.5, "num_ctx": 4000}

    def test_no_override_returns_provider_params(self, tmp_path):
        """Model without sub-table gets provider-level params only."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b", "llama3"]

[models.providers.ollama.params]
temperature = 0
num_ctx = 8192

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="llama3")
        assert kwargs == {"temperature": 0, "num_ctx": 8192}

    def test_model_adds_new_keys(self, tmp_path):
        """Per-model sub-table can introduce keys not in provider params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0

[models.providers.ollama.params."qwen3:4b"]
top_p = 0.9
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="qwen3:4b")
        assert kwargs == {"temperature": 0, "top_p": 0.9}

    def test_shallow_merge(self, tmp_path):
        """Merge is shallow — provider keys not in sub-table are preserved."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0
num_ctx = 8192
seed = 42

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="qwen3:4b")
        assert kwargs == {"temperature": 0.5, "num_ctx": 8192, "seed": 42}

    def test_none_model_name_returns_provider_params(self, tmp_path):
        """model_name=None returns provider params without merging."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name=None)
        assert kwargs == {"temperature": 0}

    def test_returns_copy_with_model_override(self, tmp_path):
        """Returned dict is a copy — mutations don't affect config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params]
temperature = 0

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="qwen3:4b")
        kwargs["injected"] = True
        fresh = config.get_kwargs("ollama", model_name="qwen3:4b")
        assert "injected" not in fresh

    def test_no_provider_params_only_model_subtable(self, tmp_path):
        """Works when provider has no flat params, only model sub-table."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        config = ModelConfig.load(config_path)
        kwargs = config.get_kwargs("ollama", model_name="qwen3:4b")
        assert kwargs == {"temperature": 0.5}


class TestModelConfigValidateParams:
    """Tests for _validate() params warnings."""

    def test_warns_on_unknown_model_in_params(self, tmp_path, caplog):
        """Warns when params sub-table references a model not in models list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            ModelConfig.load(config_path)

        assert any(
            "params for 'qwen3:4b'" in record.message for record in caplog.records
        )

    def test_no_warning_when_model_in_list(self, tmp_path, caplog):
        """No warning when params sub-table references a model in models list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["qwen3:4b"]

[models.providers.ollama.params."qwen3:4b"]
temperature = 0.5
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            ModelConfig.load(config_path)

        assert not any("params for" in record.message for record in caplog.records)

    def test_no_warning_when_no_model_overrides(self, tmp_path, caplog):
        """No warning when params has no model sub-tables."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]

[models.providers.ollama.params]
temperature = 0
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            ModelConfig.load(config_path)

        assert not any("params for" in record.message for record in caplog.records)


class TestModelConfigValidateClassPath:
    """Tests for _validate() class_path validation."""

    def test_warns_on_invalid_class_path_format(self, tmp_path, caplog):
        """Warns when class_path lacks colon separator."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.bad]
class_path = "my_package.MyChatModel"
models = ["my-model"]
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            ModelConfig.load(config_path)

        assert any("invalid class_path" in record.message for record in caplog.records)

    def test_no_warning_on_valid_class_path(self, tmp_path, caplog):
        """No warning when class_path has colon separator."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.good]
class_path = "my_package.models:MyChatModel"
models = ["my-model"]
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            ModelConfig.load(config_path)

        assert not any(
            "invalid class_path" in record.message for record in caplog.records
        )


class TestGetProviderProfileModules:
    """Tests for _get_provider_profile_modules()."""

    def test_builds_from_builtin_providers(self):
        """Derives profile module paths from _BUILTIN_PROVIDERS registry."""
        fake_registry = {
            "anthropic": ("langchain_anthropic", "ChatAnthropic", None),
            "openai": ("langchain_openai", "ChatOpenAI", None),
            "ollama": ("langchain_ollama", "ChatOllama", None),
            "fireworks": ("langchain_fireworks", "ChatFireworks", None),
        }
        with patch(
            "deepagents_cli.model_config._get_builtin_providers",
            return_value=fake_registry,
        ):
            result = _get_provider_profile_modules()

        assert ("anthropic", "langchain_anthropic.data._profiles") in result
        assert ("openai", "langchain_openai.data._profiles") in result
        assert ("ollama", "langchain_ollama.data._profiles") in result
        assert ("fireworks", "langchain_fireworks.data._profiles") in result
        assert len(result) == 4

    def test_handles_submodule_paths(self):
        """Extracts package root from dotted module paths like 'pkg.submodule'."""
        fake_registry = {
            "google_anthropic_vertex": (
                "langchain_google_vertexai.model_garden",
                "ChatAnthropicVertex",
                None,
            ),
        }
        with patch(
            "deepagents_cli.model_config._get_builtin_providers",
            return_value=fake_registry,
        ):
            result = _get_provider_profile_modules()

        assert result == [
            ("google_anthropic_vertex", "langchain_google_vertexai.data._profiles"),
        ]


class TestGetBuiltinProviders:
    """Tests for _get_builtin_providers() forward-compat helper."""

    def test_prefers_builtin_providers(self):
        """Uses _BUILTIN_PROVIDERS when both attributes exist."""
        import langchain.chat_models.base as base_module

        builtin = {"anthropic": ("langchain_anthropic", "ChatAnthropic", None)}
        legacy = {"openai": ("langchain_openai", "ChatOpenAI", None)}

        with (
            patch.object(base_module, "_BUILTIN_PROVIDERS", builtin, create=True),
            patch.object(base_module, "_SUPPORTED_PROVIDERS", legacy, create=True),
        ):
            result = _get_builtin_providers()

        assert result is builtin

    def test_falls_back_to_supported_providers(self):
        """Falls back to _SUPPORTED_PROVIDERS when _BUILTIN_PROVIDERS is absent."""
        import langchain.chat_models.base as base_module

        legacy = {"openai": ("langchain_openai", "ChatOpenAI", None)}

        # Delete _BUILTIN_PROVIDERS if it exists so fallback is exercised
        had_builtin = hasattr(base_module, "_BUILTIN_PROVIDERS")
        if had_builtin:
            saved = base_module._BUILTIN_PROVIDERS
            delattr(base_module, "_BUILTIN_PROVIDERS")

        try:
            with patch.object(base_module, "_SUPPORTED_PROVIDERS", legacy, create=True):
                result = _get_builtin_providers()
            assert result is legacy
        finally:
            if had_builtin:
                base_module._BUILTIN_PROVIDERS = saved

    def test_returns_empty_when_neither_exists(self):
        """Returns empty dict when neither attribute exists."""
        import langchain.chat_models.base as base_module

        # Temporarily remove both attributes
        saved_attrs: dict[str, Any] = {}
        for attr in ("_BUILTIN_PROVIDERS", "_SUPPORTED_PROVIDERS"):
            if hasattr(base_module, attr):
                saved_attrs[attr] = getattr(base_module, attr)
                delattr(base_module, attr)

        try:
            result = _get_builtin_providers()
            assert result == {}
        finally:
            for attr, value in saved_attrs.items():
                setattr(base_module, attr, value)


class TestLoadProviderProfiles:
    """Tests for _load_provider_profiles() direct-file loading."""

    def test_loads_profiles_from_file(self, tmp_path):
        """Loads _PROFILES dict from a standalone .py file."""
        pkg_dir = tmp_path / "fake_provider"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        data_dir = pkg_dir / "data"
        data_dir.mkdir()
        (data_dir / "_profiles.py").write_text(
            '_PROFILES = {"model-a": {"tool_calling": True}}\n'
        )

        fake_spec = type(
            "FakeSpec",
            (),
            {
                "origin": str(pkg_dir / "__init__.py"),
                "submodule_search_locations": None,
            },
        )()
        with patch("importlib.util.find_spec", return_value=fake_spec):
            result = _load_provider_profiles("fake_provider.data._profiles")

        assert result == {"model-a": {"tool_calling": True}}

    def test_raises_import_error_when_package_not_found(self):
        """Raises ImportError when find_spec returns None."""
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(ImportError, match="not installed"),
        ):
            _load_provider_profiles("nonexistent.data._profiles")

    def test_raises_import_error_when_profiles_missing(self, tmp_path):
        """Raises ImportError when _profiles.py doesn't exist on disk."""
        pkg_dir = tmp_path / "fake_provider"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "data").mkdir()
        # No _profiles.py created

        fake_spec = type(
            "FakeSpec",
            (),
            {
                "origin": str(pkg_dir / "__init__.py"),
                "submodule_search_locations": None,
            },
        )()
        with (
            patch("importlib.util.find_spec", return_value=fake_spec),
            pytest.raises(ImportError, match="not found"),
        ):
            _load_provider_profiles("fake_provider.data._profiles")

    def test_returns_empty_dict_when_no_profiles_attr(self, tmp_path):
        """Returns empty dict when the module has no _PROFILES attribute."""
        pkg_dir = tmp_path / "fake_provider"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        data_dir = pkg_dir / "data"
        data_dir.mkdir()
        (data_dir / "_profiles.py").write_text("# no _PROFILES here\n")

        fake_spec = type(
            "FakeSpec",
            (),
            {
                "origin": str(pkg_dir / "__init__.py"),
                "submodule_search_locations": None,
            },
        )()
        with patch("importlib.util.find_spec", return_value=fake_spec):
            result = _load_provider_profiles("fake_provider.data._profiles")

        assert result == {}

    def test_uses_submodule_search_locations_fallback(self, tmp_path):
        """Falls back to submodule_search_locations when origin is None."""
        pkg_dir = tmp_path / "ns_provider"
        pkg_dir.mkdir()
        data_dir = pkg_dir / "data"
        data_dir.mkdir()
        (data_dir / "_profiles.py").write_text(
            '_PROFILES = {"ns-model": {"tool_calling": True}}\n'
        )

        fake_spec = type(
            "FakeSpec",
            (),
            {
                "origin": None,
                "submodule_search_locations": [str(pkg_dir)],
            },
        )()
        with patch("importlib.util.find_spec", return_value=fake_spec):
            result = _load_provider_profiles("ns_provider.data._profiles")

        assert result == {"ns-model": {"tool_calling": True}}


class TestGetAvailableModelsTextIO:
    """Tests for text_inputs / text_outputs filtering in get_available_models()."""

    def test_excludes_model_without_text_inputs(self):
        """Models with text_inputs=False are excluded."""
        fake_profiles = {
            "good-model": {"tool_calling": True},
            "image-only": {"tool_calling": True, "text_inputs": False},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with patch(
            "deepagents_cli.model_config._load_provider_profiles",
            side_effect=mock_load,
        ):
            models = get_available_models()

        assert "good-model" in models["anthropic"]
        assert "image-only" not in models["anthropic"]

    def test_excludes_model_without_text_outputs(self):
        """Models with text_outputs=False are excluded."""
        fake_profiles = {
            "good-model": {"tool_calling": True},
            "embedding-only": {"tool_calling": True, "text_outputs": False},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with patch(
            "deepagents_cli.model_config._load_provider_profiles",
            side_effect=mock_load,
        ):
            models = get_available_models()

        assert "good-model" in models["anthropic"]
        assert "embedding-only" not in models["anthropic"]

    def test_includes_model_with_text_io_true(self):
        """Models with explicit text_inputs=True and text_outputs=True pass."""
        fake_profiles = {
            "explicit-true": {
                "tool_calling": True,
                "text_inputs": True,
                "text_outputs": True,
            },
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with patch(
            "deepagents_cli.model_config._load_provider_profiles",
            side_effect=mock_load,
        ):
            models = get_available_models()

        assert "explicit-true" in models["anthropic"]

    def test_includes_model_without_text_io_fields(self):
        """Models missing text_inputs/text_outputs fields default to included."""
        fake_profiles = {
            "no-fields": {"tool_calling": True},
        }

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_anthropic.data._profiles":
                return fake_profiles
            msg = "not installed"
            raise ImportError(msg)

        with patch(
            "deepagents_cli.model_config._load_provider_profiles",
            side_effect=mock_load,
        ):
            models = get_available_models()

        assert "no-fields" in models["anthropic"]


class TestModelConfigError:
    """Tests for ModelConfigError exception class."""

    def test_is_exception(self):
        """ModelConfigError is an Exception subclass."""
        assert issubclass(ModelConfigError, Exception)

    def test_carries_message(self):
        """ModelConfigError carries the error message."""
        err = ModelConfigError("test error message")
        assert str(err) == "test error message"


class TestSaveRecentModel:
    """Tests for save_recent_model() function."""

    def test_creates_new_file(self, tmp_path):
        """Creates config file when it doesn't exist."""
        config_path = tmp_path / "config.toml"
        save_recent_model("anthropic:claude-sonnet-4-5", config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert 'recent = "anthropic:claude-sonnet-4-5"' in content

    def test_updates_existing_recent(self, tmp_path):
        """Updates existing recent model."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
recent = "old-model"

[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        save_recent_model("new-model", config_path)

        content = config_path.read_text()
        assert 'recent = "new-model"' in content
        assert "old-model" not in content
        # Should preserve other config
        assert "[models.providers.anthropic]" in content

    def test_preserves_existing_default(self, tmp_path):
        """Does not overwrite [models].default when saving recent."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "ollama:qwen3:4b"
""")
        save_recent_model("anthropic:claude-sonnet-4-5", config_path)

        content = config_path.read_text()
        assert 'default = "ollama:qwen3:4b"' in content
        assert 'recent = "anthropic:claude-sonnet-4-5"' in content

    def test_creates_parent_directory(self, tmp_path):
        """Creates parent directory if needed."""
        config_path = tmp_path / "subdir" / "config.toml"
        save_recent_model("anthropic:claude-sonnet-4-5", config_path)

        assert config_path.exists()


class TestModelConfigLoadRecent:
    """Tests for ModelConfig.load() reading recent_model."""

    def test_loads_recent_model(self, tmp_path):
        """Loads recent model from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
recent = "anthropic:claude-sonnet-4-5"
""")
        config = ModelConfig.load(config_path)

        assert config.recent_model == "anthropic:claude-sonnet-4-5"

    def test_recent_model_none_when_absent(self, tmp_path):
        """recent_model is None when [models].recent key is absent."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "anthropic:claude-sonnet-4-5"
""")
        config = ModelConfig.load(config_path)

        assert config.recent_model is None

    def test_loads_both_default_and_recent(self, tmp_path):
        """Loads both default_model and recent_model from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "ollama:qwen3:4b"
recent = "anthropic:claude-sonnet-4-5"
""")
        config = ModelConfig.load(config_path)

        assert config.default_model == "ollama:qwen3:4b"
        assert config.recent_model == "anthropic:claude-sonnet-4-5"


class TestModelPrecedenceOrder:
    """Tests for model selection precedence: default > recent > env."""

    def test_default_takes_priority_over_recent(self, tmp_path):
        """[models].default takes priority over [models].recent."""
        from deepagents_cli.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
default = "ollama:qwen3:4b"
recent = "anthropic:claude-sonnet-4-5"
""")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key"},
                clear=False,
            ),
        ):
            result = _get_default_model_spec()

        assert result == "ollama:qwen3:4b"

    def test_recent_takes_priority_over_env(self, tmp_path):
        """[models].recent takes priority over env var auto-detection."""
        from deepagents_cli.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models]
recent = "openai:gpt-5.2"
""")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key"},
                clear=False,
            ),
        ):
            result = _get_default_model_spec()

        assert result == "openai:gpt-5.2"

    def test_env_used_when_neither_set(self, tmp_path):
        """Falls back to env var auto-detection when neither default nor recent set."""
        from deepagents_cli.config import _get_default_model_spec, settings

        config_path = tmp_path / "config.toml"
        config_path.write_text("")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.object(settings, "openai_api_key", None),
            patch.object(settings, "anthropic_api_key", "test-key"),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key"},
                clear=False,
            ),
        ):
            result = _get_default_model_spec()

        assert result == "anthropic:claude-sonnet-4-5-20250929"


class TestIsWarningSuppressed:
    """Tests for is_warning_suppressed() function."""

    def test_returns_true_when_key_present(self, tmp_path) -> None:
        """Returns True when key is in [warnings].suppress list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        assert is_warning_suppressed("ripgrep", config_path) is True

    def test_returns_false_when_key_absent(self, tmp_path) -> None:
        """Returns False when key is not in [warnings].suppress list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["other"]\n')

        assert is_warning_suppressed("ripgrep", config_path) is False

    def test_returns_false_when_file_missing(self, tmp_path) -> None:
        """Returns False when config file does not exist."""
        config_path = tmp_path / "nonexistent.toml"

        assert is_warning_suppressed("ripgrep", config_path) is False

    def test_returns_false_on_corrupt_toml(self, tmp_path) -> None:
        """Returns False when config file has invalid TOML."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[invalid toml")

        assert is_warning_suppressed("ripgrep", config_path) is False

    def test_returns_false_when_no_warnings_section(self, tmp_path) -> None:
        """Returns False when config has no [warnings] section."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "some:model"\n')

        assert is_warning_suppressed("ripgrep", config_path) is False


class TestSuppressWarning:
    """Tests for suppress_warning() function."""

    def test_creates_file_with_key(self, tmp_path) -> None:
        """Creates config file with [warnings].suppress list."""
        config_path = tmp_path / "config.toml"

        result = suppress_warning("ripgrep", config_path)

        assert result is True
        assert config_path.exists()
        assert is_warning_suppressed("ripgrep", config_path) is True

    def test_adds_to_existing_list(self, tmp_path) -> None:
        """Adds key to existing [warnings].suppress list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["other"]\n')

        result = suppress_warning("ripgrep", config_path)

        assert result is True
        assert is_warning_suppressed("other", config_path) is True
        assert is_warning_suppressed("ripgrep", config_path) is True

    def test_deduplicates(self, tmp_path) -> None:
        """Does not add duplicate entries."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        suppress_warning("ripgrep", config_path)

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["warnings"]["suppress"].count("ripgrep") == 1

    def test_preserves_other_config(self, tmp_path) -> None:
        """Preserves existing config sections when adding suppression."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "some:model"\n')

        suppress_warning("ripgrep", config_path)

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["models"]["default"] == "some:model"
        assert "ripgrep" in data["warnings"]["suppress"]
