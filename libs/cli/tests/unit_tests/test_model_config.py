"""Tests for model_config module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.model_config import (
    PROVIDER_API_KEY_ENV,
    ModelConfig,
    ModelSpec,
    get_curated_models,
    get_default_models,
    has_provider_credentials,
)


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


class TestGetCuratedModels:
    """Tests for get_curated_models() function."""

    def test_returns_dict_with_providers(self):
        """Returns dictionary with provider keys."""
        models = get_curated_models()
        assert isinstance(models, dict)
        assert "anthropic" in models
        assert "openai" in models
        assert "google_genai" in models

    def test_anthropic_models_include_claude(self):
        """Anthropic models include Claude variants."""
        models = get_curated_models()
        assert any("claude" in m for m in models["anthropic"])

    def test_openai_models_include_gpt(self):
        """OpenAI models include GPT variants."""
        models = get_curated_models()
        assert any("gpt" in m for m in models["openai"])

    def test_google_models_include_gemini(self):
        """Google models include Gemini variants."""
        models = get_curated_models()
        assert any("gemini" in m for m in models["google_genai"])


class TestGetDefaultModels:
    """Tests for get_default_models() function."""

    def test_returns_fallback_models(self):
        """Returns hardcoded fallback model list."""
        models = get_default_models()
        assert isinstance(models, dict)
        assert "anthropic" in models
        assert "openai" in models
        assert "google_genai" in models


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
        assert PROVIDER_API_KEY_ENV["openai"] == "OPENAI_API_KEY"
        assert PROVIDER_API_KEY_ENV["google_genai"] == "GOOGLE_API_KEY"
        assert PROVIDER_API_KEY_ENV["google_vertexai"] == "GOOGLE_CLOUD_PROJECT"


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
[default]
model = "claude-sonnet-4-5"
""")
        config = ModelConfig.load(config_path)

        assert config.default_model == "claude-sonnet-4-5"

    def test_loads_providers(self, tmp_path):
        """Loads provider configurations."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[providers.anthropic]
models = ["claude-sonnet-4-5", "claude-haiku-4-5"]
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
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
[providers.local-ollama]
base_url = "http://localhost:11434/v1"
models = ["llama3"]
""")
        config = ModelConfig.load(config_path)

        assert (
            config.providers["local-ollama"]["base_url"] == "http://localhost:11434/v1"
        )


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
[providers.anthropic]
models = ["claude-sonnet-4-5", "claude-haiku-4-5"]

[providers.openai]
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
[providers.anthropic]
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

    def test_returns_true_when_no_key_required(self, tmp_path):
        """Returns True when api_key_env not specified."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[providers.local]
models = ["llama3"]
""")
        config = ModelConfig.load(config_path)

        assert config.has_credentials("local") is True

    def test_returns_true_when_env_var_set(self, tmp_path):
        """Returns True when api_key_env is set in environment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[providers.anthropic]
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
[providers.anthropic]
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
[providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)

        assert config.get_base_url("anthropic") is None

    def test_returns_base_url(self, tmp_path):
        """Returns configured base_url."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[providers.local]
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
[providers.anthropic]
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
        assert 'model = "claude-sonnet-4-5"' in content

    def test_updates_existing_default(self, tmp_path):
        """Updates existing default model."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[default]
model = "old-model"

[providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        model_config.save_default_model("new-model", config_path)

        content = config_path.read_text()
        assert 'model = "new-model"' in content
        assert "old-model" not in content
        # Should preserve other config
        assert "[providers.anthropic]" in content

    def test_adds_default_section(self, tmp_path):
        """Adds [default] section if missing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        model_config.save_default_model("claude-sonnet-4-5", config_path)

        content = config_path.read_text()
        assert "[default]" in content
        assert 'model = "claude-sonnet-4-5"' in content

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
        assert 'model = "anthropic:claude-sonnet-4-5"' in content

    def test_updates_to_provider_model_format(self, tmp_path):
        """Updates from bare model name to provider:model format."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[default]
model = "claude-sonnet-4-5"
""")
        model_config.save_default_model("anthropic:claude-opus-4-5", config_path)

        content = config_path.read_text()
        assert 'model = "anthropic:claude-opus-4-5"' in content
        assert "claude-sonnet-4-5" not in content


class TestModelPersistenceBetweenSessions:
    """Tests for model selection persistence across app sessions.

    These tests verify that when a user switches models using /model command,
    the selection persists when the CLI is restarted (new session).
    """

    def test_saved_model_is_used_when_no_model_specified(self, tmp_path):
        """Saved default model should be used when CLI starts without --model.

        This test reproduces the bug where switching models via /model command
        saves the model to config, but the saved model is not used on restart.

        Steps:
        1. Save a model to config (simulating /model switch)
        2. Call _get_default_model_spec() without specifying a model
        3. Verify the saved model is used

        This test SHOULD PASS when the bug is fixed.
        """
        from deepagents_cli.config import _get_default_model_spec
        from deepagents_cli.model_config import DEFAULT_CONFIG_PATH, save_default_model

        # Use a temporary config path
        config_path = tmp_path / ".deepagents" / "config.toml"

        # Step 1: Save model to config (simulating /model anthropic:claude-opus-4-5)
        save_default_model("anthropic:claude-opus-4-5", config_path)

        # Verify the model was saved
        assert config_path.exists()
        content = config_path.read_text()
        assert 'model = "anthropic:claude-opus-4-5"' in content

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
            # Step 3: Get default model spec - should use saved config
            result = _get_default_model_spec()

            # BUG: Currently this returns "anthropic:claude-sonnet-4-5-20250929"
            # from env var detection, not the saved "anthropic:claude-opus-4-5"
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
