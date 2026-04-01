"""Tests for model_config module."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from deepagents_cli import model_config
from deepagents_cli.model_config import (
    PROVIDER_API_KEY_ENV,
    THREAD_COLUMN_DEFAULTS,
    ModelConfig,
    ModelConfigError,
    ModelProfileEntry,
    ModelSpec,
    _get_builtin_providers,
    _get_provider_profile_modules,
    _load_provider_profiles,
    _profile_module_from_class_path,
    clear_caches,
    clear_default_model,
    get_available_models,
    get_model_profiles,
    has_provider_credentials,
    is_warning_suppressed,
    load_thread_columns,
    save_recent_model,
    save_thread_columns,
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

    def test_returns_none_for_unknown_provider(self):
        """Returns None for unknown provider (let provider handle auth)."""
        assert has_provider_credentials("unknown") is None

    def test_returns_true_when_env_var_set(self):
        """Returns True when provider env var is set."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            assert has_provider_credentials("anthropic") is True

    def test_returns_false_when_env_var_not_set(self):
        """Returns False when provider env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert has_provider_credentials("anthropic") is False

    def test_returns_true_with_prefixed_env_var(self):
        """Returns True when only the DEEPAGENTS_CLI_ prefixed var is set."""
        with patch.dict(
            "os.environ",
            {"DEEPAGENTS_CLI_ANTHROPIC_API_KEY": "sk-prefixed"},
            clear=True,
        ):
            assert has_provider_credentials("anthropic") is True


class TestThreadColumnPersistence:
    """Tests for thread selector column visibility persistence."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Saved thread column choices should load back on the next session."""
        config_path = tmp_path / "config.toml"
        columns = {
            "thread_id": True,
            "messages": False,
            "created_at": True,
            "updated_at": False,
            "git_branch": True,
            "cwd": False,
            "initial_prompt": False,
            "agent_name": True,
        }

        assert save_thread_columns(columns, config_path) is True
        assert load_thread_columns(config_path) == columns

    def test_load_merges_partial_config_with_defaults(self, tmp_path):
        """Missing thread column keys should fall back to defaults."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[threads.columns]
thread_id = true
updated_at = false
"""
        )

        assert load_thread_columns(config_path) == {
            **THREAD_COLUMN_DEFAULTS,
            "thread_id": True,
            "updated_at": False,
        }


class TestThreadRelativeTimePersistence:
    """Tests for thread relative-time preference persistence."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """Saved relative-time preference should load back on the next session."""
        from deepagents_cli.model_config import (
            load_thread_relative_time,
            save_thread_relative_time,
        )

        config_path = tmp_path / "config.toml"
        assert save_thread_relative_time(False, config_path) is True
        assert load_thread_relative_time(config_path) is False

        assert save_thread_relative_time(True, config_path) is True
        assert load_thread_relative_time(config_path) is True

    def test_default_is_true(self, tmp_path: Path) -> None:
        """When no config file exists, relative time defaults to True."""
        from deepagents_cli.model_config import load_thread_relative_time

        config_path = tmp_path / "config.toml"
        assert load_thread_relative_time(config_path) is True

    def test_preserves_other_config_sections(self, tmp_path: Path) -> None:
        """Saving relative-time should not clobber other config sections."""
        from deepagents_cli.model_config import save_thread_relative_time

        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "anthropic:claude-sonnet-4-5"\n')

        save_thread_relative_time(False, config_path)

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["models"]["default"] == "anthropic:claude-sonnet-4-5"
        assert data["threads"]["relative_time"] is False


class TestThreadSortOrderPersistence:
    """Tests for thread sort-order preference persistence."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """Saved sort order should load back on the next session."""
        from deepagents_cli.model_config import (
            load_thread_sort_order,
            save_thread_sort_order,
        )

        config_path = tmp_path / "config.toml"
        assert save_thread_sort_order("created_at", config_path) is True
        assert load_thread_sort_order(config_path) == "created_at"

        assert save_thread_sort_order("updated_at", config_path) is True
        assert load_thread_sort_order(config_path) == "updated_at"

    def test_default_is_updated_at(self, tmp_path: Path) -> None:
        """When no config file exists, sort order defaults to updated_at."""
        from deepagents_cli.model_config import load_thread_sort_order

        config_path = tmp_path / "config.toml"
        assert load_thread_sort_order(config_path) == "updated_at"

    def test_invalid_value_falls_back_to_default(self, tmp_path: Path) -> None:
        """An unrecognized sort_order value should fall back to updated_at."""
        from deepagents_cli.model_config import load_thread_sort_order

        config_path = tmp_path / "config.toml"
        config_path.write_text('[threads]\nsort_order = "bogus"\n')
        assert load_thread_sort_order(config_path) == "updated_at"

    def test_preserves_other_config_sections(self, tmp_path: Path) -> None:
        """Saving sort order should not clobber other config sections."""
        from deepagents_cli.model_config import save_thread_sort_order

        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "anthropic:claude-sonnet-4-5"\n')

        save_thread_sort_order("created_at", config_path)

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["models"]["default"] == "anthropic:claude-sonnet-4-5"
        assert data["threads"]["sort_order"] == "created_at"


class TestThreadConfigCoalesced:
    """Tests for the coalesced `load_thread_config()` helper."""

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        """When the config file does not exist, defaults should be returned."""
        from deepagents_cli.model_config import load_thread_config

        config_path = tmp_path / "config.toml"
        cfg = load_thread_config(config_path)
        assert cfg.columns == THREAD_COLUMN_DEFAULTS
        assert cfg.relative_time is True
        assert cfg.sort_order == "updated_at"

    def test_reads_all_sections_from_one_parse(self, tmp_path: Path) -> None:
        """A single TOML read should populate columns, relative_time, and sort_order."""
        from deepagents_cli.model_config import load_thread_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[threads]
relative_time = false
sort_order = "created_at"

[threads.columns]
thread_id = true
messages = false
"""
        )
        cfg = load_thread_config(config_path)
        assert cfg.columns["thread_id"] is True
        assert cfg.columns["messages"] is False
        # unchanged defaults
        assert cfg.columns["updated_at"] is True
        assert cfg.relative_time is False
        assert cfg.sort_order == "created_at"

    def test_matches_individual_loaders(self, tmp_path: Path) -> None:
        """Coalesced result should match the three individual loaders."""
        from deepagents_cli.model_config import (
            load_thread_columns,
            load_thread_config,
            load_thread_relative_time,
            load_thread_sort_order,
        )

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[threads]
relative_time = false
sort_order = "created_at"

[threads.columns]
git_branch = true
cwd = true
"""
        )
        cfg = load_thread_config(config_path)
        assert cfg.columns == load_thread_columns(config_path)
        assert cfg.relative_time == load_thread_relative_time(config_path)
        assert cfg.sort_order == load_thread_sort_order(config_path)

    def test_corrupt_toml_returns_defaults(self, tmp_path: Path) -> None:
        """A corrupt config file should return defaults without crashing."""
        from deepagents_cli.model_config import load_thread_config

        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid TOML {{{{")
        cfg = load_thread_config(config_path)
        assert cfg.columns == THREAD_COLUMN_DEFAULTS
        assert cfg.relative_time is True
        assert cfg.sort_order == "updated_at"

    def test_default_path_uses_cache(self) -> None:
        """Second call with default path should return cached result."""
        from deepagents_cli.model_config import (
            _thread_config_cache,
            invalidate_thread_config_cache,
            load_thread_config,
        )

        invalidate_thread_config_cache()
        try:
            first = load_thread_config()
            second = load_thread_config()
            assert first is second
        finally:
            invalidate_thread_config_cache()

    def test_save_invalidates_cache(self, tmp_path: Path) -> None:
        """Saving thread config should invalidate the cached value."""
        from deepagents_cli.model_config import (
            invalidate_thread_config_cache,
            load_thread_config,
            save_thread_columns,
        )

        invalidate_thread_config_cache()
        try:
            first = load_thread_config()
            assert first is load_thread_config()

            save_thread_columns(dict(THREAD_COLUMN_DEFAULTS), tmp_path / "c.toml")
            # Cache was invalidated by save
            from deepagents_cli.model_config import _thread_config_cache

            assert _thread_config_cache is None
        finally:
            invalidate_thread_config_cache()

    def test_save_relative_time_invalidates_cache(self, tmp_path: Path) -> None:
        """Saving relative_time should invalidate the cached value."""
        from deepagents_cli.model_config import (
            _thread_config_cache,
            invalidate_thread_config_cache,
            load_thread_config,
            save_thread_relative_time,
        )

        invalidate_thread_config_cache()
        try:
            load_thread_config()
            save_thread_relative_time(False, tmp_path / "c.toml")
            from deepagents_cli.model_config import _thread_config_cache

            assert _thread_config_cache is None
        finally:
            invalidate_thread_config_cache()

    def test_save_sort_order_invalidates_cache(self, tmp_path: Path) -> None:
        """Saving sort_order should invalidate the cached value."""
        from deepagents_cli.model_config import (
            _thread_config_cache,
            invalidate_thread_config_cache,
            load_thread_config,
            save_thread_sort_order,
        )

        invalidate_thread_config_cache()
        try:
            load_thread_config()
            save_thread_sort_order("created_at", tmp_path / "c.toml")
            from deepagents_cli.model_config import _thread_config_cache

            assert _thread_config_cache is None
        finally:
            invalidate_thread_config_cache()


class TestResolveEnvVar:
    """Tests for resolve_env_var prefix override."""

    def test_returns_canonical_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to the canonical env var when no prefix is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
        monkeypatch.delenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", raising=False)
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("ANTHROPIC_API_KEY") == "sk-canonical"

    def test_prefix_beats_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DEEPAGENTS_CLI_ prefixed var takes priority over canonical."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", "sk-override")
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("ANTHROPIC_API_KEY") == "sk-override"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when neither form is set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", raising=False)
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("ANTHROPIC_API_KEY") is None

    def test_empty_string_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty strings are normalized to None."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", "")
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("ANTHROPIC_API_KEY") is None

    def test_prefix_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Works when only the prefixed var is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CLI_OPENAI_API_KEY", "sk-prefixed")
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"

    def test_empty_prefix_blocks_canonical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty prefix var blocks fallback to canonical (explicit disable)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
        monkeypatch.setenv("DEEPAGENTS_CLI_ANTHROPIC_API_KEY", "")
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("ANTHROPIC_API_KEY") is None

    def test_skips_double_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Names already carrying the prefix don't get double-prefixed."""
        monkeypatch.setenv("DEEPAGENTS_CLI_MY_KEY", "direct")
        monkeypatch.delenv("DEEPAGENTS_CLI_DEEPAGENTS_CLI_MY_KEY", raising=False)
        from deepagents_cli.model_config import resolve_env_var

        assert resolve_env_var("DEEPAGENTS_CLI_MY_KEY") == "direct"


class TestProviderApiKeyEnv:
    """Tests for PROVIDER_API_KEY_ENV constant."""

    def test_contains_major_providers(self):
        """Contains environment variables for major providers."""
        assert PROVIDER_API_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"
        assert PROVIDER_API_KEY_ENV["azure_openai"] == "AZURE_OPENAI_API_KEY"
        assert PROVIDER_API_KEY_ENV["baseten"] == "BASETEN_API_KEY"
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

    def test_returns_true_with_prefixed_env_var(self, tmp_path):
        """Returns True when only the DEEPAGENTS_CLI_ prefixed var is set."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)

        with patch.dict(
            "os.environ",
            {"DEEPAGENTS_CLI_ANTHROPIC_API_KEY": "sk-prefixed"},
            clear=True,
        ):
            assert config.has_credentials("anthropic") is True


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

    def test_skips_config_provider_with_no_models_and_no_class_path(self, tmp_path):
        """Config provider with no models and no class_path is not added."""
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


class TestDisabledProviders:
    """Tests for provider hiding via `enabled = false`."""

    def test_enabled_false_hides_registry_provider(self, tmp_path: Path) -> None:
        """Registry provider with `enabled = false` is hidden."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = false
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

        assert "anthropic" not in models

    def test_enabled_false_hides_config_only_provider(self, tmp_path: Path) -> None:
        """A config-only provider with `enabled = false` is not shown."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
enabled = false
models = ["my-model"]
api_key_env = "CUSTOM_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "custom" not in models

    def test_enabled_true_preserves_provider(self, tmp_path: Path) -> None:
        """A provider with `enabled = true` behaves normally."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = true
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

        assert "anthropic" in models
        assert "claude-sonnet-4-5" in models["anthropic"]

    def test_enabled_false_excludes_from_profiles(self, tmp_path: Path) -> None:
        """A disabled provider is excluded from get_model_profiles()."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = false
""")
        fake_profiles = {
            "claude-sonnet-4-5": {
                "tool_calling": True,
                "max_input_tokens": 200000,
            },
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
            profiles = get_model_profiles()

        assert "anthropic:claude-sonnet-4-5" not in profiles

    def test_enabled_false_excludes_config_only_from_profiles(
        self, tmp_path: Path
    ) -> None:
        """A disabled config-only provider is excluded from profiles."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
enabled = false
models = ["my-model"]
api_key_env = "CUSTOM_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        assert "custom:my-model" not in profiles

    def test_disabled_provider_does_not_affect_others(self, tmp_path: Path) -> None:
        """Disabling one provider does not affect other providers."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = false

[models.providers.custom]
models = ["my-model"]
api_key_env = "CUSTOM_KEY"
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

        assert "anthropic" not in models
        assert "custom" in models
        assert "my-model" in models["custom"]


class TestIsProviderEnabled:
    """Tests for ModelConfig.is_provider_enabled()."""

    def test_returns_true_when_not_in_config(self) -> None:
        """Providers not in config are enabled by default."""
        config = ModelConfig()
        assert config.is_provider_enabled("anthropic") is True

    def test_returns_true_when_enabled_not_set(self, tmp_path: Path) -> None:
        """Provider without `enabled` field is enabled."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)
        assert config.is_provider_enabled("anthropic") is True

    def test_returns_false_when_enabled_false(self, tmp_path: Path) -> None:
        """`enabled = false` disables the provider."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = false
""")
        config = ModelConfig.load(config_path)
        assert config.is_provider_enabled("anthropic") is False

    def test_returns_true_for_nonempty_models_list(self, tmp_path: Path) -> None:
        """Provider with models is enabled."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)
        assert config.is_provider_enabled("anthropic") is True

    def test_enabled_false_takes_precedence_over_models(self, tmp_path: Path) -> None:
        """`enabled = false` hides provider even with models listed."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = false
models = ["claude-sonnet-4-5"]
""")
        config = ModelConfig.load(config_path)
        assert config.is_provider_enabled("anthropic") is False

    def test_string_false_not_treated_as_disabled(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """String `"false"` is not bool `false`; provider stays enabled."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
enabled = "false"
""")
        with caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"):
            config = ModelConfig.load(config_path)

        assert config.is_provider_enabled("anthropic") is True
        assert any("non-boolean" in r.message for r in caplog.records)


class TestProfileModuleFromClassPath:
    """Tests for _profile_module_from_class_path() helper."""

    def test_derives_module_path(self):
        """Derives profile module from a valid class_path."""
        result = _profile_module_from_class_path(
            "langchain_baseten.chat_models:ChatBaseten"
        )
        assert result == "langchain_baseten.data._profiles"

    def test_returns_none_for_missing_colon(self):
        """Returns None when class_path has no colon separator."""
        assert _profile_module_from_class_path("my_package.MyChatModel") is None

    def test_single_segment_package(self):
        """Works with a single-segment package name."""
        result = _profile_module_from_class_path("mypkg:MyClass")
        assert result == "mypkg.data._profiles"

    def test_returns_none_for_empty_module_part(self):
        """Returns None when module part before colon is empty."""
        assert _profile_module_from_class_path(":MyClass") is None


class TestClassPathProviderAutoDiscovery:
    """Tests for auto-discovering models from class_path provider packages."""

    FAKE_BASETEN_PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        "deepseek-ai/DeepSeek-V3.2": {
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        },
        "Qwen/Qwen3-Coder": {
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        },
        "some/no-tools-model": {
            "tool_calling": False,
            "text_inputs": True,
            "text_outputs": True,
        },
    }

    def test_get_available_models_discovers_class_path_profiles(self, tmp_path):
        """class_path provider auto-discovers models from package profiles."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
""")

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_baseten.data._profiles":
                return self.FAKE_BASETEN_PROFILES
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

        assert "baseten" in models
        assert "deepseek-ai/DeepSeek-V3.2" in models["baseten"]
        assert "Qwen/Qwen3-Coder" in models["baseten"]
        # Filtered out: no tool_calling
        assert "some/no-tools-model" not in models["baseten"]

    def test_get_model_profiles_discovers_class_path_profiles(self, tmp_path):
        """class_path provider profiles are included in get_model_profiles()."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
""")

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_baseten.data._profiles":
                return self.FAKE_BASETEN_PROFILES
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        assert "baseten:deepseek-ai/DeepSeek-V3.2" in profiles
        entry = profiles["baseten:deepseek-ai/DeepSeek-V3.2"]
        assert entry["profile"]["tool_calling"] is True
        # No config overrides, so overridden_keys should be empty
        assert entry["overridden_keys"] == frozenset()
        # Unlike get_available_models(), profiles include ALL models (no filter)
        assert "baseten:some/no-tools-model" in profiles

    def test_get_model_profiles_class_path_import_failure_graceful(self, tmp_path):
        """get_model_profiles() degrades gracefully when class_path package fails."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        assert not any(key.startswith("baseten:") for key in profiles)

    def test_class_path_profiles_merged_with_config_overrides(self, tmp_path):
        """Config profile overrides are applied on top of class_path profiles."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"

[models.providers.baseten.profile]
max_input_tokens = 9999
""")

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_baseten.data._profiles":
                return {
                    "my-model": {
                        "tool_calling": True,
                        "max_input_tokens": 4096,
                    },
                }
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        entry = profiles["baseten:my-model"]
        assert entry["profile"]["max_input_tokens"] == 9999
        assert "max_input_tokens" in entry["overridden_keys"]

    def test_class_path_import_failure_graceful(self, tmp_path):
        """Gracefully handles class_path package not being installed."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
""")
        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "baseten" not in models

    def test_class_path_non_import_error_logs_warning(self, tmp_path, caplog):
        """Non-ImportError from class_path package logs warning, not debug."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
""")

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_baseten.data._profiles":
                msg = "broken profiles module"
                raise RuntimeError(msg)
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_cli.model_config"),
        ):
            models = get_available_models()

        assert "baseten" not in models
        assert any(
            "Failed to load profiles" in record.message and "baseten" in record.message
            for record in caplog.records
        )

    def test_explicit_models_list_skips_auto_discovery(self, tmp_path):
        """Explicit models list bypasses auto-discovery even when profiles exist."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
class_path = "langchain_baseten.chat_models:ChatBaseten"
api_key_env = "BASETEN_API_KEY"
models = ["my-explicit-model"]
""")

        def mock_load(module_path: str) -> dict[str, Any]:
            if module_path == "langchain_baseten.data._profiles":
                return self.FAKE_BASETEN_PROFILES
            msg = "not installed"
            raise ImportError(msg)

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_cli.model_config._get_builtin_providers",
                return_value={},
            ),
        ):
            models = get_available_models()

        assert "baseten" in models
        assert models["baseten"] == ["my-explicit-model"]

    def test_skips_builtin_registry_providers(self, tmp_path):
        """Does not double-load profiles for providers in the built-in registry."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
class_path = "langchain_anthropic.chat_models:ChatAnthropic"
""")
        fake_profiles = {"claude-sonnet-4-5": {"tool_calling": True}}

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

        # Should only appear once (from registry path, not double-loaded)
        assert models["anthropic"].count("claude-sonnet-4-5") == 1


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

    def test_class_path_provider_without_api_key_env_returns_true(self, tmp_path):
        """Returns True for class_path provider with no api_key_env.

        class_path providers manage their own auth (e.g., custom headers, JWT)
        so they should be treated as having credentials available.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.cis]
class_path = "agent_forge.integrations:CISChat"
models = ["aviato-turbo"]
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert has_provider_credentials("cis") is True

    def test_class_path_with_api_key_env_respects_env_var(self, tmp_path):
        """api_key_env takes precedence over class_path for credential check."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.cis]
class_path = "agent_forge.integrations:CISChat"
models = ["aviato-turbo"]
api_key_env = "CIS_API_KEY"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert has_provider_credentials("cis") is False

    def test_returns_none_for_totally_unknown_provider(self):
        """Returns None for provider not in hardcoded map or config.

        Unknown providers are let through so the provider itself can report
        auth failures at model-creation time.
        """
        assert has_provider_credentials("nonexistent_provider_xyz") is None


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


class TestModelConfigGetProfileOverrides:
    """Tests for ModelConfig.get_profile_overrides() method."""

    def test_returns_empty_for_unknown_provider(self):
        """Returns empty dict for unknown provider."""
        config = ModelConfig()
        assert config.get_profile_overrides("unknown") == {}

    def test_returns_empty_when_no_profile(self, tmp_path):
        """Returns empty dict when profile not in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
""")
        config = ModelConfig.load(config_path)
        assert config.get_profile_overrides("custom") == {}

    def test_returns_provider_wide_overrides(self, tmp_path):
        """Returns flat profile overrides."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]

[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        config = ModelConfig.load(config_path)
        overrides = config.get_profile_overrides("anthropic")
        assert overrides == {"max_input_tokens": 4096}

    def test_per_model_override_takes_precedence(self, tmp_path):
        """Per-model sub-table overrides provider-wide value."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5", "claude-opus-4-6"]

[models.providers.anthropic.profile]
max_input_tokens = 4096

[models.providers.anthropic.profile."claude-sonnet-4-5"]
max_input_tokens = 8192
""")
        config = ModelConfig.load(config_path)
        overrides = config.get_profile_overrides(
            "anthropic", model_name="claude-sonnet-4-5"
        )
        assert overrides == {"max_input_tokens": 8192}

    def test_model_without_subtable_gets_provider_defaults(self, tmp_path):
        """Model not in sub-table gets provider-level profile only."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5", "claude-opus-4-6"]

[models.providers.anthropic.profile]
max_input_tokens = 4096

[models.providers.anthropic.profile."claude-sonnet-4-5"]
max_input_tokens = 8192
""")
        config = ModelConfig.load(config_path)
        overrides = config.get_profile_overrides(
            "anthropic", model_name="claude-opus-4-6"
        )
        assert overrides == {"max_input_tokens": 4096}

    def test_none_model_name_returns_provider_defaults(self, tmp_path):
        """model_name=None returns provider-wide profile only."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]

[models.providers.anthropic.profile]
max_input_tokens = 4096

[models.providers.anthropic.profile."claude-sonnet-4-5"]
max_input_tokens = 8192
""")
        config = ModelConfig.load(config_path)
        overrides = config.get_profile_overrides("anthropic", model_name=None)
        assert overrides == {"max_input_tokens": 4096}

    def test_multiple_flat_keys_with_model_subtable(self, tmp_path):
        """Multiple flat keys returned; model sub-table merges on top."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]

[models.providers.anthropic.profile]
max_input_tokens = 4096
supports_thinking = true

[models.providers.anthropic.profile."claude-sonnet-4-5"]
max_input_tokens = 8192
""")
        config = ModelConfig.load(config_path)
        overrides = config.get_profile_overrides(
            "anthropic", model_name="claude-sonnet-4-5"
        )
        assert overrides == {"max_input_tokens": 8192, "supports_thinking": True}


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

        assert result == "anthropic:claude-sonnet-4-6"


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


class TestGetModelProfiles:
    """Tests for get_model_profiles() function."""

    def test_returns_upstream_profiles(self) -> None:
        """Returns profiles keyed by provider:model spec."""
        fake_profiles = {
            "claude-sonnet-4-5": {
                "tool_calling": True,
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
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
            profiles = get_model_profiles()

        assert "anthropic:claude-sonnet-4-5" in profiles
        entry = profiles["anthropic:claude-sonnet-4-5"]
        assert entry["profile"]["max_input_tokens"] == 200000
        assert entry["profile"]["tool_calling"] is True
        assert entry["overridden_keys"] == frozenset()

    def test_merges_config_overrides(self, tmp_path: Path) -> None:
        """Config.toml profile overrides are merged and tracked."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
[models.providers.anthropic.profile]
max_input_tokens = 100000
""")
        fake_profiles = {
            "claude-sonnet-4-5": {
                "tool_calling": True,
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
            },
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
            profiles = get_model_profiles()

        entry = profiles["anthropic:claude-sonnet-4-5"]
        assert entry["profile"]["max_input_tokens"] == 100000
        assert entry["profile"]["max_output_tokens"] == 64000
        assert "max_input_tokens" in entry["overridden_keys"]
        assert "max_output_tokens" not in entry["overridden_keys"]

    def test_config_only_model_no_upstream(self, tmp_path: Path) -> None:
        """Config-only model with no upstream profile creates an entry."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
[models.providers.custom.profile]
max_input_tokens = 4096
""")

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        assert "custom:my-model" in profiles
        entry = profiles["custom:my-model"]
        assert entry["profile"]["max_input_tokens"] == 4096
        assert "max_input_tokens" in entry["overridden_keys"]

    def test_cache_cleared(self) -> None:
        """clear_caches() resets the profiles cache."""
        fake_profiles = {
            "test-model": {"tool_calling": True},
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
            get_model_profiles()

        assert model_config._profiles_cache is not None
        clear_caches()
        assert model_config._profiles_cache is None

    def test_overridden_keys_subset_of_profile(self, tmp_path: Path) -> None:
        """overridden_keys is always a subset of profile keys."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
[models.providers.anthropic.profile]
max_input_tokens = 100000
""")
        fake_profiles = {
            "claude-sonnet-4-5": {
                "tool_calling": True,
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
            },
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
            profiles = get_model_profiles()

        for spec, entry in profiles.items():
            assert entry["overridden_keys"] <= entry["profile"].keys(), (
                f"{spec}: overridden_keys {entry['overridden_keys']} "
                f"not a subset of profile keys {set(entry['profile'].keys())}"
            )

    def test_cli_override_merged_on_top(self) -> None:
        """CLI override is merged on top of upstream + config.toml."""
        fake_profiles = {
            "claude-sonnet-4-5": {
                "tool_calling": True,
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
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
            profiles = get_model_profiles(cli_override={"max_input_tokens": 4096})

        entry = profiles["anthropic:claude-sonnet-4-5"]
        assert entry["profile"]["max_input_tokens"] == 4096
        assert entry["profile"]["max_output_tokens"] == 64000
        assert "max_input_tokens" in entry["overridden_keys"]

    def test_cli_override_skips_cache(self) -> None:
        """cli_override path does not populate module-level cache."""
        fake_profiles = {
            "test-model": {"tool_calling": True},
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
            get_model_profiles(cli_override={"max_input_tokens": 4096})

        assert model_config._profiles_cache is None

    def test_cli_override_on_config_only_model(self, tmp_path: Path) -> None:
        """CLI override applies to config-only models with no upstream profile."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
[models.providers.custom.profile]
max_input_tokens = 8192
""")

        with (
            patch(
                "deepagents_cli.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles(cli_override={"max_output_tokens": 2048})

        entry = profiles["custom:my-model"]
        assert entry["profile"]["max_input_tokens"] == 8192
        assert entry["profile"]["max_output_tokens"] == 2048
        assert "max_output_tokens" in entry["overridden_keys"]
        assert "max_input_tokens" in entry["overridden_keys"]
