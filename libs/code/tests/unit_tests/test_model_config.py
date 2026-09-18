"""Tests for model_config module."""

import io
import logging
import sys
import threading
import tomllib
from collections.abc import Iterator
from contextlib import AbstractContextManager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code import model_config
from deepagents_code.json_types import JsonObject
from deepagents_code.model_config import (
    IMPLICIT_AUTH_PROVIDERS,
    NO_AUTH_REQUIRED_PROVIDERS,
    PROVIDER_API_KEY_ENV,
    RETRY_PARAM_BY_PROVIDER,
    STARTUP_MODE_AUTO,
    STARTUP_MODE_MANUAL,
    STARTUP_MODE_YOLO,
    THREAD_COLUMN_DEFAULTS,
    McpProjectServerApproval,
    McpServerTrustLists,
    ModelConfig,
    ModelConfigError,
    ModelNotAllowedError,
    ModelSpec,
    NoAllowedModelCredentialsError,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    _is_local_endpoint,
    clear_caches,
    clear_default_agent,
    default_cache_dir,
    fingerprint_mcp_server_config,
    get_available_models,
    get_model_profiles,
    get_provider_auth_status,
    has_provider_credentials,
    is_warning_suppressed,
    load_default_agent,
    load_effort_for_model,
    load_mcp_server_trust_lists,
    load_recent_agent,
    load_recent_models,
    load_startup_mode,
    load_thread_columns,
    normalize_mcp_project_root,
    parse_model_allowlist,
    save_default_agent,
    save_effort_for_model,
    save_recent_agent,
    save_recent_model,
    save_recent_startup_mode,
    save_thread_columns,
    suppress_warning_reason,
    touch_recent_model,
    unsuppress_warning,
)


def _create_git_common_dir(common_dir: Path) -> Path:
    """Create the minimal shared metadata required by Git trust resolution."""
    (common_dir / "objects").mkdir(parents=True)
    (common_dir / "refs").mkdir()
    (common_dir / "worktrees").mkdir()
    (common_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (common_dir / "config").write_text("[core]\n\tbare = false\n")
    return common_dir


class TestDefaultCacheDir:
    """`default_cache_dir` resolves the OS-appropriate cache root."""

    def test_xdg_cache_home_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set `XDG_CACHE_HOME` wins on Linux."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
        assert default_cache_dir() == tmp_path / "xdg-cache"

    def test_xdg_cache_home_unset_falls_back_to_dot_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without `XDG_CACHE_HOME`, Linux falls back to `~/.cache`."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / ".cache"

    def test_xdg_cache_home_empty_falls_back_to_dot_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty `XDG_CACHE_HOME` is treated as unset."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / ".cache"

    def test_macos_uses_library_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MacOS resolves to `~/Library/Caches`, ignoring `XDG_CACHE_HOME`."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / "Library" / "Caches"

    def test_macos_without_xdg_cache_home_uses_library_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MacOS resolves to `~/Library/Caches` when `XDG_CACHE_HOME` is unset."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / "Library" / "Caches"

    def test_xdg_cache_home_relative_falls_back_to_dot_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative `XDG_CACHE_HOME` is invalid per the XDG spec and ignored."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "cache")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / ".cache"

    def test_windows_uses_local_app_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows uses its native local application-data directory."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
        assert default_cache_dir() == tmp_path / "local-app-data"

    def test_windows_without_local_app_data_uses_home_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows retains a predictable fallback when `LOCALAPPDATA` is unavailable."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_cache_dir() == tmp_path / "AppData" / "Local"


def _create_git_repository(root: Path) -> Path:
    """Create a worktree with an in-tree Git common directory."""
    root.mkdir()
    return _create_git_common_dir(root / ".git")


def _create_git_worktree(common_dir: Path, root: Path, name: str) -> Path:
    """Create reciprocal linked-worktree metadata under `common_dir`."""
    root.mkdir()
    git_entry = root / ".git"
    git_dir = common_dir / "worktrees" / name
    git_dir.mkdir()
    git_entry.write_text(f"gitdir: {git_dir}\n")
    (git_dir / "commondir").write_text("../..\n")
    (git_dir / "gitdir").write_text(f"{git_entry}\n")
    (git_dir / "HEAD").write_text(f"ref: refs/heads/{name}\n")
    return git_dir


@pytest.fixture(autouse=True)
def _clear_model_caches() -> Iterator[None]:
    """Clear module-level caches before and after each test."""
    clear_caches()
    yield
    clear_caches()


class TestRetryParamByProvider:
    """Tests for retry-parameter registry drift."""

    def test_all_retry_providers_are_known(self) -> None:
        """Every retry-enabled provider is a known provider."""
        known_providers = (
            set(PROVIDER_API_KEY_ENV)
            | set(IMPLICIT_AUTH_PROVIDERS)
            | set(NO_AUTH_REQUIRED_PROVIDERS)
            | {"bedrock", model_config.CODEX_PROVIDER}
        )
        assert set(RETRY_PARAM_BY_PROVIDER) <= known_providers


class TestModelSpec:
    """Tests for ModelSpec value type."""

    def test_parse_with_colons_in_model_name(self) -> None:
        """parse() handles model names that contain colons."""
        spec = ModelSpec.parse("custom:model:with:colons")
        assert spec.provider == "custom"
        assert spec.model == "model:with:colons"


class TestModelAllowlist:
    """Tests for exact model policy parsing and matching."""

    @pytest.mark.parametrize(
        "value",
        ["openai:gpt-5.6-terra", ["openai"], ["openai: gpt"], [3], [""]],
    )
    def test_parser_rejects_malformed_values(self, value: object) -> None:
        """Wrong shapes and noncanonical entries reject the declaration."""
        with pytest.raises((TypeError, ValueError), match=r"expected|entry|invalid"):
            parse_model_allowlist(value)

    def test_empty_policy_denies_every_model(self) -> None:
        """An explicitly empty allowlist is a total lockdown."""
        config = ModelConfig(allowed_models=(), allowed_models_source="config.toml")
        assert config.is_model_allowed("openai:gpt-5.6-terra") is False
        with pytest.raises(ModelNotAllowedError, match="allows no models"):
            config.require_model_allowed("openai:gpt-5.6-terra")

    def test_parser_rejects_bare_bedrock_ids(self) -> None:
        """A bare Bedrock ID would split at its version colon and never match.

        `create_model` normalizes the same input to `bedrock:<id>`, so accepting
        it here would turn the allowlist into a silent deny-all.
        """
        with pytest.raises(ValueError, match="bedrock:"):
            parse_model_allowlist(["anthropic.claude-3-5-sonnet-20241022-v2:0"])

    def test_error_context_names_the_declaring_file(self) -> None:
        """A rejection inside a loop over files says which file to edit."""
        config = ModelConfig(
            allowed_models=("openai:gpt-5.6-terra",),
            allowed_models_source="config.toml",
        )

        with pytest.raises(ModelNotAllowedError, match=r"subagent 'rev' \(a/b\.md\)"):
            config.require_model_allowed(
                "openai:blocked", context="subagent 'rev' (a/b.md)"
            )

    def test_canonicalize_accepts_an_inferable_bare_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare name whose provider is inferable matches like the canonical form.

        `create_model` checks the *resolved* spec, so a preflight that checked
        the raw text rejected supported bare names the authoritative gate would
        have allowed.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-real-key")
        config = ModelConfig(
            allowed_models=("openai:gpt-5.6-terra",),
            allowed_models_source="config.toml",
        )

        assert config.canonical_model_spec("gpt-5.6-terra") == "openai:gpt-5.6-terra"
        assert config.policy_error("gpt-5.6-terra", canonicalize=True) is None
        # Without canonicalization the raw text is unmatchable, which is why the
        # flag exists rather than being the unconditional behavior.
        assert config.policy_error("gpt-5.6-terra") is not None

    def test_canonicalize_still_blocks_a_disallowed_bare_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inference is not a bypass: the resolved spec must still be allowed."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-real-key")
        config = ModelConfig(
            allowed_models=("anthropic:claude-opus-5",),
            allowed_models_source="config.toml",
        )

        assert config.canonical_model_spec("gpt-5.6-terra") == "openai:gpt-5.6-terra"
        error = config.policy_error("gpt-5.6-terra", canonicalize=True)
        assert error is not None
        # The message echoes what the user typed, not the canonical form.
        assert "'gpt-5.6-terra'" in str(error)

    def test_parser_accepts_provider_wildcard(self) -> None:
        """`provider:*` round-trips and deduplicates like an exact entry."""
        assert parse_model_allowlist(
            ["openai:*", "anthropic:claude-opus-5", "openai:*"]
        ) == ("openai:*", "anthropic:claude-opus-5")

    @pytest.mark.parametrize(
        "entry",
        ["*", ":*", "gpt-*", "openai:gpt-*", "openai:*:extra", "o penai:*"],
    )
    def test_parser_rejects_misplaced_wildcards(self, entry: str) -> None:
        """Only a whole-provider `provider:*` suffix is a valid wildcard.

        A bare `*` or a `*` inside the model part would either match
        everything or silently match nothing, so the parser demands the
        explicit provider-scoped form.
        """
        with pytest.raises(ValueError, match=r"wildcard|provider:\*"):
            parse_model_allowlist([entry])

    def test_error_lists_the_allowed_models(self) -> None:
        """The message names what *is* permitted, not only what is not."""
        config = ModelConfig(
            allowed_models=("openai:a", "anthropic:b"),
            allowed_models_source="config.toml",
        )

        error = config.policy_error("openai:blocked")

        assert error is not None
        assert "openai:a, anthropic:b" in str(error)


class TestHasProviderCredentials:
    """Tests for has_provider_credentials() function."""

    def test_returns_true_when_env_var_set(self):
        """Returns True when provider env var is set."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            assert has_provider_credentials("anthropic") is True

    def test_returns_false_when_env_var_not_set(self):
        """Returns False when provider env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert has_provider_credentials("anthropic") is False

    def test_returns_true_with_prefixed_env_var(self):
        """Returns True when only the DEEPAGENTS_CODE_ prefixed var is set."""
        with patch.dict(
            "os.environ",
            {"DEEPAGENTS_CODE_ANTHROPIC_API_KEY": "sk-prefixed"},
            clear=True,
        ):
            assert has_provider_credentials("anthropic") is True

    @pytest.mark.parametrize(
        "provider", ["anthropic", "baseten", "fireworks", "google_genai", "openai"]
    )
    def test_returns_true_with_langsmith_gateway(self, provider: str) -> None:
        """Returns True for providers supported by LangSmith Gateway."""
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_GATEWAY": "true",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-key",
            },
            clear=True,
        ):
            assert has_provider_credentials(provider) is True

    def test_returns_true_with_custom_langsmith_gateway_url(self) -> None:
        """Returns True when the gateway setting is a custom URL."""
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_GATEWAY": "https://gateway.example.com",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-key",
            },
            clear=True,
        ):
            status = get_provider_auth_status("openai")

        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.ENV
        assert status.env_var == "LANGSMITH_GATEWAY_API_KEY"

    @pytest.mark.parametrize("gateway", ["false", "0", "no", ""])
    def test_returns_false_when_langsmith_gateway_disabled(self, gateway: str) -> None:
        """Returns False when the gateway setting is disabled."""
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_GATEWAY": gateway,
                "LANGSMITH_GATEWAY_API_KEY": "gateway-key",
            },
            clear=True,
        ):
            assert has_provider_credentials("anthropic") is False

    def test_returns_false_when_langsmith_gateway_key_missing(self) -> None:
        """Returns False when the enabled gateway has no API key."""
        with patch.dict("os.environ", {"LANGSMITH_GATEWAY": "true"}, clear=True):
            assert has_provider_credentials("openai") is False

    def test_returns_false_for_unsupported_gateway_provider(self) -> None:
        """Returns False when the provider integration lacks gateway support."""
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_GATEWAY": "true",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-key",
            },
            clear=True,
        ):
            assert has_provider_credentials("groq") is False

    def test_class_path_override_does_not_borrow_gateway(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `class_path` override must not report gateway auth.

        Overriding a built-in gateway provider name with a custom `class_path`
        builds an arbitrary class (via `_create_model_from_class`) that need not
        consume the gateway variables, so its own `api_key_env` preflight must
        stand rather than reporting CONFIGURED off the gateway.
        """
        state_dir = tmp_path / ".state"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[models.providers.openai]\n"
            'class_path = "my_package:CustomChat"\n'
            'api_key_env = "CUSTOM_KEY"\n'
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_GATEWAY": "true",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-key",
            },
            clear=True,
        ):
            status = get_provider_auth_status("openai")

        assert status.state is ProviderAuthState.MISSING
        assert status.env_var == "CUSTOM_KEY"


@pytest.fixture
def fake_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the credential store into a temp directory."""
    state_dir = tmp_path / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)
    return state_dir


class TestStoredCredentials:
    """Stored API keys (added via /auth) integrate into auth resolution."""

    @pytest.fixture(autouse=True)
    def _clear_dotenv_prefixed_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strip `DEEPAGENTS_CODE_*` keys preloaded from `~/.deepagents/.env`.

        `dotenv.load_dotenv()` runs at config-import time and may inject
        prefixed variants that win over `monkeypatch.setenv` in
        `resolve_env_var`'s lookup order.
        """
        for var in (
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_resolve_provider_credential_prefers_stored_over_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stored credential beats env var (matches pi-mono ordering)."""
        from deepagents_code import auth_store
        from deepagents_code.model_config import resolve_provider_credential

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        auth_store.set_stored_key("anthropic", "from-store")

        assert resolve_provider_credential("anthropic") == "from-store"

    def test_resolve_provider_credential_falls_back_to_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Env var is used when no stored credential exists."""
        from deepagents_code.model_config import resolve_provider_credential

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert resolve_provider_credential("anthropic") == "from-env"

    def test_resolve_provider_credential_returns_none_for_unknown_provider(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider with no env-var binding and no stored key returns None."""
        from deepagents_code.model_config import resolve_provider_credential

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert resolve_provider_credential("totally-unknown") is None

    @pytest.mark.parametrize("override", [None, "", "from-prefix"])
    def test_status_reports_stored_credential(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
        override: str | None,
    ) -> None:
        """A stored key flips status to CONFIGURED with a stored detail."""
        from deepagents_code import auth_store

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        if override is not None:
            monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", override)
        auth_store.set_stored_key("anthropic", "from-store")

        status = get_provider_auth_status("anthropic")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED
        assert status.env_var == "ANTHROPIC_API_KEY"

    def test_apply_stored_credentials_sets_env_var(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`apply_stored_credentials` exports the stored key into os.environ."""
        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        auth_store.set_stored_key("openai", "from-store")
        applied = apply_stored_credentials("openai")

        assert applied is True
        import os

        assert os.environ["OPENAI_API_KEY"] == "from-store"

    def test_apply_stored_credentials_overrides_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stored credential takes precedence over an already-set env var."""
        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        auth_store.set_stored_key("openai", "from-store")

        assert apply_stored_credentials("openai") is True
        import os

        assert os.environ["OPENAI_API_KEY"] == "from-store"

    def test_apply_stored_credentials_noop_when_no_store(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No stored key means no environment mutation."""
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert apply_stored_credentials("anthropic") is False
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "from-env"

    def test_apply_stored_credentials_sets_base_url(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored base_url is exported alongside the key, alt name cleared."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_BASE", "https://stale.example/v1")
        auth_store.set_stored_key(
            "openai", "from-store", base_url="https://mine.example/v1"
        )

        assert apply_stored_credentials("openai") is True
        assert os.environ["OPENAI_BASE_URL"] == "https://mine.example/v1"
        # The alternate name the SDK also reads must not retain a stale value.
        assert "OPENAI_API_BASE" not in os.environ

    def test_apply_stored_credentials_sets_baseten_base_url(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored Baseten endpoint writes `BASETEN_BASE_URL` and clears legacy."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("BASETEN_API_KEY", raising=False)
        monkeypatch.setenv("BASETEN_API_BASE", "https://stale.example/v1")
        auth_store.set_stored_key(
            "baseten", "from-store", base_url="https://mine.example/v1"
        )

        assert apply_stored_credentials("baseten") is True
        assert os.environ["BASETEN_BASE_URL"] == "https://mine.example/v1"
        assert "BASETEN_API_BASE" not in os.environ

    def test_apply_stored_credentials_blank_base_url_clears_gateway(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored key with no base_url clears the inherited (gateway) URL.

        This is what stops a personal key from being shipped to the gateway.
        """
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv(
            "OPENAI_BASE_URL", "https://gateway.smith.langchain.com/openai/v1"
        )
        auth_store.set_stored_key("openai", "sk-personal")

        assert apply_stored_credentials("openai") is True
        assert os.environ["OPENAI_API_KEY"] == "sk-personal"
        assert "OPENAI_BASE_URL" not in os.environ
        assert "OPENAI_API_BASE" not in os.environ

    def test_apply_stored_credentials_blank_base_url_clears_gemini_gateway(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gemini routes via GOOGLE_GEMINI_BASE_URL, so the pairing applies too.

        The google-genai SDK reads GOOGLE_GEMINI_BASE_URL natively, so a stored
        key with no base_url must clear it or a personal key reaches the gateway.
        """
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv(
            "GOOGLE_GEMINI_BASE_URL", "https://gateway.smith.langchain.com/gemini"
        )
        auth_store.set_stored_key("google_genai", "personal-gemini-key")

        assert apply_stored_credentials("google_genai") is True
        assert "GOOGLE_GEMINI_BASE_URL" not in os.environ

    def test_apply_stored_credentials_blank_base_url_clears_anthropic_custom_headers(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored Anthropic key with no base_url clears `ANTHROPIC_CUSTOM_HEADERS`.

        The Anthropic SDK reads `ANTHROPIC_CUSTOM_HEADERS` and injects the
        headers into every request. A gateway-provisioned environment sets
        this to `X-Api-Key: <gateway-key>`, which overrides the SDK's own
        `api_key`-derived header. When switching to a personal key, the
        custom header must also be cleared or the gateway key is sent to
        the provider's native endpoint and rejected.
        """
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", "https://gateway.smith.langchain.com/anthropic"
        )
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: lsv2_sk_gateway_key")
        auth_store.set_stored_key("anthropic", "sk-ant-personal")

        assert apply_stored_credentials("anthropic") is True
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-personal"
        assert "ANTHROPIC_BASE_URL" not in os.environ
        assert "ANTHROPIC_API_URL" not in os.environ
        assert "ANTHROPIC_CUSTOM_HEADERS" not in os.environ

    def test_apply_stored_credentials_with_base_url_preserves_anthropic_custom_headers(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored Anthropic key *with* a base_url preserves custom headers.

        When the user stores a gateway endpoint in `/auth`, the custom
        headers env var should be left in place — it carries the gateway
        auth header that the gateway expects.
        """
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: lsv2_sk_gateway_key")
        auth_store.set_stored_key(
            "anthropic",
            "lsv2_sk_gateway_key",
            base_url="https://gateway.smith.langchain.com/anthropic",
        )

        assert apply_stored_credentials("anthropic") is True
        assert (
            os.environ["ANTHROPIC_BASE_URL"]
            == "https://gateway.smith.langchain.com/anthropic"
        )
        assert (
            os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "X-Api-Key: lsv2_sk_gateway_key"
        )

    def test_apply_stored_credentials_config_base_url_preserves_anthropic_headers(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A config-routed Anthropic gateway keeps its custom headers."""
        import os

        from deepagents_code import auth_store, model_config
        from deepagents_code.model_config import apply_stored_credentials, clear_caches

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
base_url = "https://configured.gateway.example/anthropic"
models = ["claude-sonnet-4-5"]
""")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", "https://stale.gateway.example/anthropic"
        )
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: lsv2_sk_gateway_key")
        auth_store.set_stored_key("anthropic", "sk-ant-personal")

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            clear_caches()
            assert apply_stored_credentials("anthropic") is True
            assert (
                model_config.ModelConfig.load().get_base_url("anthropic")
                == "https://configured.gateway.example/anthropic"
            )

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-personal"
        assert "ANTHROPIC_BASE_URL" not in os.environ
        assert "ANTHROPIC_API_URL" not in os.environ
        assert (
            os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "X-Api-Key: lsv2_sk_gateway_key"
        )

    def test_apply_stored_credentials_prefixed_base_url_preserves_anthropic_headers(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scoped Anthropic endpoint override keeps its gateway headers."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_credentials

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", "https://stale.gateway.example/anthropic"
        )
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_ANTHROPIC_BASE_URL",
            "https://scoped.gateway.example/anthropic",
        )
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: lsv2_sk_gateway_key")
        auth_store.set_stored_key("anthropic", "sk-ant-personal")

        assert apply_stored_credentials("anthropic") is True

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-personal"
        assert "ANTHROPIC_BASE_URL" not in os.environ
        assert "ANTHROPIC_API_URL" not in os.environ
        assert (
            os.environ["DEEPAGENTS_CODE_ANTHROPIC_BASE_URL"]
            == "https://scoped.gateway.example/anthropic"
        )
        assert (
            os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "X-Api-Key: lsv2_sk_gateway_key"
        )

    def test_apply_stored_credentials_clears_config_base_url_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A config-declared `base_url_env` participates in the pairing.

        Lets a provider outside the hardcoded set clear an inherited gateway
        URL when a `/auth` key with no base URL is applied.
        """
        import os

        from deepagents_code import auth_store, model_config
        from deepagents_code.model_config import apply_stored_credentials, clear_caches

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.myco]
api_key_env = "MYCO_KEY"
base_url_env = "MYCO_BASE_URL"
models = ["m1"]
""")
        monkeypatch.delenv("MYCO_KEY", raising=False)
        monkeypatch.setenv("MYCO_BASE_URL", "https://gateway.example/myco")
        auth_store.set_stored_key("myco", "myco-personal")

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            clear_caches()
            assert apply_stored_credentials("myco") is True

        assert os.environ["MYCO_KEY"] == "myco-personal"
        assert "MYCO_BASE_URL" not in os.environ

    def test_corrupt_store_does_not_block_status(
        self,
        fake_state_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt auth.json doesn't poison `get_provider_auth_status`."""
        path = fake_state_dir / "auth.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        # Status should still resolve via env var without raising.
        status = get_provider_auth_status("anthropic")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.ENV


class TestServiceCredentials:
    """Non-model services (e.g. Tavily) resolve and apply stored keys."""

    @pytest.fixture(autouse=True)
    def _clear_service_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strip service env vars so each test controls its own state."""
        for var in (
            "TAVILY_API_KEY",
            "DEEPAGENTS_CODE_TAVILY_API_KEY",
            "LANGSMITH_API_KEY",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_apply_exports_stored_langsmith_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored LangSmith key is copied onto LANGSMITH_API_KEY."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_service_credentials

        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test")
        apply_stored_service_credentials()
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"

    def test_status_missing_when_unset(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """No stored or env key reports MISSING with the canonical env var."""
        from deepagents_code.model_config import get_service_auth_status

        status = get_service_auth_status("tavily")
        assert status.state is ProviderAuthState.MISSING
        assert status.env_var == "TAVILY_API_KEY"
        assert status.detail == "TAVILY_API_KEY is not set or is empty"

    def test_status_configured_from_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An env var reports CONFIGURED from the env source."""
        from deepagents_code.model_config import get_service_auth_status

        monkeypatch.setenv("TAVILY_API_KEY", "from-env")
        status = get_service_auth_status("tavily")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.ENV

    def test_langsmith_status_primary_wins_over_fallback(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LangSmith status reports its primary env var when both are set."""
        from deepagents_code.model_config import get_service_auth_status

        monkeypatch.setenv("LANGSMITH_API_KEY", "from-primary")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "from-fallback")
        status = get_service_auth_status("langsmith")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.ENV
        assert status.env_var == "LANGSMITH_API_KEY"

    def test_prefixed_override_outranks_stored_service_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prefixed override reports ENV, because the store never applies."""
        from deepagents_code import auth_store
        from deepagents_code.model_config import get_service_auth_status

        auth_store.set_stored_key("langsmith", "from-store")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "from-prefix")
        status = get_service_auth_status("langsmith")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.ENV
        assert status.env_var == "LANGSMITH_API_KEY"

    def test_empty_prefixed_override_hides_stored_service_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty override suppresses the store, matching the runtime.

        `apply_stored_service_credentials` skips the copy whenever the prefixed
        name is present, so the stored key never reaches the SDK. Reporting it
        as configured would be a promise the app cannot keep.
        """
        from deepagents_code import auth_store
        from deepagents_code.model_config import get_service_auth_status

        auth_store.set_stored_key("langsmith", "from-store")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        assert get_service_auth_status("langsmith").state is ProviderAuthState.MISSING

    def test_missing_langsmith_detail_names_the_fallback(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """The MISSING message names every env var that would have worked."""
        from deepagents_code.model_config import get_service_auth_status

        status = get_service_auth_status("langsmith")
        assert status.state is ProviderAuthState.MISSING
        assert status.detail == (
            "LANGSMITH_API_KEY or LANGCHAIN_API_KEY is not set or is empty"
        )

    def test_service_fallback_status_env_var_stays_canonical(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prefixed fallback also records its canonical spelling."""
        from deepagents_code.model_config import (
            get_service_auth_status,
            resolved_env_var_name,
        )

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "from-prefix")
        status = get_service_auth_status("langsmith")
        assert status.env_var == "LANGCHAIN_API_KEY"
        assert (
            resolved_env_var_name(status.env_var) == "DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
        )

    def test_status_configured_from_store(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """A stored key reports CONFIGURED from the stored source."""
        from deepagents_code import auth_store
        from deepagents_code.model_config import get_service_auth_status

        auth_store.set_stored_key("tavily", "from-store")
        status = get_service_auth_status("tavily")
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED

    def test_apply_exports_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """`apply_stored_service_credentials` copies the stored key to env."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_service_credentials

        auth_store.set_stored_key("tavily", "from-store")
        apply_stored_service_credentials()
        assert os.environ["TAVILY_API_KEY"] == "from-store"

    def test_apply_noop_without_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No stored key leaves an existing env var untouched."""
        import os

        from deepagents_code.model_config import apply_stored_service_credentials

        monkeypatch.setenv("TAVILY_API_KEY", "from-env")
        apply_stored_service_credentials()
        assert os.environ["TAVILY_API_KEY"] == "from-env"

    def test_apply_stored_key_overrides_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored key wins over a conflicting existing env var.

        Guards the documented precedence (matching `apply_stored_credentials`):
        a key entered via `/auth` must beat a plain `TAVILY_API_KEY` already in
        the environment, otherwise the stored key would be silently ignored.
        """
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_service_credentials

        monkeypatch.setenv("TAVILY_API_KEY", "from-env")
        auth_store.set_stored_key("tavily", "from-store")
        apply_stored_service_credentials()
        assert os.environ["TAVILY_API_KEY"] == "from-store"

    def test_apply_stored_key_respects_prefixed_override(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scoped service key is not overwritten by a stored key."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.model_config import apply_stored_service_credentials

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_prefixed")
        auth_store.set_stored_key("langsmith", "lsv2_stored")
        apply_stored_service_credentials()
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_prefixed"


class TestSplitCredentialSource:
    """`warn_on_split_credential_source` flags key/endpoint env-tier mismatches."""

    @pytest.fixture(autouse=True)
    def _isolate_openai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear every OpenAI key/endpoint env var so each test sets its own.

        `dotenv.load_dotenv()` runs during the first credentials access and may
        inject prefixed variants from a developer's
        `~/.deepagents/.env` that would otherwise leak into these assertions.
        """
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "DEEPAGENTS_CODE_OPENAI_BASE_URL",
            "DEEPAGENTS_CODE_OPENAI_API_BASE",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_warns_when_key_prefixed_but_base_url_plain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Prefixed key + plain base URL (no prefixed base URL) emits a DEBUG line."""
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("openai")

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "DEEPAGENTS_CODE_OPENAI_API_KEY" in m and "OPENAI_BASE_URL" in m
            for m in messages
        )
        # The secret value and the URL value must never appear in the log.
        assert all("sk-secret-value" not in m for m in messages)
        assert all("https://gateway.example/v1" not in m for m in messages)

    def test_no_warning_when_both_prefixed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A matching prefixed base URL means the pair shares a source: no warning."""
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_OPENAI_BASE_URL", "https://gateway.example/v1"
        )

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("openai")

        assert not caplog.records

    def test_no_warning_when_key_is_plain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plain key with a plain base URL is a same-tier pair: no warning."""
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("openai")

        assert not caplog.records

    def test_no_warning_when_no_base_url_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A prefixed key with no endpoint at all has nothing to mismatch."""
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("openai")

        assert not caplog.records

    def test_no_warning_when_prefixed_key_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An empty prefixed key does not resolve from the prefixed tier: no warning.

        Symmetric to `test_empty_prefixed_base_url_is_not_treated_as_plain`: the
        key half of the pair must be *present and non-empty* for a split to exist.
        """
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("openai")

        assert not caplog.records

    def test_no_warning_when_provider_has_no_base_url_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A provider with a key env var but no base-URL env var returns early.

        `google_vertexai` maps to `GOOGLE_CLOUD_PROJECT` for credentials but has
        no entry in `PROVIDER_BASE_URL_ENV`, so there is no endpoint variable to
        compare against.
        """
        from deepagents_code.model_config import warn_on_split_credential_source

        monkeypatch.setenv("DEEPAGENTS_CODE_GOOGLE_CLOUD_PROJECT", "my-project")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            warn_on_split_credential_source("google_vertexai")

        assert not caplog.records

    def test_warns_for_config_declared_env_vars(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """The prefix is applied to config-declared env names, not just built-ins.

        A `config.toml` provider that declares its own `api_key_env` /
        `base_url_env` participates in the same split-source detection.
        """
        from deepagents_code import model_config
        from deepagents_code.model_config import (
            clear_caches,
            warn_on_split_credential_source,
        )

        for var in (
            "MYCO_KEY",
            "DEEPAGENTS_CODE_MYCO_KEY",
            "MYCO_BASE_URL",
            "DEEPAGENTS_CODE_MYCO_BASE_URL",
        ):
            monkeypatch.delenv(var, raising=False)

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.myco]
api_key_env = "MYCO_KEY"
base_url_env = "MYCO_BASE_URL"
models = ["m1"]
""")
        monkeypatch.setenv("DEEPAGENTS_CODE_MYCO_KEY", "sk-secret-value")
        monkeypatch.setenv("MYCO_BASE_URL", "https://gateway.example/myco")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"),
        ):
            clear_caches()
            warn_on_split_credential_source("myco")

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "DEEPAGENTS_CODE_MYCO_KEY" in m and "MYCO_BASE_URL" in m for m in messages
        )

    def test_no_warning_when_config_base_url_literal_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """A `config.toml` `base_url` literal wins over env vars: no env split."""
        from deepagents_code import model_config
        from deepagents_code.model_config import (
            clear_caches,
            warn_on_split_credential_source,
        )

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openai]
base_url = "https://configured.example/v1"
""")
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"),
        ):
            clear_caches()
            warn_on_split_credential_source("openai")

        assert not caplog.records


class TestThreadColumnPersistence:
    """Tests for thread selector column visibility persistence."""


class TestThreadRelativeTimePersistence:
    """Tests for thread relative-time preference persistence."""


class TestThreadSortOrderPersistence:
    """Tests for thread sort-order preference persistence."""


class TestThreadScopePersistence:
    """Tests for thread-selector directory-scope persistence."""


class TestThreadConfigCoalesced:
    """Tests for the coalesced `load_thread_config()` helper."""

    def test_corrupt_toml_returns_defaults(self, tmp_path: Path) -> None:
        """A corrupt config file should return defaults without crashing."""
        from deepagents_code.model_config import load_thread_config

        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid TOML {{{{")
        cfg = load_thread_config(config_path)
        assert cfg.columns == THREAD_COLUMN_DEFAULTS
        assert cfg.relative_time is True
        assert cfg.sort_order == "updated_at"
        assert cfg.scope == "cwd"


class TestResolveEnvVar:
    """Tests for resolve_env_var prefix override."""

    def test_prefix_beats_canonical_and_logs_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Prefixed variables take priority and log their source only once."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
        monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-override")
        caplog.set_level(logging.DEBUG, logger="deepagents_code.model_config")
        from deepagents_code.model_config import (
            reset_env_resolution_log,
            resolve_env_var,
        )

        reset_env_resolution_log()
        try:
            assert resolve_env_var("ANTHROPIC_API_KEY") == "sk-override"
            assert resolve_env_var("ANTHROPIC_API_KEY") == "sk-override"
            assert (
                caplog.messages.count(
                    "Resolved ANTHROPIC_API_KEY from DEEPAGENTS_CODE_ANTHROPIC_API_KEY"
                )
                == 1
            )
        finally:
            reset_env_resolution_log()

    def test_reset_allows_resolution_to_be_logged_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Resetting resolution diagnostics starts a new logging generation."""
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-prefixed")
        caplog.set_level(logging.DEBUG, logger="deepagents_code.model_config")
        from deepagents_code.model_config import (
            reset_env_resolution_log,
            resolve_env_var,
        )

        reset_env_resolution_log()
        try:
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            reset_env_resolution_log()
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert (
                caplog.messages.count(
                    "Resolved OPENAI_API_KEY from DEEPAGENTS_CODE_OPENAI_API_KEY"
                )
                == 2
            )
        finally:
            reset_env_resolution_log()

    def test_debug_disabled_resolution_still_logs_once_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A resolve while DEBUG is off must not consume the one-time log slot."""
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-prefixed")
        from deepagents_code.model_config import (
            reset_env_resolution_log,
            resolve_env_var,
        )

        reset_env_resolution_log()
        try:
            # DEBUG disabled: resolve succeeds but records nothing, so the name
            # must not be marked as already-logged.
            caplog.set_level(logging.INFO, logger="deepagents_code.model_config")
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert caplog.messages == []

            # DEBUG enabled: the first resolution should still emit exactly once.
            caplog.set_level(logging.DEBUG, logger="deepagents_code.model_config")
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert (
                caplog.messages.count(
                    "Resolved OPENAI_API_KEY from DEEPAGENTS_CODE_OPENAI_API_KEY"
                )
                == 1
            )
        finally:
            reset_env_resolution_log()


class TestUnknownProviderError:
    """Tests for the structured `UnknownProviderError` exception."""


class TestModelConfigLoad:
    """Tests for ModelConfig.load() method."""

    def test_returns_empty_config_when_models_section_is_not_a_table(
        self, tmp_path, caplog
    ):
        """Valid TOML with a scalar `models` falls back instead of raising.

        `load()` must be total: a structurally wrong config surfaces as an
        AttributeError from `.get(...)` after a clean parse, which callers like
        the /auth modal do not guard against.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('models = "oops"\n')

        with caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"):
            config = ModelConfig.load(config_path)

        assert config.default_model is None
        assert config.providers == {}
        assert any("structurally invalid" in r.getMessage() for r in caplog.records)

    def test_loads_model_allowlist(self, tmp_path: Path) -> None:
        """Loads an ordered allowlist and records its user source."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["openai:gpt-5.6-terra", "ollama:qwen3:4b"]\n'
        )

        config = ModelConfig.load(config_path)

        assert config.allowed_models == (
            "openai:gpt-5.6-terra",
            "ollama:qwen3:4b",
        )
        assert config.allowed_models_source == "config.toml"

    def test_empty_model_allowlist_is_distinct_from_absent(
        self, tmp_path: Path
    ) -> None:
        """An explicit empty list remains an active deny-all policy."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[models]\nallowed = []\n")

        config = ModelConfig.load(config_path)

        assert config.allowed_models == ()
        assert config.allowed_models_source == "config.toml"

    @pytest.mark.parametrize(
        "declaration",
        [
            pytest.param('allowed = ["openai:gpt", "broken"]', id="bad-entry"),
            pytest.param('allowed = "openai:gpt"', id="string-not-list"),
            pytest.param("allowed = 3", id="wrong-type"),
        ],
    )
    def test_malformed_user_allowlist_denies_all(
        self, tmp_path: Path, declaration: str
    ) -> None:
        """A malformed voluntary list fails closed instead of disabling policy.

        `models.allowed` has no manifest default, so an unparseable list would
        otherwise resolve to `None` -- which means *unrestricted*. A typo must
        not silently switch off the guardrail the user asked for.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"[models]\n{declaration}\n")

        config = ModelConfig.load(config_path)

        assert config.allowed_models == ()
        assert config.allowed_models_source is not None
        assert "malformed" in config.allowed_models_source
        assert not config.is_model_allowed("openai:gpt")

    def test_malformed_allowlist_error_names_the_defect(self, tmp_path: Path) -> None:
        """The resulting error says the list is malformed, not merely empty."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nallowed = ["broken"]\n')

        error = ModelConfig.load(config_path).policy_error("openai:gpt-5")

        assert error is not None
        assert "[models].allowed is malformed" in str(error)

    def test_absent_allowlist_stays_unrestricted(self, tmp_path: Path) -> None:
        """No declaration at all is unrestricted, unlike a malformed one."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[models]\ndefault = 'openai:gpt-5'\n")

        config = ModelConfig.load(config_path)

        assert config.allowed_models is None
        assert config.allowed_models_source is None
        assert config.is_model_allowed("anything:at-all")

    def test_loads_provider_display_metadata(self, tmp_path):
        """Loads provider metadata used by auth UI."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
display_name = "My Gateway"
short_name = "Gateway"
api_key_url = "https://gateway.example/keys"
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        config = ModelConfig.load(config_path)

        assert config.get_provider_display_name("my_gateway") == "My Gateway"
        assert config.get_provider_short_name("my_gateway") == "Gateway"
        assert (
            config.get_provider_api_key_url("my_gateway")
            == "https://gateway.example/keys"
        )
        assert config.get_provider_display_name("missing") is None
        assert config.get_provider_short_name("missing") is None
        assert config.get_provider_api_key_url("missing") is None

    def test_corrupt_toml_returns_empty_config(self, tmp_path, caplog):
        """Corrupt TOML file returns empty config and logs a warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[invalid toml content")

        with caplog.at_level(logging.WARNING):
            config = ModelConfig.load(config_path)

        assert config.default_model is None
        assert config.providers == {}
        assert any("invalid TOML syntax" in r.message for r in caplog.records)

    def test_bare_summarization_default_warns_but_loads(self, tmp_path, caplog):
        """A bare name is legitimate -- `create_model` auto-detects the provider.

        So this warns rather than rejecting, matching the sibling
        `auto_classifier` check. The value must still load: an unresolvable
        spec is caught later, at the first compaction.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nsummarization_default = "gpt-5.4-mini"\n')

        with caplog.at_level(logging.WARNING):
            config = ModelConfig.load(config_path)

        assert config.summarization_default_model == "gpt-5.4-mini"
        assert "summarization_default_model" in caplog.text
        assert "provider:model" in caplog.text

    def test_loads_summarization_default_model(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\ndefault = "anthropic:claude-sonnet-4-5"\n'
            'summarization_default = "openai:gpt-5.4-mini"\n'
        )

        config = ModelConfig.load(config_path)

        assert config.default_model == "anthropic:claude-sonnet-4-5"
        assert config.summarization_default_model == "openai:gpt-5.4-mini"

    def test_qualified_summarization_default_does_not_warn(self, tmp_path, caplog):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nsummarization_default = "openai:gpt-5.4-mini"\n'
        )

        with caplog.at_level(logging.WARNING):
            ModelConfig.load(config_path)

        assert "summarization_default_model" not in caplog.text


class TestModelConfigGetAllModels:
    """Tests for ModelConfig.get_all_models() method."""


class TestModelConfigGetProviderForModel:
    """Tests for ModelConfig.get_provider_for_model() method."""


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
        """Returns True when only the DEEPAGENTS_CODE_ prefixed var is set."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
models = ["claude-sonnet-4-5"]
api_key_env = "ANTHROPIC_API_KEY"
""")
        config = ModelConfig.load(config_path)

        with patch.dict(
            "os.environ",
            {"DEEPAGENTS_CODE_ANTHROPIC_API_KEY": "sk-prefixed"},
            clear=True,
        ):
            assert config.has_credentials("anthropic") is True


class TestModelConfigGetBaseUrl:
    """Tests for ModelConfig.get_base_url() method."""

    def test_baseten_base_url_precedes_legacy_api_base(self, monkeypatch):
        """Baseten follows `langchain-baseten` endpoint env precedence."""
        monkeypatch.setenv("BASETEN_BASE_URL", "https://new.example/v1")
        monkeypatch.setenv("BASETEN_API_BASE", "https://legacy.example/v1")
        config = ModelConfig()

        assert config.get_base_url("baseten") == "https://new.example/v1"

    def test_baseten_falls_back_to_legacy_api_base(self, monkeypatch):
        """Baseten still honors the legacy endpoint env var."""
        monkeypatch.setenv("BASETEN_API_BASE", "https://legacy.example/v1")
        config = ModelConfig()

        assert config.get_base_url("baseten") == "https://legacy.example/v1"

    def test_falls_back_to_stored_base_url_for_provider_without_env_var(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """A `/auth` endpoint resolves for a provider with no base-URL env var.

        Some providers have an API-key env var but no dedicated base-URL env var,
        so steps 1-2 find nothing. The stored endpoint must still resolve here so
        it reaches the model as the `base_url` kwarg — otherwise a value saved in
        `/auth` is silently lost.
        """
        from deepagents_code import auth_store

        auth_store.set_stored_key("litellm", "k", base_url="https://proxy.example/v1")
        config = ModelConfig()

        assert config.get_base_url("litellm") == "https://proxy.example/v1"

    def test_config_literal_wins_over_stored_base_url(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """A `config.toml` literal still wins over the stored endpoint."""
        from deepagents_code import auth_store

        auth_store.set_stored_key("baseten", "k", base_url="https://stored.example/v1")
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.baseten]
base_url = "https://config.example/v1"
models = ["m1"]
""")
        config = ModelConfig.load(config_path)

        assert config.get_base_url("baseten") == "https://config.example/v1"

    def test_blank_stored_base_url_yields_none(
        self,
        fake_state_dir: Path,  # noqa: ARG002
    ) -> None:
        """A stored key with no endpoint leaves `get_base_url` at the default."""
        from deepagents_code import auth_store

        auth_store.set_stored_key("baseten", "k")
        config = ModelConfig()

        assert config.get_base_url("baseten") is None

    def test_corrupt_store_does_not_raise(
        self,
        fake_state_dir: Path,
    ) -> None:
        """A corrupt credential store resolves to None, never propagating."""
        fake_state_dir.mkdir(parents=True, exist_ok=True)
        (fake_state_dir / "auth.json").write_text("{ not valid json")
        config = ModelConfig()

        assert config.get_base_url("baseten") is None


class TestGetDefaultBaseUrlEnv:
    """Tests for `get_default_base_url_env` — the var a blank save falls back to.

    A blank save clears the *plain* endpoint vars, so only the
    `DEEPAGENTS_CODE_`-prefixed name still supplies a value afterward. The
    helper returns that name (for display), never the plain name or a value.
    """

    def test_returns_prefixed_alternate_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prefixed alternate is named when it supplies the blank fallback."""
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_BASETEN_API_BASE", "https://legacy.example/v1"
        )
        assert (
            model_config.get_default_base_url_env("baseten")
            == "DEEPAGENTS_CODE_BASETEN_API_BASE"
        )

    def test_canonical_prefixed_name_precedes_alternate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The helper matches `get_base_url` provider env precedence."""
        monkeypatch.setenv("DEEPAGENTS_CODE_BASETEN_BASE_URL", "https://new.example/v1")
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_BASETEN_API_BASE", "https://legacy.example/v1"
        )
        assert (
            model_config.get_default_base_url_env("baseten")
            == "DEEPAGENTS_CODE_BASETEN_BASE_URL"
        )


class TestModelConfigGetApiKeyEnv:
    """Tests for ModelConfig.get_api_key_env() method."""


class TestSaveDefaultModel:
    """Tests for save_default_model() function."""


class TestSaveGoalAutoAcceptCriteria:
    """Tests for the first-run goal criteria preference writer."""

    def test_returns_false_when_config_cannot_be_written(self, tmp_path) -> None:
        """An unwritable config path should preserve the boolean failure contract."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("blocked", encoding="utf-8")

        assert (
            model_config.save_goal_auto_accept_criteria(
                True,
                blocker / "config.toml",
            )
            is False
        )


class TestClearDefaultModel:
    """Tests for clear_default_model() function."""


class TestAutoClassifierModelPersistence:
    """Tests for the `[models].auto_classifier` writer/reader pair."""

    @pytest.mark.parametrize(
        ("contents", "label"),
        [("", "empty file"), ('[permissions]\nmode = "auto"\n', "no [models] table")],
    )
    def test_clear_is_a_silent_noop_when_no_models_table_exists(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        contents: str,
        label: str,
    ) -> None:
        """An ordinary config with nothing stored clears cleanly and quietly.

        `data.get("models")` yields `None` here, which must not be mistaken for
        the wrong-shape branch: telling the user their `[models]` section is
        broken when they simply never stored a model sends them to repair a file
        that is fine.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text(contents, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert model_config.clear_auto_classifier_model(config_path) is True
            assert model_config.clear_default_model(config_path) is True

        assert "non-table [models] section" not in caplog.text, label


class TestEffortPersistence:
    """Tests for per-model reasoning effort persistence."""

    def test_concurrent_config_writer_preserves_effort(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concurrent thread-preference save cannot drop an effort save."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "openai:gpt-5.5"\n')
        barrier = threading.Barrier(2)
        original_load = tomllib.load

        def synchronized_load(file: Any) -> dict[str, Any]:  # noqa: ANN401
            data = original_load(file)
            # With the shared lock, the first writer times out before the
            # second can read. An unlocked implementation reaches both sides
            # and deterministically exposes the lost update.
            with suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=1)
            return data

        monkeypatch.setattr(model_config.tomllib, "load", synchronized_load)
        columns = {**THREAD_COLUMN_DEFAULTS, "messages": False}
        results: list[bool] = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    save_effort_for_model("openai:gpt-5.6-luna", "max", config_path)
                )
            ),
            threading.Thread(
                target=lambda: results.append(save_thread_columns(columns, config_path))
            ),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert all(results)
        assert load_effort_for_model("openai:gpt-5.6-luna", config_path) == "max"
        assert load_thread_columns(config_path) == columns


class TestModelPersistenceBetweenSessions:
    """Tests for model selection persistence across app sessions.

    These tests verify that when a user switches models using /model command,
    the selection persists when the CLI is restarted (new session).
    """


class TestGetAvailableModels:
    """Tests for get_available_models() function."""


class TestGetAvailableModelsMergesConfig:
    """Tests for get_available_models() merging config-file providers."""

    def test_allowlist_filters_discovery_and_additive_config_models(
        self, tmp_path: Path
    ) -> None:
        """Registration remains additive before the final policy filter."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["anthropic:claude-custom"]\n\n'
            '[models.providers.anthropic]\nmodels = ["claude-custom"]\n'
        )
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
                "deepagents_code.model_config._load_provider_profiles",
                side_effect=mock_load,
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert models == {"anthropic": ["claude-custom"]}


class TestOllamaModelDiscovery:
    """Tests for auto-populating the switcher from a running Ollama daemon."""

    @staticmethod
    def _patch_registry() -> AbstractContextManager[object]:
        """Patch the langchain registry so `ollama` is a known provider."""
        return patch(
            "deepagents_code.model_config._get_builtin_providers",
            return_value={
                "ollama": ("langchain_ollama.chat_models", "ChatOllama"),
            },
        )

    @staticmethod
    def _empty_profiles_loader(module_path: str) -> dict[str, Any]:
        """Pretend `langchain_ollama` ships no profile data."""
        if module_path == "langchain_ollama.data._profiles":
            return {}
        msg = "not installed"
        raise ImportError(msg)

    def test_unreachable_daemon_probed_and_logged_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unreachable daemon is probed and logged once per reload."""
        reachable = MagicMock(return_value=False)
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable", reachable
        )

        with (
            patch("urllib.request.urlopen") as urlopen,
            caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"),
        ):
            assert (
                model_config._get_ollama_installed_models("http://localhost:11434")
                == []
            )
            assert (
                model_config._get_ollama_installed_models("http://localhost:11434")
                == []
            )

        # Preflight ran once; the negative result was cached for the second call.
        reachable.assert_called_once()
        urlopen.assert_not_called()
        assert caplog.text.count("Ollama daemon not detected") == 1

    def test_unreachable_daemon_logged_once_across_callers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The two startup callers share one probe and one "not detected" log.

        Regression: an unreachable daemon was probed -- and logged "not
        detected" -- once by `get_available_models` and again by
        `get_model_profiles`, so the line appeared twice per reload. Drives the
        real callers rather than `_get_ollama_installed_models` directly.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        monkeypatch.delenv("DEEPAGENTS_CODE_OLLAMA_DISCOVERY", raising=False)
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable",
            MagicMock(return_value=False),
        )

        with (
            self._patch_registry(),
            patch(
                "deepagents_code.model_config._load_provider_profiles",
                side_effect=self._empty_profiles_loader,
            ),
            patch(
                "deepagents_code.model_config.importlib.util.find_spec",
                return_value=object(),
            ),
            patch("urllib.request.urlopen") as urlopen,
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"),
        ):
            get_available_models()
            get_model_profiles()

        urlopen.assert_not_called()
        assert caplog.text.count("Ollama daemon not detected") == 1

    @pytest.mark.parametrize("endpoint", [None, "http://localhost:11434/"])
    def test_unreachable_cache_key_matches_across_normalization(
        self, endpoint: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`None` and a trailing slash resolve to the same negative-cache key.

        The add-site (`_fetch_ollama_installed_models`) and check-site
        (`_get_ollama_installed_models`) must normalize identically, else the
        empty result is keyed differently from the lookup and the daemon is
        re-probed every call instead of once per reload.
        """
        reachable = MagicMock(return_value=False)
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable", reachable
        )

        with patch("urllib.request.urlopen"):
            assert model_config._get_ollama_installed_models(endpoint) == []
            assert model_config._get_ollama_installed_models(endpoint) == []

        reachable.assert_called_once()


class _BytesContext:
    """Minimal context manager wrapping a bytes payload for fake `urlopen`."""

    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)

    def __enter__(self) -> io.BytesIO:
        return self._body

    def __exit__(self, *_exc: object) -> None:
        self._body.close()


class TestFetchOllamaInstalledModels:
    """Tests for the `_fetch_ollama_installed_models` HTTP probe."""

    @pytest.fixture(autouse=True)
    def _assume_host_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bypass the TCP presence preflight so HTTP parsing paths are exercised."""
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable",
            lambda *_args, **_kwargs: True,
        )

    def test_skips_http_probe_when_host_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable daemon short-circuits before any HTTP request."""
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable",
            lambda *_args, **_kwargs: False,
        )

        with patch("urllib.request.urlopen") as fake:
            assert (
                model_config._fetch_ollama_installed_models("http://localhost:11434")
                == []
            )

        fake.assert_not_called()

    def test_hosted_endpoint_skips_tcp_preflight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hosted endpoints proceed through the proxy-aware HTTP probe."""
        import json

        reachable = MagicMock(return_value=False)
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable", reachable
        )

        with patch(
            "urllib.request.urlopen",
            return_value=_BytesContext(json.dumps({"models": []}).encode("utf-8")),
        ) as urlopen:
            assert (
                model_config._fetch_ollama_installed_models(
                    "https://ollama.example.com"
                )
                == []
            )

        reachable.assert_not_called()
        urlopen.assert_called_once()

    def test_forwards_normalized_base_and_timeout_to_preflight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rstrip'd base and the caller's timeout reach the preflight."""
        import json

        reachable = MagicMock(return_value=True)
        monkeypatch.setattr(
            "deepagents_code.model_config._ollama_host_reachable", reachable
        )
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_OLLAMA_API_KEY", raising=False)

        with patch(
            "urllib.request.urlopen",
            return_value=_BytesContext(json.dumps({"models": []}).encode("utf-8")),
        ):
            assert (
                model_config._fetch_ollama_installed_models(
                    "http://localhost:11434/", timeout=0.5
                )
                == []
            )

        reachable.assert_called_once_with("http://localhost:11434", timeout=0.5)

    def test_returns_sorted_names_from_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parses `{"models": [{"name": ...}]}` and sorts results."""
        import json
        from urllib.request import Request

        captured_url: list[str] = []
        captured_timeout: list[float] = []
        captured_headers: list[dict[str, str]] = []

        def fake_urlopen(request: Request, timeout: float) -> _BytesContext:
            captured_url.append(request.full_url)
            captured_timeout.append(timeout)
            captured_headers.append(dict(request.header_items()))
            payload = {"models": [{"name": "qwen3:4b"}, {"name": "llama3"}]}
            return _BytesContext(json.dumps(payload).encode("utf-8"))

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_OLLAMA_API_KEY", raising=False)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = model_config._fetch_ollama_installed_models(
                "http://localhost:11434"
            )

        assert result == ["llama3", "qwen3:4b"]
        assert captured_url == ["http://localhost:11434/api/tags"]
        assert captured_timeout == [model_config.OLLAMA_DISCOVERY_TIMEOUT_SECONDS]
        assert "Authorization" not in {k.title() for k in captured_headers[0]}

    def test_forwards_optional_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`OLLAMA_API_KEY` is forwarded to local discovery endpoints."""
        import json
        from urllib.request import Request

        captured_headers: list[dict[str, str]] = []

        def fake_urlopen(
            request: Request,
            timeout: float,  # noqa: ARG001
        ) -> _BytesContext:
            captured_headers.append(dict(request.header_items()))
            return _BytesContext(json.dumps({"models": []}).encode("utf-8"))

        monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            model_config._fetch_ollama_installed_models("http://localhost:11434")

        # Header names are title-cased by urllib.
        assert captured_headers[0].get("Authorization") == "Bearer secret-token"

    def test_does_not_forward_optional_api_key_to_remote_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discovery does not send credentials to non-local endpoints."""
        import json
        from urllib.request import Request

        captured_headers: list[dict[str, str]] = []

        def fake_urlopen(
            request: Request,
            timeout: float,  # noqa: ARG001
        ) -> _BytesContext:
            captured_headers.append(dict(request.header_items()))
            return _BytesContext(json.dumps({"models": []}).encode("utf-8"))

        monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            model_config._fetch_ollama_installed_models("https://ollama.example.com")

        assert "Authorization" not in captured_headers[0]


class TestOllamaHostReachable:
    """Tests for the `_ollama_host_reachable` TCP presence preflight."""

    def test_true_when_connection_succeeds(self) -> None:
        """A successful TCP connection reports the daemon as present.

        Also pins that the default timeout reaches `socket.create_connection`:
        the preflight exists to fail *fast*, so a dropped timeout would let an
        absent host stall on the OS connect timeout -- the hang this removes.
        """
        captured: list[tuple[tuple[str, int], float]] = []

        def fake_create_connection(
            address: tuple[str, int], *, timeout: float
        ) -> MagicMock:
            captured.append((address, timeout))
            return MagicMock()

        with patch("socket.create_connection", side_effect=fake_create_connection):
            assert model_config._ollama_host_reachable("http://localhost:11434") is True

        assert captured == [
            (("localhost", 11434), model_config.OLLAMA_DISCOVERY_TIMEOUT_SECONDS)
        ]

    def test_forwards_explicit_timeout(self) -> None:
        """A caller-supplied timeout is forwarded to the socket connect."""
        captured: list[float] = []

        def fake_create_connection(
            address: tuple[str, int],  # noqa: ARG001
            *,
            timeout: float,
        ) -> MagicMock:
            captured.append(timeout)
            return MagicMock()

        with patch("socket.create_connection", side_effect=fake_create_connection):
            assert (
                model_config._ollama_host_reachable(
                    "http://localhost:11434", timeout=0.25
                )
                is True
            )

        assert captured == [0.25]

    def test_false_and_silent_when_connection_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Connection refused (an `OSError`) reports absent without warning."""
        refused = ConnectionRefusedError(61, "Connection refused")

        def boom(*_args: object, **_kwargs: object) -> None:
            raise refused

        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"),
            patch("socket.create_connection", side_effect=boom),
        ):
            assert (
                model_config._ollama_host_reachable("http://localhost:11434") is False
            )

        assert caplog.records == []

    def test_defers_to_probe_on_connect_timeout(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A connect timeout is ambiguous, so it defers to the HTTP probe.

        A present-but-slow or still-booting daemon times out just like an
        absent one; reporting it absent here would negatively cache the empty
        result and hide a working daemon until the next reload. Returning
        "reachable" lets the HTTP probe -- which may now succeed -- decide, and
        leaves the empty result uncached. Silent, since a timeout is expected.
        """

        def boom(*_args: object, **_kwargs: object) -> None:
            raise TimeoutError

        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"),
            patch("socket.create_connection", side_effect=boom),
        ):
            assert model_config._ollama_host_reachable("http://localhost:11434") is True

        assert caplog.records == []

    def test_false_and_warns_when_error_unexpected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-`OSError` failure reports absent and surfaces a warning.

        Covers `pytest-socket`'s `SocketBlockedError`, which inherits from
        `Exception` (not `OSError`); an unexpected error here is a possible
        real bug, so it is logged rather than silently swallowed.
        """
        blocked = RuntimeError("sockets disabled")

        def boom(*_args: object, **_kwargs: object) -> None:
            raise blocked

        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"),
            patch("socket.create_connection", side_effect=boom),
        ):
            assert (
                model_config._ollama_host_reachable("http://localhost:11434") is False
            )

        assert any("unexpected RuntimeError" in r.getMessage() for r in caplog.records)

    def test_defers_to_probe_when_host_unparseable(self) -> None:
        """A URL without a host defers to the HTTP probe instead of blocking it."""
        with patch("socket.create_connection") as fake:
            assert model_config._ollama_host_reachable("http://") is True

        fake.assert_not_called()

    @pytest.mark.parametrize(
        "endpoint",
        ["http://localhost:notaport", "http://localhost:99999"],
    )
    def test_defers_to_probe_when_port_invalid(self, endpoint: str) -> None:
        """An invalid port defers to the best-effort HTTP probe."""
        with patch("socket.create_connection") as fake:
            assert model_config._ollama_host_reachable(endpoint) is True

        fake.assert_not_called()

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("https://ollama.example.com", ("ollama.example.com", 443)),
            ("http://ollama.internal", ("ollama.internal", 80)),
        ],
    )
    def test_defaults_scheme_port_when_absent(
        self, endpoint: str, expected: tuple[str, int]
    ) -> None:
        """A schemed URL without an explicit port falls back to the scheme default.

        The preflight and the HTTP probe must agree on the target, so a portless
        `http` host resolves to 80 and `https` to 443 -- matching what urllib
        would connect to.
        """
        captured: list[tuple[str, int]] = []

        def fake_create_connection(
            address: tuple[str, int],
            *,
            timeout: float,  # noqa: ARG001
        ) -> MagicMock:
            captured.append(address)
            return MagicMock()

        with patch("socket.create_connection", side_effect=fake_create_connection):
            assert model_config._ollama_host_reachable(endpoint) is True

        assert captured == [expected]


class TestFetchOllamaInstalledModelProfiles:
    """Tests for Ollama `/api/show` profile discovery."""

    def test_posts_model_names_to_show_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fetches local `/api/show` with bearer auth and parses context length."""
        import json
        from urllib.request import Request

        captured_url: list[str] = []
        captured_body: list[dict[str, str]] = []
        captured_headers: list[dict[str, str]] = []

        def fake_urlopen(request: Request, timeout: float) -> _BytesContext:
            assert timeout == model_config.OLLAMA_DISCOVERY_TIMEOUT_SECONDS
            captured_url.append(request.full_url)
            captured_headers.append(dict(request.header_items()))
            data = cast("bytes", request.data)
            captured_body.append(json.loads(data.decode("utf-8")))
            payload = {
                "model_info": {"qwen3.context_length": 262144},
                "capabilities": ["completion", "tools"],
            }
            return _BytesContext(json.dumps(payload).encode("utf-8"))

        monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            profiles = model_config._fetch_ollama_installed_model_profiles(
                "http://localhost:11434",
                ["qwen3:4b"],
            )

        assert profiles["qwen3:4b"]["max_input_tokens"] == 262144
        assert profiles["qwen3:4b"]["tool_calling"] is True
        assert captured_url == ["http://localhost:11434/api/show"]
        assert captured_body == [{"model": "qwen3:4b"}]
        assert captured_headers[0].get("Authorization") == "Bearer secret-token"

    def test_show_does_not_forward_optional_api_key_to_remote_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Profile discovery does not send credentials to non-local endpoints."""
        import json
        from urllib.request import Request

        captured_headers: list[dict[str, str]] = []

        def fake_urlopen(
            request: Request,
            timeout: float,  # noqa: ARG001
        ) -> _BytesContext:
            captured_headers.append(dict(request.header_items()))
            payload = {"model_info": {"qwen3.context_length": 262144}}
            return _BytesContext(json.dumps(payload).encode("utf-8"))

        monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            profiles = model_config._fetch_ollama_installed_model_profiles(
                "https://ollama.example.com",
                ["qwen3:4b"],
            )

        assert profiles["qwen3:4b"]["max_input_tokens"] == 262144
        assert "Authorization" not in captured_headers[0]


class TestDisabledProviders:
    """Tests for provider hiding via `enabled = false`."""


class TestIsProviderEnabled:
    """Tests for ModelConfig.is_provider_enabled()."""


class TestProfileModuleFromClassPath:
    """Tests for _profile_module_from_class_path() helper."""


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
                "deepagents_code.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            profiles = get_model_profiles()

        assert not any(key.startswith("baseten:") for key in profiles)

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
                "deepagents_code.model_config._load_provider_profiles",
                side_effect=ImportError("not installed"),
            ),
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
        ):
            models = get_available_models()

        assert "baseten" not in models


class TestHasProviderCredentialsFallback:
    """Tests for has_provider_credentials() falling back to ModelConfig."""

    def test_falls_back_to_config_no_key_required(self, tmp_path):
        """Returns True for local Ollama with no api_key_env."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert has_provider_credentials("ollama") is True

    def test_ollama_remote_without_key_is_unknown(self, tmp_path):
        """Remote Ollama without optional auth should not claim local readiness."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
base_url = "https://ollama.example.com"
models = ["llama3"]
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            status = get_provider_auth_status("ollama")
            legacy = has_provider_credentials("ollama")

        assert status.state is ProviderAuthState.UNKNOWN
        assert status.env_var == "OLLAMA_API_KEY"
        assert "OLLAMA_API_KEY" in (status.detail or "")
        assert legacy is None

    def test_ollama_optional_api_key_is_configured(self, tmp_path):
        """OLLAMA_API_KEY marks Ollama as configured for cloud/hosted use."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
base_url = "https://ollama.example.com"
models = ["llama3"]
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}, clear=True),
        ):
            status = get_provider_auth_status("ollama")
            legacy = has_provider_credentials("ollama")

        assert status.state is ProviderAuthState.CONFIGURED
        assert status.env_var == "OLLAMA_API_KEY"
        assert legacy is True

    @pytest.mark.parametrize("provider", ["google_anthropic_vertex", "google_vertexai"])
    def test_vertex_missing_project_uses_implicit_auth(self, provider: str):
        """Vertex providers should allow ADC when project env vars are unset."""
        with patch.dict("os.environ", {}, clear=True):
            status = get_provider_auth_status(provider)
            legacy = has_provider_credentials(provider)

        assert status.state is ProviderAuthState.IMPLICIT
        assert legacy is True

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


class TestIsLangsmithGatewayHost:
    """Tests for the shared LangSmith gateway host predicate.

    Two modules gate behavior on this (`doctor` classifies tracing endpoints,
    `app` decides whether a provider key mismatches the gateway it is being
    sent through), so its boundary cases are pinned directly rather than only
    through callers. `cold_cache` deliberately does *not* use it: its
    cross-format decision comes from the model-name prefix alone and is
    host-independent.
    """


class TestIsLocalEndpoint:
    """Tests for _is_local_endpoint URL classification."""


class TestProviderAuthStatusBranches:
    """Direct coverage of get_provider_auth_status states beyond Ollama."""

    def test_managed_state_for_class_path_provider(self, tmp_path: Path) -> None:
        """class_path without api_key_env returns MANAGED with custom-auth detail."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.cis]
class_path = "agent_forge.integrations:CISChat"
models = ["aviato-turbo"]
""")
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            status = get_provider_auth_status("cis")

        assert status.state is ProviderAuthState.MANAGED
        assert status.detail == "custom auth"
        assert status.env_var is None

    def test_missing_state_for_known_provider_without_env(self) -> None:
        """Hardcoded provider with no env set returns MISSING with the env name."""
        with patch.dict("os.environ", {}, clear=True):
            status = get_provider_auth_status("anthropic")

        assert status.state is ProviderAuthState.MISSING
        assert status.env_var == "ANTHROPIC_API_KEY"
        assert status.blocks_start is True

    def test_missing_state_for_config_provider_with_empty_env(
        self,
        tmp_path: Path,
    ) -> None:
        """Config provider with api_key_env set but unset env returns MISSING."""
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
            status = get_provider_auth_status("fireworks")

        assert status.state is ProviderAuthState.MISSING
        assert status.env_var == "FIREWORKS_API_KEY"

    def test_ollama_host_env_drives_locality(self, tmp_path: Path) -> None:
        """OLLAMA_HOST env var controls local vs. remote when no base_url is set."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {"OLLAMA_HOST": "https://ollama.example.com"},
                clear=True,
            ),
        ):
            status = get_provider_auth_status("ollama")

        assert status.state is ProviderAuthState.UNKNOWN
        assert status.env_var == "OLLAMA_API_KEY"


class TestProviderAuthStatusMissingDetail:
    """Tests for ProviderAuthStatus.missing_detail() rendering."""

    def test_with_env_var_uses_env_var_message(self) -> None:
        """env_var presence yields a 'not set or is empty' message."""
        status = ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider="anthropic",
            env_var="ANTHROPIC_API_KEY",
        )
        assert status.missing_detail() == "ANTHROPIC_API_KEY is not set or is empty"

    def test_with_detail_only_falls_back_to_detail(self) -> None:
        """Without env_var but with a detail string, returns the detail."""
        status = ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider="custom",
            detail="bespoke auth missing",
        )
        assert status.missing_detail() == "bespoke auth missing"

    def test_without_env_var_or_detail_returns_unknown_provider_hint(self) -> None:
        """Bare MISSING falls back to a 'not recognized' hint."""
        status = ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider="phantom",
        )
        message = status.missing_detail()
        assert "phantom" in message
        assert "not recognized" in message


class TestModelConfigGetClassPath:
    """Tests for ModelConfig.get_class_path() method."""


class TestModelConfigGetKwargs:
    """Tests for ModelConfig.get_kwargs() method."""


class TestModelConfigGetKwargsPerModel:
    """Tests for ModelConfig.get_kwargs() with per-model overrides."""

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


class TestModelConfigGetEffectiveKwargs:
    """Tests for effective request kwargs used by model construction and policy."""


class TestModelConfigGetProfileOverrides:
    """Tests for ModelConfig.get_profile_overrides() method."""


class TestModelConfigValidateParams:
    """Tests for _validate() params warnings."""


class TestModelConfigValidateClassPath:
    """Tests for _validate() class_path validation."""


class TestGetProviderProfileModules:
    """Tests for _get_provider_profile_modules()."""


class TestGetBuiltinProviders:
    """Tests for _get_builtin_providers() forward-compat helper."""


class TestLoadProviderProfiles:
    """Tests for _load_provider_profiles() direct-file loading."""


class TestGetAvailableModelsTextIO:
    """Tests for text_inputs / text_outputs filtering in get_available_models()."""


class TestModelConfigError:
    """Tests for ModelConfigError exception class."""


class TestSaveRecentModel:
    """Tests for save_recent_model() function."""

    def test_refuses_model_outside_allowlist(self, tmp_path: Path) -> None:
        """Persistence raises rather than reporting a policy block as I/O failure.

        A `False` return is indistinguishable from an unwritable file, which is
        how callers came to tell the user to check directory permissions that
        were already correct.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nallowed = ["anthropic:claude-sonnet-5"]\n')

        with pytest.raises(ModelNotAllowedError) as excinfo:
            save_recent_model("openai:gpt-5.6-terra", config_path)

        assert "not included in" in str(excinfo.value)
        assert "anthropic:claude-sonnet-5" in str(excinfo.value)
        assert "recent" not in config_path.read_text()


class TestRecentModelsMRU:
    """`load_recent_models` / `touch_recent_model` round-trip + MRU semantics."""

    def test_load_filters_entries_outside_allowlist(self, tmp_path: Path) -> None:
        """Stale MRU entries cannot reappear in the selector."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nallowed = ["anthropic:allowed"]\n')
        (tmp_path / "recent_models.json").write_text(
            '{"models": ["openai:blocked", "anthropic:allowed"]}'
        )

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert load_recent_models(state_dir=tmp_path) == ["anthropic:allowed"]

    def test_touch_refuses_entry_outside_allowlist(self, tmp_path: Path) -> None:
        """The write side of the MRU is gated too, not just the read side.

        Without this, a blocked spec could re-enter `recent_models.json` and be
        offered by the selector on the next launch.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nallowed = ["anthropic:allowed"]\n')

        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            assert touch_recent_model("openai:blocked", state_dir=tmp_path) is False
            assert touch_recent_model("anthropic:allowed", state_dir=tmp_path) is True
            assert load_recent_models(state_dir=tmp_path) == ["anthropic:allowed"]

    def test_touch_rejects_spec_without_provider_prefix(self, tmp_path):
        """Specs missing the `provider:` prefix should not be persisted."""
        assert touch_recent_model("just-a-model", state_dir=tmp_path) is False
        assert load_recent_models(state_dir=tmp_path) == []

    def test_load_ignores_malformed_payload(self, tmp_path):
        """A corrupt cache file should be treated as empty, not crash."""
        (tmp_path / "recent_models.json").write_text("not json{{", encoding="utf-8")
        assert load_recent_models(state_dir=tmp_path) == []


class TestRecentAgent:
    """save_recent_agent + load_recent_agent round-trip."""

    def test_save_same_value_preserves_file_identity(self, tmp_path):
        config_path = tmp_path / "config.toml"
        content = b'[agents]\nrecent = "coder"\n'
        config_path.write_bytes(content)
        inode = config_path.stat().st_ino

        assert save_recent_agent("coder", config_path) is True

        assert config_path.stat().st_ino == inode
        assert config_path.read_bytes() == content


class TestDefaultAgent:
    """save_default_agent + clear_default_agent + load_default_agent round-trip."""


class TestModelConfigLoadRecent:
    """Tests for ModelConfig.load() reading recent_model."""


class TestModelPrecedenceOrder:
    """Tests for model selection precedence: default > recent > env."""

    def test_disallowed_saved_values_fall_back_in_allowlist_order(
        self, tmp_path: Path
    ) -> None:
        """Stale defaults cannot outrank an allowed authenticated fallback."""
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["anthropic:claude-opus-5"]\n'
            'default = "openai:gpt-5.6-terra"\n'
        )

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
        ):
            result = _get_default_model_spec()

        assert result == "anthropic:claude-opus-5"

    def test_empty_allowlist_blocks_default_resolution(self, tmp_path: Path) -> None:
        """A deny-all list never reaches unrestricted credential fallback."""
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("[models]\nallowed = []\n")

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            pytest.raises(ModelNotAllowedError, match="allows no models"),
        ):
            _get_default_model_spec()

    def test_allowlist_falls_back_to_remote_no_auth_provider(
        self, tmp_path: Path
    ) -> None:
        """A remote Ollama endpoint with unknown auth remains a viable fallback.

        `get_provider_auth_status` reports UNKNOWN for remote no-auth providers
        because a LAN/hosted endpoint may not require credentials. Rejecting
        that state here would block startup even though `create_model()`
        deliberately permits it.
        """
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["ollama:qwen3:4b"]\n\n'
            "[models.providers.ollama]\n"
            'base_url = "https://ollama.example.com"\n'
            'models = ["qwen3:4b"]\n'
        )

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = _get_default_model_spec()

        assert result == "ollama:qwen3:4b"

    def test_allowlist_wildcard_falls_back_to_configured_models(
        self, tmp_path: Path
    ) -> None:
        """A `provider:*` entry expands to the provider's configured models.

        The wildcard itself names no model, so default resolution picks the
        first credentialed model the provider declares rather than selecting
        the wildcard literally.
        """
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["my_gateway:*"]\n\n'
            "[models.providers.my_gateway]\n"
            'base_url = "https://gateway.example.com/v1"\n'
            'api_key_env = "MY_GATEWAY_API_KEY"\n'
            'models = ["model-a", "model-b"]\n'
        )

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"MY_GATEWAY_API_KEY": "test-key"}, clear=True),
        ):
            result = _get_default_model_spec()

        assert result == "my_gateway:model-a"

    def test_allowlist_wildcard_expands_discovered_builtin_lineup(
        self, tmp_path: Path
    ) -> None:
        """A built-in `provider:*` wildcard expands registry-discovered models.

        `openai` declares no `[models.providers.openai].models`; its lineup is
        discovered from the installed provider package's profile data. With a
        credential present, default resolution must pick a discovered model
        rather than reporting "No discoverable models".
        """
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\nallowed = ["openai:*"]\n')

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
        ):
            result = _get_default_model_spec()

        assert result is not None
        provider, _, model = result.partition(":")
        assert provider == "openai"
        assert model

    def test_allowlist_wildcard_without_models_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A wildcard for a provider with no discoverable models selects nothing.

        `my_gateway` is declared with no `models` list and no registry or
        `class_path` lineup to discover, so the wildcard contributes no
        candidates. (A built-in like `openai` would expand to its discovered
        profile lineup instead.)
        """
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nallowed = ["my_gateway:*"]\n\n'
            "[models.providers.my_gateway]\n"
            'base_url = "https://gateway.example.com/v1"\n'
            'api_key_env = "MY_GATEWAY_API_KEY"\n'
        )

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"MY_GATEWAY_API_KEY": "test-key"}, clear=True),
            pytest.raises(NoAllowedModelCredentialsError, match="No discoverable"),
        ):
            _get_default_model_spec()

    def test_env_used_when_neither_set(self, tmp_path):
        """Falls back to env var auto-detection when neither default nor recent set."""
        from deepagents_code.config import _get_credentials, _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("")

        owner = _get_credentials()
        replacement = replace(
            owner.active,
            openai_api_key=None,
            anthropic_api_key="test-key",
        )
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_code.auth_store.get_stored_key", return_value=None),
            patch.object(owner, "_active", replacement),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "test-key"},
                clear=True,
            ),
        ):
            result = _get_default_model_spec()

        assert result == "anthropic:claude-opus-5"

    def test_stored_key_used_when_neither_model_set(self, tmp_path):
        """Falls back to stored TUI credentials when no env vars are set."""
        from deepagents_code.config import _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("")

        def stored_key(provider: str) -> str | None:
            return "test-key" if provider == "anthropic" else None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_code.auth_store.get_stored_key", side_effect=stored_key),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = _get_default_model_spec()

        assert result == "anthropic:claude-opus-5"

    def test_vertex_project_does_not_drive_env_default(self, tmp_path):
        """Vertex project alone should not select an automatic default model."""
        from deepagents_code.config import _get_credentials, _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("")

        owner = _get_credentials()
        replacement = replace(
            owner.active,
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            google_cloud_project="test-project",
            nvidia_api_key=None,
        )
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_code.auth_store.get_stored_key", return_value=None),
            patch.dict("os.environ", {}, clear=True),
            patch.object(owner, "_active", replacement),
            pytest.raises(ModelConfigError),
        ):
            _get_default_model_spec()

    def test_nvidia_key_does_not_drive_env_default(self, tmp_path):
        """NVIDIA key alone should not select an automatic default model."""
        from deepagents_code.config import _get_credentials, _get_default_model_spec

        config_path = tmp_path / "config.toml"
        config_path.write_text("")

        owner = _get_credentials()
        replacement = replace(
            owner.active,
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            google_cloud_project=None,
            nvidia_api_key="test-key",
        )
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch("deepagents_code.auth_store.get_stored_key", return_value=None),
            patch.dict("os.environ", {}, clear=True),
            patch.object(owner, "_active", replacement),
            pytest.raises(ModelConfigError),
        ):
            _get_default_model_spec()


class TestIsWarningSuppressed:
    """Tests for is_warning_suppressed() function."""

    def test_returns_false_on_corrupt_toml(self, tmp_path) -> None:
        """Returns False when config file has invalid TOML."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[invalid toml")

        assert is_warning_suppressed("ripgrep", config_path) is False


class TestSuppressWarning:
    """Tests for suppress_warning() function."""

    def test_reason_names_a_malformed_warnings_table(self, tmp_path: Path) -> None:
        """The cause must be distinguishable from an I/O failure.

        The two need different fixes, and a bare `False` supports only the
        generic "check file permissions" advice -- which sends a user with one
        line of bad TOML to `chmod` a file that was never unwritable.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('warnings = ["ripgrep"]\n')

        reason = suppress_warning_reason("ripgrep", config_path)

        assert reason is not None
        assert "not a table" in reason


class TestUnsuppressWarning:
    """Tests for unsuppress_warning() function."""

    def test_returns_false_on_corrupt_toml(self, tmp_path: Path) -> None:
        """Returns False when config file contains malformed TOML."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml [[[")

        result = unsuppress_warning("tavily", config_path)

        assert result is False


class TestMcpServerTrustLists:
    """Tests for the McpServerTrustLists value object itself."""

    def test_post_init_enforces_disjointness_on_direct_construction(self) -> None:
        """A name in both lists is dropped from enabled, however constructed.

        The docstring promises the invariant holds "no matter how it was
        constructed; callers need not pre-subtract" — pin that at the type level,
        independent of the loader.
        """
        lists = McpServerTrustLists(
            enabled=frozenset({"keep", "both"}),
            disabled=frozenset({"both"}),
        )

        assert lists.enabled == frozenset({"keep"})
        assert lists.disabled == frozenset({"both"})

    def test_read_error_excluded_from_equality(self) -> None:
        """`read_error` is diagnostic only and does not affect equality."""
        assert McpServerTrustLists(
            frozenset(), frozenset(), read_error="boom"
        ) == McpServerTrustLists(frozenset(), frozenset())

    def test_third_positional_argument_remains_read_error(self) -> None:
        """The pre-approval constructor position remains backward compatible."""
        lists = McpServerTrustLists(frozenset(), frozenset(), "boom")

        assert lists.read_error == "boom"
        assert lists.approvals == frozenset()

    def test_load_failed_tracks_read_error(self) -> None:
        """`load_failed` names the fail-closed contract for `read_error`."""
        assert not McpServerTrustLists(frozenset(), frozenset()).load_failed
        assert McpServerTrustLists(
            frozenset(), frozenset(), read_error="boom"
        ).load_failed


class TestFingerprintMcpServerConfig:
    """Independent oracle for the definition fingerprint.

    Every trust round-trip test builds its expected TOML with
    `fingerprint_mcp_server_config`, so those tests are self-referential: a
    regression that narrowed the fingerprint (e.g. hashing only `command`) would
    pass all of them. These pin the field-completeness and canonicalization
    contract directly, since a narrowed fingerprint is a silent security
    downgrade — an attacker could keep an approved name while mutating `args`,
    `env`, or `headers`.
    """


class TestNormalizeMcpProjectRoot:
    """Tests for normalize_mcp_project_root()."""

    def test_symlink_resolves_to_target(self, tmp_path: Path) -> None:
        """A symlinked root and its target normalize to the same string.

        Root matching is exact-string over normalized output, so write-side and
        read-side must agree whether the path is reached via a link or directly.
        """
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)

        assert normalize_mcp_project_root(link) == normalize_mcp_project_root(target)

    def test_main_and_linked_worktree_keep_exact_roots(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")

        assert normalize_mcp_project_root(main) == str(main.resolve())
        assert normalize_mcp_project_root(worktree) == str(worktree.resolve())
        assert normalize_mcp_project_root(main) != normalize_mcp_project_root(worktree)

    def test_sibling_worktrees_keep_distinct_roots(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")

        assert normalize_mcp_project_root(first) == str(first.resolve())
        assert normalize_mcp_project_root(second) == str(second.resolve())
        assert normalize_mcp_project_root(first) != normalize_mcp_project_root(second)

    def test_independent_clones_use_distinct_local_identities(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        _create_git_repository(first)
        _create_git_repository(second)

        assert normalize_mcp_project_root(first) != normalize_mcp_project_root(second)

    def test_non_git_roots_keep_exact_resolved_paths(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        assert normalize_mcp_project_root(first) == str(first.resolve())
        assert normalize_mcp_project_root(second) == str(second.resolve())
        assert normalize_mcp_project_root(first) != normalize_mcp_project_root(second)

    def test_missing_worktree_metadata_falls_back_to_exact_root(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        git_dir = _create_git_worktree(common_dir, worktree, "worktree")
        (git_dir / "commondir").unlink()

        assert normalize_mcp_project_root(worktree) == str(worktree.resolve())

    def test_malformed_worktree_metadata_falls_back_to_exact_root(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        git_dir = _create_git_worktree(common_dir, worktree, "worktree")
        (git_dir / "commondir").write_text("../..\nunexpected\n")

        assert normalize_mcp_project_root(worktree) == str(worktree.resolve())

    def test_git_metadata_does_not_change_exact_root(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        genuine = tmp_path / "genuine"
        forged = tmp_path / "forged"
        common_dir = _create_git_repository(main)
        git_dir = _create_git_worktree(common_dir, genuine, "genuine")
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {git_dir}\n")

        assert normalize_mcp_project_root(genuine) == str(genuine.resolve())
        assert normalize_mcp_project_root(forged) == str(forged.resolve())

    def test_oserror_falls_back_to_expanded_unresolved_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `resolve()` raises, the expanded-but-unresolved path is returned.

        Documented as fail-closed: a transient failure on only one side yields a
        different string and a spurious re-prompt, never a false match.
        """

        def _boom(*_args: object, **_kwargs: object) -> Path:
            msg = "nope"
            raise OSError(msg)

        monkeypatch.setattr(Path, "resolve", _boom)

        result = normalize_mcp_project_root("~/proj")

        assert result is not None
        assert "~" not in result  # still expanded
        assert result == str(Path("~/proj").expanduser())


class TestMcpProjectServerApproval:
    """Tests for the approval value object and its normalizing factory."""

    def test_local_server_uses_exact_worktree_root(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")

        approval = McpProjectServerApproval.create(
            project_root=worktree,
            name="docs",
            server={"command": "python", "args": ["server.py"]},
        )

        assert approval is not None
        assert approval.project_root == str(worktree.resolve())
        assert approval.git_common_dir is False

    def test_remote_server_uses_shared_git_identity(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")

        approval = McpProjectServerApproval.create(
            project_root=worktree,
            name="docs",
            server={"url": "https://example.test/mcp"},
        )

        assert approval is not None
        assert approval.project_root == str(common_dir.resolve())
        assert approval.git_common_dir is True

    def test_interpolated_remote_url_uses_exact_worktree_root(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")

        approval = McpProjectServerApproval.create(
            project_root=worktree,
            name="docs",
            server={"url": "https://${MCP_HOST}/mcp"},
        )

        assert approval is not None
        assert approval.project_root == str(worktree.resolve())
        assert approval.git_common_dir is False

    def test_scope_marker_participates_in_equality(self) -> None:
        exact = McpProjectServerApproval(
            project_root="/repo/.git",
            name="docs",
            fingerprint="sha256:value",
        )
        shared = McpProjectServerApproval(
            project_root="/repo/.git",
            name="docs",
            fingerprint="sha256:value",
            git_common_dir=True,
        )

        assert exact != shared

    def test_marked_git_identity_does_not_rebind_when_metadata_is_stale(
        self, tmp_path: Path
    ) -> None:
        outer = tmp_path / "outer"
        _create_git_repository(outer)
        nested_common_dir = _create_git_common_dir(outer / "nested.git")
        worktree = tmp_path / "worktree"
        _create_git_worktree(nested_common_dir, worktree, "worktree")
        server = {"type": "http", "url": "https://example.test/mcp"}
        runtime = McpProjectServerApproval.create(
            project_root=worktree, name="docs", server=server
        )
        assert runtime is not None
        persisted = runtime.as_toml()
        assert persisted["git_common_dir"] is True
        (nested_common_dir / "HEAD").unlink()

        restored = McpProjectServerApproval.from_toml(persisted)
        outer_approval = McpProjectServerApproval.create(
            project_root=outer, name="docs", server=server
        )

        assert restored is not None
        assert restored.project_root == str(nested_common_dir.resolve())
        assert restored.git_common_dir is True
        assert restored != outer_approval

    def test_from_toml_rejects_non_boolean_git_marker(self) -> None:
        assert (
            McpProjectServerApproval.from_toml(
                {
                    "project_root": "/project/.git",
                    "name": "docs",
                    "fingerprint": "sha256:value",
                    "git_common_dir": "true",
                }
            )
            is None
        )

    def test_from_toml_rejects_relative_marked_root(self) -> None:
        """A marked Git identity with a relative root fails closed."""
        assert (
            McpProjectServerApproval.from_toml(
                {
                    "project_root": "relative/project/.git",
                    "name": "docs",
                    "fingerprint": "sha256:value",
                    "git_common_dir": True,
                }
            )
            is None
        )

    def test_from_toml_normalizes_unresolved_root(self, tmp_path: Path) -> None:
        """A persisted, not-yet-resolved root is normalized on read.

        So it lines up with the resolved root `create` produces at write time.
        """
        server = {"command": "echo"}
        runtime = McpProjectServerApproval.create(
            project_root=tmp_path / "proj", name="docs", server=server
        )
        assert runtime is not None

        restored = McpProjectServerApproval.from_toml(
            {
                "project_root": str(tmp_path / "proj"),
                "name": "docs",
                "fingerprint": fingerprint_mcp_server_config(server),
            }
        )

        assert restored == runtime

    def test_legacy_exact_root_is_not_broadened_to_git_identity(
        self, tmp_path: Path
    ) -> None:
        project = tmp_path / "project"
        _create_git_repository(project)

        restored = McpProjectServerApproval.from_toml(
            {
                "project_root": str(project),
                "name": "docs",
                "fingerprint": "sha256:value",
            }
        )

        assert restored is not None
        assert restored.project_root == str(project.resolve())
        assert restored.git_common_dir is False


class TestMcpServerTrustListsIsEnabled:
    """Direct tests of the per-server trust decision (`is_enabled`).

    The consumers only reach `is_enabled` transitively, so these pin the
    contract branches directly: name/root/fingerprint scoping, the
    project-agnostic env allowlist, and the disabled short-circuit.
    """

    @staticmethod
    def _server() -> dict[str, object]:
        return {"command": "echo", "args": ["run"]}

    @staticmethod
    def _remote_server() -> dict[str, object]:
        return {"type": "http", "url": "https://example.test/mcp"}

    def _approval_for(
        self,
        root: Path,
        name: str,
        server: dict[str, object] | None = None,
    ) -> McpProjectServerApproval:
        approval = McpProjectServerApproval.create(
            project_root=root, name=name, server=server or self._server()
        )
        assert approval is not None
        return approval

    def test_exact_scoped_match_is_enabled(self, tmp_path: Path) -> None:
        """Matching name, root, and fingerprint together approve the server."""
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert lists.is_enabled("docs", project_root=tmp_path, server=self._server())

    def test_local_approval_does_not_enable_linked_worktree(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(main, "docs")}),
        )

        assert not lists.is_enabled(
            "docs", project_root=worktree, server=self._server()
        )

    def test_remote_approval_is_shared_by_sibling_worktrees(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")
        server = self._remote_server()
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(first, "docs", server)}),
        )

        assert lists.is_enabled("docs", project_root=second, server=server)

    def test_interpolated_remote_url_is_not_shared_by_sibling_worktrees(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")
        server: dict[str, object] = {"url": "https://${MCP_HOST}/mcp"}
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(first, "docs", server)}),
        )

        assert not lists.is_enabled("docs", project_root=second, server=server)

    def test_marked_git_approval_does_not_enable_local_server(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        worktree = tmp_path / "worktree"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, worktree, "worktree")
        server = self._server()
        stale = McpProjectServerApproval(
            project_root=str(common_dir.resolve()),
            name="docs",
            fingerprint=fingerprint_mcp_server_config(server),
            git_common_dir=True,
        )
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({stale}),
        )

        assert not lists.is_enabled("docs", project_root=main, server=server)
        assert not lists.is_enabled("docs", project_root=worktree, server=server)

    def test_legacy_remote_approval_stays_in_original_worktree(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")
        server = self._remote_server()
        legacy = McpProjectServerApproval(
            project_root=str(first.resolve()),
            name="docs",
            fingerprint=fingerprint_mcp_server_config(server),
        )
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({legacy}),
        )

        assert lists.is_enabled("docs", project_root=first, server=server)
        assert not lists.is_enabled("docs", project_root=second, server=server)

    def test_independent_clone_does_not_share_remote_approval(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        _create_git_repository(first)
        _create_git_repository(second)
        server = self._remote_server()
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(first, "docs", server)}),
        )

        assert not lists.is_enabled("docs", project_root=second, server=server)

    def test_forged_worktree_pointer_cannot_borrow_approval(
        self, tmp_path: Path
    ) -> None:
        main = tmp_path / "main"
        genuine = tmp_path / "genuine"
        forged = tmp_path / "forged"
        common_dir = _create_git_repository(main)
        git_dir = _create_git_worktree(common_dir, genuine, "genuine")
        forged.mkdir()
        (forged / ".git").write_text(f"gitdir: {git_dir}\n")
        server = self._remote_server()
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(genuine, "docs", server)}),
        )

        assert not lists.is_enabled("docs", project_root=forged, server=server)

    def test_blank_name_is_not_enabled(self, tmp_path: Path) -> None:
        """A blank server name (only from a malformed config) fails closed.

        `is_enabled` short-circuits rather than let
        `McpProjectServerApproval.create` raise its non-empty `ValueError` out
        of the trust filter on adversarial `.mcp.json` input.
        """
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert not lists.is_enabled("", project_root=tmp_path, server=self._server())
        assert not lists.is_enabled("   ", project_root=tmp_path, server=self._server())

    def test_different_project_root_not_enabled(self, tmp_path: Path) -> None:
        """An approval for one repo does not carry to another."""
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path / "a", "docs")}),
        )

        assert not lists.is_enabled(
            "docs", project_root=tmp_path / "b", server=self._server()
        )

    def test_changed_definition_not_enabled(self, tmp_path: Path) -> None:
        """A changed server definition (new fingerprint) re-prompts."""
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert not lists.is_enabled(
            "docs",
            project_root=tmp_path,
            server={"command": "echo", "args": ["--exfiltrate"]},
        )

    @pytest.mark.parametrize(
        ("approved", "current"),
        [
            (
                {"command": "echo", "args": ["run"]},
                {"type": "http", "url": "https://example.test/mcp"},
            ),
            (
                {"type": "http", "url": "https://example.test/mcp"},
                {"command": "echo", "args": ["run"]},
            ),
        ],
    )
    def test_transport_change_is_not_enabled(
        self,
        tmp_path: Path,
        approved: dict[str, object],
        current: dict[str, object],
    ) -> None:
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs", approved)}),
        )

        assert not lists.is_enabled("docs", project_root=tmp_path, server=current)

    def test_env_enabled_is_project_agnostic(self, tmp_path: Path) -> None:
        """An env-enabled name matches any project, even with no root at all."""
        lists = McpServerTrustLists(enabled=frozenset({"docs"}), disabled=frozenset())

        assert lists.is_enabled("docs", project_root=None, server=self._server())
        assert lists.is_enabled(
            "docs", project_root=tmp_path / "anywhere", server=self._server()
        )
        assert lists.is_enabled("docs", project_root=None, server=self._remote_server())

    def test_disabled_name_never_enabled(self, tmp_path: Path) -> None:
        """A disabled name is rejected regardless of approvals/env."""
        lists = McpServerTrustLists(enabled=frozenset(), disabled=frozenset({"docs"}))

        assert not lists.is_enabled(
            "docs", project_root=tmp_path, server=self._server()
        )

    def test_scoped_approval_needs_a_root(self, tmp_path: Path) -> None:
        """A scoped approval cannot match when the caller has no project root."""
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert not lists.is_enabled("docs", project_root=None, server=self._server())

    def test_padded_name_matches_stripped_approval(self, tmp_path: Path) -> None:
        """A whitespace-padded config name still matches its stripped approval.

        `create`/`from_toml` persist a stripped name, so `is_enabled` must
        normalize the same way or a padded `.mcp.json` key would never match its
        own saved approval. Pins that intended normalization.
        """
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset(),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert lists.is_enabled(" docs ", project_root=tmp_path, server=self._server())

    def test_padded_name_cannot_bypass_deny(self, tmp_path: Path) -> None:
        """A padded name cannot slip a denied server past reject precedence.

        `is_enabled`'s `name in self.disabled` check uses the raw name, so a
        padded `" docs "` sails past it; the deny holds only because
        `__post_init__` stripped the matching approval out of `approvals`. This
        pins that fail-closed guarantee so a refactor that changed either side
        (dropped the post-init stripping, or naively "fixed" the raw check)
        cannot reopen the bypass with the suite still green.
        """
        lists = McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset({"docs"}),
            approvals=frozenset({self._approval_for(tmp_path, "docs")}),
        )

        assert not lists.is_enabled(
            " docs ", project_root=tmp_path, server=self._server()
        )


class TestLoadMcpServerApprovalsParsing:
    """Fail-closed parsing of `[mcp].enabled_project_server_approvals`.

    Dropping a malformed entry only reduces trust, but the real hazard is a
    regression that *accepts* an entry missing a fingerprint — silently
    degrading definition-bound scoping to name+root matching. These pin the drop.
    """

    def test_non_list_value_yields_no_approvals(self, tmp_path: Path) -> None:
        """A scalar (wrong-typed) approvals value degrades to no approvals."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\nenabled_project_server_approvals = "nope"\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result.approvals == frozenset()
        # The wrong-typed key counts as one dropped diagnostic so callers can
        # surface it instead of it only reaching an unseen debug log.
        assert result.malformed_approvals == 1

    def test_malformed_entries_are_dropped(self, tmp_path: Path) -> None:
        """Non-table and fingerprint-less entries drop; well-formed ones survive."""
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = [\n"
            '  "not-a-table",\n'
            f'  {{ project_root = "{project_root}", name = "missing-fp" }},\n'
            f'  {{ project_root = "{project_root}", name = "good", '
            f'fingerprint = "{fingerprint}" }},\n'
            "]\n"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.approvals == frozenset(
            {
                McpProjectServerApproval(
                    project_root=project_root,
                    name="good",
                    fingerprint=fingerprint,
                )
            }
        )
        # Both the non-table entry and the fingerprint-less entry are counted so
        # a corrupt saved approval is surfaced rather than silently re-prompting.
        assert result.malformed_approvals == 2


class TestLoadMcpServerTrustLists:
    """Tests for load_mcp_server_trust_lists()."""

    def test_reads_approvals_and_disabled_list_from_toml(self, tmp_path: Path) -> None:
        """Parses scoped approvals and disabled lists from the [mcp] table."""
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "docs", '
            f'fingerprint = "{fingerprint}" }}]\n'
            'disabled_project_servers = ["blocked"]\n'
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result == McpServerTrustLists(
            enabled=frozenset(),
            disabled=frozenset({"blocked"}),
            approvals=frozenset(
                {
                    McpProjectServerApproval(
                        project_root=project_root,
                        name="docs",
                        fingerprint=fingerprint,
                    )
                }
            ),
        )

    def test_legacy_worktree_approvals_remain_exact(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")
        fingerprint = fingerprint_mcp_server_config({"command": "echo"})
        roots = [main, first, second]
        entries = ",\n".join(
            f'  {{ project_root = "{root}", name = "docs", '
            f'fingerprint = "{fingerprint}" }}'
            for root in roots
        )
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f"[mcp]\nenabled_project_server_approvals = [\n{entries}\n]\n"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert {approval.project_root for approval in result.approvals} == {
            str(root.resolve()) for root in roots
        }
        assert not any(approval.git_common_dir for approval in result.approvals)
        assert result.malformed_approvals == 0

    def test_marked_git_identity_is_read_from_toml(self, tmp_path: Path) -> None:
        """A persisted `git_common_dir = true` row loads as a marked approval."""
        common_dir = _create_git_repository(tmp_path / "main")
        fingerprint = fingerprint_mcp_server_config({"url": "https://example.test"})
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = [\n"
            f'  {{ project_root = "{common_dir}", name = "docs", '
            f'fingerprint = "{fingerprint}", git_common_dir = true }}\n'
            "]\n"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.malformed_approvals == 0
        assert len(result.approvals) == 1
        (approval,) = result.approvals
        assert approval.git_common_dir is True
        assert approval.project_root == str(common_dir)

    def test_unresolvable_approval_root_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale approval whose root becomes a symlink loop fails closed."""
        config_path = tmp_path / "config.toml"
        loop = tmp_path / "loop"
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{loop}", name = "docs", '
            f'fingerprint = "{fingerprint}" }}]\n'
        )

        def _boom(*_args: object, **_kwargs: object) -> Path:
            msg = "symlink loop"
            raise RuntimeError(msg)

        monkeypatch.setattr(Path, "resolve", _boom)

        result = load_mcp_server_trust_lists(config_path)

        assert result.approvals == frozenset()
        assert result.malformed_approvals == 1

    def test_reject_precedence_removes_from_approvals(self, tmp_path: Path) -> None:
        """A name in approvals and disabled is reported only as disabled."""
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "both", '
            f'fingerprint = "{fingerprint}" }}]\n'
            'disabled_project_servers = ["both"]\n'
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.approvals == frozenset()
        assert result.disabled == frozenset({"both"})

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """A missing config file yields empty lists, not an error.

        A missing file is the normal "unset" case and must NOT set `read_error`
        (that is reserved for a file that exists but cannot be read/parsed), so
        callers do not fail closed just because the user has no config.toml.
        """
        result = load_mcp_server_trust_lists(tmp_path / "nonexistent.toml")

        assert result == McpServerTrustLists(frozenset(), frozenset())
        assert result.read_error is None

    def test_env_deny_beats_toml_allow_same_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject wins across sources: env-disabled beats TOML-enabled by name.

        Proves the disjointness invariant runs on the final merged frozensets
        (after env/TOML resolution), not merely within a single source.
        """
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "srv", '
            f'fingerprint = "{fingerprint}" }}]\n'
        )
        monkeypatch.setenv(model_config._env_vars.DISABLED_PROJECT_MCP_SERVERS, "srv")

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.approvals == frozenset()
        assert result.disabled == frozenset({"srv"})

    def test_missing_mcp_section_returns_empty(self, tmp_path: Path) -> None:
        """A config without an [mcp] table yields empty lists."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[models]\ndefault = "some:model"\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result == McpServerTrustLists(frozenset(), frozenset())

    def test_corrupt_toml_falls_back_to_empty_and_sets_read_error(
        self, tmp_path: Path
    ) -> None:
        """Malformed TOML degrades to empty lists but records a read error.

        The empty lists compare equal to a clean empty result (`read_error` is
        excluded from equality), but `read_error` is set so callers can fail
        closed instead of treating a broken config as "nothing denied."
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[invalid toml")

        result = load_mcp_server_trust_lists(config_path)

        assert result == McpServerTrustLists(frozenset(), frozenset())
        assert result.read_error is not None
        assert str(config_path) in result.read_error

    def test_dangerous_env_survives_toml_deny_when_config_unreadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Characterize the documented reject-wins corner (accepted footgun).

        When `config.toml` is unreadable, `toml_disabled` is lost, so a name that
        is both TOML-`disabled` and exported in the `DANGEROUSLY_` enable env var
        survives — "reject wins" does NOT hold in this one corner. This pins that
        intentional behavior (and its `read_error` surfacing) so a future change
        that closes it is a deliberate decision, not an accidental regression.
        Contrast `test_env_deny_beats_toml_allow_same_name`, where a readable
        config keeps the deny.
        """
        config_path = tmp_path / "config.toml"
        # The deny lives here but is lost because the file cannot be parsed.
        config_path.write_text('[[invalid toml\ndisabled_project_servers = ["srv"]')
        monkeypatch.setenv(
            model_config._env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS, "srv"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.read_error is not None
        # The footgun: the name survives despite the (unreadable) TOML deny.
        assert result.enabled == frozenset({"srv"})
        assert "srv" not in result.disabled
        assert result.is_enabled(
            "srv", project_root=tmp_path, server={"command": "echo", "args": []}
        )

    def test_legacy_enabled_ignored_and_mixed_disabled_dropped(
        self, tmp_path: Path
    ) -> None:
        """Legacy flat enabled names are ignored; disabled list still parses."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[mcp]\n"
            'enabled_project_servers = "docs"\n'
            'disabled_project_servers = [1, "blocked", true]\n'
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.approvals == frozenset()
        assert result.disabled == frozenset({"blocked"})

    def test_env_composes_with_toml_approvals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Process-wide names and project-scoped approvals both remain active."""
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        approval = McpProjectServerApproval(
            project_root=project_root,
            name="toml-enabled",
            fingerprint=fingerprint,
        )
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "toml-enabled", '
            f'fingerprint = "{fingerprint}" }}]\n'
            'disabled_project_servers = ["toml-disabled"]\n'
        )
        monkeypatch.setenv(
            model_config._env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            "env-enabled, env-two",
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset({"env-enabled", "env-two"})
        assert result.approvals == frozenset({approval})
        assert result.disabled == frozenset({"toml-disabled"})

    def test_empty_env_keeps_toml_approvals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty process-wide allowlist does not erase remembered approvals."""
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        approval = McpProjectServerApproval(
            project_root=project_root,
            name="toml-enabled",
            fingerprint=fingerprint,
        )
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "toml-enabled", '
            f'fingerprint = "{fingerprint}" }}]\n'
        )
        monkeypatch.setenv(
            model_config._env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            "",
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.approvals == frozenset({approval})

    def test_defaults_to_user_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no argument, the loader reads the user-level config path only."""
        user_config = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        user_config.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "docs", '
            f'fingerprint = "{fingerprint}" }}]\n'
        )
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_config)

        result = load_mcp_server_trust_lists()

        assert result.enabled == frozenset()
        assert result.approvals == frozenset(
            {
                McpProjectServerApproval(
                    project_root=project_root,
                    name="docs",
                    fingerprint=fingerprint,
                )
            }
        )

    def test_disabled_env_honored_without_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deny list can be set purely from the env var."""
        config_path = tmp_path / "config.toml"  # no [mcp] table
        monkeypatch.setenv(
            model_config._env_vars.DISABLED_PROJECT_MCP_SERVERS, "blocked, other"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset({"blocked", "other"})

    def test_disabled_env_unions_with_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The disabled env list UNIONS with the TOML deny list (denies accrue).

        A deny must never be silently dropped by the other source, so both
        contribute.
        """
        config_path = tmp_path / "config.toml"
        project_root = str(tmp_path / "project")
        fingerprint = fingerprint_mcp_server_config({"command": "echo", "args": []})
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "toml-enabled", '
            f'fingerprint = "{fingerprint}" }}]\n'
            'disabled_project_servers = ["toml-disabled"]\n'
        )
        monkeypatch.setenv(
            model_config._env_vars.DISABLED_PROJECT_MCP_SERVERS, "env-disabled"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset({"toml-disabled", "env-disabled"})
        assert result.enabled == frozenset()
        assert result.approvals == frozenset(
            {
                McpProjectServerApproval(
                    project_root=project_root,
                    name="toml-enabled",
                    fingerprint=fingerprint,
                )
            }
        )

    def test_empty_disabled_env_preserves_toml_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set-but-empty disabled env var cannot clear the TOML deny list.

        Because disabled unions across sources, an empty env value contributes
        nothing and the configured deny survives — closing the fail-open where
        `DISABLED=""` (e.g. from an attacker-adjacent source) would silently
        neutralize the user's deny list.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\ndisabled_project_servers = ["toml-disabled"]\n')
        monkeypatch.setenv(model_config._env_vars.DISABLED_PROJECT_MCP_SERVERS, "")

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset({"toml-disabled"})

    def test_legacy_enabled_toml_list_is_ignored(self, tmp_path: Path) -> None:
        """Legacy flat TOML approvals no longer auto-approve project servers."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\nenabled_project_servers = [" docs ", "  "]\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.approvals == frozenset()
        # The dropped names are surfaced so non-interactive paths can explain
        # why those servers stopped loading.
        assert result.legacy_ignored == frozenset({"docs"})

    def test_legacy_env_var_flagged_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The removed env var, still exported, is flagged (not honored)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[mcp]\n")
        monkeypatch.setenv("DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS", "docs")

        result = load_mcp_server_trust_lists(config_path)

        # The old name never pre-approves; it only sets the diagnostic flag.
        assert result.legacy_env_ignored is True
        assert result.enabled == frozenset()

    def test_legacy_env_var_absent_leaves_flag_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the old env var unset, the diagnostic flag stays `False`."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[mcp]\n")
        monkeypatch.delenv("DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS", raising=False)

        result = load_mcp_server_trust_lists(config_path)

        assert result.legacy_env_ignored is False

    def test_bare_string_disabled_is_coerced_to_single_name(
        self, tmp_path: Path
    ) -> None:
        """A bare-string deny list is one server name, not a dropped-to-empty typo.

        Coercing (rather than silently dropping) is the safe direction for the
        deny list: it keeps enforcing the user's rejection instead of failing
        open on a scalar-instead-of-list mistake.
        """
        config_path = tmp_path / "config.toml"
        # Valid TOML, but a string rather than a list — the [mcp] table is still
        # a dict, so the "should be a table" branch does not fire.
        config_path.write_text('[mcp]\ndisabled_project_servers = "blocked"\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset({"blocked"})

    def test_bare_string_disabled_splits_on_commas(self, tmp_path: Path) -> None:
        """A comma-separated bare-string deny list parses like the env form.

        Without splitting, `"a, b"` would become one bogus token matching no
        server and silently enforce nothing — a fail-open for the deny list.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\ndisabled_project_servers = "evil, backdoor"\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset({"evil", "backdoor"})
        assert result.read_error is None

    def test_wrong_typed_disabled_fails_closed_with_read_error(
        self, tmp_path: Path
    ) -> None:
        """A wrong-typed deny list blocks TOML approvals and sets `read_error`.

        Preserving a saved approval when the deny list cannot be read would let
        that server load despite an unenforced rejection policy. Only explicit
        environment approvals may survive this config read failure.
        """
        config_path = tmp_path / "config.toml"
        project_root = tmp_path / "project"
        server = {"command": "echo", "args": []}
        fingerprint = fingerprint_mcp_server_config(server)
        config_path.write_text(
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_root}", name = "docs", '
            f'fingerprint = "{fingerprint}" }}]\n'
            "disabled_project_servers = 123\n"
        )

        result = load_mcp_server_trust_lists(config_path)

        assert result.disabled == frozenset()
        assert result.approvals == frozenset()
        assert not result.is_enabled("docs", project_root=project_root, server=server)
        assert result.load_failed
        assert "disabled_project_servers" in (result.read_error or "")

    def test_wrong_typed_enabled_stays_empty_without_read_error(
        self, tmp_path: Path
    ) -> None:
        """A wrong-typed allow list degrades to empty (already fail-closed).

        Unlike the deny list, an unreadable allow list approves nothing extra, so
        it must NOT set `read_error` (which would block even trusted configs).
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("[mcp]\nenabled_project_servers = 123\n")

        result = load_mcp_server_trust_lists(config_path)

        assert result.enabled == frozenset()
        assert result.read_error is None

    def test_non_table_mcp_sets_read_error(self, tmp_path: Path) -> None:
        """An `[mcp]` value that is not a table fails closed (deny unreadable)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('mcp = "oops"\n')

        result = load_mcp_server_trust_lists(config_path)

        assert result == McpServerTrustLists(frozenset(), frozenset())
        assert result.load_failed


class TestGetModelProfiles:
    """Tests for get_model_profiles() function."""


class TestCodexProviderMirror:
    """`openai_codex` mirrors the curated `CODEX_MODELS` subset of `openai`.

    The Codex backend serves a narrower lineup than the full `openai` API, so
    only models in the `CODEX_MODELS` allowlist are exposed under
    `openai_codex`; other openai models are not mirrored.
    """

    def test_profiles_mirror_codex_allowlist_under_codex(self) -> None:
        model_config.clear_caches()
        profiles = model_config.get_model_profiles()
        openai_models = [
            spec.split(":", 1)[1] for spec in profiles if spec.startswith("openai:")
        ]
        assert openai_models, "expected openai profiles to load"
        for model_name in openai_models:
            codex_spec = f"{model_config.CODEX_PROVIDER}:{model_name}"
            if model_name in model_config.CODEX_MODELS:
                assert codex_spec in profiles
            else:
                assert codex_spec not in profiles


class TestAddEnabledProjectMcpServers:
    """Tests for persisting the approval prompt's "always allow" choice."""

    @staticmethod
    def _server_configs() -> JsonObject:
        return {
            "docs": {"command": "echo", "args": ["docs"]},
            "reference": {"type": "http", "url": "https://example.test/mcp"},
            "github": {"command": "gh", "args": ["api"]},
        }

    def _approvals(self, config_path: Path) -> list[dict[str, str | bool]]:
        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return data["mcp"]["enabled_project_server_approvals"]

    def test_scopes_mixed_transports_per_server_across_worktrees(
        self, tmp_path: Path
    ) -> None:
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        main = tmp_path / "main"
        first = tmp_path / "first"
        second = tmp_path / "second"
        common_dir = _create_git_repository(main)
        _create_git_worktree(common_dir, first, "first")
        _create_git_worktree(common_dir, second, "second")
        server_configs = self._server_configs()
        config_path = tmp_path / "config.toml"

        for project_root in (first, second):
            assert add_enabled_project_mcp_servers(
                ["docs", "reference"],
                config_path,
                project_root=project_root,
                server_configs=server_configs,
            )

        approvals = self._approvals(config_path)
        local = [approval for approval in approvals if approval["name"] == "docs"]
        remote = [approval for approval in approvals if approval["name"] == "reference"]
        assert {approval["project_root"] for approval in local} == {
            str(first.resolve()),
            str(second.resolve()),
        }
        assert all("git_common_dir" not in approval for approval in local)
        assert remote == [
            {
                "project_root": str(common_dir.resolve()),
                "name": "reference",
                "fingerprint": fingerprint_mcp_server_config(
                    server_configs["reference"]
                ),
                "git_common_dir": True,
            }
        ]

    def test_nested_external_common_identity_stays_idempotent(
        self, tmp_path: Path
    ) -> None:
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        outer = tmp_path / "outer"
        _create_git_repository(outer)
        nested_common_dir = _create_git_common_dir(outer / "nested.git")
        worktree = tmp_path / "nested-worktree"
        _create_git_worktree(nested_common_dir, worktree, "nested")
        server_configs = self._server_configs()
        config_path = tmp_path / "config.toml"

        assert add_enabled_project_mcp_servers(
            ["reference"],
            config_path,
            project_root=worktree,
            server_configs=server_configs,
        )

        approvals = self._approvals(config_path)
        assert approvals[0]["project_root"] == str(nested_common_dir.resolve())
        assert approvals[0]["git_common_dir"] is True
        lists = load_mcp_server_trust_lists(config_path)
        assert lists.is_enabled(
            "reference",
            project_root=worktree,
            server=server_configs["reference"],
        )
        assert not lists.is_enabled(
            "reference", project_root=outer, server=server_configs["reference"]
        )

    def test_removes_migrated_names_from_legacy_approvals(self, tmp_path: Path) -> None:
        """Scoped approvals consume matching names from the legacy allowlist."""
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\nenabled_project_servers = ["docs", "github"]\n')

        assert add_enabled_project_mcp_servers(
            ["docs"],
            config_path,
            project_root=tmp_path / "project",
            server_configs=self._server_configs(),
        )

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["mcp"]["enabled_project_servers"] == ["github"]
        assert load_mcp_server_trust_lists(config_path).legacy_ignored == frozenset(
            {"github"}
        )

    def test_deletes_empty_legacy_approval_key(self, tmp_path: Path) -> None:
        """Migrating the final legacy name removes its warning source."""
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp]\nenabled_project_servers = ["docs"]\n')

        assert add_enabled_project_mcp_servers(
            ["docs"],
            config_path,
            project_root=tmp_path / "project",
            server_configs=self._server_configs(),
        )

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert "enabled_project_servers" not in data["mcp"]
        assert not load_mcp_server_trust_lists(config_path).legacy_ignored

    def test_heals_non_table_mcp_value(self, tmp_path: Path) -> None:
        """An existing scalar `mcp` value is overwritten with a proper table.

        The write-side analog of the read-side `test_non_table_mcp_sets_read_error`:
        a corrupt `[mcp]` must not abort the save, and unrelated config survives.
        """
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'mcp = "oops"\n[models]\ndefault = "anthropic:claude-sonnet-4-5"\n'
        )

        assert add_enabled_project_mcp_servers(
            ["docs"],
            config_path,
            project_root=tmp_path / "project",
            server_configs=self._server_configs(),
        )

        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["models"]["default"] == "anthropic:claude-sonnet-4-5"
        assert data["mcp"]["enabled_project_server_approvals"][0]["name"] == "docs"

    def test_round_trips_through_loader(self, tmp_path: Path) -> None:
        """Persisted approvals are read back by `load_mcp_server_trust_lists`."""
        from deepagents_code.model_config import (
            add_enabled_project_mcp_servers,
            load_mcp_server_trust_lists,
        )

        config_path = tmp_path / "config.toml"
        project_root = tmp_path / "project"
        server_configs = self._server_configs()
        assert add_enabled_project_mcp_servers(
            ["docs", "reference"],
            config_path,
            project_root=project_root,
            server_configs=server_configs,
        )
        lists = load_mcp_server_trust_lists(config_path)

        assert lists.enabled == frozenset()
        assert lists.approvals == frozenset(
            {
                McpProjectServerApproval(
                    project_root=str(project_root),
                    name="docs",
                    fingerprint=fingerprint_mcp_server_config(server_configs["docs"]),
                ),
                McpProjectServerApproval(
                    project_root=str(project_root),
                    name="reference",
                    fingerprint=fingerprint_mcp_server_config(
                        server_configs["reference"]
                    ),
                ),
            }
        )

    def test_returns_false_on_unparseable_config(self, tmp_path: Path) -> None:
        """A corrupt existing config fails closed (returns False) and is untouched."""
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        config_path = tmp_path / "config.toml"
        corrupt = "[mcp]\nenabled_project_server_approvals = [\n"
        config_path.write_text(corrupt)
        assert (
            add_enabled_project_mcp_servers(
                ["docs"],
                config_path,
                project_root=tmp_path / "project",
                server_configs=self._server_configs(),
            )
            is False
        )
        # The unparseable file is left exactly as-is — no partial atomic clobber.
        assert config_path.read_text() == corrupt

    def test_returns_false_on_os_error(self, tmp_path: Path) -> None:
        """An I/O failure while writing fails closed (returns False).

        Direct coverage of the `OSError` arm the docstring promises: the config
        directory cannot be created because a regular file sits where a
        directory must go.
        """
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        blocker = tmp_path / "afile"
        blocker.write_text("")  # a file where a parent directory is needed
        config_path = blocker / "config.toml"
        assert (
            add_enabled_project_mcp_servers(
                ["docs"],
                config_path,
                project_root=tmp_path / "project",
                server_configs=self._server_configs(),
            )
            is False
        )

    def test_failed_write_leaves_no_stray_tmp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write that fails mid-flight cleans up its atomic temp file.

        Covers the `except BaseException: unlink; raise` arm: a serialization
        failure after `mkstemp` must not leave a `.tmp` turd in the config dir.
        """
        from deepagents_code import model_config
        from deepagents_code.model_config import add_enabled_project_mcp_servers

        config_path = tmp_path / "config.toml"

        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "serialize failed"
            raise ValueError(msg)

        monkeypatch.setattr(model_config.tomli_w, "dump", _boom)

        assert (
            add_enabled_project_mcp_servers(
                ["docs"],
                config_path,
                project_root=tmp_path / "project",
                server_configs=self._server_configs(),
            )
            is False
        )
        assert not config_path.exists()
        assert list(tmp_path.glob("*.tmp")) == []


class TestLoadStartupMode:
    """Tests for the `[startup]` approval-mode read and its recent-mode write.

    Covers `load_startup_mode` over both `mode` and `recent`, and
    `save_recent_startup_mode`.
    """

    @pytest.mark.parametrize("mode", [STARTUP_MODE_MANUAL, STARTUP_MODE_AUTO])
    def test_recent_mode_is_restored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        """A stored recent mode is restored on a bare launch."""
        from deepagents_code.approval_mode import save_auto_mode_notice

        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
        if mode == STARTUP_MODE_AUTO:
            assert save_auto_mode_notice()
        config = tmp_path / "config.toml"
        config.write_text(f"[startup]\nrecent = '{mode}'\n")
        assert load_startup_mode(config) == mode

    @pytest.mark.parametrize("notice_state", ["missing", "stale"])
    def test_recent_auto_requires_current_notice(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        notice_state: str,
    ) -> None:
        """Implicit Auto restoration fails closed until its notice is current."""
        state_dir = tmp_path / ".state"
        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", state_dir)
        if notice_state == "stale":
            state_dir.mkdir()
            (state_dir / "approval.json").write_text(
                '{"auto_notice_shown":true,"auto_notice_version":"old"}\n'
            )
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nrecent = 'auto'\n")

        assert load_startup_mode(config) == STARTUP_MODE_MANUAL

    def test_recent_auto_blocked_by_notice_warns_and_queues_notice(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A declined Auto restore is diagnosable, not silent.

        This is the one exit that discards a *valid* preference, so without a
        log line and a queued toast an `AUTO_NOTICE_VERSION` bump looks exactly
        like the persistence feature being broken.
        """
        from deepagents_code.model_config import (
            consume_recent_auto_not_restored_notice,
        )

        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nrecent = 'auto'\n")
        # The notice is module state; clear anything an earlier test queued.
        consume_recent_auto_not_restored_notice()

        with caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"):
            assert load_startup_mode(config) == STARTUP_MODE_MANUAL

        assert any(
            "Not restoring [startup].recent" in record.getMessage()
            for record in caplog.records
        )
        notice = consume_recent_auto_not_restored_notice()
        assert notice is not None
        assert "Shift+Tab" in notice
        # One-shot: a second consumer must not re-toast the same launch.
        assert consume_recent_auto_not_restored_notice() is None

    def test_restored_recent_auto_queues_no_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful restore leaves nothing to explain."""
        from deepagents_code.approval_mode import save_auto_mode_notice
        from deepagents_code.model_config import (
            consume_recent_auto_not_restored_notice,
        )

        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
        assert save_auto_mode_notice()
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nrecent = 'auto'\n")
        consume_recent_auto_not_restored_notice()

        assert load_startup_mode(config) == STARTUP_MODE_AUTO
        assert consume_recent_auto_not_restored_notice() is None

    def test_explicit_mode_outranks_recent(self, tmp_path: Path) -> None:
        """An explicit mode is an intentional default and wins."""
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nmode = 'manual'\nrecent = 'auto'\n")
        assert load_startup_mode(config) == STARTUP_MODE_MANUAL

    @pytest.mark.parametrize("recent", ["yolo", "hands-off"])
    def test_unsafe_or_invalid_recent_mode_fails_closed(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        recent: str,
    ) -> None:
        """Only Manual and Auto restore; anything else warns and fails closed."""
        config = tmp_path / "config.toml"
        config.write_text(f"[startup]\nrecent = '{recent}'\n")
        with caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"):
            assert load_startup_mode(config) == STARTUP_MODE_MANUAL
        assert any(
            "[startup].recent" in record.getMessage() for record in caplog.records
        )

    @pytest.mark.parametrize("literal", ["['auto']", "{ a = 'auto' }", "3", "true"])
    def test_non_scalar_recent_returns_default(
        self, tmp_path: Path, literal: str
    ) -> None:
        """A non-string `recent` must not reach the frozenset membership test.

        `recent in RECENT_STARTUP_MODES` raises `TypeError: unhashable type` on
        a list or table, which `except (OSError, TOMLDecodeError)` does not
        catch, so dropping the isinstance guard aborts launch. This mirrors
        `test_non_scalar_mode_returns_default` for the newer key.
        """
        config = tmp_path / "config.toml"
        config.write_text(f"[startup]\nrecent = {literal}\n")
        assert load_startup_mode(config) == STARTUP_MODE_MANUAL

    @pytest.mark.parametrize("mode", [STARTUP_MODE_MANUAL, STARTUP_MODE_AUTO])
    def test_save_recent_startup_mode_round_trip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        """A saved mode reloads, and neighbouring config keys survive the write."""
        from deepagents_code.approval_mode import save_auto_mode_notice

        monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", tmp_path / ".state")
        if mode == STARTUP_MODE_AUTO:
            assert save_auto_mode_notice()
        config = tmp_path / "config.toml"
        config.write_text("[models]\ndefault = 'openai:gpt-5.5'\n")

        assert save_recent_startup_mode(mode, config) is True
        with config.open("rb") as file:
            data = tomllib.load(file)
        assert data["startup"]["recent"] == mode
        assert data["models"]["default"] == "openai:gpt-5.5"
        assert load_startup_mode(config) == mode

    def test_save_recent_startup_mode_rejects_yolo(self, tmp_path: Path) -> None:
        """YOLO must never be restored implicitly, so it cannot be stored.

        The guard is the write-side half of `RECENT_STARTUP_MODES`; the read
        side is covered above.
        """
        with pytest.raises(ValueError, match="Invalid recent startup mode"):
            save_recent_startup_mode(STARTUP_MODE_YOLO, tmp_path / "config.toml")

    def test_dangerously_auto_is_rejected(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nmode = 'dangerously-auto'\n")
        assert load_startup_mode(config) == STARTUP_MODE_MANUAL

    def test_invalid_explicit_mode_ignores_recent(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An invalid explicit mode fails closed instead of restoring recent Auto."""
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nmode = 'hands-off'\nrecent = 'auto'\n")
        with caplog.at_level(logging.WARNING, logger="deepagents_code.model_config"):
            assert load_startup_mode(config) == STARTUP_MODE_MANUAL
        # Assert on the `mode` warning specifically: matching bare "startup"
        # would also pass on the `recent` warning, which must not fire here.
        assert any("[startup].mode" in r.getMessage() for r in caplog.records)
        assert not any("[startup].recent" in r.getMessage() for r in caplog.records)

    def test_non_scalar_mode_returns_default(self, tmp_path: Path) -> None:
        """A non-string `mode` (e.g. array) falls back instead of raising.

        `value in VALID_STARTUP_MODES` (a frozenset) would raise `TypeError:
        unhashable type` on a list/dict; the isinstance guard must prevent that.
        """
        config = tmp_path / "config.toml"
        config.write_text("[startup]\nmode = ['dangerously-auto']\n")
        assert load_startup_mode(config) == STARTUP_MODE_MANUAL

    def test_unparseable_file_returns_default(self, tmp_path: Path) -> None:
        """Syntactically invalid TOML is swallowed and falls back to default.

        Exercises the `except (OSError, tomllib.TOMLDecodeError)` branch, which
        must fail closed (to `manual`) rather than propagate and abort startup.
        """
        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[\n")
        assert load_startup_mode(config) == STARTUP_MODE_MANUAL


class TestWritesReachTheSharedResolver:
    """Every `model_config` writer must advance the shared config generation.

    These writers edit `config.toml` directly instead of going through
    `configuration.writer.update_user_config`, so they own the refresh
    themselves. A writer that forgets it leaves the resolver serving the
    pre-write generation for the life of the process: the file on disk is
    correct, the UI reports "saved", and the setting never takes effect until
    the user restarts or runs `/reload`.
    """

    def test_a_failed_refresh_does_not_escape_a_committed_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The bytes are on disk, so the writer must still report success.

        `refresh_shared_resolver` caught only `OSError`, but a reload also
        raises `ValueError` from the snapshot invariants and `RuntimeError`
        from a provider with no snapshot. Those escaped a `-> bool` writer into
        UI code after the write had already landed.
        """
        import logging

        from deepagents_code.configuration import resolver as resolver_module

        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()

        def explode(**_kwargs: object) -> None:
            msg = "synthetic reload failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(resolver_module, "get_config_resolver", explode)

        with caplog.at_level(logging.WARNING):
            assert model_config.save_thread_sort_order("created_at") is True

        assert "created_at" in config_path.read_text(encoding="utf-8")
        assert "could not refresh the shared config resolver" in caplog.text

    def test_every_config_writer_invalidates_the_shared_generation(self) -> None:
        """No `model_config` writer may skip the refresh.

        The behavioral cases above cover five writers by name, so removing the
        refresh from `save_thread_columns` or `clear_default_agent` left the
        suite green. There are eleven, and the failure is invisible at runtime
        -- the file on disk is correct and the UI reports "saved" -- so the
        class needs a structural guard rather than one case per writer.

        A writer here is a function that takes a `config_path` and commits it
        with an atomic replace. `touch_recent_model` writes the MRU state file
        the same way but takes a `state_dir`, so it is correctly not a writer.
        """
        import ast
        from pathlib import Path

        source = Path(model_config.__file__).read_text(encoding="utf-8")
        missing: set[str] = set()
        writers: set[str] = set()
        for function in ast.walk(ast.parse(source)):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {argument.arg for argument in function.args.args} | {
                argument.arg for argument in function.args.kwonlyargs
            }
            if "config_path" not in params:
                continue
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            commits = any(
                isinstance(call.func, ast.Attribute) and call.func.attr == "replace"
                for call in calls
            )
            if not commits:
                continue
            writers.add(function.name)
            # A *call*, not the name: the helper is referenced in comments and
            # docstrings, so a substring check passes after the call is gone.
            refreshes = any(
                isinstance(call.func, ast.Name)
                and call.func.id == "_invalidate_config_caches"
                for call in calls
            )
            if not refreshes:
                missing.add(function.name)

        assert writers, "the AST probe stopped recognizing any config writer"
        assert not missing, (
            f"`model_config` writers that never refresh the resolver: "
            f"{sorted(missing)}. Call `_invalidate_config_caches(config_path)` "
            "after a committed write, or the saved value stays invisible to "
            "every reader for the life of the process."
        )

    def test_noop_save_refreshes_externally_changed_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An equal on-disk value still invalidates stale cached views."""
        from deepagents_code.config_manifest import get_option
        from deepagents_code.configuration.resolver import get_config_resolver

        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        config_path.write_text(
            '[models]\nauto_classifier = "openai:old"\n', encoding="utf-8"
        )
        model_config.clear_caches()
        option = get_option("models.auto_classifier")
        assert option is not None
        assert model_config.ModelConfig.load().auto_classifier_model == "openai:old"
        assert get_config_resolver().get(option).value == "openai:old"

        config_path.write_text(
            '[models]\nauto_classifier = "openai:new"\n', encoding="utf-8"
        )
        inode = config_path.stat().st_ino

        assert model_config.save_auto_classifier_model("openai:new") is True

        assert config_path.stat().st_ino == inode
        assert model_config.ModelConfig.load().auto_classifier_model == "openai:new"
        assert get_config_resolver().get(option).value == "openai:new"
