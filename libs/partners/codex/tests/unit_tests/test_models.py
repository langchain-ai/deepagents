from deepagents_codex.models import (
    CODEX_MODELS,
    get_available_codex_models,
    get_model_profile,
)


class TestCodexModels:
    def test_models_have_required_fields(self) -> None:
        required = {"max_input_tokens", "tool_calling", "text_inputs", "text_outputs"}
        for model_id, profile in CODEX_MODELS.items():
            missing = required - set(profile.keys())
            assert not missing, f"{model_id} missing fields: {missing}"

    def test_all_models_support_tool_calling(self) -> None:
        for model_id, profile in CODEX_MODELS.items():
            assert profile["tool_calling"] is True, (
                f"{model_id} must support tool_calling"
            )

    def test_get_available_returns_sorted(self) -> None:
        models = get_available_codex_models()
        assert models == sorted(models)
        assert len(models) == len(CODEX_MODELS)

    def test_known_models_present(self) -> None:
        models = get_available_codex_models()
        assert "gpt-5.4" in models
        assert "gpt-5.1-codex" in models
        assert "gpt-5.1-codex-mini" in models
        assert "codex-mini-latest" in models

    def test_get_model_profile_known(self) -> None:
        profile = get_model_profile("gpt-5.4")
        assert profile["max_input_tokens"] == 400_000

    def test_get_model_profile_unknown_returns_default(self) -> None:
        """Unknown model IDs get a sensible default profile."""
        profile = get_model_profile("gpt-99-codex-future")
        assert profile["tool_calling"] is True
        assert profile["max_input_tokens"] == 192_000
