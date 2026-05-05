"""Unit tests for `DeepAgentsWrapper` initialization guards."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from deepagents_harbor.deepagents_wrapper import (
    DeepAgentsWrapper,
    _parse_openrouter_providers,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestModelNameRequired:
    """The wrapper rejects empty/whitespace `model_name` at construction."""

    def test_empty_string_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="")

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            DeepAgentsWrapper(logs_dir=tmp_path, model_name="   ")


class TestOpenRouterPrefix:
    """`openrouter_provider` requires an `openrouter:` prefixed model."""

    def test_mismatched_prefix_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="openrouter_provider requires an openrouter: model prefix"
        ):
            DeepAgentsWrapper(
                logs_dir=tmp_path,
                model_name="claude-sonnet-4-6",
                openrouter_provider="MiniMax",
            )


class TestHappyPathConstruction:
    """Wrapper construction succeeds with a valid model name and stashes state."""

    def test_constructs_with_valid_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub the credential so `init_chat_model` doesn't reject us.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        wrapper = DeepAgentsWrapper(logs_dir=tmp_path, model_name="claude-sonnet-4-6")

        assert wrapper._model_name == "claude-sonnet-4-6"
        assert wrapper._model is not None
        assert wrapper._temperature == 0.0


class TestParseOpenrouterProviders:
    """The comma-separated provider parser drops empty tokens and rejects empty input."""

    def test_single_value(self) -> None:
        assert _parse_openrouter_providers("MiniMax") == ["MiniMax"]

    def test_comma_separated_list(self) -> None:
        assert _parse_openrouter_providers("MiniMax,Fireworks") == ["MiniMax", "Fireworks"]

    def test_strips_whitespace_around_each_token(self) -> None:
        assert _parse_openrouter_providers("  MiniMax , Fireworks  ") == [
            "MiniMax",
            "Fireworks",
        ]

    def test_drops_empty_tokens(self) -> None:
        assert _parse_openrouter_providers("MiniMax,,Fireworks,") == ["MiniMax", "Fireworks"]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one non-empty provider name"):
            _parse_openrouter_providers("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one non-empty provider name"):
            _parse_openrouter_providers("   ")

    def test_all_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one non-empty provider name"):
            _parse_openrouter_providers(" , , ")


class TestOpenRouterRoutingKwargs:
    """`openrouter_provider` + `openrouter_allow_fallbacks` shape `init_chat_model` kwargs.

    These tests capture what gets handed to `init_chat_model` so we can assert
    on the routing config (`only` list + `allow_fallbacks` flag) without
    spinning up a real model client.
    """

    @staticmethod
    def _capture_init(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[dict[str, object]]:
        captured: list[dict[str, object]] = []

        def fake_init(model: str, **kwargs: object) -> object:  # noqa: ARG001
            captured.append(kwargs)
            return object()

        monkeypatch.setattr(
            "deepagents_harbor.deepagents_wrapper.init_chat_model",
            fake_init,
        )
        return captured

    def test_strict_pin_with_single_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_init(monkeypatch)
        DeepAgentsWrapper(
            logs_dir=tmp_path,
            model_name="openrouter:minimax/minimax-m2",
            openrouter_provider="MiniMax",
        )
        assert captured[0]["openrouter_provider"] == {
            "only": ["MiniMax"],
            "allow_fallbacks": False,
        }

    def test_allowlist_with_fallbacks_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_init(monkeypatch)
        DeepAgentsWrapper(
            logs_dir=tmp_path,
            model_name="openrouter:minimax/minimax-m2",
            openrouter_provider="MiniMax,Fireworks",
            openrouter_allow_fallbacks=False,
        )
        assert captured[0]["openrouter_provider"] == {
            "only": ["MiniMax", "Fireworks"],
            "allow_fallbacks": False,
        }

    def test_allowlist_with_fallbacks_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_init(monkeypatch)
        DeepAgentsWrapper(
            logs_dir=tmp_path,
            model_name="openrouter:minimax/minimax-m2",
            openrouter_provider="MiniMax,Fireworks",
            openrouter_allow_fallbacks=True,
        )
        assert captured[0]["openrouter_provider"] == {
            "only": ["MiniMax", "Fireworks"],
            "allow_fallbacks": True,
        }

    def test_no_routing_kwargs_when_provider_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_init(monkeypatch)
        DeepAgentsWrapper(
            logs_dir=tmp_path,
            model_name="openrouter:minimax/minimax-m2",
        )
        assert "openrouter_provider" not in captured[0]

    def test_empty_provider_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one non-empty provider name"):
            DeepAgentsWrapper(
                logs_dir=tmp_path,
                model_name="openrouter:minimax/minimax-m2",
                openrouter_provider=" , , ",
            )

    def test_allow_fallbacks_without_provider_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="openrouter_allow_fallbacks requires openrouter_provider"
        ):
            DeepAgentsWrapper(
                logs_dir=tmp_path,
                model_name="openrouter:minimax/minimax-m2",
                openrouter_allow_fallbacks=True,
            )

    def test_allow_fallbacks_is_keyword_only(self) -> None:
        sig = inspect.signature(DeepAgentsWrapper.__init__)
        param = sig.parameters["openrouter_allow_fallbacks"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
