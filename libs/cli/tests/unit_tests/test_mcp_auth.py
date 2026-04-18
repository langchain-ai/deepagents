"""Tests for deepagents_cli.mcp_auth."""

import pytest

from deepagents_cli.mcp_auth import resolve_headers


class TestResolveHeaders:
    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "secret-value")
        assert resolve_headers({"Authorization": "Bearer ${FOO}"}) == {
            "Authorization": "Bearer secret-value"
        }

    def test_multiple_vars_in_one_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")
        assert resolve_headers({"X-Combo": "${A}-${B}"}) == {"X-Combo": "alpha-beta"}

    def test_missing_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            resolve_headers(
                {"Authorization": "Bearer ${MISSING_VAR}"},
                server_name="linear",
            )
        msg = str(exc_info.value)
        assert "MISSING_VAR" in msg
        assert "Authorization" in msg
        assert "linear" in msg

    def test_escape_double_dollar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOT_LOOKED_UP", raising=False)
        assert resolve_headers({"X-Literal": "$${NOT_LOOKED_UP}"}) == {
            "X-Literal": "${NOT_LOOKED_UP}"
        }

    def test_dollar_not_followed_by_brace_untouched(self) -> None:
        assert resolve_headers({"X-Price": "price=$5"}) == {"X-Price": "price=$5"}

    def test_non_string_value_rejected(self) -> None:
        with pytest.raises(TypeError):
            resolve_headers({"X-Bad": 123}, server_name="srv")  # type: ignore[dict-item]

    def test_empty_headers_returns_empty_dict(self) -> None:
        assert resolve_headers({}) == {}

    def test_no_substitution_when_no_placeholders(self) -> None:
        assert resolve_headers({"X-Plain": "hello"}) == {"X-Plain": "hello"}
