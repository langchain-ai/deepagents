"""Tests for SwarmFilter evaluation and dotted-path column access."""

from __future__ import annotations

from deepagents_repl._swarm.filter import MISSING, evaluate_filter, read_column


class TestReadColumn:
    def test_reads_flat_column(self) -> None:
        assert read_column({"x": 1}, "x") == 1

    def test_reads_dotted_path(self) -> None:
        row = {"meta": {"sector": "tech"}}
        assert read_column(row, "meta.sector") == "tech"

    def test_deep_dotted_path(self) -> None:
        row = {"a": {"b": {"c": "deep"}}}
        assert read_column(row, "a.b.c") == "deep"

    def test_missing_returns_sentinel(self) -> None:
        assert read_column({"x": 1}, "y") is MISSING

    def test_missing_intermediate_returns_sentinel(self) -> None:
        assert read_column({"a": 1}, "a.b.c") is MISSING

    def test_null_value_is_not_missing(self) -> None:
        row = {"x": None}
        assert read_column(row, "x") is None  # present, explicitly null
        assert read_column(row, "x") is not MISSING


class TestEquals:
    def test_matches_exact_value(self) -> None:
        assert evaluate_filter({"column": "x", "equals": 1}, {"x": 1}) is True
        assert evaluate_filter({"column": "x", "equals": 1}, {"x": 2}) is False

    def test_matches_null_for_missing_column(self) -> None:
        # Common usage pattern: `{column, equals: null}` to find unprocessed rows.
        assert evaluate_filter({"column": "result", "equals": None}, {"id": "a"}) is True

    def test_structural_equality_for_objects(self) -> None:
        row = {"meta": {"a": 1, "b": 2}}
        assert (
            evaluate_filter({"column": "meta", "equals": {"b": 2, "a": 1}}, row) is True
        )


class TestNotEquals:
    def test_inverted(self) -> None:
        assert evaluate_filter({"column": "x", "notEquals": 1}, {"x": 2}) is True
        assert evaluate_filter({"column": "x", "notEquals": 1}, {"x": 1}) is False


class TestIn:
    def test_membership(self) -> None:
        clause = {"column": "status", "in": ["pending", "active"]}
        assert evaluate_filter(clause, {"status": "pending"}) is True
        assert evaluate_filter(clause, {"status": "done"}) is False


class TestExists:
    def test_true_matches_present_value(self) -> None:
        assert evaluate_filter({"column": "x", "exists": True}, {"x": 0}) is True

    def test_true_does_not_match_null(self) -> None:
        assert evaluate_filter({"column": "x", "exists": True}, {"x": None}) is False

    def test_true_does_not_match_missing(self) -> None:
        assert evaluate_filter({"column": "x", "exists": True}, {}) is False

    def test_false_matches_missing_and_null(self) -> None:
        assert evaluate_filter({"column": "x", "exists": False}, {}) is True
        assert evaluate_filter({"column": "x", "exists": False}, {"x": None}) is True


class TestAnd:
    def test_all_must_match(self) -> None:
        clause = {
            "and": [
                {"column": "status", "equals": "active"},
                {"column": "priority", "in": ["high", "critical"]},
            ]
        }
        assert evaluate_filter(clause, {"status": "active", "priority": "high"}) is True
        assert evaluate_filter(clause, {"status": "active", "priority": "low"}) is False
        assert evaluate_filter(clause, {"status": "done", "priority": "high"}) is False

    def test_empty_and_is_vacuously_true(self) -> None:
        assert evaluate_filter({"and": []}, {}) is True


class TestOr:
    def test_any_match_succeeds(self) -> None:
        clause = {
            "or": [
                {"column": "x", "equals": 1},
                {"column": "y", "equals": 2},
            ]
        }
        assert evaluate_filter(clause, {"x": 1, "y": 99}) is True
        assert evaluate_filter(clause, {"x": 99, "y": 2}) is True
        assert evaluate_filter(clause, {"x": 99, "y": 99}) is False

    def test_empty_or_is_vacuously_false(self) -> None:
        assert evaluate_filter({"or": []}, {}) is False


class TestNested:
    def test_and_of_ors(self) -> None:
        clause = {
            "and": [
                {"or": [
                    {"column": "a", "equals": 1},
                    {"column": "b", "equals": 2},
                ]},
                {"column": "flag", "exists": True},
            ]
        }
        assert evaluate_filter(clause, {"a": 1, "flag": True}) is True
        assert evaluate_filter(clause, {"b": 2, "flag": True}) is True
        assert evaluate_filter(clause, {"a": 1}) is False  # no flag
