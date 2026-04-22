"""Tests for instruction template interpolation."""

from __future__ import annotations

import pytest

from deepagents_repl._swarm.interpolate import interpolate_instruction


def test_substitutes_string_bare() -> None:
    row = {"name": "Acme"}
    assert interpolate_instruction("Analyze {name}.", row) == "Analyze Acme."


def test_substitutes_dotted_path() -> None:
    row = {"meta": {"sector": "tech"}}
    assert (
        interpolate_instruction("Sector: {meta.sector}.", row) == "Sector: tech."
    )


def test_numeric_values_stringified() -> None:
    row = {"revenue": 1200, "ratio": 0.5}
    assert (
        interpolate_instruction("Revenue {revenue}, ratio {ratio}.", row)
        == "Revenue 1200, ratio 0.5."
    )


def test_booleans() -> None:
    row = {"flag": True, "other": False}
    assert (
        interpolate_instruction("{flag} and {other}", row) == "true and false"
    )


def test_objects_json_serialized() -> None:
    row = {"meta": {"a": 1, "b": 2}}
    result = interpolate_instruction("Meta: {meta}", row)
    assert result.startswith("Meta: {")
    assert '"a"' in result and '"b"' in result


def test_arrays_json_serialized() -> None:
    row = {"tags": ["a", "b"]}
    assert interpolate_instruction("Tags: {tags}", row) == 'Tags: ["a", "b"]'


def test_explicit_null_renders_as_null() -> None:
    row = {"status": None}
    assert interpolate_instruction("Status: {status}", row) == "Status: null"


def test_raises_on_missing_column() -> None:
    with pytest.raises(ValueError, match="Missing column\\(s\\) in row: name"):
        interpolate_instruction("Hi {name}", {})


def test_aggregates_multiple_missing() -> None:
    with pytest.raises(ValueError) as exc:
        interpolate_instruction("{a} {b} {c}", {"b": "ok"})
    msg = str(exc.value)
    assert "a" in msg
    assert "c" in msg
    assert "b" not in msg


def test_no_placeholders_passes_through() -> None:
    assert interpolate_instruction("no placeholders here", {}) == "no placeholders here"


def test_placeholder_is_trimmed() -> None:
    row = {"x": 1}
    assert interpolate_instruction("val={ x }", row) == "val=1"


def test_non_identifier_braces_left_alone() -> None:
    """The interpolator must only match identifier/dotted paths so that
    natural-language curly braces pass through untouched."""
    # JSON fragment in an instruction template — should not be interpolated.
    template = 'Output schema: { "label": string, "score": number }'
    assert interpolate_instruction(template, {}) == template
    # Expression with an operator — shouldn't parse as a column.
    assert interpolate_instruction("{n + 1}", {"n": 5}) == "{n + 1}"
