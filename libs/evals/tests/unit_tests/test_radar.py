from __future__ import annotations

import json

import pytest

from deepagents_evals.radar import (
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    EVAL_CATEGORIES,
    ModelResult,
    _safe_filename,
    _short_model_name,
    generate_individual_radars,
    generate_radar,
    load_results_from_summary,
    toy_data,
)

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def test_toy_data_covers_all_categories():
    results = toy_data()
    assert len(results) >= 2
    for r in results:
        for cat in EVAL_CATEGORIES:
            assert cat in r.scores, f"{r.model} missing category {cat}"
            assert 0.0 <= r.scores[cat] <= 1.0


def test_category_labels_cover_all_categories():
    assert set(CATEGORY_LABELS.keys()) == set(ALL_CATEGORIES)


def test_short_model_name_strips_provider():
    assert _short_model_name("anthropic:claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert _short_model_name("openai:gpt-5.4") == "gpt-5.4"


def test_short_model_name_truncates_long():
    assert _short_model_name("a" * 50) == "a" * 27 + "..."


def test_short_model_name_exact_boundary():
    assert _short_model_name("a" * 30) == "a" * 30
    assert _short_model_name("a" * 31) == "a" * 27 + "..."


def test_short_model_name_no_provider():
    assert _short_model_name("gpt-5.4") == "gpt-5.4"


def test_short_model_name_provider_and_long():
    assert _short_model_name("provider:" + "x" * 50) == "x" * 27 + "..."


# --- generate_radar ---


def test_generate_radar_returns_figure():
    results = toy_data()
    fig = generate_radar(results, title="Test")
    assert fig is not None
    assert len(fig.get_axes()) == 1


def test_generate_radar_saves_to_file(tmp_path):
    out = tmp_path / "radar.png"
    results = toy_data()
    generate_radar(results, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_radar_saves_nested_directory(tmp_path):
    out = tmp_path / "nested" / "dir" / "radar.png"
    results = toy_data()
    generate_radar(results, output=out)
    assert out.exists()


def test_generate_radar_custom_categories():
    results = [ModelResult(model="test", scores={"a": 0.5, "b": 0.8, "c": 0.3})]
    fig = generate_radar(results, categories=["a", "b", "c"])
    assert fig is not None


def test_generate_radar_missing_scores_default_zero():
    results = [ModelResult(model="test", scores={"file_operations": 0.9})]
    fig = generate_radar(results)
    assert fig is not None


def test_generate_radar_many_models_color_cycling():
    results = [ModelResult(model=f"model-{i}", scores={"a": 0.5, "b": 0.8}) for i in range(10)]
    fig = generate_radar(results, categories=["a", "b"])
    assert fig is not None


# --- generate_individual_radars ---


def test_generate_individual_radars_creates_per_model_files(tmp_path):
    results = toy_data()
    paths = generate_individual_radars(results, output_dir=tmp_path)
    assert len(paths) == len(results)
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"


def test_generate_individual_radars_filenames_are_safe(tmp_path):
    results = [
        ModelResult(model="anthropic:claude-sonnet-4-6", scores={"a": 0.5, "b": 0.8, "c": 0.3}),
        ModelResult(model="openai:gpt-5.4", scores={"a": 0.6, "b": 0.7, "c": 0.4}),
    ]
    paths = generate_individual_radars(results, output_dir=tmp_path, categories=["a", "b", "c"])
    names = [p.stem for p in paths]
    assert "anthropic-claude-sonnet-4-6" in names
    assert "openai-gpt-5.4" in names


def test_generate_individual_radars_single_model(tmp_path):
    results = [ModelResult(model="test", scores={"a": 0.5, "b": 0.8, "c": 0.3})]
    paths = generate_individual_radars(results, output_dir=tmp_path, categories=["a", "b", "c"])
    assert len(paths) == 1


# --- _safe_filename ---


def test_safe_filename_replaces_colons():
    assert _safe_filename("anthropic:claude-sonnet-4-6") == "anthropic-claude-sonnet-4-6"


def test_safe_filename_replaces_slashes():
    assert _safe_filename("org/model/v1") == "org-model-v1"


def test_safe_filename_empty_string():
    assert _safe_filename("") == "unknown"


def test_safe_filename_only_special_chars():
    assert _safe_filename(":::") == "unknown"


# --- load_results_from_summary ---


def test_load_results_from_summary_happy_path(tmp_path):
    data = [
        {
            "model": "anthropic:claude-sonnet-4-6",
            "category_scores": {"file_operations": 0.85, "memory": 0.90},
        },
        {
            "model": "openai:gpt-5.4",
            "category_scores": {"file_operations": 0.72, "memory": 0.80},
        },
    ]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    results = load_results_from_summary(path)
    assert len(results) == 2
    assert results[0].model == "anthropic:claude-sonnet-4-6"
    assert results[0].scores == {"file_operations": 0.85, "memory": 0.90}
    assert results[1].scores == {"file_operations": 0.72, "memory": 0.80}


def test_load_results_from_summary_missing_category_scores_raises(tmp_path):
    data = [{"model": "test-model"}]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(KeyError):
        load_results_from_summary(path)


def test_load_results_from_summary_missing_model_defaults(tmp_path):
    data = [{"category_scores": {"memory": 0.9}}]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    results = load_results_from_summary(path)
    assert results[0].model == "unknown"


def test_load_results_from_summary_empty_array(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("[]", encoding="utf-8")

    results = load_results_from_summary(path)
    assert results == []


def test_load_results_from_summary_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_results_from_summary("/nonexistent/path.json")


def test_load_results_from_summary_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_results_from_summary(path)
