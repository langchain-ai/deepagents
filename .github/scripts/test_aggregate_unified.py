import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified as au  # noqa: E402


def _summary(
    model, k, pass_k, avg_k, tasks, passed, incomplete=False, category="context"
):
    # model + category are recorded authoritatively by aggregate_shards.py.
    return {
        "model": model,
        "category": category,
        "dataset": "d",
        "rollouts_per_task": k,
        "incomplete": incomplete,
        "totals": {
            "tasks": tasks,
            "trials": tasks * k,
            "expected_trials": tasks * k,
            "passed": passed,
            "errored": 0,
        },
        f"pass@{k}": pass_k,
        f"avg@{k}": avg_k,
    }


def test_read_leaf_reads_model_and_category_from_summary(tmp_path):
    d = tmp_path / "harbor-anthropic-opus-context"
    d.mkdir()
    (d / "summary.json").write_text(
        json.dumps(_summary("anthropic:opus", 3, 0.8, 0.5, 30, 45, category="context"))
    )
    leaf = au.read_leaf(d)
    assert leaf["model"] == "anthropic:opus"
    assert leaf["category"] == "context"
    assert leaf["pass_at_k"] == 0.8
    assert leaf["avg_at_k"] == 0.5


def test_read_leaf_coerces_null_model(tmp_path):
    # summary.json's model is null only if --model wasn't passed; guard anyway.
    d = tmp_path / "leaf"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_summary(None, 3, None, None, 0, 0)))
    assert au.read_leaf(d)["model"] == "unknown"


def test_discover_leaves_reads_summary_from_download_root(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(_summary("anthropic:opus", 3, 0.8, 0.5, 30, 45))
    )

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["anthropic:opus"]


def test_discover_leaves_treats_root_summary_layout_as_exclusive(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(_summary("root-model", 3, 0.8, 0.5, 30, 45))
    )
    child = tmp_path / "child-artifact"
    child.mkdir()
    (child / "summary.json").write_text(
        json.dumps(_summary("child-model", 3, 0.7, 0.4, 30, 36))
    )

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["root-model"]


def test_discover_leaves_warns_and_skips_truncated_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "a-malformed"
    malformed.mkdir()
    (malformed / "summary.json").write_text('{"rollouts_per_task":')
    valid = tmp_path / "b-valid"
    valid.mkdir()
    (valid / "summary.json").write_text(
        json.dumps(_summary("anthropic:opus", 3, 0.8, 0.5, 30, 45))
    )

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["anthropic:opus"]
    warning = capsys.readouterr().out
    assert "::warning::" in warning
    assert str(malformed / "summary.json") in warning


def test_discover_leaves_skips_summary_with_invalid_aggregation_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "a-malformed"
    malformed.mkdir()
    malformed_summary = _summary("poison", 3, 0.8, 0.5, 30, 45)
    malformed_summary["totals"]["tasks"] = "thirty"
    (malformed / "summary.json").write_text(json.dumps(malformed_summary))
    valid = tmp_path / "b-valid"
    valid.mkdir()
    (valid / "summary.json").write_text(
        json.dumps(_summary("anthropic:opus", 3, 0.8, 0.5, 30, 45))
    )

    combined = au.combine(au._discover_leaves(tmp_path))

    assert list(combined["models"]) == ["anthropic:opus"]
    warning = capsys.readouterr().out
    assert "::warning::" in warning
    assert str(malformed / "summary.json") in warning


def test_load_list_env_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", "[")

    with pytest.raises(SystemExit) as exc_info:
        au._load_list_env("EXPECTED_MODELS")

    assert str(exc_info.value) == "EXPECTED_MODELS must be a JSON list of strings"


@pytest.mark.parametrize("value", [{"model": "opus"}, ["opus", 3]])
def test_load_list_env_rejects_invalid_decoded_shape(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", json.dumps(value))

    with pytest.raises(SystemExit) as exc_info:
        au._load_list_env("EXPECTED_MODELS")

    assert str(exc_info.value) == "EXPECTED_MODELS must be a JSON list of strings"


def test_combine_computes_macro_and_micro():
    leaves = [
        {
            "model": "m",
            "category": "autonomous",
            "pass_at_k": 1.0,
            "avg_at_k": 1.0,
            "tasks": 10,
            "passed": 30,
            "incomplete": False,
        },
        {
            "model": "m",
            "category": "context",
            "pass_at_k": 0.0,
            "avg_at_k": 0.0,
            "tasks": 30,
            "passed": 0,
            "incomplete": False,
        },
    ]
    out = au.combine(leaves)
    m = out["models"]["m"]
    # macro = mean of category pass@k = (1.0 + 0.0)/2 = 0.5
    assert m["macro"]["pass_at_k"] == 0.5
    # Both micro metrics are task-weighted: (1.0*10 + 0.0*30)/40 = 0.25.
    assert abs(m["micro"]["pass_at_k"] - 0.25) < 1e-9
    assert abs(m["micro"]["avg_at_k"] - 0.25) < 1e-9


def test_combine_ignores_none_metrics_from_zero_task_leaf() -> None:
    leaves = [
        {
            "model": "m",
            "category": "empty",
            "pass_at_k": None,
            "avg_at_k": None,
            "tasks": 0,
            "passed": 0,
            "incomplete": False,
        },
        {
            "model": "m",
            "category": "populated",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
    ]

    model = au.combine(leaves)["models"]["m"]

    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["micro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["categories"]["empty"]["incomplete"] is True
    assert model["incomplete"] is True


def test_combine_rejects_duplicate_model_category_leaves() -> None:
    leaves = [
        {
            "model": "m",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "m",
            "category": "context",
            "pass_at_k": 0.7,
            "avg_at_k": 0.4,
            "tasks": 10,
            "passed": 12,
            "incomplete": False,
        },
    ]

    with pytest.raises(
        ValueError, match="Duplicate leaf for model 'm' and category 'context'"
    ):
        au.combine(leaves)


def test_render_markdown_sorts_by_macro_desc():
    combined = {
        "categories": ["autonomous", "context"],
        "models": {
            "lo": {
                "categories": {
                    "autonomous": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                    "context": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                },
                "macro": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                "micro": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                "incomplete": False,
            },
            "hi": {
                "categories": {
                    "autonomous": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                    "context": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                },
                "macro": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                "micro": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                "incomplete": False,
            },
        },
    }
    md = au.render_markdown(combined, k=3)
    assert md.index("hi") < md.index("lo")  # higher macro ranked first
    assert "pass@3" in md


def test_render_markdown_describes_expected_model_without_leaf_summaries() -> None:
    combined = au.combine([], expected_models=["ghost"])

    markdown = au.render_markdown(combined, k=3)

    assert "`ghost` — no leaf summaries found" in markdown


def test_write_outputs_describes_expected_model_without_leaf_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    combined = au.combine([], expected_models=["ghost"])

    au.write_outputs(combined, k=3, out_dir=tmp_path, step_summary_path=None)

    output = capsys.readouterr().out
    assert "no leaf summaries found" in output
    assert "a category reported incomplete data" not in output


def test_radar_results_shape():
    combined = {
        "categories": ["autonomous", "context"],
        "models": {
            "m": {
                "categories": {
                    "autonomous": {"pass_at_k": 0.5},
                    "context": {"pass_at_k": 0.7},
                },
                "macro": {"pass_at_k": 0.6},
                "micro": {"pass_at_k": 0.6},
                "incomplete": False,
            }
        },
    }
    rr = au.radar_results(combined)
    assert rr == [{"model": "m", "scores": {"autonomous": 0.5, "context": 0.7}}]


def _leaf_dir(
    tmp_path,
    model,
    category,
    k=3,
    pass_k=0.5,
    avg_k=0.5,
    tasks=10,
    passed=15,
    incomplete=False,
):
    d = tmp_path / f"harbor-{model}-{category}".replace(":", "-").replace("/", "-")
    d.mkdir()
    (d / "summary.json").write_text(
        json.dumps(
            _summary(
                model, k, pass_k, avg_k, tasks, passed, incomplete, category=category
            )
        )
    )
    return d


def test_discover_leaves_rejects_numeric_metrics_for_zero_task_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = _leaf_dir(
        tmp_path, "m", "empty", pass_k=1.0, avg_k=1.0, tasks=0, passed=0
    )
    _leaf_dir(tmp_path, "m", "populated", pass_k=0.8, avg_k=0.5)

    model = au.combine(au._discover_leaves(tmp_path))["models"]["m"]

    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert str(malformed / "summary.json") in capsys.readouterr().out


@pytest.mark.parametrize(
    ("pass_k", "avg_k"),
    [(-0.01, 0.5), (0.5, 1.01)],
)
def test_discover_leaves_rejects_metrics_outside_unit_interval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pass_k: float,
    avg_k: float,
) -> None:
    malformed = _leaf_dir(tmp_path, "bad", "context", pass_k=pass_k, avg_k=avg_k)
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["good"]
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_discover_leaves_warns_and_skips_unconvertibly_large_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = _leaf_dir(tmp_path, "bad", "context", pass_k=10**400, avg_k=0.5)
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["good"]
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_discover_leaves_warns_and_skips_oversized_task_weight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = _leaf_dir(
        tmp_path,
        "bad",
        "context",
        pass_k=0.5,
        avg_k=0.5,
        tasks=10**400,
        passed=0,
    )
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    combined = au.combine(au._discover_leaves(tmp_path))

    assert list(combined["models"]) == ["good"]
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_discover_leaves_warns_and_skips_json_integer_over_digit_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "a-malformed"
    malformed.mkdir()
    oversized_integer = "9" * 5000
    (malformed / "summary.json").write_text(
        f'{{"rollouts_per_task": {oversized_integer}}}'
    )
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["good"]
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_discover_leaves_requires_producer_incomplete_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "harbor-bad-context"
    malformed.mkdir()
    malformed_summary = _summary("bad", 3, 0.8, 0.5, 10, 15)
    del malformed_summary["incomplete"]
    (malformed / "summary.json").write_text(json.dumps(malformed_summary))
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["good"]
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_main_writes_outputs_and_skips_radar_for_subset(tmp_path):
    for cat, pk in [("autonomous", 0.5), ("context", 0.6)]:
        _leaf_dir(tmp_path, "m", cat, pass_k=pk, avg_k=pk)
    out = tmp_path / "combined"
    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "unified_summary.json").exists()
    # only 2 categories -> radar input not emitted (workflow radar step is gated on it)
    assert not (out / "radar_results.json").exists()


def test_main_emits_radar_input_for_three_categories(tmp_path):
    for cat in ("autonomous", "conversation", "context"):
        _leaf_dir(tmp_path, "m", cat)
    out = tmp_path / "combined"
    au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])
    assert (out / "radar_results.json").exists()


@pytest.mark.parametrize("rollouts", [0, -1])
def test_main_rejects_nonpositive_rollouts(
    tmp_path: Path, rollouts: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        au.main([str(tmp_path), "--rollouts", str(rollouts)])

    assert exc_info.value.code == 2
    assert "--rollouts must be >= 1" in capsys.readouterr().err


def test_main_fails_only_when_every_expected_model_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", '["m"]')
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    assert rc == 1
    assert (out / "unified_summary.json").exists()
    assert "::error::Every expected model is incomplete" in capsys.readouterr().out

    _leaf_dir(tmp_path, "m", "context")
    assert au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)]) == 0


def test_main_fails_when_required_leaf_has_no_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", '["m"]')
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    _leaf_dir(
        tmp_path,
        "m",
        "context",
        pass_k=None,
        avg_k=None,
        tasks=0,
        passed=0,
    )
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    combined = json.loads((out / "unified_summary.json").read_text())
    model = combined["models"]["m"]
    assert rc == 1
    assert model["categories"]["context"]["incomplete"] is True
    assert model["incomplete"] is True
    assert "::error::Every expected model is incomplete" in capsys.readouterr().out


def test_main_ignores_unexpected_complete_model_for_expected_grid_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", '["missing"]')
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    _leaf_dir(tmp_path, "extra", "context")
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    assert rc == 1
    assert "::error::Every expected model is incomplete" in capsys.readouterr().out


def test_main_does_not_apply_expected_grid_failure_without_expected_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("EXPECTED_MODELS", raising=False)
    monkeypatch.delenv("EXPECTED_CATEGORIES", raising=False)
    _leaf_dir(tmp_path, "m", "context", incomplete=True)
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "::warning::Model m incomplete" in output
    assert "::error::Every expected model is incomplete" not in output


def test_main_warns_and_skips_leaf_with_mismatched_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXPECTED_MODELS", '["m"]')
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    leaf_dir = _leaf_dir(tmp_path, "m", "context", k=2, pass_k=0.8, avg_k=0.5)
    assert au.read_leaf(leaf_dir)["pass_at_k"] == 0.8
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    output = capsys.readouterr().out
    combined = json.loads((out / "unified_summary.json").read_text())
    assert rc == 1
    assert "rollouts_per_task is 2; expected 3" in output
    assert combined["models"]["m"]["missing_categories"] == ["context"]


def test_combine_flags_missing_leaves_against_expected_grid():
    # "b" ran only autonomous; "context" leaf never uploaded.
    leaves = [
        {
            "model": "a",
            "category": "autonomous",
            "pass_at_k": 0.9,
            "avg_at_k": 0.9,
            "tasks": 10,
            "passed": 9,
            "incomplete": False,
        },
        {
            "model": "a",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.8,
            "tasks": 10,
            "passed": 8,
            "incomplete": False,
        },
        {
            "model": "b",
            "category": "autonomous",
            "pass_at_k": 1.0,
            "avg_at_k": 1.0,
            "tasks": 10,
            "passed": 10,
            "incomplete": False,
        },
    ]
    out = au.combine(
        leaves,
        expected_models=["a", "b"],
        expected_categories=["autonomous", "context"],
    )
    assert out["categories"] == ["autonomous", "context"]
    assert out["models"]["a"]["incomplete"] is False
    assert out["models"]["b"]["incomplete"] is True
    assert out["models"]["b"]["missing_categories"] == ["context"]


def test_combine_excludes_display_only_categories_from_expected_completeness() -> None:
    leaves = [
        {
            "model": "m",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "extra",
            "category": "autonomous",
            "pass_at_k": 0.7,
            "avg_at_k": 0.4,
            "tasks": 10,
            "passed": 12,
            "incomplete": False,
        },
    ]

    out = au.combine(leaves, expected_models=["m"], expected_categories=["context"])

    assert out["categories"] == ["context", "autonomous"]
    assert out["models"]["m"]["missing_categories"] == []
    assert out["models"]["m"]["incomplete"] is False


@pytest.mark.parametrize("unexpected_incomplete", [False, True])
def test_combine_scores_only_expected_categories(
    unexpected_incomplete: bool,
) -> None:
    leaves = [
        {
            "model": "m",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "m",
            "category": "autonomous",
            "pass_at_k": 0.0,
            "avg_at_k": 0.0,
            "tasks": 30,
            "passed": 0,
            "incomplete": unexpected_incomplete,
        },
    ]

    model = au.combine(leaves, expected_categories=["context"])["models"]["m"]

    assert set(model["categories"]) == {"context", "autonomous"}
    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["micro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["incomplete"] is False


def test_combine_includes_expected_model_with_no_leaves():
    out = au.combine([], expected_models=["ghost"], expected_categories=["autonomous"])
    assert "ghost" in out["models"]
    assert out["models"]["ghost"]["incomplete"] is True
    assert out["models"]["ghost"]["missing_categories"] == ["autonomous"]


def test_combine_marks_expected_model_without_leaves_incomplete_without_categories() -> (
    None
):
    out = au.combine([], expected_models=["ghost"])

    assert out["models"]["ghost"]["missing_categories"] == []
    assert out["models"]["ghost"]["incomplete"] is True
