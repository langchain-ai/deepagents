import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified as au  # noqa: E402


def _summary(
    model,
    k,
    pass_k,
    avg_k,
    tasks,
    passed,
    incomplete=False,
    category="context",
    config="bare",
):
    # model + category + config are recorded authoritatively by aggregate_shards.py.
    return {
        "model": model,
        "category": category,
        "config": config,
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

    assert [r["model"] for r in combined["rows"]] == ["anthropic:opus"]
    warning = capsys.readouterr().out
    assert "::warning::" in warning
    assert str(malformed / "summary.json") in warning


def test_load_list_env_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _load_list_env still backs EXPECTED_CATEGORIES.
    monkeypatch.setenv("EXPECTED_CATEGORIES", "[")

    with pytest.raises(SystemExit) as exc_info:
        au._load_list_env("EXPECTED_CATEGORIES")

    assert str(exc_info.value) == "EXPECTED_CATEGORIES must be a JSON list of strings"


@pytest.mark.parametrize("value", [{"model": "opus"}, ["opus", 3]])
def test_load_list_env_rejects_invalid_decoded_shape(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setenv("EXPECTED_CATEGORIES", json.dumps(value))

    with pytest.raises(SystemExit) as exc_info:
        au._load_list_env("EXPECTED_CATEGORIES")

    assert str(exc_info.value) == "EXPECTED_CATEGORIES must be a JSON list of strings"


def test_load_leaves_env_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPECTED_LEAVES", "[")

    with pytest.raises(SystemExit) as exc_info:
        au._load_leaves_env("EXPECTED_LEAVES")

    assert str(exc_info.value) == (
        "EXPECTED_LEAVES must be a JSON list of {model, branch, config, category} objects"
    )


def test_load_leaves_env_preserves_explicit_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPECTED_LEAVES", "[]")

    assert au._load_leaves_env("EXPECTED_LEAVES") == []


@pytest.mark.parametrize(
    "value",
    [
        {"model": "opus"},
        [{"model": "m", "branch": "main", "config": "bare"}],
        ["m"],
    ],
)
def test_load_leaves_env_rejects_invalid_decoded_shape(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setenv("EXPECTED_LEAVES", json.dumps(value))

    with pytest.raises(SystemExit) as exc_info:
        au._load_leaves_env("EXPECTED_LEAVES")

    assert str(exc_info.value) == (
        "EXPECTED_LEAVES must be a JSON list of {model, branch, config, category} objects"
    )


def test_combine_computes_macro_and_micro():
    leaves = [
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
            "pass_at_k": 1.0,
            "avg_at_k": 1.0,
            "tasks": 10,
            "passed": 30,
            "incomplete": False,
        },
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.0,
            "avg_at_k": 0.0,
            "tasks": 30,
            "passed": 0,
            "incomplete": False,
        },
    ]
    out = au.combine(leaves)
    (m,) = out["rows"]
    # macro = mean of category pass@k = (1.0 + 0.0)/2 = 0.5
    assert m["macro"]["pass_at_k"] == 0.5
    # Both micro metrics are task-weighted: (1.0*10 + 0.0*30)/40 = 0.25.
    assert abs(m["micro"]["pass_at_k"] - 0.25) < 1e-9
    assert abs(m["micro"]["avg_at_k"] - 0.25) < 1e-9


def test_combine_ignores_none_metrics_from_zero_task_leaf() -> None:
    leaves = [
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "empty",
            "pass_at_k": None,
            "avg_at_k": None,
            "tasks": 0,
            "passed": 0,
            "incomplete": False,
        },
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "populated",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
    ]

    (model,) = au.combine(leaves)["rows"]

    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["micro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["categories"]["empty"]["incomplete"] is True
    assert model["incomplete"] is True


def test_combine_quarantines_duplicate_model_config_category_leaves() -> None:
    leaves = [
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.7,
            "avg_at_k": 0.4,
            "tasks": 10,
            "passed": 12,
            "incomplete": False,
        },
    ]

    result = au.combine(leaves)

    assert result["rows"][0]["categories"] == {}
    assert result["rows"][0]["incomplete"] is True
    assert result["issues"][0]["code"] == "duplicate_leaf"


def test_render_markdown_sorts_by_macro_desc():
    combined = {
        "categories": ["autonomous", "context"],
        "rows": [
            {
                "model": "lo",
                "branch": "current",
                "config": "bare",
                "categories": {
                    "autonomous": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                    "context": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                },
                "macro": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                "micro": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                "incomplete": False,
            },
            {
                "model": "hi",
                "branch": "current",
                "config": "bare",
                "categories": {
                    "autonomous": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                    "context": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                },
                "macro": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                "micro": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                "incomplete": False,
            },
        ],
    }
    md = au.render_markdown(combined, k=3)
    assert md.index("hi") < md.index("lo")  # higher macro ranked first
    assert "pass@3" in md


def test_render_markdown_ranks_none_macro_last():
    # A row with an incomplete/ghost result has a None macro (_mean([]) is
    # None). The sort key pushes None to the bottom; if that element were
    # dropped or inverted, incomplete rows would rank above scored ones.
    combined = {
        "categories": ["context"],
        "rows": [
            {
                "model": "ghost",
                "branch": "current",
                "config": "bare",
                "categories": {},
                "macro": {"pass_at_k": None, "avg_at_k": None},
                "micro": {"pass_at_k": None, "avg_at_k": None},
                "incomplete": True,
            },
            {
                "model": "scored",
                "branch": "current",
                "config": "bare",
                "categories": {"context": {"pass_at_k": 0.2, "avg_at_k": 0.2}},
                "macro": {"pass_at_k": 0.2, "avg_at_k": 0.2},
                "micro": {"pass_at_k": 0.2, "avg_at_k": 0.2},
                "incomplete": False,
            },
        ],
    }
    md = au.render_markdown(combined, k=3)
    assert md.index("scored") < md.index("ghost")  # None macro ranked last


def test_render_markdown_describes_expected_row_without_leaf_summaries() -> None:
    combined = au.combine(
        [],
        expected_leaves=[
            {
                "model": "ghost",
                "branch": "current",
                "config": "bare",
                "category": "autonomous",
            }
        ],
    )

    markdown = au.render_markdown(combined, k=3)

    assert "`ghost / current / bare` — no leaf summaries found" in markdown


def test_write_outputs_describes_expected_row_without_leaf_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    combined = au.combine(
        [],
        expected_leaves=[
            {
                "model": "ghost",
                "branch": "current",
                "config": "bare",
                "category": "autonomous",
            }
        ],
    )

    au.write_outputs(combined, k=3, out_dir=tmp_path, step_summary_path=None)

    output = capsys.readouterr().out
    assert "no leaf summaries found" in output
    assert "a category reported incomplete data" not in output


def test_radar_results_shape():
    combined = {
        "categories": ["autonomous", "context"],
        "rows": [
            {
                "model": "m",
                "branch": "current",
                "config": "bare",
                "categories": {
                    "autonomous": {"pass_at_k": 0.5},
                    "context": {"pass_at_k": 0.7},
                },
                "macro": {"pass_at_k": 0.6},
                "micro": {"pass_at_k": 0.6},
                "incomplete": False,
            }
        ],
    }
    rr = au.radar_results(combined)
    assert rr == [
        {
            "model": "m / current / bare",
            "scores": {"autonomous": 0.5, "context": 0.7},
        }
    ]


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
    config="bare",
):
    d = tmp_path / f"harbor-{model}-{category}".replace(":", "-").replace("/", "-")
    d.mkdir()
    (d / "summary.json").write_text(
        json.dumps(
            _summary(
                model,
                k,
                pass_k,
                avg_k,
                tasks,
                passed,
                incomplete,
                category=category,
                config=config,
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

    (model,) = au.combine(au._discover_leaves(tmp_path))["rows"]

    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert str(malformed / "summary.json") in capsys.readouterr().out


def test_discover_leaves_rejects_null_metric_for_populated_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Inverse of the zero-task rule: a populated leaf (tasks > 0) with a null
    # metric must be rejected. Otherwise combine would treat the null as 0.0
    # (`leaf["pass_at_k"] or 0.0`) weighted by a real task count, silently
    # deflating the model's micro score instead of surfacing the bad leaf.
    malformed = _leaf_dir(tmp_path, "bad", "context", pass_k=None, avg_k=0.5)
    _leaf_dir(tmp_path, "good", "context", pass_k=0.8, avg_k=0.5)

    leaves = au._discover_leaves(tmp_path)

    assert [leaf["model"] for leaf in leaves] == ["good"]
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

    assert [r["model"] for r in combined["rows"]] == ["good"]
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


def test_main_records_invalid_expected_grid_as_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _leaf_dir(tmp_path, "m", "context")
    monkeypatch.setenv("EXPECTED_LEAVES", "not-json")
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    out = tmp_path / "combined"

    assert au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)]) == 0

    combined = json.loads((out / "unified_summary.json").read_text())
    assert combined["issues"][0]["code"] == "invalid_expected_leaves"


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


def test_main_reports_when_every_expected_row_has_no_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A run with no usable leaves emits a diagnostic scorecard without failing.
    monkeypatch.setenv(
        "EXPECTED_LEAVES",
        '[{"model": "m", "branch": "current", "config": "bare", "category": "context"}]',
    )
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    assert rc == 0
    assert (out / "unified_summary.json").exists()
    combined = json.loads((out / "unified_summary.json").read_text())
    assert combined["issues"][0]["code"] == "no_usable_leaf_summaries"
    assert (
        "::warning::No usable eval leaf summaries were found" in capsys.readouterr().out
    )

    _leaf_dir(tmp_path, "m", "context")
    assert au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)]) == 0


def test_main_warns_when_required_leaf_has_no_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(
        "EXPECTED_LEAVES",
        '[{"model": "m", "branch": "current", "config": "bare", "category": "context"}]',
    )
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
    (model,) = combined["rows"]
    assert rc == 0
    assert model["categories"]["context"]["incomplete"] is True
    assert model["incomplete"] is True
    assert (
        "::warning::Every expected (model, branch, config) row is incomplete"
        in capsys.readouterr().out
    )


def test_main_passes_when_an_unexpected_row_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # New contract: main fails only when EVERY row (expected or not) is incomplete.
    # A complete unexpected (model, branch, config) row therefore keeps the run green,
    # while
    # the missing expected leaf is still surfaced as a warning.
    monkeypatch.setenv(
        "EXPECTED_LEAVES",
        '[{"model": "missing", "branch": "current", "config": "bare", "category": "context"}]',
    )
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    _leaf_dir(tmp_path, "extra", "context")
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "::warning::missing / current / bare incomplete" in output
    assert (
        "::warning::Every expected (model, branch, config) row is incomplete"
        not in output
    )


def test_main_does_not_apply_expected_grid_failure_without_expected_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("EXPECTED_LEAVES", raising=False)
    monkeypatch.delenv("EXPECTED_CATEGORIES", raising=False)
    _leaf_dir(tmp_path, "m", "context", incomplete=True)
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    output = capsys.readouterr().out
    assert rc == 0
    assert "::warning::m / current / bare incomplete" in output
    assert (
        "::error::Every expected (model, branch, config) row is incomplete"
        not in output
    )


def test_main_preserves_explicit_empty_expected_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXPECTED_LEAVES", "[]")
    monkeypatch.setenv("EXPECTED_CATEGORIES", "[]")
    _leaf_dir(tmp_path, "m", "context", incomplete=True)
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    assert rc == 0
    assert (
        "::warning::Every expected (model, branch, config) row is incomplete"
        in capsys.readouterr().out
    )


def test_main_warns_and_skips_leaf_with_mismatched_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(
        "EXPECTED_LEAVES",
        '[{"model": "m", "branch": "current", "config": "bare", "category": "context"}]',
    )
    monkeypatch.setenv("EXPECTED_CATEGORIES", '["context"]')
    leaf_dir = _leaf_dir(tmp_path, "m", "context", k=2, pass_k=0.8, avg_k=0.5)
    assert au.read_leaf(leaf_dir)["pass_at_k"] == 0.8
    out = tmp_path / "combined"

    rc = au.main([str(tmp_path), "--rollouts", "3", "--out-dir", str(out)])

    output = capsys.readouterr().out
    combined = json.loads((out / "unified_summary.json").read_text())
    (model,) = combined["rows"]
    assert rc == 0
    assert "::warning::No usable eval leaf summaries were found" in output
    assert "rollouts_per_task is 2; expected 3" in output
    assert model["missing_categories"] == ["context"]


def test_combine_flags_missing_leaves_against_expected_grid():
    # "b" ran only autonomous; "context" leaf never uploaded.
    leaves = [
        {
            "model": "a",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
            "pass_at_k": 0.9,
            "avg_at_k": 0.9,
            "tasks": 10,
            "passed": 9,
            "incomplete": False,
        },
        {
            "model": "a",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.8,
            "tasks": 10,
            "passed": 8,
            "incomplete": False,
        },
        {
            "model": "b",
            "branch": "current",
            "config": "bare",
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
        expected_leaves=[
            {
                "model": "a",
                "branch": "current",
                "config": "bare",
                "category": "autonomous",
            },
            {
                "model": "a",
                "branch": "current",
                "config": "bare",
                "category": "context",
            },
            {
                "model": "b",
                "branch": "current",
                "config": "bare",
                "category": "autonomous",
            },
            {
                "model": "b",
                "branch": "current",
                "config": "bare",
                "category": "context",
            },
        ],
        expected_categories=["autonomous", "context"],
    )
    rows = {r["model"]: r for r in out["rows"]}
    assert out["categories"] == ["autonomous", "context"]
    assert rows["a"]["incomplete"] is False
    assert rows["b"]["incomplete"] is True
    assert rows["b"]["missing_categories"] == ["context"]


def test_combine_excludes_display_only_categories_from_expected_completeness() -> None:
    leaves = [
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "extra",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
            "pass_at_k": 0.7,
            "avg_at_k": 0.4,
            "tasks": 10,
            "passed": 12,
            "incomplete": False,
        },
    ]

    out = au.combine(
        leaves,
        expected_leaves=[
            {"model": "m", "branch": "current", "config": "bare", "category": "context"}
        ],
        expected_categories=["context"],
    )

    rows = {r["model"]: r for r in out["rows"]}
    assert out["categories"] == ["context", "autonomous"]
    assert rows["m"]["missing_categories"] == []
    assert rows["m"]["incomplete"] is False


@pytest.mark.parametrize("unexpected_incomplete", [False, True])
def test_combine_scores_only_expected_categories(
    unexpected_incomplete: bool,
) -> None:
    leaves = [
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "context",
            "pass_at_k": 0.8,
            "avg_at_k": 0.5,
            "tasks": 10,
            "passed": 15,
            "incomplete": False,
        },
        {
            "model": "m",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
            "pass_at_k": 0.0,
            "avg_at_k": 0.0,
            "tasks": 30,
            "passed": 0,
            "incomplete": unexpected_incomplete,
        },
    ]

    # The required category set now comes from the expected-leaf grid, so scoring
    # is scoped to "context" even though the "autonomous" leaf is present.
    (model,) = au.combine(
        leaves,
        expected_leaves=[
            {"model": "m", "branch": "current", "config": "bare", "category": "context"}
        ],
        expected_categories=["context"],
    )["rows"]

    assert set(model["categories"]) == {"context", "autonomous"}
    assert model["macro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["micro"] == {"pass_at_k": 0.8, "avg_at_k": 0.5}
    assert model["incomplete"] is False


def test_combine_includes_expected_row_with_no_leaves():
    out = au.combine(
        [],
        expected_leaves=[
            {
                "model": "ghost",
                "branch": "current",
                "config": "bare",
                "category": "autonomous",
            }
        ],
        expected_categories=["autonomous"],
    )
    rows = {(r["model"], r["branch"], r["config"]): r for r in out["rows"]}
    assert ("ghost", "current", "bare") in rows
    assert rows[("ghost", "current", "bare")]["incomplete"] is True
    assert rows[("ghost", "current", "bare")]["missing_categories"] == ["autonomous"]


# --- Task 2: (model, config, category) rows -----------------------------------


def test_read_leaf_includes_config(tmp_path):
    import json

    import aggregate_unified

    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "summary.json").write_text(
        json.dumps(
            {
                "model": "openai:gpt",
                "category": "autonomous",
                "config": "bare",
                "rollouts_per_task": 3,
                "totals": {"tasks": 1, "passed": 1},
                "incomplete": False,
                "pass@3": 1.0,
                "avg@3": 1.0,
            }
        )
    )
    result = aggregate_unified.read_leaf(leaf, expected_rollouts=3)
    assert result["config"] == "bare"


def _leaf(model, config, category, pass_at_k):
    return {
        "model": model,
        "branch": "current",
        "config": config,
        "category": category,
        "pass_at_k": pass_at_k,
        "avg_at_k": pass_at_k,
        "tasks": 1,
        "passed": int(pass_at_k),
        "incomplete": False,
    }


def test_combine_rows_are_model_config_pairs():
    import aggregate_unified

    leaves = [
        _leaf("openai:gpt", "bare", "autonomous", 1.0),
        _leaf("openai:gpt", "dcode", "autonomous", 0.0),
    ]
    expected_leaves = [
        {
            "model": "openai:gpt",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
        },
        {
            "model": "openai:gpt",
            "branch": "current",
            "config": "dcode",
            "category": "autonomous",
        },
    ]
    combined = aggregate_unified.combine(
        leaves, expected_leaves=expected_leaves, expected_categories=["autonomous"]
    )
    keys = {(r["model"], r["config"]) for r in combined["rows"]}
    assert keys == {("openai:gpt", "bare"), ("openai:gpt", "dcode")}


def test_combine_same_model_configs_do_not_collide():
    import aggregate_unified

    # Two configs, same model+category: must NOT raise a duplicate-leaf error.
    leaves = [
        _leaf("openai:gpt", "bare", "autonomous", 1.0),
        _leaf("openai:gpt", "dcode", "autonomous", 0.5),
    ]
    combined = aggregate_unified.combine(leaves)
    assert len(combined["rows"]) == 2


def test_combine_quarantines_true_triple_duplicate():
    import aggregate_unified

    leaves = [
        _leaf("openai:gpt", "bare", "autonomous", 1.0),
        _leaf("openai:gpt", "bare", "autonomous", 0.0),
    ]
    result = aggregate_unified.combine(leaves)

    assert result["rows"][0]["categories"] == {}
    assert result["issues"][0]["code"] == "duplicate_leaf"


# --- Task 2: (model, branch, config, category) rows ---------------------------


def _bleaf(model, branch, config, category, pass_at_k):
    return {
        "model": model,
        "branch": branch,
        "config": config,
        "category": category,
        "pass_at_k": pass_at_k,
        "avg_at_k": pass_at_k,
        "tasks": 1,
        "passed": int(pass_at_k),
        "incomplete": False,
    }


def test_combined_row_records_expected_source_sha():
    expected = [
        {
            "model": "openai:gpt",
            "branch": "feature",
            "source_sha": "a" * 40,
            "config": "bare",
            "category": "autonomous",
        }
    ]
    combined = au.combine(
        [_bleaf("openai:gpt", "feature", "bare", "autonomous", 1.0)],
        expected_leaves=expected,
        expected_categories=["autonomous"],
    )
    assert combined["rows"][0]["source_sha"] == "a" * 40


def test_combine_rows_split_by_branch():
    import aggregate_unified

    leaves = [
        _bleaf("openai:gpt", "main", "bare", "autonomous", 1.0),
        _bleaf("openai:gpt", "feature", "bare", "autonomous", 0.0),
    ]
    combined = aggregate_unified.combine(leaves)
    keys = {(r["model"], r["branch"], r["config"]) for r in combined["rows"]}
    assert keys == {
        ("openai:gpt", "main", "bare"),
        ("openai:gpt", "feature", "bare"),
    }


def test_combine_same_model_config_different_branch_no_collision():
    import aggregate_unified

    leaves = [
        _bleaf("openai:gpt", "main", "bare", "autonomous", 1.0),
        _bleaf("openai:gpt", "feature", "bare", "autonomous", 0.5),
    ]
    combined = aggregate_unified.combine(leaves)
    assert len(combined["rows"]) == 2


def test_combine_supports_heterogeneous_required_categories():
    import aggregate_unified

    leaves = [
        _bleaf("openai:gpt", "current", "bare", "autonomous", 1.0),
        _bleaf("openai:gpt", "current", "tau3", "conversation", 0.5),
    ]
    expected_leaves = [
        {
            "model": "openai:gpt",
            "branch": "current",
            "config": "bare",
            "category": "autonomous",
        },
        {
            "model": "openai:gpt",
            "branch": "current",
            "config": "tau3",
            "category": "conversation",
        },
    ]

    combined = aggregate_unified.combine(
        leaves,
        expected_leaves=expected_leaves,
        expected_categories=["autonomous", "conversation"],
    )

    rows = {
        (row["model"], row["branch"], row["config"]): row for row in combined["rows"]
    }
    assert rows[("openai:gpt", "current", "bare")]["missing_categories"] == []
    assert rows[("openai:gpt", "current", "tau3")]["missing_categories"] == []


# ---------------------------------------------------------------------------
# Token usage and cost
# ---------------------------------------------------------------------------


def _leaf_exp(model, config, category, pass_at_k, experiment):
    leaf = _leaf(model, config, category, pass_at_k)
    leaf["langsmith_experiment"] = experiment
    return leaf


def _usage_block(
    *,
    status,
    expected,
    observed,
    completed,
    errored,
    prompt,
    completion,
    cost,
    c_prompt,
    c_completion,
    c_cost,
):
    return {
        "status": status,
        "coverage": {
            "expected_rollouts": expected,
            "observed_rollouts": observed,
            "token_rollouts": observed,
            "priced_rollouts": observed,
            "completed_rollouts": completed,
            "errored_rollouts": errored,
        },
        "totals": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": (None if prompt is None else prompt + completion),
            "cost_usd": cost,
        },
        "completed_totals": {
            "prompt_tokens": c_prompt,
            "completion_tokens": c_completion,
            "total_tokens": (None if c_prompt is None else c_prompt + c_completion),
            "cost_usd": c_cost,
        },
    }


def test_read_leaf_exposes_langsmith_experiment(tmp_path: Path):
    d = tmp_path / "leaf"
    d.mkdir()
    summary = _summary("m", 2, 0.5, 0.5, 2, 1)
    summary["langsmith_experiment"] = "exp-xyz"
    (d / "summary.json").write_text(json.dumps(summary))
    assert au.read_leaf(d)["langsmith_experiment"] == "exp-xyz"


def test_read_leaf_defaults_missing_experiment_to_none(tmp_path: Path):
    d = tmp_path / "leaf"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_summary("m", 2, 0.5, 0.5, 2, 1)))
    assert au.read_leaf(d)["langsmith_experiment"] is None


def test_combine_rolls_usage_up_across_a_rows_experiments():
    leaves = [
        _leaf_exp("m", "bare", "autonomous", 0.5, "exp-A"),
        _leaf_exp("m", "bare", "context", 0.5, "exp-B"),
    ]
    experiments = {
        "exp-A": _usage_block(
            status="complete",
            expected=4,
            observed=4,
            completed=3,
            errored=1,
            prompt=1000,
            completion=500,
            cost=0.5,
            c_prompt=700,
            c_completion=350,
            c_cost=0.35,
        ),
        "exp-B": _usage_block(
            status="partial",
            expected=4,
            observed=3,
            completed=3,
            errored=0,
            prompt=200,
            completion=100,
            cost=0.1,
            c_prompt=200,
            c_completion=100,
            c_cost=0.1,
        ),
    }
    (row,) = au.combine(leaves, experiments=experiments)["rows"]
    usage = row["usage"]
    assert usage["experiments"] == ["exp-A", "exp-B"]
    # Completed-only totals sum across both experiments.
    assert usage["completed_totals"]["prompt_tokens"] == 900
    assert usage["completed_totals"]["completion_tokens"] == 450
    assert usage["completed_totals"]["cost_usd"] == pytest.approx(0.45)
    # True spend keeps the errored rollout's tokens/cost.
    assert usage["totals"]["cost_usd"] == pytest.approx(0.6)
    assert usage["coverage"]["expected_rollouts"] == 8
    assert usage["coverage"]["completed_rollouts"] == 6
    assert usage["coverage"]["errored_rollouts"] == 1
    # Status is the worst across the row's experiments.
    assert usage["status"] == "partial"


def test_combine_without_experiments_omits_usage():
    leaves = [_leaf("m", "bare", "autonomous", 1.0)]
    combined = au.combine(leaves)
    assert combined["usage_available"] is False
    assert "usage" not in combined["rows"][0]


def test_combine_usage_empty_when_leaf_has_no_experiment():
    leaves = [_leaf("m", "bare", "autonomous", 1.0)]  # no langsmith_experiment
    (row,) = au.combine(leaves, experiments={})["rows"]
    assert row["usage"]["status"] == "unavailable"
    assert row["usage"]["experiments"] == []
    assert row["usage"]["completed_totals"]["cost_usd"] is None


def test_render_usage_markdown_formats_completed_dashes_and_status():
    leaves = [
        _leaf_exp("m", "bare", "autonomous", 1.0, "exp-A"),
        _leaf_exp("m", "dcode", "autonomous", 0.0, "exp-missing"),
    ]
    experiments = {
        "exp-A": _usage_block(
            status="complete",
            expected=2,
            observed=2,
            completed=2,
            errored=0,
            prompt=1500,
            completion=250,
            cost=1.25,
            c_prompt=1500,
            c_completion=250,
            c_cost=1.25,
        ),
    }
    combined = au.combine(leaves, experiments=experiments)
    md = au.render_usage_markdown(combined)
    assert "## Token usage and cost" not in md  # heading is written by write_outputs
    assert "| Model / branch / config | Completed |" in md
    # Present experiment: thousands separator + 6-decimal cost.
    assert "| 1,500 | 250 | 1.250000 | 1.250000 | complete |" in md
    assert "2/2 (0 err)" in md
    # Missing experiment: em dashes and unavailable status.
    assert "| — | — | — | — | unavailable |" in md


def test_write_outputs_appends_usage_table_only_when_available(tmp_path: Path):
    leaves = [_leaf_exp("m", "bare", "autonomous", 1.0, "exp-A")]
    experiments = {
        "exp-A": _usage_block(
            status="complete",
            expected=1,
            observed=1,
            completed=1,
            errored=0,
            prompt=10,
            completion=5,
            cost=0.02,
            c_prompt=10,
            c_completion=5,
            c_cost=0.02,
        ),
    }
    step = tmp_path / "step.md"
    step.touch()
    combined = au.combine(leaves, experiments=experiments)
    au.write_outputs(combined, 1, tmp_path / "out", str(step))
    rendered = step.read_text()
    assert "## Unified evals — cross-model comparison" in rendered
    assert "## Token usage and cost" in rendered

    # Without experiments, the usage section must be absent.
    step2 = tmp_path / "step2.md"
    step2.touch()
    combined_no_usage = au.combine(leaves)
    au.write_outputs(combined_no_usage, 1, tmp_path / "out2", str(step2))
    assert "## Token usage and cost" not in step2.read_text()


def test_load_usage_reads_experiments_map(tmp_path: Path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"schema_version": 1, "experiments": {"exp-A": {}}}))
    assert au._load_usage(p) == {"exp-A": {}}


def test_load_usage_rejects_bad_schema(tmp_path: Path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"schema_version": 2, "experiments": {}}))
    with pytest.raises(SystemExit):
        au._load_usage(p)


def test_main_with_usage_json_writes_usage_table(tmp_path: Path, monkeypatch):
    root = tmp_path / "_leaves" / "leaf"
    root.mkdir(parents=True)
    summary = _summary("m", 2, 0.5, 0.5, 2, 1)
    summary["langsmith_experiment"] = "exp-A"
    (root / "summary.json").write_text(json.dumps(summary))
    usage = {
        "schema_version": 1,
        "experiments": {
            "exp-A": _usage_block(
                status="complete",
                expected=4,
                observed=4,
                completed=4,
                errored=0,
                prompt=100,
                completion=50,
                cost=0.3,
                c_prompt=100,
                c_completion=50,
                c_cost=0.3,
            )
        },
    }
    up = tmp_path / "u.json"
    up.write_text(json.dumps(usage))
    step = tmp_path / "step.md"
    step.touch()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    out = tmp_path / "_combined"
    rc = au.main(
        [str(tmp_path / "_leaves"), "--rollouts", "2", "--out-dir", str(out), "--usage-json", str(up)]
    )
    assert rc == 0
    combined = json.loads((out / "unified_summary.json").read_text())
    assert combined["usage_available"] is True
    assert combined["rows"][0]["usage"]["completed_totals"]["cost_usd"] == 0.3
    assert "## Token usage and cost" in step.read_text()
