import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_unified as au  # noqa: E402


def _summary(model, k, pass_k, avg_k, tasks, passed, incomplete=False):
    return {
        "model": model, "dataset": "d", "rollouts_per_task": k,
        "incomplete": incomplete,
        "totals": {"tasks": tasks, "trials": tasks * k, "expected_trials": tasks * k,
                   "passed": passed, "errored": 0},
        f"pass@{k}": pass_k, f"avg@{k}": avg_k,
    }


def test_read_leaf_extracts_dynamic_keys(tmp_path):
    d = tmp_path / "harbor-anthropic-opus-context"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_summary("anthropic:opus", 3, 0.8, 0.5, 30, 45)))
    (d / "category.txt").write_text("context")
    leaf = au.read_leaf(d)
    assert leaf["model"] == "anthropic:opus"
    assert leaf["category"] == "context"
    assert leaf["pass_at_k"] == 0.8
    assert leaf["avg_at_k"] == 0.5


def test_combine_computes_macro_and_micro():
    leaves = [
        {"model": "m", "category": "autonomous", "pass_at_k": 1.0, "avg_at_k": 1.0, "tasks": 10, "passed": 30, "incomplete": False},
        {"model": "m", "category": "context", "pass_at_k": 0.0, "avg_at_k": 0.0, "tasks": 30, "passed": 0, "incomplete": False},
    ]
    out = au.combine(leaves)
    m = out["models"]["m"]
    # macro = mean of category pass@k = (1.0 + 0.0)/2 = 0.5
    assert m["macro"]["pass_at_k"] == 0.5
    # micro pass fraction pooled by tasks passed-any: not derivable from pass_at_k alone,
    # micro uses avg@k weighted by tasks: (1.0*10 + 0.0*30)/40 = 0.25
    assert abs(m["micro"]["avg_at_k"] - 0.25) < 1e-9


def test_render_markdown_sorts_by_macro_desc():
    combined = {
        "categories": ["autonomous", "context"],
        "models": {
            "lo": {"categories": {"autonomous": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                                   "context": {"pass_at_k": 0.1, "avg_at_k": 0.1}},
                    "macro": {"pass_at_k": 0.1, "avg_at_k": 0.1},
                    "micro": {"pass_at_k": 0.1, "avg_at_k": 0.1}, "incomplete": False},
            "hi": {"categories": {"autonomous": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                                   "context": {"pass_at_k": 0.9, "avg_at_k": 0.9}},
                    "macro": {"pass_at_k": 0.9, "avg_at_k": 0.9},
                    "micro": {"pass_at_k": 0.9, "avg_at_k": 0.9}, "incomplete": False},
        },
    }
    md = au.render_markdown(combined, k=3)
    assert md.index("hi") < md.index("lo")  # higher macro ranked first
    assert "pass@3" in md


def test_radar_results_shape():
    combined = {
        "categories": ["autonomous", "context"],
        "models": {"m": {"categories": {"autonomous": {"pass_at_k": 0.5},
                                         "context": {"pass_at_k": 0.7}},
                          "macro": {"pass_at_k": 0.6}, "micro": {"pass_at_k": 0.6},
                          "incomplete": False}},
    }
    rr = au.radar_results(combined)
    assert rr == [{"model": "m", "scores": {"autonomous": 0.5, "context": 0.7}}]


def _leaf_dir(tmp_path, model, category, k=3, pass_k=0.5, avg_k=0.5, tasks=10, passed=15,
              incomplete=False, model_txt=True, model_in_summary=None):
    d = tmp_path / f"harbor-{model}-{category}".replace(":", "-").replace("/", "-")
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(
        _summary(model if model_in_summary is None else model_in_summary,
                 k, pass_k, avg_k, tasks, passed, incomplete)))
    (d / "category.txt").write_text(category)
    if model_txt:
        (d / "model.txt").write_text(model + "\n")
    return d


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


def test_read_leaf_prefers_model_txt_over_null_summary_model(tmp_path):
    # An all-errored leaf writes model: null in summary.json; model.txt is authoritative.
    d = _leaf_dir(tmp_path, "openai:gpt-5.6-luna", "context", model_in_summary=None)
    # force summary model to null explicitly
    s = json.loads((d / "summary.json").read_text())
    s["model"] = None
    (d / "summary.json").write_text(json.dumps(s))
    leaf = au.read_leaf(d)
    assert leaf["model"] == "openai:gpt-5.6-luna"


def test_combine_flags_missing_leaves_against_expected_grid():
    # "b" ran only autonomous; "context" leaf never uploaded.
    leaves = [
        {"model": "a", "category": "autonomous", "pass_at_k": 0.9, "avg_at_k": 0.9, "tasks": 10, "passed": 9, "incomplete": False},
        {"model": "a", "category": "context", "pass_at_k": 0.8, "avg_at_k": 0.8, "tasks": 10, "passed": 8, "incomplete": False},
        {"model": "b", "category": "autonomous", "pass_at_k": 1.0, "avg_at_k": 1.0, "tasks": 10, "passed": 10, "incomplete": False},
    ]
    out = au.combine(leaves, expected_models=["a", "b"], expected_categories=["autonomous", "context"])
    assert out["categories"] == ["autonomous", "context"]
    assert out["models"]["a"]["incomplete"] is False
    assert out["models"]["b"]["incomplete"] is True
    assert out["models"]["b"]["missing_categories"] == ["context"]


def test_combine_includes_expected_model_with_no_leaves():
    out = au.combine([], expected_models=["ghost"], expected_categories=["autonomous"])
    assert "ghost" in out["models"]
    assert out["models"]["ghost"]["incomplete"] is True
    assert out["models"]["ghost"]["missing_categories"] == ["autonomous"]
