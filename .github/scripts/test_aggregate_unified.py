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
