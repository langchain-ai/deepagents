"""Combine per-(model x category) Harbor summary.json files into a cross-model
comparison (macro + micro overalls), a leaderboard, combined JSON, and radar input.

Each leaf directory contains:
  summary.json  -- written by aggregate_shards.py (dynamic pass@{K}/avg@{K} keys)
  category.txt  -- the category name for this leaf (autonomous|conversation|context)
"""
from __future__ import annotations

import json
from pathlib import Path


def read_leaf(leaf_dir: Path) -> dict:
    summary = json.loads((leaf_dir / "summary.json").read_text())
    category = (leaf_dir / "category.txt").read_text().strip()
    k = summary["rollouts_per_task"]
    totals = summary.get("totals", {})
    return {
        "model": summary.get("model"),
        "category": category,
        "pass_at_k": summary.get(f"pass@{k}"),
        "avg_at_k": summary.get(f"avg@{k}"),
        "tasks": totals.get("tasks", 0),
        "passed": totals.get("passed", 0),
        "incomplete": bool(summary.get("incomplete", False)),
    }


def _mean(vals: list[float]) -> float | None:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


def combine(leaves: list[dict]) -> dict:
    categories = sorted({leaf["category"] for leaf in leaves})
    models_out: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = {}
    for leaf in leaves:
        by_model.setdefault(leaf["model"], []).append(leaf)

    for model, model_leaves in by_model.items():
        cats = {leaf["category"]: {
            "pass_at_k": leaf["pass_at_k"], "avg_at_k": leaf["avg_at_k"],
            "tasks": leaf["tasks"], "incomplete": leaf["incomplete"],
        } for leaf in model_leaves}
        macro = {
            "pass_at_k": _mean([leaf["pass_at_k"] for leaf in model_leaves]),
            "avg_at_k": _mean([leaf["avg_at_k"] for leaf in model_leaves]),
        }
        total_tasks = sum(leaf["tasks"] for leaf in model_leaves) or 0
        micro_avg = (
            sum((leaf["avg_at_k"] or 0.0) * leaf["tasks"] for leaf in model_leaves) / total_tasks
            if total_tasks else None
        )
        micro_pass = (
            sum((leaf["pass_at_k"] or 0.0) * leaf["tasks"] for leaf in model_leaves) / total_tasks
            if total_tasks else None
        )
        models_out[model] = {
            "categories": cats,
            "macro": macro,
            "micro": {"pass_at_k": micro_pass, "avg_at_k": micro_avg},
            "incomplete": any(leaf["incomplete"] for leaf in model_leaves),
        }
    return {"models": models_out, "categories": categories}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_markdown(combined: dict, k: int) -> str:
    cats = combined["categories"]
    header = ["Model"] + [f"{c} pass@{k}/avg@{k}" for c in cats] + [
        f"Overall macro pass@{k}", f"macro avg@{k}", f"micro pass@{k}", f"micro avg@{k}"]
    rows = []
    ranked = sorted(
        combined["models"].items(),
        key=lambda kv: (kv[1]["macro"]["pass_at_k"] is None, -(kv[1]["macro"]["pass_at_k"] or 0.0)),
    )
    for model, m in ranked:
        cells = [model + (" ⚠️" if m["incomplete"] else "")]
        for c in cats:
            cat = m["categories"].get(c)
            cells.append(f"{_fmt(cat['pass_at_k'])}/{_fmt(cat['avg_at_k'])}" if cat else "—")
        cells += [_fmt(m["macro"]["pass_at_k"]), _fmt(m["macro"]["avg_at_k"]),
                  _fmt(m["micro"]["pass_at_k"]), _fmt(m["micro"]["avg_at_k"])]
        rows.append(cells)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def radar_results(combined: dict) -> list[dict]:
    out = []
    for model, m in combined["models"].items():
        scores = {c: v["pass_at_k"] for c, v in m["categories"].items() if v.get("pass_at_k") is not None}
        out.append({"model": model, "scores": scores})
    return out


def write_outputs(combined: dict, k: int, out_dir: Path, step_summary_path: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unified_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    (out_dir / "radar_results.json").write_text(json.dumps(radar_results(combined), indent=2) + "\n")
    md = render_markdown(combined, k)
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write("## Unified evals — cross-model comparison\n\n")
            f.write(md)
