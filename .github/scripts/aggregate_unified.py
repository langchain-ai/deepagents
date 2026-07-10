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
