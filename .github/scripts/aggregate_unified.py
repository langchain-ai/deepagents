"""Combine per-(model x category) Harbor summary.json files into a cross-model
comparison (macro + micro overalls), a leaderboard, combined JSON, and radar input.

Each leaf directory (one per model x category) contains:
  summary.json  -- written by aggregate_shards.py (dynamic pass@{K}/avg@{K} keys)
  category.txt  -- the category name for this leaf
  model.txt     -- the model spec for this leaf; authoritative, because
                   summary.json's model is null when every trial errored

The combiner is given the expected model x category grid (EXPECTED_MODELS /
EXPECTED_CATEGORIES) so a model whose leaf failed to upload is still shown and
flagged incomplete, rather than silently ranking on fewer categories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_leaf(leaf_dir: Path) -> dict:
    summary = json.loads((leaf_dir / "summary.json").read_text())
    category = (leaf_dir / "category.txt").read_text().strip()
    model_file = leaf_dir / "model.txt"
    model = model_file.read_text().strip() if model_file.exists() else summary.get("model")
    k = summary["rollouts_per_task"]
    totals = summary.get("totals", {})
    return {
        "model": model or "unknown",
        "category": category,
        "pass_at_k": summary.get(f"pass@{k}"),
        "avg_at_k": summary.get(f"avg@{k}"),
        "tasks": totals.get("tasks", 0),
        "passed": totals.get("passed", 0),
        "incomplete": bool(summary.get("incomplete", False)),
    }


def _mean(vals: list[float | None]) -> float | None:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


def combine(
    leaves: list[dict],
    expected_models: list[str] | None = None,
    expected_categories: list[str] | None = None,
) -> dict:
    present_cats = {leaf["category"] for leaf in leaves}
    if expected_categories:
        categories = list(expected_categories)
        categories += sorted(present_cats - set(categories))
    else:
        categories = sorted(present_cats)

    by_model: dict[str, list[dict]] = {}
    for leaf in leaves:
        by_model.setdefault(leaf["model"], []).append(leaf)

    if expected_models:
        models = list(expected_models)
        models += [m for m in by_model if m not in models]
    else:
        models = list(by_model)

    models_out: dict[str, dict] = {}
    for model in models:
        model_leaves = by_model.get(model, [])
        cats = {
            leaf["category"]: {
                "pass_at_k": leaf["pass_at_k"],
                "avg_at_k": leaf["avg_at_k"],
                "tasks": leaf["tasks"],
                "incomplete": leaf["incomplete"],
            }
            for leaf in model_leaves
        }
        missing = [c for c in categories if c not in cats]
        macro = {
            "pass_at_k": _mean([leaf["pass_at_k"] for leaf in model_leaves]),
            "avg_at_k": _mean([leaf["avg_at_k"] for leaf in model_leaves]),
        }
        total_tasks = sum(leaf["tasks"] for leaf in model_leaves) or 0
        micro_pass = (
            sum((leaf["pass_at_k"] or 0.0) * leaf["tasks"] for leaf in model_leaves) / total_tasks
            if total_tasks else None
        )
        micro_avg = (
            sum((leaf["avg_at_k"] or 0.0) * leaf["tasks"] for leaf in model_leaves) / total_tasks
            if total_tasks else None
        )
        models_out[model] = {
            "categories": cats,
            "macro": macro,
            "micro": {"pass_at_k": micro_pass, "avg_at_k": micro_avg},
            "missing_categories": missing,
            "incomplete": bool(missing) or any(leaf["incomplete"] for leaf in model_leaves),
        }
    return {"models": models_out, "categories": categories}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_markdown(combined: dict, k: int) -> str:
    cats = combined["categories"]
    header = ["Model"] + [f"{c} pass@{k}/avg@{k}" for c in cats] + [
        f"Overall macro pass@{k}", f"macro avg@{k}", f"micro pass@{k}", f"micro avg@{k}"]
    ranked = sorted(
        combined["models"].items(),
        key=lambda kv: (kv[1]["macro"]["pass_at_k"] is None, -(kv[1]["macro"]["pass_at_k"] or 0.0)),
    )
    rows = []
    for model, m in ranked:
        cells = [str(model) + (" ⚠️" if m["incomplete"] else "")]
        for c in cats:
            cat = m["categories"].get(c)
            cells.append(f"{_fmt(cat['pass_at_k'])}/{_fmt(cat['avg_at_k'])}" if cat else "—")
        cells += [_fmt(m["macro"]["pass_at_k"]), _fmt(m["macro"]["avg_at_k"]),
                  _fmt(m["micro"]["pass_at_k"]), _fmt(m["micro"]["avg_at_k"])]
        rows.append(cells)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    md = "\n".join(lines) + "\n"

    incompletes = [(mo, mm) for mo, mm in combined["models"].items() if mm["incomplete"]]
    if incompletes:
        md += "\n> ⚠️ **Ranked on partial data** — treat these rows with caution:\n"
        for mo, mm in incompletes:
            miss = mm.get("missing_categories") or []
            note = f"missing categories: {', '.join(miss)}" if miss else "a category reported incomplete data"
            md += f"> - `{mo}` — {note}\n"
    return md


def radar_results(combined: dict) -> list[dict]:
    out = []
    for model, m in combined["models"].items():
        scores = {c: v["pass_at_k"] for c, v in m["categories"].items() if v.get("pass_at_k") is not None}
        out.append({"model": model, "scores": scores})
    return out


def write_outputs(combined: dict, k: int, out_dir: Path, step_summary_path: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unified_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    # Radar needs >= 3 axes to be meaningful. Emit its input only then; the
    # workflow's radar step keys off this file's existence.
    if len(combined["categories"]) >= 3:
        (out_dir / "radar_results.json").write_text(
            json.dumps(radar_results(combined), indent=2) + "\n"
        )
    md = render_markdown(combined, k)
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write("## Unified evals — cross-model comparison\n\n")
            f.write(md)
    # A machine-visible signal so a partially-covered ranking isn't taken at face value.
    for model, m in combined["models"].items():
        if m["incomplete"]:
            miss = ", ".join(m.get("missing_categories") or []) or "a category reported incomplete data"
            print(f"::warning::Model {model} incomplete (missing: {miss}); ranked on partial data.")


def _discover_leaves(root: Path) -> list[dict]:
    leaves = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "summary.json").exists() and (child / "category.txt").exists():
            leaves.append(read_leaf(child))
    return leaves


def _load_list_env(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    value = json.loads(raw)
    return value or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.root

    leaves = _discover_leaves(args.root)
    combined = combine(
        leaves,
        _load_list_env("EXPECTED_MODELS"),
        _load_list_env("EXPECTED_CATEGORIES"),
    )
    write_outputs(combined, args.rollouts, out_dir, os.environ.get("GITHUB_STEP_SUMMARY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
