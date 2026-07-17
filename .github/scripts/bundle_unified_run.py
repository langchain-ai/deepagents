"""Assemble one self-contained Unified Eval run for a branch/model/config."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TypedDict, cast


class CategoryRecord(TypedDict):
    """Location and concrete runtime for one category result."""

    runtime: str
    path: str | None


def _load_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, object], value)


def _leaf_index(root: Path) -> dict[tuple[str, str, str], Path]:
    """Index downloaded aggregate leaves by model, runtime, and category."""
    leaves: dict[tuple[str, str, str], Path] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        summary = _load_object(summary_path, "summary")
        values = (summary.get("model"), summary.get("config"), summary.get("category"))
        if not all(isinstance(value, str) and value for value in values):
            continue
        key = cast(tuple[str, str, str], values)
        if key in leaves:
            raise ValueError(f"duplicate aggregate leaf for {key!r}")
        leaves[key] = summary_path.parent
    return leaves


def build_bundle(
    leaves_root: Path,
    product_manifest: Path,
    out_dir: Path,
    *,
    model: str,
    config: str,
    conversation_runtime: str,
    categories: list[str],
) -> dict[str, object]:
    """Copy selected leaves and immutable product metadata into one directory."""
    product = _load_object(product_manifest, "product manifest")
    required = ("version_id", "source_branch", "source_sha", "packages")
    if not all(field in product for field in required):
        raise ValueError("product manifest is missing source identity fields")

    index = _leaf_index(leaves_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product_manifest, out_dir / "package-manifest.json")
    category_records: dict[str, CategoryRecord] = {}
    missing: list[str] = []
    for category in categories:
        runtime = conversation_runtime if category == "conversation" else config
        source = index.get((model, runtime, category))
        if source is None:
            category_records[category] = {"runtime": runtime, "path": None}
            missing.append(category)
            continue
        destination = out_dir / "categories" / category
        shutil.copytree(source, destination)
        category_records[category] = {
            "runtime": runtime,
            "path": destination.relative_to(out_dir).as_posix(),
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "version_id": product["version_id"],
        "source_branch": product["source_branch"],
        "source_sha": product["source_sha"],
        "model": model,
        "config": config,
        "conversation_runtime": conversation_runtime,
        "categories": category_records,
        "missing_categories": missing,
        "packages": product["packages"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _json_strings(raw: str, label: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON list of strings")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI used by the reusable Harbor workflow's bundle job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaves-root", type=Path, required=True)
    parser.add_argument("--product-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--conversation-runtime", required=True)
    parser.add_argument("--categories-json", required=True)
    args = parser.parse_args(argv)
    build_bundle(
        args.leaves_root,
        args.product_manifest,
        args.out_dir,
        model=args.model,
        config=args.config,
        conversation_runtime=args.conversation_runtime,
        categories=_json_strings(args.categories_json, "--categories-json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
