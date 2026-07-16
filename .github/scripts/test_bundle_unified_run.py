import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle_unified_run as bundle  # noqa: E402


def _leaf(root: Path, model: str, config: str, category: str) -> None:
    leaf = root / f"{config}-{category}"
    leaf.mkdir(parents=True)
    (leaf / "summary.json").write_text(
        json.dumps({"model": model, "config": config, "category": category})
    )
    (leaf / "per_task.jsonl").write_text('{"task":"one","pass@2":1.0}\n')
    (leaf / "harbor-shard.tar.zst").write_bytes(b"raw")


def test_bundle_maps_shared_conversation_and_preserves_raw_archives(
    tmp_path: Path,
) -> None:
    leaves = tmp_path / "leaves"
    _leaf(leaves, "openai:gpt", "bare", "autonomous")
    _leaf(leaves, "openai:gpt", "tau3", "conversation")
    product = tmp_path / "package-manifest.json"
    product.write_text(
        json.dumps(
            {
                "version_id": "v2",
                "source_branch": "feature/todos",
                "source_sha": "a" * 40,
                "packages": [{"distribution": "deepagents"}],
            }
        )
    )

    manifest = bundle.build_bundle(
        leaves,
        product,
        tmp_path / "run",
        model="openai:gpt",
        config="bare",
        conversation_runtime="tau3",
        categories=["autonomous", "conversation", "context"],
    )

    assert manifest["missing_categories"] == ["context"]
    assert manifest["categories"]["conversation"] == {
        "runtime": "tau3",
        "path": "categories/conversation",
    }
    assert (tmp_path / "run/categories/autonomous/harbor-shard.tar.zst").is_file()
    assert json.loads((tmp_path / "run/manifest.json").read_text())["source_sha"] == (
        "a" * 40
    )


def test_leaf_index_rejects_duplicate_identity(tmp_path: Path) -> None:
    import pytest

    _leaf(tmp_path / "a", "model", "bare", "context")
    _leaf(tmp_path / "b", "model", "bare", "context")
    with pytest.raises(ValueError, match="duplicate aggregate leaf"):
        bundle._leaf_index(tmp_path)
