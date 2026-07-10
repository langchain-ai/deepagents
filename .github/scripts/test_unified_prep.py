import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_prep as up  # noqa: E402


def test_slugify_replaces_colons_and_slashes():
    assert up.slugify("anthropic:claude-opus-4-7") == "anthropic-claude-opus-4-7"
    assert up.slugify("fireworks:accounts/fireworks/models/glm-5p2") == (
        "fireworks-accounts-fireworks-models-glm-5p2"
    )


def test_provider_of_uses_prefix_and_falls_back_to_other():
    known = {"anthropic", "openai"}
    assert up.provider_of("anthropic:claude-opus-4-7", known) == "anthropic"
    assert up.provider_of("weirdvendor:x", known) == "other"


def test_clamp_shard_parallel_enforces_both_invariants():
    # per-model: conc*sp<=40 ; global: P*sp<=64
    assert up.clamp_shard_parallel(10, num_providers=1, concurrency=4) == 10  # 4*10=40, 1*10=64-ok
    assert up.clamp_shard_parallel(10, num_providers=7, concurrency=4) == 9   # floor(64/7)=9
    assert up.clamp_shard_parallel(10, num_providers=2, concurrency=8) == 5   # floor(40/8)=5
    assert up.clamp_shard_parallel(10, num_providers=100, concurrency=4) == 1  # floor(64/100)=0 -> 1


def test_build_provider_matrices_cross_products_models_and_categories():
    models = ["anthropic:opus", "openai:gpt", "anthropic:sonnet"]
    cats = ["autonomous", "context"]
    mats = up.build_provider_matrices(
        models, cats, shard_parallel=10, n_shards_by_cat={"autonomous": 10, "context": 3}
    )
    # bucketed by provider prefix
    assert set(mats) == {"anthropic", "openai"}
    # 2 anthropic models x 2 categories = 4 entries
    assert len(mats["anthropic"]) == 4
    entry = next(e for e in mats["anthropic"] if e["model"] == "anthropic:opus" and e["category"] == "context")
    assert entry["dataset_path"] == "datasets/context-retrieval-evals"
    assert entry["agent_impl"] == "dcode"
    assert entry["langsmith_dataset"] == "context-retrieval-evals__anthropic-opus"
    assert entry["n_shards"] == 3
