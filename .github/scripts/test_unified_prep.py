import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard_matrix  # noqa: E402
import unified_prep as up  # noqa: E402


def test_parse_int_input_enforces_inclusive_range():
    import pytest

    assert up.parse_int_input("UNIFIED_CONCURRENCY", "1", minimum=1, maximum=40) == 1
    assert up.parse_int_input("UNIFIED_CONCURRENCY", "40", minimum=1, maximum=40) == 40
    for raw in ("0", "41", "-1", "", "1.5", "many"):
        with pytest.raises(SystemExit, match=r"UNIFIED_CONCURRENCY.*1\.\.40"):
            up.parse_int_input("UNIFIED_CONCURRENCY", raw, minimum=1, maximum=40)


def test_provider_of_uses_prefix_and_falls_back_to_other():
    known = {"anthropic", "openai"}
    assert up.provider_of("anthropic:claude-opus-4-7", known) == "anthropic"
    assert up.provider_of("weirdvendor:x", known) == "other"


def test_derive_pool_from_concurrency_and_rollouts():
    # concurrency 4, rollouts 3 -> per_shard=3 -> 40//3=13 ; 80//13=6
    assert up.derive_pool(concurrency=4, rollouts=3, n_shards=34, n_models=1) == (13, 1)
    # concurrency 1 -> per_shard=1 -> 40 ; 80//40=2
    assert up.derive_pool(concurrency=1, rollouts=3, n_shards=100, n_models=5) == (40, 2)
    # rollouts < concurrency clamps per_shard to rollouts (the utilization win)
    assert up.derive_pool(concurrency=4, rollouts=2, n_shards=100, n_models=1)[0] == 20
    # clamp max_parallel to n_shards when few tasks; model_parallel clamps to n_models
    assert up.derive_pool(concurrency=1, rollouts=3, n_shards=8, n_models=1) == (8, 1)


def test_main_rejects_invalid_agent_impl(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.setenv("UNIFIED_AGENT_IMPL", "deepagent")
    with pytest.raises(SystemExit, match=r"UNIFIED_AGENT_IMPL must be one of"):
        up.main()


def test_main_rejects_invalid_profile(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.setenv("UNIFIED_PROFILE", "medium")
    with pytest.raises(SystemExit, match=r"UNIFIED_PROFILE must be one of"):
        up.main()


def test_main_dedupes_repeated_categories(tmp_path, monkeypatch):
    import lite_tasks

    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context,context,context")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    assert up.main([]) == 0
    import json as _j

    lines = dict(
        line.split("=", 1) for line in (tmp_path / "o").read_text().splitlines()
    )
    assert _j.loads(lines["categories"]) == ["context"]  # one entry, not three
    # the flat matrix isn't tripled either: one shard per context task
    eval_matrix = _j.loads(lines["eval_matrix"])["include"]
    matrix = _j.loads(eval_matrix[0]["flat_matrix"])["include"]
    assert len(matrix) == len(lite_tasks.LITE_TASKS["context"])


def test_main_rejects_invalid_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    for raw in ("not-an-integer", "", "1.5", "-1", "0", "41"):
        monkeypatch.setenv("UNIFIED_CONCURRENCY", raw)
        with pytest.raises(SystemExit, match=r"UNIFIED_CONCURRENCY.*1\.\.40"):
            up.main([])


def test_main_rejects_bad_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "no-colon-here")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    with pytest.raises(SystemExit):
        up.main([])


def test_main_rejects_empty_categories(tmp_path, monkeypatch):
    # whitespace/comma-only resolves to an empty category list; must not silently
    # skip every job and emit a "successful" empty artifact.
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", " , ")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    with pytest.raises(SystemExit):
        up.main([])


def test_main_emits_expected_models_and_categories(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus, openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous,context")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    assert up.main([]) == 0
    import json as _j

    lines = dict(
        line.split("=", 1) for line in (tmp_path / "o").read_text().splitlines()
    )
    assert _j.loads(lines["models"]) == ["anthropic:opus", "openai:gpt"]
    assert _j.loads(lines["categories"]) == ["autonomous", "context"]


def test_total_job_guard_allows_within_budget():
    up.total_job_guard(n_models=2, est_tasks_per_model=142)  # 284 <= 400, no raise


def test_total_job_guard_rejects_over_budget():
    import pytest
    with pytest.raises(SystemExit, match=r"worker pool"):
        up.total_job_guard(n_models=3, est_tasks_per_model=142)  # 426 > 400


def test_build_flat_matrix_expands_code_categories_over_configs():
    tasks = {"autonomous": ["a1", "a2"], "context": ["c1"]}
    entries = up.build_flat_matrix(
        "openai:gpt", ["autonomous", "context"], tasks, code_impls=["bare", "dcode"]
    )
    autos = [e for e in entries if e["category"] == "autonomous"]
    impls = sorted({e["agent_impl"] for e in autos})
    assert impls == ["bare", "dcode"]
    # Each config gets the full task set for the category.
    assert sum(len(e["include_tasks"].split()) for e in autos if e["agent_impl"] == "bare") == 2
    # The rest of the entry schema is untouched: existing keys keep the
    # category's CATEGORY_MAP values and the fixed 1-task/shard fields.
    cm = up.CATEGORY_MAP["autonomous"]
    entry = autos[0]
    assert entry["dataset"] == cm["dataset"]
    assert entry["dataset_path"] == cm["dataset_path"]
    assert entry["langsmith_dataset"] == ""
    assert entry["n_shards"] == 1
    assert entry["shard"] == 0


def test_build_flat_matrix_conversation_not_multiplied_by_configs():
    tasks = {"autonomous": ["a1"], "conversation": ["t1", "t2"]}
    entries = up.build_flat_matrix(
        "openai:gpt", ["autonomous", "conversation"], tasks, code_impls=["bare", "dcode"]
    )
    conv = [e for e in entries if e["category"] == "conversation"]
    assert {e["agent_impl"] for e in conv} == {"tau3"}
    assert len(conv) == 2  # two tasks, one config, one task per shard


def test_build_flat_matrix_defaults_to_bare_single_config():
    tasks = {"autonomous": ["a1"], "conversation": ["t1"]}
    entries = up.build_flat_matrix("openai:gpt", ["autonomous", "conversation"], tasks)
    auto = next(e for e in entries if e["category"] == "autonomous")
    conv = next(e for e in entries if e["category"] == "conversation")
    assert auto["agent_impl"] == "bare"
    assert conv["agent_impl"] == "tau3"


def test_build_flat_matrix_caps_entries_at_max_shards():
    # Two code categories x two configs x 60 tasks = 240 groups pre-pack, over
    # MAX_SHARDS; packing must keep the emitted entry count within the cap while
    # preserving every task exactly once per (category, config) group.
    tasks = {
        "autonomous": [f"a{i}" for i in range(60)],
        "context": [f"c{i}" for i in range(60)],
    }
    entries = up.build_flat_matrix(
        "openai:gpt",
        ["autonomous", "context"],
        tasks,
        code_impls=["bare", "dcode"],
    )
    assert len(entries) <= shard_matrix.MAX_SHARDS
    # Task fidelity: for each (category, config), the union of the group's
    # packed include_tasks equals the original task list exactly (order
    # preserved, every task present once, no drops or duplication).
    for cat in ("autonomous", "context"):
        for impl in ("bare", "dcode"):
            seen = [
                t
                for e in entries
                if e["category"] == cat and e["agent_impl"] == impl
                for t in e["include_tasks"].split()
            ]
            assert seen == tasks[cat]


def test_build_flat_matrix_dedupes_duplicate_configs():
    # A repeated config collapses to a single config: ["bare", "bare"] yields
    # the same entries as ["bare"] (guards the cap invariant, see Fix 1).
    tasks = {"autonomous": ["a1", "a2"], "context": ["c1"]}
    once = up.build_flat_matrix(
        "openai:gpt", ["autonomous", "context"], tasks, code_impls=["bare"]
    )
    twice = up.build_flat_matrix(
        "openai:gpt", ["autonomous", "context"], tasks, code_impls=["bare", "bare"]
    )
    assert twice == once


def test_main_emits_per_model_flat_matrix_lite(tmp_path, monkeypatch):
    import json as _j
    out = tmp_path / "o"
    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt, anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous,conversation,context")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("UNIFIED_CONCURRENCY", "4")
    monkeypatch.setenv("UNIFIED_ROLLOUTS", "3")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert up.main([]) == 0
    lines = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert lines["max_parallel"] == "13"      # conc4,roll3 -> 40//3
    assert lines["model_parallel"] == "2"     # 80//13=6 -> min(6, 2 models)
    eval_matrix = _j.loads(lines["eval_matrix"])["include"]
    assert len(eval_matrix) == 2  # one entry per model
    assert {e["model"] for e in eval_matrix} == {"openai:gpt", "anthropic:opus"}
    for entry in eval_matrix:
        assert set(entry) == {"model", "slug", "flat_matrix"}
        flat = _j.loads(entry["flat_matrix"])["include"]
        # lite totals 15+11+8 = 34 single-task shards per model
        assert len(flat) == 34
        assert {e["category"] for e in flat} == {
            "autonomous",
            "conversation",
            "context",
        }
    # No other output carries per-model or per-provider data; eval_matrix is
    # the single source for both.
    assert "model_slugs" not in lines
    assert "model_0_matrix" not in lines
    assert "openai_matrix" not in lines
