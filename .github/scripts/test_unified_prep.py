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
    assert up.derive_pool(concurrency=4, rollouts=3, n_shards=34, n_groups=1) == (13, 1)
    # concurrency 1 -> per_shard=1 -> 40 ; 80//40=2
    assert up.derive_pool(concurrency=1, rollouts=3, n_shards=100, n_groups=5) == (40, 2)
    # rollouts < concurrency clamps per_shard to rollouts (the utilization win)
    assert up.derive_pool(concurrency=4, rollouts=2, n_shards=100, n_groups=1)[0] == 20
    # clamp max_parallel to n_shards when few tasks; model_parallel clamps to n_groups
    assert up.derive_pool(concurrency=1, rollouts=3, n_shards=8, n_groups=1) == (8, 1)


def test_main_rejects_invalid_agent_impl(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.setenv("UNIFIED_AGENT_IMPLS", "deepagent")
    with pytest.raises(SystemExit, match=r"UNIFIED_AGENT_IMPLS entries must be in"):
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
    up.total_job_guard(total_jobs=360)  # <= 400, no raise


def test_total_job_guard_allows_at_budget():
    # Boundary is `> TOTAL_JOB_BUDGET`, so exactly at the budget must not raise.
    up.total_job_guard(total_jobs=up.TOTAL_JOB_BUDGET)


def test_total_job_guard_rejects_over_budget():
    import pytest

    with pytest.raises(SystemExit, match=r"TOTAL_JOB_BUDGET"):
        up.total_job_guard(total_jobs=up.TOTAL_JOB_BUDGET + 1)


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
    assert len(eval_matrix) == 2  # one entry per (model, branch); default branch=current
    assert {e["model"] for e in eval_matrix} == {"openai:gpt", "anthropic:opus"}
    assert {e["branch"] for e in eval_matrix} == {"current"}
    for entry in eval_matrix:
        assert set(entry) == {"model", "branch", "slug", "flat_matrix"}
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


def test_main_rejects_unknown_agent_impl(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt-5.6-luna")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_AGENT_IMPLS", "bare,bogus")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    with pytest.raises(SystemExit):
        up.main()


def test_derive_pool_divides_inner_by_branches():
    # per_shard=3 -> M=40//3=13; two branches -> inner=13//2=6; outer bounded by groups.
    inner, outer = up.derive_pool(
        concurrency=4, rollouts=3, n_shards=100, n_groups=2, n_branches=2
    )
    assert inner == 6
    assert outer == 2


def test_derive_pool_single_branch_matches_legacy():
    inner, outer = up.derive_pool(
        concurrency=4, rollouts=3, n_shards=100, n_groups=5, n_branches=1
    )
    assert (inner, outer) == (13, 5)


def test_main_rejects_more_branches_than_budget(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt-5.6-luna")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    monkeypatch.setenv("UNIFIED_CONCURRENCY", "40")  # per_shard=min(40,3)=3 -> M=13
    monkeypatch.setenv("UNIFIED_BRANCHES", ",".join(f"b{i}" for i in range(14)))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    with pytest.raises(SystemExit):
        up.main()


def test_main_emits_model_branch_matrix(tmp_path, monkeypatch):
    import json as _j
    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt-5.6-luna")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_AGENT_IMPLS", "bare,dcode")
    monkeypatch.setenv("UNIFIED_BRANCHES", "main,feature")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    up.main()
    text = out.read_text()
    matrix = _j.loads(
        next(l for l in text.splitlines() if l.startswith("eval_matrix=")).split("=", 1)[1]
    )
    pairs = {(e["model"], e["branch"]) for e in matrix["include"]}
    assert pairs == {("openai:gpt-5.6-luna", "main"), ("openai:gpt-5.6-luna", "feature")}
    leaves = _j.loads(
        next(l for l in text.splitlines() if l.startswith("expected_leaves=")).split("=", 1)[1]
    )
    assert {l["branch"] for l in leaves} == {"main", "feature"}
    assert all({"model", "branch", "config", "category"} <= set(l) for l in leaves)


def test_main_default_branch_is_current(tmp_path, monkeypatch):
    import json as _j
    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt-5.6-luna")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    up.main()
    text = out.read_text()
    matrix = _j.loads(
        next(l for l in text.splitlines() if l.startswith("eval_matrix=")).split("=", 1)[1]
    )
    assert {e["branch"] for e in matrix["include"]} == {"current"}


def test_main_emits_expected_leaves_per_config(tmp_path, monkeypatch):
    import json as _j
    from collections import Counter

    import lite_tasks

    # `autonomous` has >1 lite task, so build_flat_matrix emits one entry per
    # task (many entries sharing the same (model, config, category) triple).
    # main's `seen_leaves` set must collapse each config's many entries to a
    # single leaf; this test fails if that dedup is removed.
    assert len(lite_tasks.LITE_TASKS["autonomous"]) > 1
    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt-5.6-luna")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_AGENT_IMPLS", "bare,dcode")
    monkeypatch.setenv("UNIFIED_PROFILE", "lite")
    out = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    up.main()
    text = out.read_text()
    line = next(ln for ln in text.splitlines() if ln.startswith("expected_leaves="))
    leaves = _j.loads(line.split("=", 1)[1])
    configs = {leaf["config"] for leaf in leaves if leaf["category"] == "autonomous"}
    assert configs == {"bare", "dcode"}
    # No duplicate (model, config, category) triples: dedup collapses the many
    # per-task shard entries to exactly one leaf each.
    triples = [(leaf["model"], leaf["config"], leaf["category"]) for leaf in leaves]
    assert len(triples) == len(set(triples))
    # Each config for the multi-task `autonomous` category appears exactly once
    # (would be len(LITE_TASKS["autonomous"]) per config if dedup were removed).
    autonomous_config_counts = Counter(
        leaf["config"] for leaf in leaves if leaf["category"] == "autonomous"
    )
    assert autonomous_config_counts == Counter({"bare": 1, "dcode": 1})
