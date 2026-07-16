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
    # Packed shards can use all 4 slots: 40//4=10; 80//10=8.
    assert up.derive_pool(concurrency=4, rollouts=3, n_shards=34, n_models=1) == (10, 1)
    # concurrency 1 -> 40 shard jobs; 80//40=2 models
    assert up.derive_pool(concurrency=1, rollouts=3, n_shards=100, n_models=5) == (
        40,
        2,
    )
    # Lower rollouts do not reduce a packed shard's peak concurrency.
    assert up.derive_pool(concurrency=4, rollouts=2, n_shards=100, n_models=1)[0] == 10
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
    up.total_job_guard(n_models=2, shards_per_model=142)  # 284 <= 400, no raise


def test_total_job_guard_rejects_over_budget():
    import pytest

    with pytest.raises(SystemExit, match=r"worker pool"):
        up.total_job_guard(n_models=3, shards_per_model=142)  # 426 > 400


def test_main_full_multi_model_guard_counts_packed_shards(tmp_path, monkeypatch):
    # The guard must count the packed flat-matrix entries, not the raw task
    # count. 260 tasks pack (via ceil division) to 130 shards/model, so 3 models
    # emit 3 x 130 = 390 jobs, within the 400 budget. Guarding on the raw count
    # (3 x 260 = 780) -- or on the loose min(tasks, MAX_SHARDS)=200 bound
    # (3 x 200 = 600) -- would wrongly reject this valid run.
    import json as _j

    n_tasks = 260
    packed = len(
        up.build_flat_matrix("x:y", ["autonomous"], {"autonomous": ["t"] * n_tasks})
    )
    assert packed == 130  # pins the packing so the budget arithmetic below is real
    assert (
        3 * packed <= up.TOTAL_JOB_BUDGET < 3 * min(n_tasks, up.shard_matrix.MAX_SHARDS)
    )

    tasks_json = tmp_path / "tasks.json"
    tasks_json.write_text(
        _j.dumps({"autonomous": [f"harbor-index/t-{i}" for i in range(n_tasks)]})
    )
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:a,anthropic:b,anthropic:c")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("UNIFIED_PROFILE", "full")
    monkeypatch.setenv("UNIFIED_TASKS_JSON", str(tasks_json))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))

    assert up.main([]) == 0  # SystemExit if the guard over-counts (raw or loose bound)
    lines = dict(
        line.split("=", 1) for line in (tmp_path / "o").read_text().splitlines()
    )
    entries = _j.loads(lines["eval_matrix"])["include"]
    assert len(entries) == 3
    for entry in entries:
        assert len(_j.loads(entry["flat_matrix"])["include"]) == packed


def test_build_flat_matrix_one_entry_per_task_across_categories():
    tasks = {
        "autonomous": ["harbor-index/a", "harbor-index/b"],
        "context": ["cb-cloud-0"],
    }
    entries = up.build_flat_matrix(
        "openai:gpt", ["autonomous", "context"], tasks, code_impl="bare"
    )
    # 2 autonomous + 1 context = 3 single-task shards
    assert len(entries) == 3
    auto = [e for e in entries if e["category"] == "autonomous"]
    assert {e["include_tasks"] for e in auto} == {"harbor-index/a", "harbor-index/b"}
    assert all(e["n_shards"] == 1 and e["shard"] == 0 for e in entries)
    assert all(e["agent_impl"] == "bare" for e in auto)  # code override applies
    ctx = next(e for e in entries if e["category"] == "context")
    assert ctx["dataset_path"] == "datasets/context-retrieval-evals"
    assert ctx["agent_impl"] == "bare"
    assert all(e["langsmith_dataset"] == "" for e in entries)


def test_build_flat_matrix_defaults_to_bare():
    # No override: the code categories fall back to DEFAULT_AGENT_IMPL (bare),
    # while conversation stays pinned to tau3.
    assert up.DEFAULT_AGENT_IMPL == "bare"
    tasks = {
        "autonomous": ["harbor-index/a"],
        "conversation": ["sierra-research/tau3-bench__x"],
    }
    entries = up.build_flat_matrix("openai:gpt", ["autonomous", "conversation"], tasks)
    auto = next(e for e in entries if e["category"] == "autonomous")
    conv = next(e for e in entries if e["category"] == "conversation")
    assert auto["agent_impl"] == "bare"
    assert conv["agent_impl"] == "tau3"


def test_build_flat_matrix_conversation_stays_tau3():
    tasks = {"conversation": ["sierra-research/tau3-bench__x"]}
    entries = up.build_flat_matrix(
        "openai:gpt", ["conversation"], tasks, code_impl="bare"
    )
    assert entries[0]["agent_impl"] == "tau3"


def test_build_flat_matrix_packs_above_cap():
    tasks = {
        "autonomous": [f"harbor-index/t{i}" for i in range(shard_matrix.MAX_SHARDS + 5)]
    }
    entries = up.build_flat_matrix(
        "openai:gpt", ["autonomous"], tasks, code_impl="dcode"
    )
    assert len(entries) <= shard_matrix.MAX_SHARDS
    # every task still present, split across include_tasks strings
    seen = " ".join(e["include_tasks"] for e in entries).split()
    assert seen == tasks["autonomous"]


def test_build_flat_matrix_bounds_per_model_total_across_categories():
    # 120 + 120 + 40 = 280 > MAX_SHARDS (200): the per-model TOTAL must be
    # packed down to the cap via proportional per-category budgets, not just
    # each category independently capped (which would allow up to 3x200).
    tasks = {
        "autonomous": [f"harbor-index/a{i}" for i in range(120)],
        "conversation": [f"sierra-research/tau3-bench__c{i}" for i in range(120)],
        "context": [f"cb-cloud-{i}" for i in range(40)],
    }
    entries = up.build_flat_matrix(
        "openai:gpt",
        ["autonomous", "conversation", "context"],
        tasks,
        code_impl="dcode",
    )
    assert len(entries) <= shard_matrix.MAX_SHARDS
    for cat, expected in tasks.items():
        cat_entries = [e for e in entries if e["category"] == cat]
        seen = " ".join(e["include_tasks"] for e in cat_entries).split()
        assert seen == expected  # every task present exactly once, order preserved


def test_derive_pool_budgets_packed_shards_at_full_concurrency():
    tasks = {"autonomous": [f"harbor-index/a{i}" for i in range(260)]}
    entries = up.build_flat_matrix("openai:gpt", ["autonomous"], tasks)
    assert any(len(entry["include_tasks"].split()) > 1 for entry in entries)

    max_parallel, _ = up.derive_pool(
        concurrency=4, rollouts=3, n_shards=len(entries), n_models=1
    )
    assert max_parallel == 10
    assert max_parallel * 4 == up.MAX_TASKS_PER_MODEL


def test_build_flat_matrix_below_cap_stays_one_task_per_shard():
    # Lite-like: total is well under MAX_SHARDS, so behavior is unchanged —
    # one task per matrix entry, no packing.
    tasks = {
        "autonomous": [f"harbor-index/a{i}" for i in range(15)],
        "conversation": [f"sierra-research/tau3-bench__c{i}" for i in range(11)],
        "context": [f"cb-cloud-{i}" for i in range(8)],
    }
    entries = up.build_flat_matrix(
        "openai:gpt",
        ["autonomous", "conversation", "context"],
        tasks,
        code_impl="dcode",
    )
    assert len(entries) == 15 + 11 + 8
    assert all(len(e["include_tasks"].split()) == 1 for e in entries)


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
    assert lines["max_parallel"] == "10"  # conc4 -> 40//4
    assert lines["model_parallel"] == "2"  # 80//10=8 -> min(8, 2 models)
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
