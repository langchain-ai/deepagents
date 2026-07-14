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


def test_clamp_shard_parallel_enforces_both_invariants():
    # per-model: conc*sp<=40 ; global: P*sp<=64
    assert (
        up.clamp_shard_parallel(10, num_providers=1, concurrency=4) == 10
    )  # 4*10=40, 1*10=64-ok
    assert (
        up.clamp_shard_parallel(10, num_providers=7, concurrency=4) == 9
    )  # floor(64/7)=9
    assert (
        up.clamp_shard_parallel(10, num_providers=2, concurrency=8) == 5
    )  # floor(40/8)=5
    assert (
        up.clamp_shard_parallel(10, num_providers=100, concurrency=4) == 1
    )  # floor(64/100)=0 -> 1


def test_build_provider_matrices_cross_products_models_and_categories():
    models = ["anthropic:opus", "openai:gpt", "anthropic:sonnet"]
    cats = ["autonomous", "context"]
    mats = up.build_provider_matrices(
        models,
        cats,
        shard_parallel=10,
        n_shards_by_cat={"autonomous": 10, "context": 3},
    )
    # bucketed by provider prefix
    assert set(mats) == {"anthropic", "openai"}
    # 2 anthropic models x 2 categories = 4 entries
    assert len(mats["anthropic"]) == 4
    entry = next(
        e
        for e in mats["anthropic"]
        if e["model"] == "anthropic:opus" and e["category"] == "context"
    )
    assert entry["dataset_path"] == "datasets/context-retrieval-evals"
    assert entry["agent_impl"] == "dcode"
    # Shared dataset: empty so the leaf derives the canonical per-category name.
    assert entry["langsmith_dataset"] == ""
    assert entry["n_shards"] == 3


def test_dcode_impl_override_swaps_code_categories_only():
    # `dcode_impl` replaces the harness for the code categories (autonomous,
    # context) but must leave the conversation category on tau3.
    mats = up.build_provider_matrices(
        ["openai:gpt"],
        ["autonomous", "conversation", "context"],
        shard_parallel=10,
        n_shards_by_cat={"autonomous": 10, "conversation": 3, "context": 3},
        dcode_impl="bare",
    )
    impls = {e["category"]: e["agent_impl"] for e in mats["openai"]}
    assert impls == {
        "autonomous": "bare",
        "conversation": "tau3",
        "context": "bare",
    }


def test_dcode_impl_defaults_to_dcode():
    mats = up.build_provider_matrices(
        ["openai:gpt"],
        ["autonomous", "context"],
        shard_parallel=10,
        n_shards_by_cat={"autonomous": 10, "context": 3},
    )
    assert {e["agent_impl"] for e in mats["openai"]} == {"dcode"}


def test_main_rejects_invalid_agent_impl(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.setenv("UNIFIED_AGENT_IMPL", "deepagent")
    with pytest.raises(SystemExit, match=r"UNIFIED_AGENT_IMPL must be one of"):
        up.main()


def test_full_profile_has_empty_include_tasks():
    mats = up.build_provider_matrices(
        ["openai:gpt"], ["autonomous"], shard_parallel=10, n_shards_by_cat={"autonomous": 10}
    )
    assert mats["openai"][0]["include_tasks"] == ""


def test_lite_profile_sets_include_tasks_and_right_sizes_shards():
    import lite_tasks

    defaults = {"autonomous": 10, "conversation": 3, "context": 3}
    mats = up.build_provider_matrices(
        ["openai:gpt"],
        ["autonomous", "conversation", "context"],
        shard_parallel=10,
        n_shards_by_cat=defaults,
        dcode_impl="bare",
        profile="lite",
    )
    by_cat = {e["category"]: e for e in mats["openai"]}
    for cat in ("autonomous", "conversation", "context"):
        n = len(lite_tasks.LITE_TASKS[cat])
        assert by_cat[cat]["include_tasks"] == " ".join(lite_tasks.LITE_TASKS[cat])
        assert by_cat[cat]["n_shards"] == min(defaults[cat], max(1, (n + 4) // 5))


def test_main_rejects_invalid_profile(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("UNIFIED_MODELS", "openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.setenv("UNIFIED_PROFILE", "medium")
    with pytest.raises(SystemExit, match=r"UNIFIED_PROFILE must be one of"):
        up.main()


def test_langsmith_dataset_is_empty_for_shared_dataset():
    # No per-model dataset name: the leaf derives the canonical shared dataset,
    # so every entry's langsmith_dataset is empty regardless of the model spec.
    mats = up.build_provider_matrices(
        ["openrouter:foo/bar", "openrouter:foo-bar"],
        ["context"],
        shard_parallel=10,
        n_shards_by_cat={"context": 3},
    )
    assert all(e["langsmith_dataset"] == "" for e in mats["openrouter"])


def test_main_dedupes_repeated_categories(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context,context,context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    assert up.main([]) == 0
    import json as _j

    lines = dict(
        line.split("=", 1) for line in (tmp_path / "o").read_text().splitlines()
    )
    anth = _j.loads(lines["anthropic_matrix"])
    assert len(anth["include"]) == 1  # one entry, not three


def test_main_emits_per_provider_outputs(tmp_path, monkeypatch):
    out = tmp_path / "gh_out"
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus, openai:gpt")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "autonomous,context")
    monkeypatch.setenv("UNIFIED_CONCURRENCY", "4")
    monkeypatch.setenv("UNIFIED_SHARD_PARALLEL", "10")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    rc = up.main([])
    assert rc == 0
    text = out.read_text()
    lines = dict(line.split("=", 1) for line in text.splitlines())
    assert lines["anthropic_has_models"] == "true"
    assert lines["openai_has_models"] == "true"
    assert lines["fireworks_has_models"] == "false"
    assert lines["effective_shard_parallel"] == "10"
    import json as _j

    anth = _j.loads(lines["anthropic_matrix"])
    assert len(anth["include"]) == 2  # 1 model x 2 categories
    assert "providers" not in lines


def test_main_rejects_invalid_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    for raw in ("not-an-integer", "", "1.5", "-1", "0", "41"):
        monkeypatch.setenv("UNIFIED_CONCURRENCY", raw)
        with pytest.raises(SystemExit, match=r"UNIFIED_CONCURRENCY.*1\.\.40"):
            up.main([])


def test_main_rejects_invalid_shard_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    for raw in ("not-an-integer", "", "1.5", "-1", "0"):
        monkeypatch.setenv("UNIFIED_SHARD_PARALLEL", raw)
        with pytest.raises(SystemExit, match=r"UNIFIED_SHARD_PARALLEL.*>= 1"):
            up.main([])


def test_main_accepts_large_shard_parallel_and_clamps_to_resource_limits(
    tmp_path, monkeypatch
):
    cases = [
        ("anthropic:a", "8", "5"),
        (
            "anthropic:a,baseten:b,fireworks:c,google_genai:d,groq:e,"
            "nvidia:f,openai:g,xai:h",
            "1",
            "8",
        ),
    ]
    for index, (models, concurrency, expected) in enumerate(cases):
        out = tmp_path / f"o{index}"
        monkeypatch.setenv("UNIFIED_MODELS", models)
        monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
        monkeypatch.setenv("UNIFIED_CONCURRENCY", concurrency)
        monkeypatch.setenv("UNIFIED_SHARD_PARALLEL", "1000")
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))

        assert up.main([]) == 0
        lines = dict(line.split("=", 1) for line in out.read_text().splitlines())
        assert lines["effective_shard_parallel"] == expected


def test_main_rejects_invalid_category_shard_counts(tmp_path, monkeypatch):
    variables = (
        "UNIFIED_N_SHARDS_AUTONOMOUS",
        "UNIFIED_N_SHARDS_CONVERSATION",
        "UNIFIED_N_SHARDS_CONTEXT",
    )
    invalid = (
        "not-an-integer",
        "",
        "1.5",
        "-1",
        "0",
        str(shard_matrix.MAX_SHARDS + 1),
    )
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    import pytest

    for variable in variables:
        for raw in invalid:
            for current in variables:
                monkeypatch.setenv(current, "1")
            monkeypatch.setenv(variable, raw)
            with pytest.raises(
                SystemExit,
                match=rf"{variable}.*1\.\.{shard_matrix.MAX_SHARDS}",
            ):
                up.main([])


def test_main_accepts_category_shard_count_boundaries(tmp_path, monkeypatch):
    variables = {
        "autonomous": "UNIFIED_N_SHARDS_AUTONOMOUS",
        "conversation": "UNIFIED_N_SHARDS_CONVERSATION",
        "context": "UNIFIED_N_SHARDS_CONTEXT",
    }
    monkeypatch.setenv("UNIFIED_MODELS", "anthropic:opus")
    import json as _j

    for category, variable in variables.items():
        for boundary in (1, shard_matrix.MAX_SHARDS):
            out = tmp_path / f"{category}-{boundary}"
            for current in variables.values():
                monkeypatch.setenv(current, "1")
            monkeypatch.setenv(variable, str(boundary))
            monkeypatch.setenv("UNIFIED_CATEGORIES", category)
            monkeypatch.setenv("GITHUB_OUTPUT", str(out))

            assert up.main([]) == 0
            lines = dict(line.split("=", 1) for line in out.read_text().splitlines())
            matrix = _j.loads(lines["anthropic_matrix"])
            assert matrix["include"][0]["n_shards"] == boundary


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
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    assert up.main([]) == 0
    import json as _j

    lines = dict(
        line.split("=", 1) for line in (tmp_path / "o").read_text().splitlines()
    )
    assert _j.loads(lines["models"]) == ["anthropic:opus", "openai:gpt"]
    assert _j.loads(lines["categories"]) == ["autonomous", "context"]


def test_main_routes_unknown_provider_into_other_bucket(tmp_path, monkeypatch):
    # A spec whose prefix isn't a KNOWN_PROVIDER still resolves (the resolver
    # doesn't allowlist providers), and must land in the "other" matrix rather
    # than being silently dropped from every provider bucket.
    out = tmp_path / "o"
    monkeypatch.setenv("UNIFIED_MODELS", "weirdvendor:x")
    monkeypatch.setenv("UNIFIED_CATEGORIES", "context")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert up.main([]) == 0
    import json as _j

    lines = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert lines["other_has_models"] == "true"
    assert lines["anthropic_has_models"] == "false"
    other = _j.loads(lines["other_matrix"])
    assert [e["model"] for e in other["include"]] == ["weirdvendor:x"]
    assert other["include"][0]["provider"] == "other"
