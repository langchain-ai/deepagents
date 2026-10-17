from __future__ import annotations

import experiment_name as en


def test_matches_the_historical_shell_derivation() -> None:
    # A real name produced by the previous _harbor_run.yml bash, so the migration
    # to the shared helper is name-stable.
    assert (
        en.experiment_name(
            model="openai:gpt-5.6-terra",
            branch="main",
            config="bare",
            category="context",
            run_id="30032245537",
            run_attempt="1",
        )
        == "deepagents-harbor-main-0d6e4079-bare-openai-gpt-5.6-terra-context-30032245537-1"
    )


def test_slashes_and_colons_become_dashes() -> None:
    name = en.experiment_name(
        model="openai:gpt-5.6-terra",
        branch="ss/benchmark-unified-evals-no-todos",
        config="bare",
        category="context",
        run_id="42",
        run_attempt="1",
    )
    # branch slug = sanitized branch + sha8 of the RAW branch.
    assert name.startswith(
        "deepagents-harbor-ss-benchmark-unified-evals-no-todos-4e570448-bare-"
        "openai-gpt-5.6-terra-context-"
    )
    assert name.endswith("-42-1")
    assert ":" not in name and "/" not in name


def test_category_is_optional() -> None:
    with_cat = en.experiment_name(
        model="m", branch="b", config="bare", category="conversation", run_id="1", run_attempt="1"
    )
    without_cat = en.experiment_name(
        model="m", branch="b", config="bare", category=None, run_id="1", run_attempt="1"
    )
    empty_cat = en.experiment_name(
        model="m", branch="b", config="bare", category="", run_id="1", run_attempt="1"
    )
    assert "-conversation-" in with_cat
    assert without_cat == empty_cat  # empty and None both omit the category segment


def test_deterministic() -> None:
    kwargs = dict(
        model="m", branch="b", config="bare", category="context", run_id="9", run_attempt="2"
    )
    assert en.experiment_name(**kwargs) == en.experiment_name(**kwargs)


def test_cli_reads_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HARBOR_MODEL", "openai:gpt-5.6-terra")
    monkeypatch.setenv("HARBOR_BRANCH", "main")
    monkeypatch.setenv("HARBOR_AGENT_IMPL", "bare")
    monkeypatch.setenv("HARBOR_CATEGORY", "context")
    monkeypatch.setenv("GITHUB_RUN_ID", "30032245537")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    assert en.main([]) == 0
    out = capsys.readouterr().out.strip()
    assert out == "deepagents-harbor-main-0d6e4079-bare-openai-gpt-5.6-terra-context-30032245537-1"
