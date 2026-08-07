"""Tests for the DRBench verifier templates (`judge.py`, `extract_text.py`).

The templates run inside the Harbor sandbox with no `deepagents_evals` on the path, so
they are loaded here by file path rather than imported as package modules.

`judge.py` now delegates all scoring to upstream `drbench`, which is installed only in the
verifier image. It imports `drbench` lazily inside functions precisely so it stays
loadable (and testable) without it, and the tests below inject fake `drbench` modules to
exercise the wiring: which metrics are requested, how the result is combined, and what
happens when the report or a score is missing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_TEMPLATES = Path(__file__).resolve().parents[2] / "harbor_adapters" / "drbench" / "templates"


def _load(name: str) -> ModuleType:
    """Load a template module by path, the way the sandbox runs it."""
    spec = importlib.util.spec_from_file_location(f"drbench_{name}", _TEMPLATES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def judge() -> ModuleType:
    return _load("judge")


@pytest.fixture(scope="module")
def extract_text() -> ModuleType:
    return _load("extract_text")


class _FakeTask:
    """Stands in for `drbench.task_loader.Task`; only identity matters here."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


def _install_fake_drbench(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scores: dict[str, float] | object,
    openai_models: list[str] | None = None,
    service_to_models: dict[str, list[str]] | None = None,
    calls: dict[str, Any] | None = None,
) -> None:
    """Inject the `drbench` modules `judge.py` imports lazily.

    Mirrors the real import surface: `drbench.task_loader.get_task_from_id`,
    `drbench.score_report.score_report`, and `drbench.agents.utils.OPENAI_MODELS`.
    """
    recorded = calls if calls is not None else {}

    # Annotated `Any` because these are synthesized modules whose attributes are added
    # here; a `ModuleType` has no such members to check against.
    task_loader: Any = types.ModuleType("drbench.task_loader")

    def get_task_from_id(task_id: str) -> _FakeTask:
        recorded["task_id"] = task_id
        return _FakeTask(task_id)

    task_loader.get_task_from_id = get_task_from_id

    score_report_mod: Any = types.ModuleType("drbench.score_report")
    metrics_mod: Any = types.ModuleType("drbench.metrics")

    class _Metric:
        def __init__(self, name: str) -> None:
            self.name = name

        def compute(self, **_kwargs: Any) -> dict[str, Any]:
            return _verdict_rows()

    def get_metric(name: str, **_kwargs: Any) -> _Metric:
        return _Metric(name)

    # Upstream binds this into `score_report`'s namespace with `from drbench.metrics import
    # get_metric`, so both names exist and only the `score_report` one is what gets called.
    metrics_mod.get_metric = get_metric
    score_report_mod.get_metric = get_metric

    def score_report(**kwargs: Any) -> object:
        recorded["score_report_kwargs"] = kwargs
        # Mirrors upstream's loop: every metric is resolved through the name in this
        # module's own namespace and computed, which is what the detail capture wraps.
        for name in kwargs.get("metrics") or []:
            score_report_mod.get_metric(name).compute()
        return scores

    score_report_mod.score_report = score_report

    utils: Any = types.ModuleType("drbench.agents.utils")

    def get_embeddings(texts: list[str], *_a: Any, **_kw: Any) -> Any:  # noqa: ANN401  # stub for an untyped upstream function
        # Mirrors upstream exactly: a C-contiguous float32 ndarray. The caller reads
        # `.shape[1]` and passes it to `faiss.normalize_L2`, which mutates in place, so a
        # stub returning plain lists would hide a real type regression.
        recorded.setdefault("embed_batches", []).append(len(texts))
        rows = np.array([[float(len(text)), 0.5] for text in texts])
        return np.ascontiguousarray(rows, dtype=np.float32)

    utils.get_embeddings = get_embeddings
    utils.OPENAI_MODELS = (
        ["gpt-4o", "gpt-4o-mini", "gpt-4.1"] if openai_models is None else openai_models
    )

    # Upstream's second, independent registry. It disagrees with the first -- `gpt-4.1` is
    # in `OPENAI_MODELS` but not here -- which is why the judge takes the intersection.
    # The cited-URL guard wraps this; a bare pass-through unless a test replaces it.
    class SourceReader:
        def parse_website(self, url: str) -> Any:  # noqa: ANN401  # stub for an untyped upstream method
            recorded.setdefault("fetched", []).append(url)
            return f"content of {url}"

    utils.SourceReader = SourceReader

    gen_agent: Any = types.ModuleType("drbench.gen_agent")
    gen_agent.SERVICE_TO_MODELS = (
        {"openai": ["gpt-4o-mini", "gpt-4o"], "vllm": [], "together": []}
        if service_to_models is None
        else service_to_models
    )
    gen_agent.AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o"]

    class AIAgentManager:
        """Mirrors upstream's signature, whose parameter order the shim's guard depends on.

        `QASimilarityV2` constructs this with only `model=`, so `max_tokens` takes the
        1000-token default that truncates a reasoning judge's verdict.
        """

        def __init__(
            self,
            api_key: str | None = None,
            api_url: str | None = None,
            model: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            max_tokens: int = 1000,
            temperature: float = 0.7,
            with_linebreak: bool = False,
        ) -> None:
            recorded.setdefault("manager_max_tokens", []).append(max_tokens)
            self.model = model
            self.max_tokens = max_tokens

    gen_agent.AIAgentManager = AIAgentManager

    agents: Any = types.ModuleType("drbench.agents")
    agents.utils = utils

    root = types.ModuleType("drbench")

    requests: Any = types.ModuleType("requests")

    class _Session:
        """Models the one thing that matters here: every hop goes through `send`.

        Real `requests` routes `request` and each redirect hop through `Session.send`, so
        `sent` is the record of what actually left the process -- the only way to show a
        blocked host was never contacted rather than merely reported after the fact.
        """

        def send(self, request: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # stub for untyped `requests` internals
            recorded.setdefault("sent", []).append((request.url, kwargs.get("timeout")))
            return type("R", (), {"url": request.url})()

        def request(self, method: str, url: str, **kwargs: Any) -> Any:  # noqa: ANN401  # stub for untyped `requests` internals
            recorded.setdefault("requests", []).append((url, kwargs.get("timeout")))
            return self.send(type("P", (), {"url": url})(), **kwargs)

        def resolve_redirects(self, resp: Any, req: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # stub for untyped `requests` internals
            for hop in recorded.get("hops", []):
                yield self.send(type("P", (), {"url": hop.url})(), **kwargs)

    requests.Session = _Session
    monkeypatch.setitem(sys.modules, "requests", requests)

    for name, module in (
        ("drbench", root),
        ("drbench.agents", agents),
        ("drbench.agents.utils", utils),
        ("drbench.gen_agent", gen_agent),
        ("drbench.task_loader", task_loader),
        ("drbench.score_report", score_report_mod),
        ("drbench.metrics", metrics_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _stage_paths(
    judge: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    report: str | None = "# Report\n\nA claim [1].\n",
    case: dict | None = None,
) -> tuple[Path, Path]:
    """Point the module's absolute sandbox paths at a temp tree."""
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case if case is not None else {"task_id": "DR0001"}))
    monkeypatch.setattr(judge, "_CASE_PATH", case_path)

    report_path = tmp_path / "report.md"
    if report is not None:
        report_path.write_text(report)
    monkeypatch.setattr(judge, "_REPORT_PATH", report_path)

    reward_path = tmp_path / "logs" / "verifier" / "reward.json"
    breakdown_path = tmp_path / "logs" / "verifier" / "drbench_metrics.json"
    monkeypatch.setattr(judge, "_REWARD_JSON_PATH", reward_path)
    monkeypatch.setattr(judge, "_BREAKDOWN_PATH", breakdown_path)
    return reward_path, breakdown_path


# --- judge/embedding model selection --------------------------------------------------


def test_requested_judge_model_reads_first_of_judge_models(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_MODELS", "gpt-4o, gpt-4o-mini")
    assert judge._requested_judge_model() == "gpt-4o"


def test_requested_judge_model_falls_back_when_unset(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DRBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODELS", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert judge._requested_judge_model() == "gpt-5.6-luna"


def test_requested_judge_model_prefers_drbenchs_own_variable(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The harness uses this variable to publish DRBench's resolved fallback without
    # changing the suite-wide judge used by the other categories.
    monkeypatch.setenv("JUDGE_MODELS", "gpt-5.6-luna")
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-4o")
    assert judge._requested_judge_model() == "gpt-4o"


def test_supported_judge_models_registers_default_in_all_upstream_allowlists(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Upstream gates the judge in two independent places that disagree: `gpt-4.1` is in
    # `agents.utils.OPENAI_MODELS` but not in `gen_agent.SERVICE_TO_MODELS["openai"]`.
    # A model present in only one gets past `prompt_llm` and then dies inside
    # QASimilarityV2, so only the intersection is actually usable.
    _install_fake_drbench(monkeypatch, scores={})
    assert judge._supported_judge_models() == {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.6-luna",
    }

    utils: Any = sys.modules["drbench.agents.utils"]
    gen_agent: Any = sys.modules["drbench.gen_agent"]
    assert "gpt-5.6-luna" in utils.OPENAI_MODELS
    assert "gpt-5.6-luna" in gen_agent.SERVICE_TO_MODELS["openai"]
    assert "gpt-5.6-luna" in gen_agent.AVAILABLE_MODELS


def test_judge_model_keeps_a_supported_request(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("JUDGE_MODELS", "gpt-4o-mini")
    assert judge._judge_model() == "gpt-4o-mini"


def test_judge_model_keeps_the_unified_default(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-5.6-luna")
    assert judge._judge_model() == "gpt-5.6-luna"


def test_judge_model_raises_on_an_unsupported_request(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Raising rather than substituting: a quiet fallback publishes scores under the wrong
    # model's name, and the message has to name the route that does work.
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-5.6-terra")

    with pytest.raises(ValueError, match=r"openrouter/openai/gpt-5\.6-terra") as excinfo:
        judge._judge_model()

    assert "gpt-5.6-terra" in str(excinfo.value)


def test_judge_model_rejects_a_model_in_only_one_registry(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `gpt-4.1` is the trap: allowlisted by `prompt_llm`, unknown to `AIAgentManager`, so it
    # would get past the first gate and die inside QASimilarityV2.
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-4.1")

    with pytest.raises(ValueError, match=r"gpt-4\.1"):
        judge._judge_model()


def test_judge_model_passes_an_openrouter_slug_through(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Upstream resolves the `openrouter/` prefix ahead of both registries (`prompt_llm`, and
    # `AIAgentManager`, which reads `actual_model` before its `AVAILABLE_MODELS` check), so a
    # model neither registry lists still scores. This is the route to a frontier judge.
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("JUDGE_MODELS", "openrouter/openai/gpt-5.6-sol")
    assert judge._judge_model() == "openrouter/openai/gpt-5.6-sol"


def test_judge_model_accepts_an_openrouter_variant_suffix(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("JUDGE_MODELS", "openrouter/anthropic/claude-opus-4-7:beta")
    assert judge._judge_model() == "openrouter/anthropic/claude-opus-4-7:beta"


@pytest.mark.parametrize(
    "slug",
    [
        "openrouter/",
        "openrouter/openai",
        "openrouter//gpt-5.6-sol",
        "openrouter/openai/gpt-5.6-sol/extra",
        "myopenrouter/openai/gpt-5.6-sol",
    ],
)
def test_judge_model_rejects_a_malformed_openrouter_slug(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, slug: str
) -> None:
    # The slug becomes the `model` field of an outbound request, so only the published
    # `<vendor>/<model>[:variant]` shape passes. A malformed one is a typo, and scoring it
    # with a substitute judge would publish numbers under the wrong model's name.
    _install_fake_drbench(monkeypatch, scores={})
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", slug)

    with pytest.raises(ValueError, match="is not one upstream DRBench can drive"):
        judge._judge_model()


def test_judge_model_picks_from_whatever_upstream_offers(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The set is read from upstream, not hardcoded, so a model a future bump adds is
    # honored without a change here.
    _install_fake_drbench(
        monkeypatch,
        scores={},
        openai_models=["gpt-6-omni"],
        service_to_models={"openai": ["gpt-6-omni"]},
    )
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-6-omni")
    assert judge._judge_model() == "gpt-6-omni"


def test_embedding_model_defaults_to_upstreams_choice(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None means "whatever the installed drbench picks", rather than pinning a model that
    # upstream's code did not choose.
    monkeypatch.delenv("JUDGE_EMBEDDING_MODEL", raising=False)
    assert judge._embedding_model() is None
    monkeypatch.setenv("JUDGE_EMBEDDING_MODEL", "text-embedding-3-large")
    assert judge._embedding_model() == "text-embedding-3-large"


# --- the composite ---------------------------------------------------------------------


def test_composite_is_the_harmonic_mean(judge: ModuleType) -> None:
    components = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    assert judge.composite(components) == pytest.approx(0.5)


def test_composite_matches_the_paper_worked_example(judge: ModuleType) -> None:
    # The four metrics from a real DR0001 trial: 4 / (1/0.1667 + 1/1 + 1/0.4839 + 1/0.66).
    components = {
        "insights_recall": 0.16666666666666666,
        "distractor_avoidance": 1.0,
        "factuality": 0.4838709677419355,
        "report_quality": 0.66,
    }
    assert judge.composite(components) == pytest.approx(0.378, abs=5e-4)


def test_composite_floors_a_zero_component_instead_of_zeroing_out(judge: ModuleType) -> None:
    # A zero must crater the headline without erasing all ranking signal, so two reports
    # that both miss one metric can still be ordered by the others.
    worse = judge.composite({"a": 0.0, "b": 0.2, "c": 0.2, "d": 0.2})
    better = judge.composite({"a": 0.0, "b": 0.9, "c": 0.9, "d": 0.9})
    assert 0.0 < worse < better < 0.1


def test_composite_of_nothing_is_zero(judge: ModuleType) -> None:
    assert judge.composite({}) == 0.0


def test_zero_rewards_keeps_distractor_avoidance_the_complement_of_recall(
    judge: ModuleType,
) -> None:
    # `distractor_avoidance` is defined as `1 - distractor_recall`, so a report that does
    # not exist -- having recalled no distractors -- avoided all of them. Zeroing both broke
    # the identity, and since the aggregation sums each component over expected trials, the
    # pair stopped summing to 1.0 on the scorecard and read as a metric bug.
    rewards = judge._zero_rewards()

    assert rewards["distractor_recall"] == 0.0
    assert rewards["distractor_avoidance"] == 1.0
    assert rewards["distractor_avoidance"] + rewards["distractor_recall"] == 1.0
    # The composite still earns nothing; it is deliberately not the mean of the components.
    assert rewards["reward"] == 0.0
    assert all(rewards[name] == 0.0 for name in ("insights_recall", "factuality", "report_quality"))


def test_zero_rewards_covers_every_reported_metric(judge: ModuleType) -> None:
    rewards = judge._zero_rewards()
    assert set(rewards) == {"reward", *judge._METRIC_NAMES}
    # Every metric is unearned except `distractor_avoidance`, which is the complement of
    # `distractor_recall` by definition rather than something the report earns.
    assert set(rewards.values()) == {0.0, 1.0}
    assert [name for name, value in rewards.items() if value] == ["distractor_avoidance"]


# --- reading the report ----------------------------------------------------------------


def test_read_report_returns_the_artifact(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage_paths(judge, monkeypatch, tmp_path, report="# Hello\n")
    assert judge._read_report() == "# Hello\n"


def test_read_report_raises_and_lists_what_is_present(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    # A silent zero here is indistinguishable from a genuinely empty report, so the
    # failure has to say what Harbor actually delivered.
    _stage_paths(judge, monkeypatch, tmp_path, report=None)
    (tmp_path / "decoy.txt").write_text("not the report")

    with pytest.raises(FileNotFoundError):
        judge._read_report()
    assert "decoy.txt" in capsys.readouterr().out


# --- grading ---------------------------------------------------------------------------


_SCORES = {
    "insights_recall": 0.5,
    "distractor_recall": 0.25,
    "factuality": 0.8,
    "report_quality": 0.6,
}


def test_grade_requests_exactly_upstreams_four_metrics(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES), calls=calls)
    _stage_paths(judge, monkeypatch, tmp_path)
    monkeypatch.setenv("JUDGE_MODELS", "gpt-4o")

    judge._grade()

    kwargs = calls["score_report_kwargs"]
    assert kwargs["metrics"] == [
        "insights_recall",
        "distractor_recall",
        "factuality",
        "report_quality",
    ]
    # The task id drives both ground-truth and corpus lookup inside the package.
    assert calls["task_id"] == "DR0001"
    assert isinstance(kwargs["task"], _FakeTask)
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["predicted_report_text"].startswith("# Report")


def test_grade_inverts_distractor_recall_and_reports_both(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path)

    rewards, breakdown = judge._grade()

    # Recalling a planted distractor is a failure, so the composite consumes avoidance
    # while both numbers are still reported.
    assert rewards["distractor_recall"] == 0.25
    assert rewards["distractor_avoidance"] == 0.75
    assert breakdown["components"]["distractor_avoidance"] == 0.75
    assert "distractor_recall" not in breakdown["components"]
    assert rewards["reward"] == pytest.approx(
        judge.composite(
            {
                "insights_recall": 0.5,
                "distractor_avoidance": 0.75,
                "factuality": 0.8,
                "report_quality": 0.6,
            }
        )
    )


def test_grade_records_the_upstream_scores_verbatim(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path)
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-4o-mini")

    _, breakdown = judge._grade()
    assert breakdown["upstream_scores"] == _SCORES
    assert breakdown["task_id"] == "DR0001"
    # Both the model asked for and the one that ran, so a score is never misattributed.
    assert breakdown["judge_model"] == "gpt-4o-mini"
    assert breakdown["requested_judge_model"] == "gpt-4o-mini"


def test_grade_rejects_a_missing_metric(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A silently-absent metric would otherwise be read as 0.0 and look like a bad report.
    partial = dict.fromkeys(("insights_recall", "factuality", "report_quality"), 0.5)
    _install_fake_drbench(monkeypatch, scores=partial)
    _stage_paths(judge, monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="distractor_recall"):
        judge._grade()


def test_grade_rejects_a_non_dict_result(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=["not", "a", "dict"])
    _stage_paths(judge, monkeypatch, tmp_path)

    with pytest.raises(TypeError, match="expected a dict"):
        judge._grade()


def test_grade_requires_a_task_id(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path, case={"upstream_sha": "abc"})

    with pytest.raises(ValueError, match="task_id"):
        judge._grade()


# --- main() ---------------------------------------------------------------------------


def test_main_writes_rewards_and_breakdown(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    reward_path, breakdown_path = _stage_paths(judge, monkeypatch, tmp_path)

    judge.main()

    rewards = json.loads(reward_path.read_text())
    # `reward` is the key the deepagents aggregation reads; the components ride alongside.
    assert set(rewards) == {"reward", *judge._METRIC_NAMES}
    assert rewards["insights_recall"] == 0.5
    assert 0.0 < rewards["reward"] < 1.0
    assert json.loads(breakdown_path.read_text())["upstream_scores"] == _SCORES


def test_main_fails_the_verifier_when_grading_crashes(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Harbor reads a missing reward file as an errored trial and a 0.0 as a real score, so
    # an infrastructure failure must write no reward: otherwise a judge outage on 3 of 30
    # tasks silently drags the model's average down instead of being reported.
    reward_path, breakdown_path = _stage_paths(judge, monkeypatch, tmp_path, report=None)

    def boom() -> None:
        msg = "judge exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_grade", boom)

    with pytest.raises(RuntimeError, match="judge exploded"):
        judge.main()

    assert not reward_path.exists(), "an infra failure must not be scored"
    # The diagnostic still lands, which is the whole reason not to just let it propagate.
    assert "judge exploded" in json.loads(breakdown_path.read_text())["error"]


def test_main_scores_zero_when_the_agent_wrote_no_report(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The one failure that is genuinely the model's: it finished without producing a report.
    # That is a failed attempt, so it is scored zero rather than errored.
    reward_path, breakdown_path = _stage_paths(judge, monkeypatch, tmp_path, report=None)

    def missing() -> None:
        msg = "no report at /app/report.md"
        raise judge._MissingReportError(msg)

    monkeypatch.setattr(judge, "_grade", missing)
    judge.main()

    assert json.loads(reward_path.read_text()) == judge._zero_rewards()
    assert "no report" in json.loads(breakdown_path.read_text())["error"]


def test_main_clamps_rewards_into_range(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_of_range = {
        "insights_recall": 1.4,
        "distractor_recall": -0.2,
        "factuality": 0.5,
        "report_quality": 0.5,
    }
    _install_fake_drbench(monkeypatch, scores=out_of_range)
    reward_path, _ = _stage_paths(judge, monkeypatch, tmp_path)

    judge.main()

    rewards = json.loads(reward_path.read_text())
    assert all(0.0 <= value <= 1.0 for value in rewards.values())


# --- extract_text.py (still shipped to the AGENT image) -------------------------------
# The verifier no longer needs it -- upstream's SourceReader parses cited documents --
# but the agent does: every document it downloads from the app stack arrives as binary.


def test_extract_text_renders_email_jsonl(extract_text: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "mail.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"type": "version", "version": 1},
                {
                    "type": "user",
                    "username": "dana.ray",
                    "first_name": "Dana",
                    "last_name": "Ray",
                    "email": "dana@acme.com",
                },
                {
                    "type": "email",
                    "subject": "Q2 review",
                    "from": "dana@acme.com",
                    "from_name": "Dana Ray",
                    "to": ["sam@acme.com"],
                    "cc": [],
                    "date": "2023-06-01 10:00:00",
                    "body": "Scheduling the review.",
                },
            )
        )
    )
    rendered = extract_text.extract(path)
    assert "Directory:" in rendered
    assert "Dana Ray (dana.ray) dana@acme.com" in rendered
    assert "Subject: Q2 review" in rendered
    assert "From: Dana Ray <dana@acme.com>" in rendered
    assert "To: sam@acme.com" in rendered
    assert "Scheduling the review." in rendered
    # An empty cc must not render a dangling header.
    assert "Cc:" not in rendered


def test_extract_text_renders_mattermost_jsonl(extract_text: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "channel",
                    "channel": {
                        "team": "sales-team",
                        "name": "sales-strategy",
                        "display_name": "Sales Strategy",
                        "purpose": "Collaborate on sales",
                    },
                },
                {
                    "type": "post",
                    "post": {
                        "team": "sales-team",
                        "channel": "sales-strategy",
                        "user": "michael.lee",
                        "message": "Weekly meeting tomorrow at 2 PM.",
                        "create_at": 1685731200000,
                    },
                },
            )
        )
    )
    rendered = extract_text.extract(path)
    assert "Channel: Sales Strategy" in rendered
    assert "User: michael.lee" in rendered
    assert "Weekly meeting tomorrow at 2 PM." in rendered
    # create_at is milliseconds since the epoch, so it must render as a 2023 date.
    assert "Date: 2023-06-02" in rendered


def test_extract_text_bounds_output(extract_text: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("y" * (extract_text.MAX_OUTPUT_CHARS + 1_000))
    rendered = extract_text.extract(path)
    assert rendered.startswith("y" * 100)
    assert "[truncated at" in rendered
    assert len(rendered) < extract_text.MAX_OUTPUT_CHARS + 100


def test_extract_text_rejects_unsupported_type(extract_text: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "thing.bin"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_text.extract(path)


def test_extract_text_reports_a_missing_file(extract_text: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a file"):
        extract_text.extract(tmp_path / "absent.pdf")


def test_extract_text_main_survives_one_bad_file(
    extract_text: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.txt"
    good.write_text("readable")
    status = extract_text.main([str(tmp_path / "absent.pdf"), str(good)])
    assert status == 1
    assert "readable" in capsys.readouterr().out


# --- distractor_recall, report_quality, factuality, and the composite -------------------


# --- citation formats, pinned against upstream ----------------------------------------


def test_documented_citation_forms_resolve_through_upstreams_normalizer() -> None:
    """The instruction's email/chat forms must survive upstream's citation pipeline.

    Runs only where `drbench` is installed (the verifier image), because it asserts
    against upstream's own `clean_citation` rather than a local copy of its rules. That is
    the point: it pins the contract to the pinned commit instead of to prose.
    """
    pytest.importorskip("drbench")
    # Deliberately inside the test: it may only be imported once importorskip has
    # confirmed the package is installed.
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    # The form `_instruction` tells the agent to use for an email.
    resolved = utils.clean_citation(
        "RoundCube-david.lee@example.com-emily.patel@example.com-Re: Q2 Compliance Update"
    )
    assert resolved == (
        "roundcube<sep>david.lee@example.com<sep>emily.patel@example.com"
        "<sep>re: q2 compliance update"
    )

    # And for a chat message.
    assert utils.clean_citation("MatterMost-fsma_compliance-compliance_team-john.doe") == (
        "mattermost<sep>fsma_compliance<sep>compliance_team<sep>john.doe"
    )

    # A display name instead of an address cannot resolve: every pattern in
    # `normalize_email_citation` requires an `@`, so it degrades to a filename lookup.
    # This is why the instruction demands the sender's address.
    assert not str(
        utils.clean_citation("**Re: Q2 Compliance Update** - Email from David Lee on 15 July 2025")
    ).startswith("roundcube<sep>")


# --- embedding request batching -------------------------------------------------------
# `get_most_relevant_chunks` embeds up to 200 chunks of 2048 characters in ONE request.
# At ~4 chars/token that is ~100k tokens, but content that tokenizes badly (a web page or
# PDF parsed into near-binary text) approaches 1 token/char and exceeds the API's 300k
# ceiling, failing factuality outright. Upstream batches in `semantic_retriever` but not
# on this path.


def _dense(count: int, size: int = 2048) -> list[str]:
    """Texts that tokenize close to one token per character."""
    return [
        "".join(chr(0x4E00 + (index * 37 + offset) % 900) for offset in range(size))
        for index in range(count)
    ]


def test_embedding_batching_splits_an_oversized_request_with_tiktoken(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Skips unless the real token estimator is usable. Importing tiktoken is not enough:
    # `get_encoding` downloads its BPE file on first use, so on a fresh CI runner with
    # sockets blocked it raises, `_install_embedding_batching` falls back to the character
    # estimate, and the same input fits in one batch. Guarding on the import alone is what
    # made an earlier version of this test pass locally (cached BPE) and fail in CI.
    tiktoken = pytest.importorskip("tiktoken")
    try:
        tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - any failure means the encoder is unusable here
        pytest.skip(f"cl100k_base unavailable ({type(exc).__name__}); fallback test covers it")
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_embedding_batching()
    vectors = utils.get_embeddings(_dense(200))

    batches = calls["embed_batches"]
    assert len(batches) > 1, "an oversized request must be split"
    assert sum(batches) == 200
    assert len(vectors) == 200


def test_embedding_batching_splits_an_oversized_request_without_tiktoken(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The character-estimate fallback, which is what runs wherever tiktoken is missing.
    # Setting the module to None makes `import tiktoken` raise, and the count is sized so
    # `len // 3` alone exceeds the budget.
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    count = (3 * judge._EMBED_TOKEN_BUDGET) // 2048 + 50
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_embedding_batching()
    vectors = utils.get_embeddings(_dense(count))

    batches = calls["embed_batches"]
    assert len(batches) > 1, "an oversized request must be split"
    assert sum(batches) == count
    assert len(vectors) == count


def test_embedding_batching_preserves_the_vectors_exactly(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point: this is a request-framing change, not a scoring change. Batched and
    # unbatched calls must produce identical vectors in identical order, or metrics move.
    _install_fake_drbench(monkeypatch, scores={})
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    texts = _dense(200)
    unbatched = utils.get_embeddings(texts)
    judge._install_embedding_batching()
    batched = utils.get_embeddings(texts)

    assert np.array_equal(batched, unbatched)
    # Type fidelity matters as much as the values: the consumer reads `.shape[1]` and
    # calls `faiss.normalize_L2`, which needs a C-contiguous float32 array.
    assert isinstance(batched, np.ndarray)
    assert batched.dtype == unbatched.dtype
    assert batched.flags["C_CONTIGUOUS"]
    assert batched.shape == unbatched.shape


def test_embedding_batching_leaves_a_normal_request_alone(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_embedding_batching()
    utils.get_embeddings(["short text"] * 10)
    assert calls["embed_batches"] == [10], "a small request must not be fanned out"


def test_embedding_batching_is_idempotent(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_grade` installs it on every call; double-wrapping would batch the batches.
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_embedding_batching()
    judge._install_embedding_batching()
    utils.get_embeddings(["a", "b"])
    assert calls["embed_batches"] == [2]


def test_grade_installs_batching_and_skips_the_per_insight_pass(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES), calls=calls)
    _stage_paths(judge, monkeypatch, tmp_path)

    judge._grade()

    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    assert getattr(utils.get_embeddings, "_deepagents_batched", False)
    # The per-insight results are only written to `savedir`, which we never pass, so the
    # pass costs a judge call per insight plus retries and returns nothing readable.
    assert calls["score_report_kwargs"]["include_per_insight_scores"] is False


# --- judge output cap -----------------------------------------------------------------


def test_install_judge_sampling_replaces_the_default_output_cap(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench import gen_agent  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_judge_sampling(max_tokens=32_000)
    # The construction `QASimilarityV2` makes: model only, everything else defaulted.
    gen_agent.AIAgentManager(model="openrouter/openai/gpt-5.6-sol")

    assert calls["manager_max_tokens"] == [32_000]


def test_install_judge_sampling_keeps_an_explicit_keyword(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench import gen_agent  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_judge_sampling(max_tokens=32_000)
    gen_agent.AIAgentManager(model="gpt-4o", max_tokens=64)

    assert calls["manager_max_tokens"] == [64]


def test_install_judge_sampling_tolerates_a_positional_max_tokens(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `max_tokens` is the fourth positional parameter after `self`. Injecting the keyword
    # on top of a positional value would raise "got multiple values for max_tokens".
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    from drbench import gen_agent  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_judge_sampling(max_tokens=32_000)
    gen_agent.AIAgentManager(None, None, "gpt-4o", 128)

    assert calls["manager_max_tokens"] == [128]


def test_install_judge_sampling_is_idempotent(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    from drbench import gen_agent  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_judge_sampling()
    first = gen_agent.AIAgentManager.__init__
    judge._install_judge_sampling()

    assert gen_agent.AIAgentManager.__init__ is first


def test_grade_raises_the_output_cap_only_for_an_openrouter_judge(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The direct-OpenAI path keeps upstream's 1000-token default, so scores from earlier
    # runs stay reproducible; only the OpenRouter path needs the larger cap.
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path)
    monkeypatch.setenv("JUDGE_MODELS", "gpt-4o")
    from drbench import gen_agent  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._grade()

    assert not getattr(gen_agent.AIAgentManager.__init__, "_deepagents_sampling", False)

    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    monkeypatch.setenv("JUDGE_MODELS", "openrouter/openai/gpt-5.6-sol")
    from drbench import gen_agent as reinstalled  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._grade()

    assert getattr(reinstalled.AIAgentManager.__init__, "_deepagents_sampling", False)


# --- a misconfigured judge is fatal, a broken image is not ----------------------------


def test_main_refuses_to_score_with_an_unusable_judge(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The check sits outside `main`'s catch-all on purpose. Inside it, the same error would
    # be written as a 0.0 reward -- and a 0.0 is exactly what a genuinely bad report scores,
    # so the misconfiguration would be invisible on the scorecard.
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    reward_path, _ = _stage_paths(judge, monkeypatch, tmp_path)
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-5.6-terra")

    with pytest.raises(ValueError, match="openrouter"):
        judge.main()

    assert not reward_path.exists()


def test_main_fails_the_verifier_when_drbench_is_missing(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A broken verifier image is infrastructure, not a model failure. The judge pre-check
    # deliberately swallows the import error so this lands on the graded path, which then
    # records the traceback and fails rather than scoring the model zero.
    for name in ("drbench", "drbench.gen_agent", "drbench.agents.utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", [])
    reward_path, breakdown_path = _stage_paths(judge, monkeypatch, tmp_path)
    monkeypatch.setenv("DRBENCH_JUDGE_MODEL", "gpt-4o")

    with pytest.raises(ModuleNotFoundError):
        judge.main()

    assert not reward_path.exists()
    assert "traceback" in json.loads(breakdown_path.read_text())


# --- per-insight verdict capture ------------------------------------------------------


def _verdict_rows(count: int = 2, justification: str = "no matching claim") -> dict[str, Any]:
    """A metric result shaped like `QASimilarityV2.compute`'s return value."""
    return {
        "score": 0.0,
        "per_question_results": [
            {
                "question": f"q{i}",
                "expected_insight": f"gold {i}",
                "predicted_insight": None,
                "expected_supporting_paths": ["/corpus/a.pdf"],
                "score": 0.0,
                "justification": justification,
                "confidence": "high",
            }
            for i in range(count)
        ],
    }


def test_trim_captured_keeps_only_the_allowlisted_fields(judge: ModuleType) -> None:
    rows = judge._trim_captured(_verdict_rows(count=1))

    assert len(rows) == 1
    # `question` and `expected_supporting_paths` are dropped: the paths are corpus text and
    # the question is already recoverable from the task.
    assert set(rows[0]) == set(judge._CAPTURED_FIELDS)
    assert rows[0]["expected_insight"] == "gold 0"
    assert rows[0]["justification"] == "no matching claim"


def test_trim_captured_bounds_length_and_count(judge: ModuleType) -> None:
    # A long report must not be able to inflate the breakdown artifact without limit.
    over = judge._MAX_CAPTURED_VERDICTS + 5
    rows = judge._trim_captured(_verdict_rows(count=over, justification="x" * 5_000))

    assert len(rows) == judge._MAX_CAPTURED_VERDICTS
    assert all(len(row["justification"]) == judge._MAX_CAPTURED_CHARS for row in rows)


@pytest.mark.parametrize(
    "result", [None, "not a dict", {}, {"score": 0.0}, {"per_question_results": "nope"}]
)
def test_trim_captured_ignores_a_result_without_verdicts(judge: ModuleType, result: object) -> None:
    # factuality and report_quality return no `per_question_results`; they are skipped
    # rather than partially recorded.
    assert judge._trim_captured(result) == []


def test_metric_detail_capture_patches_the_name_score_report_actually_calls(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `score_report` does `from drbench.metrics import get_metric`, binding the function in
    # its own namespace at import time. Patching `drbench.metrics.get_metric` would leave
    # that reference untouched and capture nothing -- the same class of trap as upstream's
    # `AVAILABLE_MODELS` fallback.
    _install_fake_drbench(monkeypatch, scores={})
    from drbench import metrics, score_report  # noqa: PLC0415  # ty: ignore[unresolved-import]

    before = metrics.get_metric
    judge._install_metric_detail_capture({})

    assert score_report.get_metric is not before
    assert metrics.get_metric is before


def test_metric_detail_capture_records_each_metrics_verdicts(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    from drbench import score_report  # noqa: PLC0415  # ty: ignore[unresolved-import]

    sink: dict[str, Any] = {}
    judge._install_metric_detail_capture(sink)
    metric = score_report.get_metric("insights_recall")
    returned = metric.compute()

    assert sink["insights_recall"][0]["expected_insight"] == "gold 0"
    # The wrapper must hand upstream its own result untouched, or scoring changes.
    assert returned == _verdict_rows()


def test_metric_detail_capture_is_idempotent(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    from drbench import score_report  # noqa: PLC0415  # ty: ignore[unresolved-import]

    judge._install_metric_detail_capture({})
    first = score_report.get_metric
    judge._install_metric_detail_capture({})

    assert score_report.get_metric is first


def test_grade_reports_the_judges_verdicts(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path)

    _, breakdown = judge._grade()

    assert breakdown["metric_detail"]["insights_recall"][0]["justification"] == (
        "no matching claim"
    )


# --- cited-URL fetch guard ------------------------------------------------------------
# `get_content` resolves an http citation through
# `drbench.agents.utils.SourceReader.parse_website`, whose live body is a bare
# `session.get(url)`: no host validation, no timeout, no exception handling. A single
# unreachable citation therefore zeroed all four metrics for a task in a real run.


def _fake_requests() -> Any:  # noqa: ANN401  # the registered module stand-in is a dynamic ModuleType
    """The `requests` stand-in `_install_fake_drbench` registered."""
    return sys.modules["requests"]


def test_url_guard_turns_a_failed_fetch_into_no_content(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_drbench(monkeypatch, scores={})
    from drbench.agents import utils  # noqa: PLC0415  # ty: ignore[unresolved-import]

    def boom(_self: Any, _url: str) -> Any:  # noqa: ANN401  # stub for an untyped upstream method
        msg = "name resolution failed"
        raise OSError(msg)

    utils.SourceReader = type("SourceReader", (), {"parse_website": boom})

    failures: list[dict[str, str]] = []
    judge._install_url_fetch_guard(failures)

    # None, not an exception: `get_content`'s contract already treats None as an
    # unresolvable citation, so the claim goes unsupported and the task still scores.
    assert utils.SourceReader().parse_website("https://example.com/x") is None
    assert len(failures) == 1
    assert "OSError" in failures[0]["error"]
    assert failures[0]["url"] == "https://example.com/x"


@pytest.mark.parametrize(
    "host",
    ["169.254.169.254", "127.0.0.1", "10.0.0.5", "192.168.1.1"],
)
def test_url_guard_refuses_a_non_public_host(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    # Citations are written by the model under test, so a cited URL is untrusted input.
    # Upstream performs no host check at all; this restores the guard.
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    requests = _fake_requests()
    monkeypatch.setattr(judge.socket, "getaddrinfo", lambda h, _p: [(0, 0, 0, "", (h, 0))])
    judge._install_url_fetch_guard([])

    with pytest.raises(judge._BlockedHostError):
        requests.Session().request("GET", f"http://{host}/meta")
    # Refused before the request was made, not after.
    assert calls.get("requests") is None


def test_url_guard_allows_a_public_host_and_sets_a_timeout(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    requests = _fake_requests()
    monkeypatch.setattr(
        judge.socket, "getaddrinfo", lambda _h, _p: [(0, 0, 0, "", ("93.184.216.34", 0))]
    )
    judge._install_url_fetch_guard([])

    requests.Session().request("GET", "https://example.com/page")
    (url, timeout) = calls["requests"][0]
    assert url == "https://example.com/page"
    # Upstream passes no timeout, so one slow host could consume the verifier budget.
    assert timeout == judge._URL_FETCH_TIMEOUT


def test_url_guard_blocks_a_redirect_hop_before_it_is_sent(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The assertion that matters is `sent`, not the exception. The previous guard validated
    # each hop's *response* URL, which `resolve_redirects` only yields after sending it --
    # so a cited public URL answering `302 Location: http://169.254.169.254/...` reached the
    # metadata endpoint and then raised. Raising is not the fix; not sending is.
    metadata = "http://169.254.169.254/latest/meta-data/"
    calls: dict[str, Any] = {"hops": [type("R", (), {"url": metadata})()]}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    requests = _fake_requests()
    monkeypatch.setattr(judge.socket, "getaddrinfo", lambda h, _p: [(0, 0, 0, "", (h, 0))])
    judge._install_url_fetch_guard([])

    with pytest.raises(judge._BlockedHostError):
        list(requests.Session().resolve_redirects(None, None))

    sent = [url for url, _timeout in calls.get("sent", [])]
    assert metadata not in sent, "the blocked host was contacted before the guard fired"
    assert sent == []


def test_url_guard_is_idempotent(judge: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _install_fake_drbench(monkeypatch, scores={}, calls=calls)
    requests = _fake_requests()
    monkeypatch.setattr(
        judge.socket, "getaddrinfo", lambda _h, _p: [(0, 0, 0, "", ("93.184.216.34", 0))]
    )
    judge._install_url_fetch_guard([])
    judge._install_url_fetch_guard([])

    requests.Session().request("GET", "https://example.com/")
    assert len(calls["requests"]) == 1, "double-wrapping would re-enter the guard"


def test_grade_reports_unfetchable_citations(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A depressed factuality is only interpretable next to the list of sources the
    # verifier could not read, so the breakdown must carry it even when empty.
    _install_fake_drbench(monkeypatch, scores=dict(_SCORES))
    _stage_paths(judge, monkeypatch, tmp_path)

    _, breakdown = judge._grade()
    assert breakdown["unfetchable_citations"] == []


def test_main_records_a_traceback_on_failure(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An earlier failure reported only "ConnectionError: ..." with no frame, which made
    # locating it far slower than reading a traceback would have been.
    reward_path, breakdown_path = _stage_paths(judge, monkeypatch, tmp_path, report=None)

    def boom() -> None:
        msg = "kaboom"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_grade", boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        judge.main()

    breakdown = json.loads(breakdown_path.read_text())
    assert "kaboom" in breakdown["error"]
    assert "Traceback" in breakdown["traceback"]
    assert "boom" in breakdown["traceback"]
    assert not reward_path.exists()
