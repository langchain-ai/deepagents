"""Tests for the DRBench verifier templates (`judge.py`, `extract_text.py`).

The templates run inside the Harbor sandbox with no `deepagents_evals` on the path, so
they are loaded here by file path rather than imported as package modules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Self

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


def test_judge_model_reads_first_of_judge_models(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_MODELS", "gpt-5.6-terra, gpt-5.6-luna")
    assert judge._judge_model() == "gpt-5.6-terra"


def test_judge_model_falls_back_when_unset(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JUDGE_MODELS", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert judge._judge_model() == "gpt-5.6-luna"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-5.6-luna", 1.0), ("o1-preview", 1.0), ("o3-mini", 1.0), ("gpt-4o", 0.0)],
)
def test_temperature_bumps_reasoning_judges(judge: ModuleType, model: str, expected: float) -> None:
    """Reasoning judges reject temperature 0.0 at the API, so upstream bumps them."""
    assert judge._temperature(model) == expected


@pytest.mark.parametrize(
    "response",
    [
        '{"answer": "yes"}',
        '```json\n{"answer": "yes"}\n```',
        'Sure!\n{"answer": "yes"}\nHope that helps.',
    ],
)
def test_extract_json_tolerates_wrapping(judge: ModuleType, response: str) -> None:
    assert judge._extract_json(response) == {"answer": "yes"}


def test_extract_json_raises_on_unparseable(judge: ModuleType) -> None:
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        judge._extract_json("no json here")


def test_format_claims_numbers_insights(judge: ModuleType) -> None:
    text = judge._format_claims([{"claim": "first"}, {"claim": "second"}])
    assert text == "Insight 1: first\nInsight 2: second"


def test_format_claims_handles_empty_report(judge: ModuleType) -> None:
    assert judge._format_claims([]) == "No claims found in the report."


def test_scoring_prompt_substitutes_and_unescapes_braces(judge: ModuleType) -> None:
    prompt = judge._QA_SCORING_PROMPT.format(
        claims_text="Insight 1: a claim", gold_insight="the gold"
    )
    assert "Insight 1: a claim" in prompt
    assert "the gold" in prompt
    # The template doubles the braces of the requested JSON object; after formatting the
    # judge must see single braces or it is not being asked for JSON.
    assert '{\n    "answer"' in prompt
    assert "{{" not in prompt


def test_claim_split_prompt_embeds_report(judge: ModuleType) -> None:
    prompt = judge._CLAIM_SPLIT_PROMPT.format(report_text="REPORT BODY")
    assert "REPORT BODY" in prompt
    assert "{{" not in prompt


def _score(judge: ModuleType, monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> float:
    """Drive `_score_insight` with canned judge answers."""
    monkeypatch.setattr(judge, "_call_judge", lambda _prompt, _model: answers.pop(0))
    return judge._score_one_entry("Insight 1: x", "gold", "gpt-4o")["score"]


def test_score_insight_maps_yes_to_one(judge: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _score(judge, monkeypatch, ['{"answer": "YES", "justification": "j"}']) == 1.0


@pytest.mark.parametrize("answer", ["no", "partially", ""])
def test_score_insight_maps_anything_else_to_zero(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    assert _score(judge, monkeypatch, [json.dumps({"answer": answer})]) == 0.0


def test_score_insight_retries_then_scores_zero(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def failing(_prompt: str, model: str) -> str:
        calls.append(model)
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_call_judge", failing)
    result = judge._score_one_entry("Insight 1: x", "gold", "gpt-4o")
    assert result["score"] == 0.0
    assert len(calls) == judge.MAX_RETRIES
    assert "Failed to parse model response" in result["justification"]


def test_grade_is_fraction_of_recalled_insights(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "task_id": "DR0001",
                "insights": [
                    {"id": "IN1", "type": "enterprise_fact", "answer": "one"},
                    {"id": "IN2", "type": "enterprise_fact", "answer": "two"},
                    {"id": "EX1", "type": "external_fact", "answer": "three"},
                    {"id": "IN3", "type": "enterprise_fact", "answer": "   "},
                ],
            }
        )
    )
    report = tmp_path / "report.md"
    report.write_text("# Report\n\nOne happened [1].\n")
    monkeypatch.setattr(judge, "_CASE_PATH", case)
    monkeypatch.setattr(judge, "_REPORT_PATH", report)
    monkeypatch.setattr(judge, "_split_report", lambda _text, _model: [{"claim": "One happened"}])
    # Two of the three scorable insights recalled; the blank insight is not scored.
    verdicts = iter(["yes", "no", "yes"])
    monkeypatch.setattr(
        judge,
        "_call_judge",
        lambda _prompt, _model: json.dumps({"answer": next(verdicts)}),
    )

    rewards, breakdown = judge._grade()
    assert rewards["insights_recall"] == pytest.approx(2 / 3)
    assert breakdown["insight_count"] == 3
    assert [e["id"] for e in breakdown["insights_recall"]["per_insight"]] == [
        "IN1",
        "IN2",
        "EX1",
    ]


def test_grade_scores_zero_without_a_report(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"task_id": "DR0001", "insights": [{"id": "IN1", "answer": "one"}]}))
    monkeypatch.setattr(judge, "_CASE_PATH", case)
    monkeypatch.setattr(judge, "_REPORT_PATH", tmp_path / "absent.md")
    rewards, breakdown = judge._grade()
    assert set(rewards) == {"reward", *judge._METRIC_NAMES}
    assert all(value == 0.0 for value in rewards.values())
    assert breakdown["error"] == "report missing"


def test_grade_truncates_an_oversized_report(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"task_id": "DR0001", "insights": [{"id": "IN1", "answer": "one"}]}))
    report = tmp_path / "report.md"
    report.write_text("x" * (judge.MAX_REPORT_LENGTH + 500))
    split_lengths: list[int] = []

    def record_split(text: str, _model: str) -> list[dict]:
        split_lengths.append(len(text))
        return []

    monkeypatch.setattr(judge, "_CASE_PATH", case)
    monkeypatch.setattr(judge, "_REPORT_PATH", report)
    monkeypatch.setattr(judge, "_split_report", record_split)
    monkeypatch.setattr(judge, "_call_judge", lambda _prompt, _model: '{"answer": "no"}')
    judge._grade()
    assert split_lengths == [judge.MAX_REPORT_LENGTH]


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


def test_distractor_recall_uses_the_same_judge_loop(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's DistractorRecall subclasses QASimilarityV2, so scoring is identical."""
    verdicts = iter(["yes", "no"])
    monkeypatch.setattr(judge, "_call_judge", lambda _p, _m: json.dumps({"answer": next(verdicts)}))
    score, details = judge._score_qa_set(
        "Insight 1: x",
        [{"id": "DI1", "answer": "planted"}, {"id": "DI2", "answer": "other"}],
        "gpt-4o",
    )
    assert score == pytest.approx(0.5)
    assert [d["id"] for d in details] == ["DI1", "DI2"]


def test_empty_ground_truth_scores_zero(judge: ModuleType) -> None:
    """Matches upstream's overall_score, which is 0.0 for an empty entry list."""
    score, details = judge._score_qa_set("Insight 1: x", [], "gpt-4o")
    assert score == 0.0
    assert details == []


def test_report_quality_averages_five_criteria(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = "<evaluation>" + "".join(
        f"<{c}><score>{score}</score><justification>j</justification></{c}>"
        for c, score in zip(judge._QUALITY_CRITERIA, (10, 8, 6, 4, 2), strict=True)
    )
    monkeypatch.setattr(judge, "_call_judge", lambda _p, _m: response)
    score, details = judge._report_quality("report", "q?", {"name": "Dana"}, "gpt-4o")
    # Mean of the five per-criterion scores, each divided by 10.
    assert score == pytest.approx(0.6)
    assert len(details["per_criterion"]) == 5


@pytest.mark.parametrize(("raw", "expected"), [(99, 1.0), (0, 0.1), (-5, 0.1)])
def test_report_quality_clamps_out_of_range_scores(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, raw: int, expected: float
) -> None:
    response = "".join(f"<{c}><score>{raw}</score></{c}>" for c in judge._QUALITY_CRITERIA)
    monkeypatch.setattr(judge, "_call_judge", lambda _p, _m: response)
    score, _ = judge._report_quality("report", "q?", {}, "gpt-4o")
    assert score == pytest.approx(expected)


def test_report_quality_scores_zero_when_criteria_are_missing(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge, "_call_judge", lambda _p, _m: "<depth_quality><score>9</score>")
    score, details = judge._report_quality("report", "q?", {}, "gpt-4o")
    assert score == 0.0
    assert "report_quality failed" in details["error"]


def test_factuality_counts_uncited_claims_as_unsupported(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        judge, "_webdav_index", lambda _e, _c: {"report.pdf": "http://x/report.pdf"}
    )
    monkeypatch.setattr(judge, "_resolve_citation", lambda _c, _i, _cr: "supporting text")
    monkeypatch.setattr(judge, "_relevant_chunks", lambda _q, content: [content])
    monkeypatch.setattr(
        judge, "_call_judge", lambda _p, _m: json.dumps({"is_factual": True, "explanation": "ok"})
    )
    case = {
        "endpoints": {"nextcloud": "http://drbench:8081"},
        "credentials": {"nextcloud": {"username": "u", "password": "p"}},
    }
    score, details = judge._factuality(
        [
            {"claim": "cited", "citations": ["report.pdf"]},
            {"claim": "uncited", "citations": []},
        ],
        case,
        "gpt-4o",
    )
    assert score == pytest.approx(0.5)
    assert details["per_claim"][1]["explanation"].startswith("No citations")


def test_factuality_fails_loudly_when_the_app_stack_is_unreachable(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken environment must not be reported as an unfactual report."""

    def unreachable(_endpoint: str, _credentials: dict) -> dict:
        msg = "could not list http://drbench:8081/...: URLError"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_webdav_index", unreachable)
    case = {
        "endpoints": {"nextcloud": "http://drbench:8081"},
        "credentials": {"nextcloud": {"username": "u", "password": "p"}},
    }
    with pytest.raises(RuntimeError, match="could not list"):
        judge._factuality([{"claim": "c", "citations": ["a.pdf"]}], case, "gpt-4o")


def _stub_resolver(monkeypatch: pytest.MonkeyPatch, judge: ModuleType, address: str) -> None:
    """Make host resolution deterministic and offline."""
    monkeypatch.setattr(
        judge.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", (address, 0))],
    )


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("169.254.169.254", False),  # cloud instance metadata
        ("127.0.0.1", False),
        ("10.0.0.5", False),
        ("192.168.1.1", False),
    ],
)
def test_is_public_host_rejects_internal_addresses(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, address: str, public: bool
) -> None:
    _stub_resolver(monkeypatch, judge, address)
    assert judge._is_public_host("example.test") is public


def test_is_public_host_rejects_an_unresolvable_host(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> list:
        raise OSError

    monkeypatch.setattr(judge.socket, "getaddrinfo", fail)
    assert judge._is_public_host("nope.invalid") is False


def test_factuality_never_fetches_a_citation_pointing_at_internal_metadata(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Citations come from the agent's report, so a cited URL is untrusted input."""
    fetched: list[str] = []
    monkeypatch.setattr(judge, "_http_get", lambda url, **_: fetched.append(url) or b"")
    _stub_resolver(monkeypatch, judge, "169.254.169.254")
    assert judge._resolve_citation("http://metadata.example/latest/meta-data/", {}, None) is None
    assert fetched == []


def test_relevant_chunks_skips_embeddings_for_small_documents(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream returns every chunk when there are at most TOP_K, with no embedding call."""

    def fail(_texts: list[str]) -> list[list[float]]:
        msg = "embeddings must not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(judge, "_embed", fail)
    chunks = judge._relevant_chunks("q", "a" * (judge.CHUNK_SIZE * judge.TOP_K))
    assert len(chunks) == judge.TOP_K


def test_relevant_chunks_ranks_by_similarity_when_there_are_many(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "".join(f"{marker}" * judge.CHUNK_SIZE for marker in "abcdefg")

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # The query matches the chunk made of "d" exactly.
        return [[1.0, 0.0] if "d" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(judge, "_embed", fake_embed)
    chunks = judge._relevant_chunks("d", content)
    assert len(chunks) == judge.TOP_K
    assert chunks[0].startswith("d")


def test_relevant_chunks_falls_back_when_embedding_fails(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_texts: list[str]) -> list[list[float]]:
        msg = "upstream down"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_embed", fail)
    content = "".join(f"{m}" * judge.CHUNK_SIZE for m in "abcdefg")
    assert len(judge._relevant_chunks("q", content)) == judge.TOP_K


def test_composite_is_a_harmonic_mean() -> None:
    module = _load("judge")
    perfect = module.composite(dict.fromkeys(("a", "b", "c", "d"), 1.0))
    assert perfect == pytest.approx(1.0)
    # Harmonic means sit below the arithmetic mean whenever components differ.
    mixed = module.composite({"a": 1.0, "b": 0.5, "c": 1.0, "d": 0.5})
    assert mixed < 0.75


def test_composite_floors_a_zero_component_instead_of_erasing_the_score() -> None:
    """A strict harmonic mean would return exactly 0.0 and lose all ranking signal."""
    module = _load("judge")
    score = module.composite(
        {"insights_recall": 0.9, "distractor_avoidance": 0.9, "factuality": 0.0, "quality": 0.9}
    )
    assert 0.0 < score < 0.05


def test_grade_inverts_distractor_recall_for_the_composite(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recalling a distractor is a failure, so the composite must use avoidance."""
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "task_id": "DR0001",
                "question": "q?",
                "persona": {"name": "Dana"},
                "insights": [{"id": "IN1", "answer": "one"}],
                "distractors": [{"id": "DI1", "answer": "planted"}],
            }
        )
    )
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    monkeypatch.setattr(judge, "_CASE_PATH", case)
    monkeypatch.setattr(judge, "_REPORT_PATH", report)
    monkeypatch.setattr(judge, "_split_report", lambda _t, _m: [{"claim": "c", "citations": []}])
    # Perfect recall, but the report also swallowed the distractor.
    monkeypatch.setattr(judge, "_score_qa_set", lambda _c, entries, _m: (1.0, list(entries)))
    monkeypatch.setattr(judge, "_report_quality", lambda *_a: (1.0, {}))
    monkeypatch.setattr(judge, "_factuality", lambda *_a: (1.0, {}))

    rewards, breakdown = judge._grade()
    assert rewards["distractor_recall"] == 1.0
    assert rewards["distractor_avoidance"] == 0.0
    assert "distractor_recall" not in breakdown["components"]
    assert breakdown["components"]["distractor_avoidance"] == 0.0
    # Full marks everywhere else cannot hide a fully-swallowed distractor set.
    assert rewards["reward"] < 0.05


def test_main_writes_named_rewards_with_the_headline_under_reward(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`reward` is the key the deepagents aggregation reads for pass@k / avg@k."""
    reward_json = tmp_path / "logs" / "verifier" / "reward.json"
    breakdown_path = tmp_path / "logs" / "verifier" / "drbench_metrics.json"
    monkeypatch.setattr(judge, "_REWARD_JSON_PATH", reward_json)
    monkeypatch.setattr(judge, "_BREAKDOWN_PATH", breakdown_path)
    monkeypatch.setattr(
        judge,
        "_grade",
        lambda: (
            {
                "reward": 0.5,
                "insights_recall": 0.6,
                "distractor_recall": 0.2,
                "distractor_avoidance": 0.8,
                "factuality": 0.4,
                "report_quality": 0.7,
            },
            {"task_id": "DR0001"},
        ),
    )
    judge.main()

    written = json.loads(reward_json.read_text())
    assert written["reward"] == 0.5
    assert set(written) == {"reward", *judge._METRIC_NAMES}
    assert json.loads(breakdown_path.read_text())["task_id"] == "DR0001"


def test_main_still_writes_a_reward_when_grading_crashes(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reward_json = tmp_path / "logs" / "verifier" / "reward.json"
    monkeypatch.setattr(judge, "_REWARD_JSON_PATH", reward_json)
    monkeypatch.setattr(judge, "_BREAKDOWN_PATH", tmp_path / "logs" / "verifier" / "b.json")

    def boom() -> None:
        msg = "app stack unreachable"
        raise RuntimeError(msg)

    monkeypatch.setattr(judge, "_grade", boom)
    judge.main()
    assert json.loads(reward_json.read_text())["reward"] == 0.0


def test_webdav_index_parses_a_realistic_nextcloud_response(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Namespaced hrefs, directory entries, and percent-encoded names."""
    body = (
        '<?xml version="1.0"?>\n'
        '<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">\n'
        " <d:response><d:href>/remote.php/dav/files/emily.patel/</d:href></d:response>\n"
        " <d:response><d:href>/remote.php/dav/files/emily.patel/shared/</d:href></d:response>\n"
        " <d:response><d:href>/remote.php/dav/files/emily.patel/shared/food-safety.pdf"
        "</d:href></d:response>\n"
        " <d:response><d:href>/remote.php/dav/files/emily.patel/shared/Q3%20report.docx"
        "</d:href></d:response>\n"
        "</d:multistatus>"
    )

    class FakeResponse:
        def read(self) -> bytes:
            return body.encode()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr(judge.urllib.request, "urlopen", lambda _req, **_kwargs: FakeResponse())
    index = judge._webdav_index(
        "http://drbench:8081", {"username": "emily.patel", "password": "pw"}
    )

    # Collections are not documents and must not be indexed.
    assert sorted(index) == ["food-safety.pdf", "q3 report.docx"]
    # A citation names the decoded basename, but the fetch URL must stay encoded.
    assert index["q3 report.docx"].endswith("/shared/Q3%20report.docx")
    assert index["food-safety.pdf"].startswith("http://drbench:8081/remote.php/dav/")
