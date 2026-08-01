"""Tests for the DRBench verifier templates (`judge.py`, `extract_text.py`).

The templates run inside the Harbor sandbox with no `deepagents_evals` on the path, so
they are loaded here by file path rather than imported as package modules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
    prompt = judge._INSIGHT_SCORING_PROMPT.format(
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
    return judge._score_insight("Insight 1: x", "gold", "gpt-4o")["score"]


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
    result = judge._score_insight("Insight 1: x", "gold", "gpt-4o")
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

    reward, breakdown = judge._grade()
    assert reward == pytest.approx(2 / 3)
    assert breakdown["insight_count"] == 3
    assert [entry["id"] for entry in breakdown["per_insight"]] == ["IN1", "IN2", "EX1"]


def test_grade_scores_zero_without_a_report(
    judge: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"task_id": "DR0001", "insights": [{"id": "IN1", "answer": "one"}]}))
    monkeypatch.setattr(judge, "_CASE_PATH", case)
    monkeypatch.setattr(judge, "_REPORT_PATH", tmp_path / "absent.md")
    reward, breakdown = judge._grade()
    assert reward == 0.0
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
